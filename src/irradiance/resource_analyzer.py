"""Solar resource statistical analysis and assessment."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .models import POAComponents, ResourceStatistics


class SolarResourceAnalyzer:
    """Statistical analysis of solar resource data.

    Provides comprehensive analysis methods for solar resource assessment:
    - Monthly and seasonal patterns
    - Interannual variability
    - P50/P90 exceedance probability analysis
    - Resource mapping and visualization data preparation

    Used for site assessment, energy yield prediction, and financial modeling.
    """

    def __init__(self, data: pd.Series, data_label: str = "Irradiance"):
        """Initialize the solar resource analyzer.

        Args:
            data: Time series data with DatetimeIndex (e.g., GHI, POA)
            data_label: Label for the data (e.g., "GHI", "POA Global")
        """
        self.data = data
        self.data_label = data_label

        # Validate data
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex")

    def monthly_averages(
        self, aggregation: str = "mean", normalize: bool = False
    ) -> pd.DataFrame:
        """Calculate monthly statistics.

        Args:
            aggregation: Aggregation method ('mean', 'sum', 'median', 'std')
            normalize: Normalize by number of hours in each month

        Returns:
            DataFrame with monthly statistics

        Example:
            >>> analyzer = SolarResourceAnalyzer(poa_global)
            >>> monthly_stats = analyzer.monthly_averages(aggregation='sum')
            >>> print(monthly_stats)
                  Mean    Std   Min    Max
            Jan  150.2   25.3  98.5  185.3
            ...
        """
        # Group by month
        monthly = self.data.groupby(self.data.index.month)

        if aggregation == "mean":
            result = monthly.mean()
        elif aggregation == "sum":
            result = monthly.sum()
            if normalize:
                # Normalize by average month length
                days_in_month = self.data.groupby(self.data.index.month).size() / 24
                result = result / days_in_month
        elif aggregation == "median":
            result = monthly.median()
        elif aggregation == "std":
            result = monthly.std()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        # Create comprehensive statistics
        stats_df = pd.DataFrame(
            {
                "Mean": monthly.mean(),
                "Median": monthly.median(),
                "Std": monthly.std(),
                "Min": monthly.min(),
                "Max": monthly.max(),
                "P10": monthly.quantile(0.10),
                "P90": monthly.quantile(0.90),
                "Count": monthly.count(),
            }
        )

        # Add month names
        stats_df.index = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ][: len(stats_df)]

        return stats_df

    def seasonal_patterns(self, hemishere: str = "north") -> pd.DataFrame:
        """Analyze seasonal patterns in solar resource.

        Args:
            hemishere: 'north' or 'south' hemisphere

        Returns:
            DataFrame with seasonal statistics

        Example:
            >>> seasonal = analyzer.seasonal_patterns()
            >>> print(seasonal.loc['Summer', 'Mean'])
        """
        # Define seasons based on hemisphere
        if hemishere == "north":
            seasons = {
                "Winter": [12, 1, 2],
                "Spring": [3, 4, 5],
                "Summer": [6, 7, 8],
                "Fall": [9, 10, 11],
            }
        else:  # southern hemisphere
            seasons = {
                "Summer": [12, 1, 2],
                "Fall": [3, 4, 5],
                "Winter": [6, 7, 8],
                "Spring": [9, 10, 11],
            }

        seasonal_stats = {}

        for season_name, months in seasons.items():
            season_data = self.data[self.data.index.month.isin(months)]

            seasonal_stats[season_name] = {
                "Mean": season_data.mean(),
                "Median": season_data.median(),
                "Std": season_data.std(),
                "Min": season_data.min(),
                "Max": season_data.max(),
                "P10": season_data.quantile(0.10),
                "P90": season_data.quantile(0.90),
                "Total": season_data.sum(),
                "Count": len(season_data),
            }

        return pd.DataFrame(seasonal_stats).T

    def interannual_variability(self) -> pd.DataFrame:
        """Analyze year-to-year variability in solar resource.

        Returns:
            DataFrame with annual statistics and variability metrics

        Example:
            >>> annual_var = analyzer.interannual_variability()
            >>> print(f"CV: {annual_var.loc['CoeffVariation', 'Value']:.2%}")
        """
        # Group by year
        annual = self.data.groupby(self.data.index.year)

        annual_totals = annual.sum()
        annual_means = annual.mean()

        # Calculate variability metrics
        variability_stats = {
            "Mean Annual Total": annual_totals.mean(),
            "Std Annual Total": annual_totals.std(),
            "Min Annual Total": annual_totals.min(),
            "Max Annual Total": annual_totals.max(),
            "CoeffVariation": annual_totals.std() / annual_totals.mean(),
            "Mean Daily Average": annual_means.mean(),
            "Std Daily Average": annual_means.std(),
        }

        # Create summary DataFrame
        summary = pd.DataFrame(
            {"Metric": list(variability_stats.keys()), "Value": list(variability_stats.values())}
        )

        # Add year-by-year details
        yearly_detail = pd.DataFrame(
            {
                "Annual Total": annual_totals,
                "Daily Mean": annual_means,
                "Daily Std": annual.std(),
                "Days": annual.count() / 24,  # Assuming hourly data
            }
        )

        return summary, yearly_detail

    def p50_p90_analysis(
        self, time_aggregation: str = "monthly", confidence_level: float = 0.90
    ) -> Dict[str, pd.DataFrame]:
        """Calculate P50 and P90 exceedance values for resource assessment.

        P50 is the median value (50% probability of exceedance)
        P90 represents a conservative estimate (90% probability of exceedance)

        These metrics are critical for financial modeling and risk assessment.

        Args:
            time_aggregation: Aggregation period ('daily', 'monthly', 'annual')
            confidence_level: Confidence level for P90 calculation

        Returns:
            Dictionary with P-value statistics and distribution parameters

        Example:
            >>> p_analysis = analyzer.p50_p90_analysis(time_aggregation='monthly')
            >>> p90_value = p_analysis['summary'].loc['P90', 'Value']
            >>> print(f"P90 estimate: {p90_value:.1f} kWh/m²/month")
        """
        # Aggregate data based on time period
        if time_aggregation == "daily":
            aggregated = self.data.resample("D").sum()
        elif time_aggregation == "monthly":
            aggregated = self.data.resample("M").sum()
        elif time_aggregation == "annual":
            aggregated = self.data.resample("Y").sum()
        else:
            raise ValueError(f"Unknown aggregation: {time_aggregation}")

        # Calculate percentiles
        p_values = {
            "P01": aggregated.quantile(0.01),
            "P10": aggregated.quantile(0.10),
            "P50": aggregated.quantile(0.50),  # Median
            "P90": aggregated.quantile(0.90),
            "P99": aggregated.quantile(0.99),
        }

        # Calculate statistical parameters
        mean = aggregated.mean()
        std = aggregated.std()

        # Fit normal distribution for comparison
        normal_params = stats.norm.fit(aggregated)

        # Calculate exceedance probabilities
        summary = pd.DataFrame(
            {
                "Percentile": list(p_values.keys()),
                "Value": list(p_values.values()),
                "Exceedance Probability": [0.99, 0.90, 0.50, 0.10, 0.01],
            }
        )

        # Add context metrics
        summary["Deviation from Mean (%)"] = (
            (summary["Value"] - mean) / mean * 100
        )

        # Distribution parameters
        distribution_params = pd.DataFrame(
            {
                "Parameter": ["Mean", "Median", "Std", "CV", "Skewness", "Kurtosis"],
                "Value": [
                    mean,
                    aggregated.median(),
                    std,
                    std / mean,
                    stats.skew(aggregated),
                    stats.kurtosis(aggregated),
                ],
            }
        )

        return {
            "summary": summary,
            "distribution": distribution_params,
            "time_series": aggregated,
        }

    def solar_resource_maps(
        self, grid_resolution: str = "1h"
    ) -> Dict[str, pd.DataFrame]:
        """Prepare data for solar resource heat maps and visualizations.

        Creates matrices suitable for heat map visualization showing
        diurnal and seasonal patterns.

        Args:
            grid_resolution: Time resolution for hourly grid ('1h', '30min', '15min')

        Returns:
            Dictionary with various map matrices

        Example:
            >>> maps = analyzer.solar_resource_maps()
            >>> hourly_by_month = maps['hourly_by_month']
            >>> # Use this matrix for heat map visualization
        """
        # Create hour-of-day by month matrix
        hourly_by_month = self.data.groupby(
            [self.data.index.month, self.data.index.hour]
        ).mean().unstack(level=0)

        # Month names for columns
        month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        if len(hourly_by_month.columns) <= 12:
            hourly_by_month.columns = month_names[: len(hourly_by_month.columns)]

        # Create day-of-year by hour matrix (for annual profile)
        daily_hourly = self.data.groupby(
            [self.data.index.dayofyear, self.data.index.hour]
        ).mean().unstack(level=1)

        # Create day-of-week by hour matrix
        dow_hourly = self.data.groupby(
            [self.data.index.dayofweek, self.data.index.hour]
        ).mean().unstack(level=0)

        dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if len(dow_hourly.columns) <= 7:
            dow_hourly.columns = dow_names[: len(dow_hourly.columns)]

        # Calculate availability matrix (% of time above threshold)
        threshold = self.data.quantile(0.25)  # 25th percentile as threshold
        availability = (
            (self.data > threshold)
            .groupby([self.data.index.month, self.data.index.hour])
            .mean()
            .unstack(level=0)
        ) * 100

        if len(availability.columns) <= 12:
            availability.columns = month_names[: len(availability.columns)]

        return {
            "hourly_by_month": hourly_by_month,
            "daily_hourly": daily_hourly,
            "dow_hourly": dow_hourly,
            "availability": availability,
        }

    def generate_resource_summary(self) -> ResourceStatistics:
        """Generate comprehensive resource statistics summary.

        Returns:
            ResourceStatistics object with key metrics

        Example:
            >>> summary = analyzer.generate_resource_summary()
            >>> print(f"P90: {summary.p90:.1f}, CV: {summary.coefficient_of_variation:.3f}")
        """
        return ResourceStatistics.from_series(self.data)

    def calculate_capacity_factor_range(
        self,
        system_capacity_kw: float,
        module_efficiency: float = 0.20,
        performance_ratio: float = 0.85,
    ) -> Dict[str, float]:
        """Estimate capacity factor range based on solar resource.

        Args:
            system_capacity_kw: System capacity in kW
            module_efficiency: Module efficiency (0-1)
            performance_ratio: System performance ratio (0-1)

        Returns:
            Dictionary with P50 and P90 capacity factors

        Example:
            >>> cf_range = analyzer.calculate_capacity_factor_range(100.0)
            >>> print(f"Expected CF: {cf_range['p50']:.1%}, Conservative: {cf_range['p90']:.1%}")
        """
        # Convert irradiance to energy
        # Assuming data is in W/m² and hourly
        p_analysis = self.p50_p90_analysis(time_aggregation="annual")

        # Standard test conditions: 1000 W/m²
        stc_irradiance = 1000.0

        # Calculate energy yield
        p50_irradiance = p_analysis["summary"].loc[
            p_analysis["summary"]["Percentile"] == "P50", "Value"
        ].values[0]
        p90_irradiance = p_analysis["summary"].loc[
            p_analysis["summary"]["Percentile"] == "P90", "Value"
        ].values[0]

        # Energy = Irradiance * Area * Efficiency * PR
        # For capacity factor: Energy / (Capacity * 8760 hours)
        # Area = Capacity / (STC_Irradiance * Efficiency)

        hours_per_year = 8760

        # Simplified capacity factor calculation
        cf_p50 = (p50_irradiance * performance_ratio) / (stc_irradiance * hours_per_year)
        cf_p90 = (p90_irradiance * performance_ratio) / (stc_irradiance * hours_per_year)

        return {
            "p50": cf_p50,
            "p90": cf_p90,
            "p50_annual_kwh": system_capacity_kw * cf_p50 * hours_per_year,
            "p90_annual_kwh": system_capacity_kw * cf_p90 * hours_per_year,
        }

    def identify_resource_anomalies(
        self, threshold_std: float = 3.0
    ) -> pd.DataFrame:
        """Identify anomalous periods in solar resource data.

        Uses statistical methods to detect unusual patterns that may
        indicate data quality issues or extreme weather events.

        Args:
            threshold_std: Number of standard deviations for anomaly detection

        Returns:
            DataFrame with anomalous periods and their characteristics

        Example:
            >>> anomalies = analyzer.identify_resource_anomalies(threshold_std=2.5)
            >>> print(f"Found {len(anomalies)} anomalous periods")
        """
        # Calculate rolling statistics
        rolling_mean = self.data.rolling(window=24 * 7, center=True).mean()
        rolling_std = self.data.rolling(window=24 * 7, center=True).std()

        # Identify anomalies
        z_scores = (self.data - rolling_mean) / rolling_std
        anomalies = self.data[np.abs(z_scores) > threshold_std]

        if len(anomalies) == 0:
            return pd.DataFrame()

        # Create anomaly report
        anomaly_report = pd.DataFrame(
            {
                "Timestamp": anomalies.index,
                "Value": anomalies.values,
                "Z-Score": z_scores[anomalies.index].values,
                "Rolling Mean": rolling_mean[anomalies.index].values,
                "Deviation": (anomalies.values - rolling_mean[anomalies.index].values),
            }
        )

        return anomaly_report.reset_index(drop=True)
