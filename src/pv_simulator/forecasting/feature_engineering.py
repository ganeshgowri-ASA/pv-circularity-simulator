"""
Feature engineering for time series forecasting.

This module provides tools for creating features from time series data including
lag features, rolling statistics, temporal features, and weather-related features.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel

from pv_simulator.core.schemas import FeatureConfig, TimeSeriesData


class FeatureEngineering:
    """
    Feature engineering toolkit for time series forecasting.

    Provides methods to create various types of features from time series data
    to improve forecast accuracy with machine learning models.
    """

    def __init__(self, config: Optional[FeatureConfig] = None) -> None:
        """
        Initialize feature engineering with configuration.

        Args:
            config: Feature engineering configuration (uses defaults if None)
        """
        self.config = config or FeatureConfig()

    def lag_features(
        self, data: pd.DataFrame, target_col: str, lag_periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Create lag features for a target column.

        Lag features represent past values of the time series and are crucial
        for capturing temporal dependencies.

        Args:
            data: Input DataFrame
            target_col: Name of target column
            lag_periods: List of lag periods to create (uses config if None)

        Returns:
            DataFrame with lag features added

        Example:
            >>> fe = FeatureEngineering()
            >>> df_with_lags = fe.lag_features(df, 'energy_output', [1, 7, 30])
        """
        df = data.copy()
        periods = lag_periods or self.config.lag_periods

        for lag in periods:
            df[f"{target_col}_lag_{lag}"] = df[target_col].shift(lag)

        return df

    def rolling_features(
        self,
        data: pd.DataFrame,
        target_col: str,
        windows: Optional[List[int]] = None,
        functions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create rolling window features.

        Rolling features capture trends and patterns over specific time windows.

        Args:
            data: Input DataFrame
            target_col: Name of target column
            windows: Window sizes (uses config if None)
            functions: Aggregation functions ('mean', 'std', 'min', 'max')

        Returns:
            DataFrame with rolling features added

        Example:
            >>> fe = FeatureEngineering()
            >>> df_with_rolling = fe.rolling_features(
            ...     df, 'energy_output', windows=[7, 30], functions=['mean', 'std']
            ... )
        """
        df = data.copy()
        windows = windows or self.config.rolling_windows
        functions = functions or ["mean", "std", "min", "max"]

        for window in windows:
            for func in functions:
                col_name = f"{target_col}_rolling_{window}_{func}"

                if func == "mean":
                    df[col_name] = df[target_col].rolling(window=window).mean()
                elif func == "std":
                    df[col_name] = df[target_col].rolling(window=window).std()
                elif func == "min":
                    df[col_name] = df[target_col].rolling(window=window).min()
                elif func == "max":
                    df[col_name] = df[target_col].rolling(window=window).max()
                elif func == "median":
                    df[col_name] = df[target_col].rolling(window=window).median()

        return df

    def temporal_features(
        self, data: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Create temporal features from timestamp.

        Extracts various time-based features like hour, day of week, month, etc.
        These features help models learn seasonal patterns.

        Args:
            data: Input DataFrame with timestamp column
            timestamp_col: Name of timestamp column

        Returns:
            DataFrame with temporal features added

        Example:
            >>> fe = FeatureEngineering()
            >>> df_with_temporal = fe.temporal_features(df)
        """
        df = data.copy()

        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])

        dt = df[timestamp_col].dt

        # Basic temporal features
        df["hour"] = dt.hour
        df["day"] = dt.day
        df["day_of_week"] = dt.dayofweek
        df["day_of_year"] = dt.dayofyear
        df["week"] = dt.isocalendar().week
        df["month"] = dt.month
        df["quarter"] = dt.quarter
        df["year"] = dt.year

        # Binary features
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
        df["is_month_start"] = dt.is_month_start.astype(int)
        df["is_month_end"] = dt.is_month_end.astype(int)
        df["is_quarter_start"] = dt.is_quarter_start.astype(int)
        df["is_quarter_end"] = dt.is_quarter_end.astype(int)

        # Cyclical encoding if enabled
        if self.config.cyclical_encoding:
            df = self._add_cyclical_features(df)

        return df

    def _add_cyclical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add cyclical encoding for temporal features.

        Cyclical encoding preserves the circular nature of time features
        (e.g., hour 23 is close to hour 0).

        Args:
            data: DataFrame with temporal features

        Returns:
            DataFrame with cyclical features added
        """
        df = data.copy()

        # Hour (0-23)
        if "hour" in df.columns:
            df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week (0-6)
        if "day_of_week" in df.columns:
            df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
            df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Day of year (1-365)
        if "day_of_year" in df.columns:
            df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
            df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)

        # Month (1-12)
        if "month" in df.columns:
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def weather_features(
        self,
        data: pd.DataFrame,
        temperature_col: Optional[str] = None,
        irradiance_col: Optional[str] = None,
        wind_speed_col: Optional[str] = None,
        humidity_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create weather-related features for PV forecasting.

        Weather features are crucial for solar energy forecasting as they
        directly impact PV system performance.

        Args:
            data: Input DataFrame
            temperature_col: Column name for temperature
            irradiance_col: Column name for solar irradiance
            wind_speed_col: Column name for wind speed
            humidity_col: Column name for humidity

        Returns:
            DataFrame with weather features added

        Example:
            >>> fe = FeatureEngineering()
            >>> df_with_weather = fe.weather_features(
            ...     df, temperature_col='temp', irradiance_col='ghi'
            ... )
        """
        df = data.copy()

        # Temperature features
        if temperature_col and temperature_col in df.columns:
            # Temperature differences
            df[f"{temperature_col}_diff_1h"] = df[temperature_col].diff(1)
            df[f"{temperature_col}_diff_24h"] = df[temperature_col].diff(24)

            # Temperature rolling statistics
            df[f"{temperature_col}_rolling_24h_mean"] = (
                df[temperature_col].rolling(window=24).mean()
            )
            df[f"{temperature_col}_rolling_24h_std"] = (
                df[temperature_col].rolling(window=24).std()
            )

        # Irradiance features
        if irradiance_col and irradiance_col in df.columns:
            # Irradiance change rate
            df[f"{irradiance_col}_diff"] = df[irradiance_col].diff(1)
            df[f"{irradiance_col}_pct_change"] = df[irradiance_col].pct_change(1)

            # Clear sky index (ratio of actual to maximum expected irradiance)
            # Simplified - actual implementation would use solar position
            max_irradiance = df[irradiance_col].max()
            if max_irradiance > 0:
                df["clear_sky_index"] = df[irradiance_col] / max_irradiance

        # Wind speed features
        if wind_speed_col and wind_speed_col in df.columns:
            df[f"{wind_speed_col}_rolling_3h_mean"] = (
                df[wind_speed_col].rolling(window=3).mean()
            )

            # Wind cooling effect (simplified)
            if temperature_col and temperature_col in df.columns:
                df["wind_chill_effect"] = df[wind_speed_col] * (
                    25 - df[temperature_col]
                ).clip(lower=0)

        # Humidity features
        if humidity_col and humidity_col in df.columns:
            df[f"{humidity_col}_rolling_6h_mean"] = (
                df[humidity_col].rolling(window=6).mean()
            )

            # Dew point approximation (simplified)
            if temperature_col and temperature_col in df.columns:
                df["dew_point_approx"] = (
                    df[temperature_col]
                    - (100 - df[humidity_col]) / 5
                )

        return df

    def create_all_features(
        self,
        data: TimeSeriesData,
        target_col: str = "value",
        weather_data: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Create all features based on configuration.

        This is a convenience method that applies all enabled feature
        engineering steps in one call.

        Args:
            data: Time series data
            target_col: Name of target column
            weather_data: Optional weather data to merge

        Returns:
            DataFrame with all engineered features

        Example:
            >>> fe = FeatureEngineering()
            >>> features = fe.create_all_features(time_series_data)
        """
        # Convert TimeSeriesData to DataFrame
        df = pd.DataFrame(
            {
                "timestamp": data.timestamps,
                target_col: data.values,
            }
        )

        # Add temporal features
        if self.config.temporal_features:
            df = self.temporal_features(df, "timestamp")

        # Add lag features
        if self.config.lag_features:
            df = self.lag_features(df, target_col, self.config.lag_periods)

        # Add rolling features
        if self.config.rolling_features:
            df = self.rolling_features(df, target_col, self.config.rolling_windows)

        # Merge and add weather features if provided
        if self.config.weather_features and weather_data is not None:
            # Merge weather data on timestamp
            df = pd.merge(df, weather_data, on="timestamp", how="left")

            # Create weather features
            weather_cols = [
                col for col in weather_data.columns if col != "timestamp"
            ]
            if weather_cols:
                df = self.weather_features(
                    df,
                    temperature_col=next(
                        (c for c in weather_cols if "temp" in c.lower()), None
                    ),
                    irradiance_col=next(
                        (c for c in weather_cols if "irr" in c.lower() or "ghi" in c.lower()),
                        None,
                    ),
                    wind_speed_col=next(
                        (c for c in weather_cols if "wind" in c.lower()), None
                    ),
                    humidity_col=next(
                        (c for c in weather_cols if "hum" in c.lower()), None
                    ),
                )

        return df

    @staticmethod
    def select_features(
        data: pd.DataFrame,
        target_col: str,
        method: str = "correlation",
        top_k: int = 20,
    ) -> List[str]:
        """
        Select most important features.

        Args:
            data: DataFrame with features
            target_col: Target column name
            method: Selection method ('correlation', 'variance', 'all')
            top_k: Number of features to select

        Returns:
            List of selected feature names

        Example:
            >>> selected = FeatureEngineering.select_features(
            ...     df, 'energy_output', method='correlation', top_k=10
            ... )
        """
        # Remove rows with NaN values for correlation calculation
        df_clean = data.dropna()

        if target_col not in df_clean.columns:
            raise ValueError(f"Target column {target_col} not found")

        # Exclude non-numeric and target column
        feature_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)

        if method == "correlation":
            # Select features with highest absolute correlation to target
            correlations = df_clean[feature_cols].corrwith(df_clean[target_col]).abs()
            selected = correlations.nlargest(top_k).index.tolist()

        elif method == "variance":
            # Select features with highest variance
            variances = df_clean[feature_cols].var()
            selected = variances.nlargest(top_k).index.tolist()

        elif method == "all":
            selected = feature_cols[:top_k]

        else:
            raise ValueError(f"Unknown selection method: {method}")

        return selected

    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        strategy: str = "forward_fill",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in feature DataFrame.

        Args:
            data: DataFrame with potential missing values
            strategy: Strategy for handling missing values
                     ('forward_fill', 'backward_fill', 'interpolate', 'drop', 'mean')
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            DataFrame with missing values handled

        Example:
            >>> df_clean = FeatureEngineering.handle_missing_values(
            ...     df, strategy='interpolate'
            ... )
        """
        df = data.copy()

        if strategy == "forward_fill":
            df = df.fillna(method="ffill", limit=limit)
        elif strategy == "backward_fill":
            df = df.fillna(method="bfill", limit=limit)
        elif strategy == "interpolate":
            df = df.interpolate(method="linear", limit=limit)
        elif strategy == "drop":
            df = df.dropna()
        elif strategy == "mean":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return df

    @staticmethod
    def normalize_features(
        data: pd.DataFrame,
        method: str = "standard",
        exclude_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Normalize features for machine learning.

        Args:
            data: DataFrame with features
            method: Normalization method ('standard', 'minmax', 'robust')
            exclude_cols: Columns to exclude from normalization

        Returns:
            DataFrame with normalized features

        Example:
            >>> df_normalized = FeatureEngineering.normalize_features(
            ...     df, method='standard', exclude_cols=['timestamp']
            ... )
        """
        from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

        df = data.copy()
        exclude_cols = exclude_cols or []

        # Select numeric columns to normalize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

        if not cols_to_normalize:
            return df

        # Choose scaler
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Normalize
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

        return df
