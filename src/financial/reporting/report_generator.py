"""
Financial Report Generator for PV Systems.

Generates comprehensive financial reports in multiple formats (PDF, Excel, HTML)
including executive summaries, detailed analyses, and visualizations.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
import io

import pandas as pd
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    PageBreak,
    Image as RLImage,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
import plotly.graph_objects as go

from ..models.financial_models import CashFlowModel, CircularityMetrics
from ..calculators.lcoe_calculator import LCOEResult, LCOECalculator
from ..calculators.sensitivity_analysis import SensitivityAnalyzer, SensitivityMetric
from ..visualization.charts import FinancialChartBuilder


class FinancialReportGenerator:
    """
    Comprehensive financial report generator.

    Generates professional reports with financial analysis, visualizations,
    and export capabilities to PDF, Excel, and HTML formats.
    """

    def __init__(
        self,
        project_name: str = "PV System Financial Analysis",
        company_name: str = "PV Circularity Simulator",
        analyst_name: Optional[str] = None,
    ):
        """
        Initialize report generator.

        Args:
            project_name: Name of the project
            company_name: Company or organization name
            analyst_name: Name of analyst preparing the report
        """
        self.project_name = project_name
        self.company_name = company_name
        self.analyst_name = analyst_name or "Financial Analysis Team"
        self.chart_builder = FinancialChartBuilder()

    def generate_executive_summary_pdf(
        self,
        cash_flow_model: CashFlowModel,
        lcoe_result: LCOEResult,
        output_path: Union[str, Path],
        include_charts: bool = True,
    ) -> Path:
        """
        Generate executive summary PDF report.

        Args:
            cash_flow_model: Cash flow model
            lcoe_result: LCOE calculation result
            output_path: Path to save PDF file
            include_charts: Include visualization charts

        Returns:
            Path to generated PDF file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create PDF document
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=0.75 * inch,
            leftMargin=0.75 * inch,
            topMargin=1 * inch,
            bottomMargin=0.75 * inch,
        )

        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()

        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1976D2'),
            spaceAfter=30,
            alignment=TA_CENTER,
        )

        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#2E7D32'),
            spaceAfter=12,
        )

        # Title Page
        elements.append(Paragraph(self.project_name, title_style))
        elements.append(Paragraph("Executive Financial Summary", styles['Heading2']))
        elements.append(Spacer(1, 0.2 * inch))

        # Report metadata
        metadata_text = f"""
        <b>Prepared by:</b> {self.analyst_name}<br/>
        <b>Organization:</b> {self.company_name}<br/>
        <b>Report Date:</b> {datetime.now().strftime('%B %d, %Y')}<br/>
        """
        elements.append(Paragraph(metadata_text, styles['Normal']))
        elements.append(Spacer(1, 0.5 * inch))

        # Key Financial Metrics
        elements.append(Paragraph("Key Financial Metrics", heading_style))

        npv = cash_flow_model.calculate_npv()
        irr = cash_flow_model.calculate_irr()
        payback = cash_flow_model.calculate_payback_period()
        roi = cash_flow_model.calculate_roi()

        metrics_data = [
            ['Metric', 'Value', 'Description'],
            ['LCOE', f'${lcoe_result.lcoe:.4f}/kWh', 'Levelized Cost of Energy'],
            ['NPV', f'${npv:,.2f}', 'Net Present Value'],
            ['IRR', f'{irr*100:.2f}%', 'Internal Rate of Return'],
            ['Payback Period', f'{payback:.1f} years', 'Simple Payback Period'],
            ['ROI', f'{roi:.1f}%', 'Return on Investment'],
            ['Total Investment', f'${cash_flow_model.cost_structure.get_total_capex():,.2f}',
             'Initial Capital Investment'],
        ]

        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 3*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1976D2')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))

        elements.append(metrics_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Circularity Impact
        elements.append(Paragraph("Circular Economy Impact", heading_style))

        circ_score = cash_flow_model.circularity_metrics.get_circularity_score()
        eol_value = cash_flow_model.circularity_metrics.get_eol_recovery_value(
            cash_flow_model.cost_structure.equipment_cost
        )

        circularity_text = f"""
        <b>Circularity Score:</b> {circ_score:.1f}/100<br/>
        <b>LCOE Benefit:</b> ${lcoe_result.circularity_benefit:.4f}/kWh
        ({lcoe_result.circularity_benefit/lcoe_result.without_circularity*100:.1f}% reduction)<br/>
        <b>End-of-Life Recovery Value:</b> ${eol_value:,.2f}<br/>
        <b>Material Recovery Rate:</b> {cash_flow_model.circularity_metrics.material_recovery_rate*100:.1f}%<br/>
        """
        elements.append(Paragraph(circularity_text, styles['Normal']))
        elements.append(Spacer(1, 0.3 * inch))

        # Cost Breakdown
        elements.append(Paragraph("Cost Breakdown", heading_style))

        breakdown_data = [['Cost Category', 'Amount ($)', 'Percentage']]
        total_cost = sum(v for v in lcoe_result.cost_breakdown.values() if v > 0)

        for category, amount in lcoe_result.cost_breakdown.items():
            if amount > 0:
                percentage = (amount / total_cost * 100) if total_cost > 0 else 0
                breakdown_data.append([
                    category,
                    f'${amount:,.2f}',
                    f'{percentage:.1f}%'
                ])

        breakdown_data.append([
            'Total',
            f'${total_cost:,.2f}',
            '100.0%'
        ])

        breakdown_table = Table(breakdown_data, colWidths=[3*inch, 2*inch, 1.5*inch])
        breakdown_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E7D32')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTNAME', (0, 1), (-1, -2), 'Helvetica'),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.lightgrey]),
            ('LINEABOVE', (0, -1), (-1, -1), 2, colors.black),
        ]))

        elements.append(breakdown_table)
        elements.append(Spacer(1, 0.3 * inch))

        # Add charts if requested
        if include_charts:
            elements.append(PageBreak())
            elements.append(Paragraph("Financial Visualizations", heading_style))
            elements.append(Spacer(1, 0.2 * inch))

            # Cash flow chart
            cf_chart = self.chart_builder.create_cash_flow_waterfall(cash_flow_model)
            chart_img = self._plotly_to_image(cf_chart)
            if chart_img:
                elements.append(chart_img)
                elements.append(Spacer(1, 0.2 * inch))

            # LCOE breakdown chart
            lcoe_chart = self.chart_builder.create_lcoe_breakdown_pie(lcoe_result)
            chart_img = self._plotly_to_image(lcoe_chart)
            if chart_img:
                elements.append(chart_img)

        # Build PDF
        doc.build(elements)

        return output_path

    def generate_detailed_excel_report(
        self,
        cash_flow_model: CashFlowModel,
        lcoe_result: LCOEResult,
        output_path: Union[str, Path],
        include_sensitivity: bool = True,
    ) -> Path:
        """
        Generate detailed Excel report with multiple sheets.

        Args:
            cash_flow_model: Cash flow model
            lcoe_result: LCOE calculation result
            output_path: Path to save Excel file
            include_sensitivity: Include sensitivity analysis sheet

        Returns:
            Path to generated Excel file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book

            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'bg_color': '#1976D2',
                'font_color': 'white',
                'border': 1,
            })

            currency_format = workbook.add_format({
                'num_format': '$#,##0.00',
                'border': 1,
            })

            percent_format = workbook.add_format({
                'num_format': '0.00%',
                'border': 1,
            })

            number_format = workbook.add_format({
                'num_format': '#,##0.00',
                'border': 1,
            })

            # Sheet 1: Executive Summary
            summary_data = {
                'Metric': [
                    'LCOE ($/kWh)',
                    'LCOE Real ($/kWh)',
                    'NPV ($)',
                    'IRR (%)',
                    'Payback Period (years)',
                    'ROI (%)',
                    'Total Investment ($)',
                    'Total Lifetime Cost ($)',
                    'Total Lifetime Energy (kWh)',
                    'Circularity Score',
                    'Circularity LCOE Benefit ($/kWh)',
                ],
                'Value': [
                    lcoe_result.lcoe,
                    lcoe_result.lcoe_real,
                    cash_flow_model.calculate_npv(),
                    cash_flow_model.calculate_irr() * 100,
                    cash_flow_model.calculate_payback_period(),
                    cash_flow_model.calculate_roi(),
                    cash_flow_model.cost_structure.get_total_capex(),
                    lcoe_result.total_lifetime_cost,
                    lcoe_result.total_lifetime_energy,
                    cash_flow_model.circularity_metrics.get_circularity_score(),
                    lcoe_result.circularity_benefit,
                ]
            }

            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Executive Summary', index=False)

            # Sheet 2: Cash Flow Analysis
            cash_flow_df = cash_flow_model.generate_cash_flow_series()
            cash_flow_df.to_excel(writer, sheet_name='Cash Flow', index=False)

            # Format cash flow sheet
            worksheet = writer.sheets['Cash Flow']
            worksheet.set_column('B:G', 15, currency_format)

            # Sheet 3: Cost Breakdown
            breakdown_df = pd.DataFrame([
                {'Category': k, 'Amount': v, 'Percentage': v / sum(lcoe_result.cost_breakdown.values()) * 100}
                for k, v in lcoe_result.cost_breakdown.items()
                if v > 0
            ])
            breakdown_df.to_excel(writer, sheet_name='Cost Breakdown', index=False)

            # Sheet 4: Circularity Metrics
            circ_data = {
                'Metric': [
                    'Material Recovery Rate (%)',
                    'System Weight (kg)',
                    'Refurbishment Potential (%)',
                    'Refurbishment Value Retention (%)',
                    'EOL Recovery Value ($)',
                    'Circularity Score',
                ],
                'Value': [
                    cash_flow_model.circularity_metrics.material_recovery_rate * 100,
                    cash_flow_model.circularity_metrics.system_weight,
                    cash_flow_model.circularity_metrics.refurbishment_potential * 100,
                    cash_flow_model.circularity_metrics.refurbishment_value * 100,
                    cash_flow_model.circularity_metrics.get_eol_recovery_value(
                        cash_flow_model.cost_structure.equipment_cost
                    ),
                    cash_flow_model.circularity_metrics.get_circularity_score(),
                ]
            }
            circ_df = pd.DataFrame(circ_data)
            circ_df.to_excel(writer, sheet_name='Circularity', index=False)

            # Sheet 5: Sensitivity Analysis (if requested)
            if include_sensitivity:
                analyzer = SensitivityAnalyzer(
                    base_cost_structure=cash_flow_model.cost_structure,
                    base_revenue_stream=cash_flow_model.revenue_stream,
                    base_circularity_metrics=cash_flow_model.circularity_metrics,
                    lifetime_years=cash_flow_model.lifetime_years,
                    discount_rate=cash_flow_model.discount_rate,
                )

                # Sample sensitivity parameters
                from ..models.financial_models import SensitivityParameter

                discount_param = SensitivityParameter(
                    name='discount_rate',
                    base_value=cash_flow_model.discount_rate,
                    min_value=0.03,
                    max_value=0.10,
                    step=0.01,
                    unit='%'
                )

                sens_result = analyzer.one_way_sensitivity(
                    discount_param,
                    SensitivityMetric.LCOE
                )

                sens_df = sens_result.to_dataframe()
                sens_df.to_excel(writer, sheet_name='Sensitivity Analysis', index=False)

        return output_path

    def generate_html_report(
        self,
        cash_flow_model: CashFlowModel,
        lcoe_result: LCOEResult,
        output_path: Union[str, Path],
    ) -> Path:
        """
        Generate interactive HTML report with embedded charts.

        Args:
            cash_flow_model: Cash flow model
            lcoe_result: LCOE calculation result
            output_path: Path to save HTML file

        Returns:
            Path to generated HTML file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate metrics
        npv = cash_flow_model.calculate_npv()
        irr = cash_flow_model.calculate_irr()
        payback = cash_flow_model.calculate_payback_period()
        roi = cash_flow_model.calculate_roi()

        # Generate charts
        cf_chart = self.chart_builder.create_cash_flow_waterfall(cash_flow_model)
        lcoe_chart = self.chart_builder.create_lcoe_breakdown_pie(lcoe_result)
        breakdown_chart = self.chart_builder.create_cash_flow_breakdown(cash_flow_model)

        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.project_name} - Financial Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #1976D2;
            border-bottom: 3px solid #1976D2;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2E7D32;
            margin-top: 30px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .chart-container {{
            margin: 30px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 12px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #1976D2;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{self.project_name}</h1>
        <p><strong>Prepared by:</strong> {self.analyst_name} |
           <strong>Organization:</strong> {self.company_name} |
           <strong>Date:</strong> {datetime.now().strftime('%B %d, %Y')}</p>

        <h2>Key Financial Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">LCOE</div>
                <div class="metric-value">${lcoe_result.lcoe:.4f}/kWh</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Net Present Value</div>
                <div class="metric-value">${npv:,.0f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Internal Rate of Return</div>
                <div class="metric-value">{irr*100:.2f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Payback Period</div>
                <div class="metric-value">{payback:.1f} years</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Return on Investment</div>
                <div class="metric-value">{roi:.1f}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Circularity Score</div>
                <div class="metric-value">{cash_flow_model.circularity_metrics.get_circularity_score():.1f}/100</div>
            </div>
        </div>

        <h2>Cash Flow Analysis</h2>
        <div class="chart-container" id="cash-flow-chart"></div>

        <h2>LCOE Cost Breakdown</h2>
        <div class="chart-container" id="lcoe-chart"></div>

        <h2>Revenue vs Costs</h2>
        <div class="chart-container" id="breakdown-chart"></div>

        <h2>Cost Details</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Amount</th>
                    <th>Percentage</th>
                </tr>
            </thead>
            <tbody>
"""

        # Add cost breakdown rows
        total_cost = sum(v for v in lcoe_result.cost_breakdown.values() if v > 0)
        for category, amount in lcoe_result.cost_breakdown.items():
            if amount > 0:
                percentage = (amount / total_cost * 100) if total_cost > 0 else 0
                html_content += f"""
                <tr>
                    <td>{category}</td>
                    <td>${amount:,.2f}</td>
                    <td>{percentage:.1f}%</td>
                </tr>
"""

        html_content += f"""
            </tbody>
        </table>

        <div class="footer">
            <p>Generated by {self.company_name} Financial Analysis System</p>
        </div>
    </div>

    <script>
        // Cash Flow Chart
        var cashFlowData = {cf_chart.to_json()};
        Plotly.newPlot('cash-flow-chart', cashFlowData.data, cashFlowData.layout);

        // LCOE Chart
        var lcoeData = {lcoe_chart.to_json()};
        Plotly.newPlot('lcoe-chart', lcoeData.data, lcoeData.layout);

        // Breakdown Chart
        var breakdownData = {breakdown_chart.to_json()};
        Plotly.newPlot('breakdown-chart', breakdownData.data, breakdownData.layout);
    </script>
</body>
</html>
"""

        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return output_path

    def _plotly_to_image(
        self,
        fig: go.Figure,
        width: int = 600,
        height: int = 400,
    ) -> Optional[RLImage]:
        """
        Convert Plotly figure to ReportLab Image.

        Args:
            fig: Plotly figure
            width: Image width in pixels
            height: Image height in pixels

        Returns:
            ReportLab Image object or None if conversion fails
        """
        try:
            # Convert to image bytes
            img_bytes = fig.to_image(format="png", width=width, height=height)

            # Create ReportLab Image
            img = RLImage(io.BytesIO(img_bytes), width=width/1.5, height=height/1.5)
            return img
        except Exception as e:
            print(f"Warning: Could not convert chart to image: {e}")
            return None

    def export_data_to_csv(
        self,
        cash_flow_model: CashFlowModel,
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Export all financial data to CSV files.

        Args:
            cash_flow_model: Cash flow model
            output_dir: Directory to save CSV files

        Returns:
            Dictionary mapping data types to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = {}

        # Cash flow data
        cf_path = output_dir / "cash_flow.csv"
        cash_flow_df = cash_flow_model.generate_cash_flow_series()
        cash_flow_df.to_csv(cf_path, index=False)
        files['cash_flow'] = cf_path

        # Cost structure
        cost_path = output_dir / "cost_structure.csv"
        cost_df = pd.DataFrame([{
            'Category': k,
            'Amount': v
        } for k, v in cash_flow_model.cost_structure.get_cost_breakdown().items()])
        cost_df.to_csv(cost_path, index=False)
        files['cost_structure'] = cost_path

        # Circularity metrics
        circ_path = output_dir / "circularity_metrics.csv"
        circ_df = pd.DataFrame([{
            'Metric': 'Material Recovery Rate',
            'Value': cash_flow_model.circularity_metrics.material_recovery_rate
        }, {
            'Metric': 'Circularity Score',
            'Value': cash_flow_model.circularity_metrics.get_circularity_score()
        }])
        circ_df.to_csv(circ_path, index=False)
        files['circularity'] = circ_path

        return files
