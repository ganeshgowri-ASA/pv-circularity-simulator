# PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🚀 Features

### Financial Analysis Dashboard (✅ Production-Ready)

Comprehensive financial modeling and analysis for PV systems with integrated circular economy considerations.

**Core Capabilities:**
- 💰 **LCOE Calculator** - Complete levelized cost of energy analysis with circularity impact
- 📈 **Cash Flow Visualization** - Interactive Plotly charts for financial projections
- 🎯 **Sensitivity Analysis** - Multi-dimensional risk assessment (tornado diagrams, Monte Carlo)
- 📄 **Report Generation** - Professional reports in PDF, Excel, HTML, and CSV formats
- ♻️ **Circularity Integration** - 3R approach (Reduce, Reuse, Recycle) value quantification

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

For detailed documentation, see [FINANCIAL_DASHBOARD_DOCUMENTATION.md](FINANCIAL_DASHBOARD_DOCUMENTATION.md)

## 📁 Project Structure

```
pv-circularity-simulator/
├── src/
│   └── financial/              # Financial analysis module (v1.0.0)
│       ├── models/            # Data models
│       ├── calculators/       # LCOE & sensitivity analysis
│       ├── visualization/     # Plotly charts
│       ├── reporting/         # Multi-format report generation
│       └── dashboard/         # Streamlit UI
├── tests/                     # Test suite
├── app.py                     # Main dashboard entry point
├── requirements.txt           # Python dependencies
└── FINANCIAL_DASHBOARD_DOCUMENTATION.md  # Detailed documentation
```

## 🛠️ Technology Stack

- **Python 3.8+** - Core language
- **NumPy & Pandas** - Numerical computing and data analysis
- **Plotly** - Interactive visualizations
- **Streamlit** - Dashboard framework
- **ReportLab** - PDF generation
- **pvlib** - PV system modeling
- **SciPy** - Scientific computing

## 📊 Financial Dashboard Components

### 1. LCOE Calculator
- Complete cost structure modeling (CAPEX, OPEX, EOL)
- Revenue stream projections with degradation
- Circular economy value quantification
- Cost breakdown analysis

### 2. Cash Flow Visualization
- NPV (Net Present Value)
- IRR (Internal Rate of Return)
- Payback period analysis
- ROI calculations
- Interactive waterfall charts

### 3. Sensitivity Analysis
- One-way sensitivity plots
- Tornado diagrams (multi-parameter impact)
- Two-way sensitivity heatmaps
- Monte Carlo simulation (probabilistic risk analysis)

### 4. Report Generator
- **PDF** - Executive summaries with charts
- **Excel** - Detailed multi-sheet workbooks
- **HTML** - Interactive web reports
- **CSV** - Raw data export

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src/financial --cov-report=html
```

## 📖 Documentation

Comprehensive documentation available:
- **[Financial Dashboard Documentation](FINANCIAL_DASHBOARD_DOCUMENTATION.md)** - Complete guide with examples
- **Inline Documentation** - All functions include detailed docstrings
- **Example Code** - See `tests/` directory for usage examples

## 🎯 Use Cases

- **Investment Analysis** - Evaluate PV project financial viability
- **Risk Assessment** - Quantify uncertainty and parameter sensitivity
- **Circular Economy** - Measure 3R benefits and EOL value recovery
- **Stakeholder Reports** - Generate professional multi-format reports
- **Academic Research** - Study PV economics and circularity impacts

## 🔄 Circular Economy Integration

The financial models explicitly quantify circular economy benefits:
- **Material Recovery** - End-of-life material value recovery
- **Refurbishment** - Component reuse and value retention
- **Recycling Economics** - Revenue vs disposal cost analysis
- **Circularity Score** - Comprehensive 0-100 rating system

## 📈 Roadmap

### Current (v1.0.0)
- ✅ Financial Analysis Dashboard
- ✅ LCOE Calculator with circularity
- ✅ Sensitivity Analysis Suite
- ✅ Multi-format Report Generation

### Upcoming
- 🔲 Cell design simulation
- 🔲 Module engineering (CTM loss analysis)
- 🔲 System planning & optimization
- 🔲 Performance monitoring & forecasting
- 🔲 SCAPS integration
- 🔲 Reliability testing

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! This is a production-ready module with:
- Comprehensive test coverage
- Full documentation
- Clean architecture
- Industry best practices

## 📧 Support

For questions or issues related to the Financial Dashboard:
1. Review the [documentation](FINANCIAL_DASHBOARD_DOCUMENTATION.md)
2. Check example code in `tests/`
3. Run the interactive dashboard for hands-on exploration

---

**Status:** Financial Analysis Dashboard - Production Ready ✅
