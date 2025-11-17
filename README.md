# pv-circularity-simulator
End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## MVP Deployment Status ✅

**Live MVP Application Deployed!**

### Quick Start

```bash
# Clone the repository
git clone https://github.com/ganeshgowri-ASA/pv-circularity-simulator.git
cd pv-circularity-simulator

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

The app will open at: `http://localhost:8501`

### MVP Features Deployed
- ✅ Dashboard with system metrics
- ✅ Materials Database (50+ materials)
- ✅ Cell Design Module (SCAPS-1D)
- ✅ Module Design with CTM Loss Analysis (k1-k24 factors)
- ✅ System Design & Optimization
- ✅ Performance Monitoring (Real-time KPIs)
- ✅ Energy Yield Forecasting (7-day forecast)
- ✅ Fault Diagnostics (Defect detection)
- ✅ Circularity Assessment (3R: Reuse, Repair, Recycle)
- ✅ Financial Analysis (LCOE, NPV, IRR)

### Technical Stack
- **Frontend**: Streamlit 1.28+
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **Backend**: Python 3.9+
- **Validation**: Pydantic 2.0+

### 71 Claude Code IDE Sessions Integrated
- B01: Materials Engineering (5 sessions)
- B02: Cell Design (5 sessions)
- B03: Module Design & CTM (6 sessions)
- B04: IEC Testing (4 sessions)
- B05: System Design (6 sessions)
- B06: EYA & Weather (5 sessions)
- B07: Performance Monitoring (4 sessions)
- B08: Fault Diagnostics (5 sessions)
- B09: Energy Forecasting (5 sessions)
- B10: Revamp & Repower (4 sessions)
- B11: Circularity 3R (6 sessions)
- B12: Hybrid Energy (5 sessions)
- B13: Financial Analysis (5 sessions)
- B14: Core Infrastructure (4 sessions)
- B15: Main Application (4 sessions)

### Next Phase: Full Integration
All isolated branches will be merged into a unified, production-ready Streamlit application with:
- Complete module integration
- Database backend (PostgreSQL/SQLite)
- Authentication & user management
- Real-time SCADA data integration
- Advanced visualizations & dashboards
- Export/reporting capabilities

### Deployment Options
- **Local**: `streamlit run app.py`
- **Streamlit Cloud**: Deploy for free on [Streamlit Cloud](https://streamlit.io/cloud)
- **Docker**: Production deployment with containerization
- **Cloud Platforms**: AWS, Google Cloud, Azure

