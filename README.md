# ☀️ PV Circularity Simulator

End-to-end PV lifecycle simulation platform: Cell design → Module engineering → System planning → Performance monitoring → Circularity (3R). Includes CTM loss analysis, SCAPS integration, reliability testing, energy forecasting, and circular economy modeling.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Running

**Option 1: Using the run script (Recommended)**
```bash
./run.sh
```

**Option 2: Manual setup**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run src/main.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📋 Features

### 15 Comprehensive Modules

#### 🔬 Design & Engineering
- **Materials Selection**: PV material selection and properties configuration
- **Cell Design**: Solar cell design with SCAPS integration
- **Module Design**: PV module configuration and layout
- **CTM Loss**: Cell-to-module loss analysis

#### 🧪 Testing & Validation
- **IEC Testing**: IEC 61215/61730 compliance testing and reliability
- **System Design**: Complete PV system configuration and planning

#### 📊 Performance & Analysis
- **EYA**: Energy yield assessment with P50/P90 analysis
- **Performance Monitoring**: Real-time system performance monitoring
- **Fault Diagnostics**: AI-powered fault detection and diagnosis

#### 🔮 Forecasting & Planning
- **Energy Forecasting**: ML-based energy production forecasting
- **Revamp/Repower**: System upgrade and repowering analysis

#### ♻️ Sustainability & Economics
- **Circularity**: 3R analysis (Reduce, Reuse, Recycle)
- **Hybrid Systems**: PV + storage and hybrid configurations
- **Financial Modeling**: Comprehensive financial analysis and ROI

## 🏗️ Project Structure

```
pv-circularity-simulator/
├── src/
│   ├── main.py                 # Main application
│   ├── modules/                # Application modules
│   │   ├── dashboard.py
│   │   ├── materials_selection.py
│   │   ├── cell_design.py
│   │   ├── module_design.py
│   │   ├── ctm_loss.py
│   │   ├── iec_testing.py
│   │   ├── system_design.py
│   │   ├── eya.py
│   │   ├── performance_monitoring.py
│   │   ├── fault_diagnostics.py
│   │   ├── energy_forecasting.py
│   │   ├── revamp_repower.py
│   │   ├── circularity.py
│   │   ├── hybrid_systems.py
│   │   └── financial_modeling.py
│   ├── utils/                  # Utility functions
│   │   └── session_manager.py
│   └── components/             # Reusable UI components
├── .streamlit/                 # Streamlit configuration
│   └── config.toml
├── projects/                   # Saved project files
├── requirements.txt            # Python dependencies
├── run.sh                      # Launch script
├── LICENSE
└── README.md
```

## 💡 Usage Guide

### Creating a New Project
1. Click "🆕 New" in the sidebar
2. Enter your project name
3. Navigate through modules using the sidebar
4. Save your work with "💾 Save"

### Loading an Existing Project
1. Use the file uploader in the sidebar
2. Select your saved `.json` project file
3. The application will load all your saved data

### Module Navigation
- Use the sidebar to access different modules
- Each module is organized for the PV lifecycle workflow
- Modules are independent but data can be shared across them

### Settings
- Click "⚙️ Settings" to customize:
  - Units (Metric/Imperial)
  - Currency
  - Language
  - Display preferences
  - Theme

## 🔧 Technical Details

### Built With
- **Streamlit** - Web application framework
- **Python 3.8+** - Programming language
- **Pandas & NumPy** - Data processing
- **Scikit-learn** - Machine learning

### Key Capabilities
- ✅ Session state management for project persistence
- ✅ Modular architecture for easy extension
- ✅ Custom CSS styling for professional UI
- ✅ Comprehensive error handling
- ✅ Real-time data visualization
- ✅ Export/import functionality

## 📚 Documentation

See the in-app Help panel (❓ Help button) for:
- Quick Start Guide
- Module descriptions
- Resources and links
- About and version info

## 🤝 Contributing

This is a private repository. For questions or issues, please contact the development team.

## 📄 License

Copyright © 2024 PV Circularity Team. All rights reserved.

## 🆘 Support

For support and questions:
- Check the Help panel in the application
- Review module-specific documentation
- Contact: [support contact]

---

**Version**: 1.0.0
**Last Updated**: 2024
