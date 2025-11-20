# PV Circularity Simulator - Setup Guide

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Virtual environment tool (optional but recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd pv-circularity-simulator
   ```

2. **Create and activate a virtual environment** (recommended)
   ```bash
   # On Linux/Mac
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## Application Structure

```
pv-circularity-simulator/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview
├── SETUP.md                    # This setup guide
├── LICENSE                     # License file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── pages/
    ├── __init__.py            # Pages package
    ├── README.md              # Pages documentation
    └── [page modules]         # Individual page implementations
```

## Core Components

### app.py

The main application file contains four core functions:

1. **`streamlit_app()`** - Main entry point
   - Configures Streamlit page settings
   - Orchestrates application flow
   - Manages rendering pipeline

2. **`session_state_management()`** - State initialization
   - Initializes all session variables
   - Maintains state across page navigation
   - Manages workflow progress tracking

3. **`page_routing()`** - Navigation handler
   - Routes between different pages
   - Updates navigation history
   - Triggers page reruns when needed

4. **`sidebar_navigation()`** - Sidebar UI
   - Renders navigation menu
   - Displays project information
   - Provides quick actions

### Configuration

Streamlit settings can be customized in `.streamlit/config.toml`:

- **Theme**: Colors, fonts, and appearance
- **Server**: Port, CORS, security settings
- **Client**: Error handling, toolbar options
- **Performance**: Upload limits, compression

## Development Workflow

### Adding a New Page

1. Create a new Python file in `pages/` directory
2. Implement the `render()` function
3. Update `PAGE_CONFIG` in `app.py`
4. Add the page to the `PageEnum` enumeration

Example:

```python
# pages/new_feature.py
import streamlit as st

def render() -> None:
    """Render the new feature page."""
    st.header("New Feature")
    # Implementation here
```

### Session State Structure

The application uses `st.session_state` to maintain data:

```python
st.session_state = {
    "current_page": str,
    "initialized": bool,
    "user_data": Dict[str, Any],
    "simulation_data": {
        "cell_design": {},
        "module_engineering": {},
        "ctm_losses": {},
        "system_config": {},
        "performance_data": {},
        "forecast_data": {},
        "reliability_data": {},
        "circularity_metrics": {},
        "circular_economy_model": {},
    },
    "workflow_progress": Dict[str, bool],
    "navigation_history": List[str],
    "project_name": str,
    "last_saved": Optional[str]
}
```

## Testing

### Manual Testing

1. Start the application
2. Navigate through all pages
3. Test state persistence across navigation
4. Verify save/export functionality
5. Check responsive design

### Automated Testing (Future)

```bash
# Unit tests
pytest tests/

# Integration tests
pytest tests/integration/

# Coverage report
pytest --cov=. --cov-report=html
```

## Deployment

### Local Deployment

```bash
streamlit run app.py --server.port 8501
```

### Production Deployment

Options include:
- **Streamlit Cloud**: Easy deployment for Streamlit apps
- **Docker**: Containerized deployment
- **Cloud Platforms**: AWS, GCP, Azure with containerization

Example Dockerfile (to be created):
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **Port already in use**
   - Change port: `streamlit run app.py --server.port 8502`
   - Or kill the process using port 8501

3. **Page not rendering**
   - Check browser console for errors
   - Verify page module exists in `pages/` directory
   - Check PAGE_CONFIG in app.py

4. **Session state issues**
   - Clear cache: Press 'C' in the app or use the menu
   - Restart the application

## Performance Optimization

- Use `@st.cache_data` for expensive computations
- Use `@st.cache_resource` for loading models/connections
- Implement lazy loading for large datasets
- Optimize data structures in session state

## Contributing

1. Create a feature branch
2. Implement changes with full docstrings
3. Test thoroughly
4. Submit pull request with description

## Support

- Documentation: See README.md and inline docstrings
- Issues: GitHub issue tracker
- Questions: Project discussion board

## License

See LICENSE file for details.
