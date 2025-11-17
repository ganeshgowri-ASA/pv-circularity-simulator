#!/bin/bash
# PV System Design UI - Startup Script

echo "🌞 Starting PV System Design Studio..."
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run the Streamlit app
echo ""
echo "🚀 Launching PV System Design Studio..."
echo "📱 Access the app at: http://localhost:8501"
echo ""

streamlit run ui/system_design_app.py
