#!/bin/bash
# PV Circularity Simulator - Launch Script

echo "Starting PV Circularity Simulator..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Run the Streamlit app
echo "=================================="
echo "Launching application..."
echo "=================================="
streamlit run src/main.py
