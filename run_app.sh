#!/bin/bash

# NBA Player Points Predictor - Startup Script

echo "🏀 Starting NBA Player Points Predictor..."
echo "=========================================="

# Check if we're in the right directory
if [ ! -f "MODEL/app.py" ]; then
    echo "❌ Error: app.py not found in MODEL directory"
    echo "Please run this script from the project root directory"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3 and try again"
    exit 1
fi

# Check if requirements are installed
echo "🔍 Checking dependencies..."
cd MODEL
python3 -c "import streamlit, pandas, numpy, sklearn, nba_api" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip3 install -r ../requirements.txt
fi

# Start the app
echo "🚀 Starting Streamlit app..."
echo "📱 The app will open in your browser at http://localhost:8501"
echo "🔄 Press Ctrl+C to stop the app"
echo ""

streamlit run app.py --server.port 8501
