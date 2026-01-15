#!/bin/bash

# Material Classification App Launcher
# This script activates the virtual environment and launches the Streamlit app

echo "Starting Material Classification Application..."
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "✓ Virtual environment found"
    echo "✓ Activating virtual environment..."
    source venv/bin/activate
else
    echo "Virtual environment not found"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    echo " Generating mock data..."
    python utils/data_generator.py
    
    echo "Training model..."
    python train_model.py
fi

# Check if model is trained
if [ ! -f "trained_models/classifier.pkl" ]; then
    echo "⚠️  Model not found. Training model..."
    python train_model.py
fi

echo ""
echo "✓ Setup complete!"
echo "🌐 Launching web application..."
echo ""
echo "The app will open automatically in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Launch Streamlit
streamlit run app.py
