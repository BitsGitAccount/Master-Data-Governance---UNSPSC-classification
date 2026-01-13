#!/bin/bash

# Material Classification PoC - Quick Start Script

echo "=================================="
echo "Material Classification PoC Setup"
echo "=================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi

echo -e "\n1. Creating virtual environment..."
python3 -m venv venv

echo -e "\n2. Activating virtual environment..."
source venv/bin/activate

echo -e "\n3. Installing dependencies..."
pip install -r requirements.txt

echo -e "\n4. Generating mock data..."
python utils/data_generator.py

echo -e "\n5. Training classification model..."
python train_model.py

echo -e "\n=================================="
echo "Setup complete!"
echo "=================================="
echo -e "\nTo run the application:"
echo "  source venv/bin/activate"
echo "  streamlit run app.py"