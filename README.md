# Material Classification and Attribute Extraction PoC

## Overview
This Proof of Concept (PoC) automates the classification of materials into appropriate UNSPSC codes using **both material descriptions and Technical Data Sheets (TDS) PDFs**. The system combines textual descriptions with extracted PDF attributes to provide accurate top 5 UNSPSC code predictions.

## Key Requirements
⚠️ **Both material description and TDS PDF are mandatory** for production classification to ensure optimal accuracy.

## Features
- **Combined Classification**: Uses both material description and PDF attributes for accurate UNSPSC classification
- **Top 5 Predictions**: Returns top 5 UNSPSC codes ranked by probability
- **Attribute Extraction**: Extracts weight, dimensions, manufacturer, MPN, and material ID from TDS PDFs
- **Confidence Scoring**: Provides confidence levels for all predictions
- **Explainability**: Shows which data influenced decisions with detailed reasoning
- **User Interface**: Interactive UI for data input and result visualization
- **Demo Modes**: Separate tabs for testing individual components

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure
```
material-classification-poc/
├── data/
│   ├── mock_materials.csv          # Mock structured data
│   └── tds_pdfs/                   # Mock TDS PDF files
├── models/
│   ├── classifier.py               # Classification model
│   └── pdf_extractor.py            # PDF attribute extraction
├── utils/
│   ├── data_generator.py           # Mock data generation
│   └── explainability.py           # Explainability utilities
├── app.py                          # Streamlit UI
├── train_model.py                  # Model training script
└── requirements.txt
```

## Usage

### 1. Generate Mock Data
```bash
cd material-classification-poc
python utils/data_generator.py
```

### 2. Train Classification Model
```bash
python train_model.py
```

### 3. Run the Application
```bash
streamlit run app.py
```

### 4. Using the Application

#### Main Classification (Production Mode)
1. Navigate to the **"Material Classification"** tab
2. **Enter material description** (Required) - provide a detailed description of the material
3. **Upload or select TDS PDF** (Required) - provide the Technical Data Sheet
4. Click **"Classify Material"** to get results

The system will:
- Extract attributes from the PDF (weight, dimensions, manufacturer, etc.)
- Combine PDF attributes with the material description
- Classify and return **top 5 UNSPSC codes** with probabilities
- Show explainability for the predictions
- Display detailed PDF extraction results

#### Demo Modes
- **Demo: Description Only** - Test classification using only material description
- **Demo: PDF Extraction** - Test PDF attribute extraction independently
- **Batch Processing** - Process multiple materials at once

## Components

### Classification Model
- Uses TF-IDF vectorization for text processing
- Implements Logistic Regression for multi-class classification
- Returns **top 5 UNSPSC predictions** with probabilities
- Provides probability-based confidence scores
- Explains predictions using feature importance analysis
- Combines material description with PDF-extracted attributes for enhanced accuracy

### PDF Attribute Extraction
- Extracts text from PDFs using PyMuPDF
- Uses regex patterns to identify key attributes:
  - Weight and dimensions
  - Manufacturer information
  - Material ID and part numbers (MPN)
- Validates extracted data with confidence scoring
- Tracks source location and context in documents
- Provides detailed extraction quality metrics

### Combined Processing
- **Mandatory dual-input system**: Requires both description and PDF
- Enhances material description with extracted PDF attributes
- Improves classification accuracy through data fusion
- Provides comprehensive explainability showing:
  - Most influential keywords from enhanced description
  - Exact PDF sources for extracted attributes
  - Confidence scores for all predictions

### Explainability
- Shows most influential words for classification decisions
- Displays exact text snippets for extracted PDF attributes
- Provides confidence scores for all predictions
- Tracks data sources (description vs PDF attributes)
- Explains top 5 UNSPSC predictions with probability rankings

## License
MIT
