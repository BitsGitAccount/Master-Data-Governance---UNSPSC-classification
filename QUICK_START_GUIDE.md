# Quick Start Guide - Material Classification PoC

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 500 MB free disk space

### Option 1: Automated Setup (Recommended)

```bash
cd material-classification-poc
chmod +x run_poc.sh
./run_poc.sh
```

This script will:
1. ✅ Create virtual environment
2. ✅ Install all dependencies
3. ✅ Generate mock data (100 materials + 20 PDFs)
4. ✅ Train the classification model
5. ✅ Prepare the system for use

**Expected time**: 3-5 minutes

### Option 2: Manual Setup

```bash
# 1. Navigate to project directory
cd material-classification-poc

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate mock data
python utils/data_generator.py

# 5. Train the model
python train_model.py

# 6. Run the application
streamlit run app.py
```

## 📱 Using the Application

### Starting the Application

```bash
cd material-classification-poc
source venv/bin/activate  # Activate virtual environment
streamlit run app.py
```

The application will open automatically in your browser at: `http://localhost:8501`

### Application Tabs

#### 1. 🔍 Classification Tab
**Purpose**: Classify individual materials into UNSPSC codes

**Steps**:
1. Enter a material description (e.g., "High-Quality Plastic Packaging")
2. Click "🔍 Classify Material"
3. View results:
   - Predicted UNSPSC code
   - Confidence score
   - Top 3 predictions
   - Influential keywords

**Example Input**:
```
Premium Steel Pipes for Industrial Use
```

**Example Output**:
```
Predicted UNSPSC: 23456789
Confidence: 85%
Category: Steel Pipes and Tubes
Keywords: 'steel', 'pipe', 'industrial'
```

#### 2. 📄 PDF Extraction Tab
**Purpose**: Extract attributes from Technical Data Sheet PDFs

**Steps**:
1. Upload a PDF or select a sample file
2. Click "🔍 Extract Attributes"
3. View extracted attributes:
   - Weight
   - Dimensions
   - Manufacturer
   - Material ID
   - Part Number

**Sample Files Available**: 20 pre-generated TDS PDFs (MAT0001 - MAT0020)

#### 3. ⚡ Batch Processing Tab
**Purpose**: Process multiple materials simultaneously

**Steps**:
1. Select number of samples (5-50)
2. Click "🚀 Process Batch"
3. View results table with accuracy metrics
4. Download results as CSV

#### 4. ℹ️ About Tab
**Purpose**: Learn about the system and its components

## 🧪 Testing the System

### Test Classification

```bash
# Test single material
cd material-classification-poc
source venv/bin/activate
python << EOF
from models.classifier import MaterialClassifier

classifier = MaterialClassifier()
classifier.load_model('trained_models')

result = classifier.predict_with_confidence([
    "Premium Plastic Packaging Material"
])[0]

print(f"Predicted: {result['predicted_unspsc']}")
print(f"Confidence: {result['confidence']:.1%}")
EOF
```

### Test PDF Extraction

```bash
# Test PDF extraction
cd material-classification-poc
source venv/bin/activate
python models/pdf_extractor.py data/tds_pdfs/MAT0001_TDS.pdf
```

## 📊 Understanding Results

### Classification Confidence Levels

| Confidence | Meaning | Action |
|------------|---------|--------|
| ≥ 80% | High confidence | ✅ Accept automatically |
| 60-79% | Medium confidence | ⚠️ Review recommended |
| < 60% | Low confidence | ❌ Manual review required |

### PDF Extraction Quality Scores

| Score | Status | Meaning |
|-------|--------|---------|
| ≥ 80% | Good | ✅ High quality extraction |
| 60-79% | Fair | ⚠️ Some attributes may need review |
| < 60% | Poor | ❌ Manual verification needed |

## 🔧 Troubleshooting

### Issue: "Module not found" error

**Solution**:
```bash
# Make sure virtual environment is activated
source venv/bin/activate
# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: "Model not found" in UI

**Solution**:
```bash
# Train the model
cd material-classification-poc
source venv/bin/activate
python train_model.py
```

### Issue: "No sample PDFs found"

**Solution**:
```bash
# Generate mock data
cd material-classification-poc
source venv/bin/activate
python utils/data_generator.py
```

### Issue: Streamlit won't start

**Solution**:
```bash
# Check if port 8501 is available
lsof -ti:8501 | xargs kill -9  # Kill any process using port 8501
streamlit run app.py
```

## 📁 Project Structure Overview

```
material-classification-poc/
│
├── 📊 data/                        # Generated data
│   ├── mock_materials.csv          # 100 material records
│   └── tds_pdfs/                   # 20 sample PDFs
│
├── 🤖 models/                      # Core models
│   ├── classifier.py               # Classification logic
│   └── pdf_extractor.py            # PDF extraction logic
│
├── 💾 trained_models/              # Saved models
│   ├── classifier.pkl              # Trained classifier
│   └── vectorizer.pkl              # TF-IDF vectorizer
│
├── 🛠️ utils/                       # Utilities
│   └── data_generator.py           # Mock data generation
│
├── 🎨 app.py                       # Web UI (Streamlit)
├── 🎓 train_model.py               # Training script
├── 📋 requirements.txt             # Dependencies
└── 🚀 run_poc.sh                   # Setup script
```

## 💡 Tips for Best Results

### Classification Tips
1. **Be Specific**: Include material type, grade, and use case
2. **Use Keywords**: Mention industry-standard terms
3. **Consistent Format**: Similar materials should have similar descriptions

**Good Example**: "Premium Stainless Steel Pipe - Grade 316L - Industrial Use"
**Poor Example**: "Metal thing for plumbing"

### PDF Upload Tips
1. **Use Standard TDS Format**: The system works best with structured TDS documents
2. **Clear Text**: Ensure PDFs have selectable text (not scanned images)
3. **Standard Units**: Use kg, cm, etc. for best extraction

## 🎯 Common Use Cases

### Use Case 1: New Material Classification
```
Input: "High-Density Polyethylene Packaging Container"
Expected: UNSPSC Code for Plastic Packaging Materials
Confidence: High (>80%)
```

### Use Case 2: Bulk Material Processing
```
Input: CSV file with 50 materials
Process: Batch processing
Output: CSV with predictions and confidence scores
```

### Use Case 3: TDS Attribute Extraction
```
Input: Technical Data Sheet PDF
Output: Weight, Dimensions, Manufacturer, Part Number
Validation: Source tracking for audit
```

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review `PROJECT_SUMMARY.md` for detailed information
3. Examine `README.md` for technical details

## ✅ Verification Checklist

Before considering setup complete, verify:

- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] Mock data generated (check `data/` folder)
- [ ] Model trained (check `trained_models/` folder)
- [ ] Application starts without errors
- [ ] Classification tab works
- [ ] PDF extraction tab works
- [ ] Batch processing works

## 🎓 Next Steps

1. **Explore the UI**: Try all tabs and features
2. **Test with Custom Data**: Upload your own PDFs
3. **Review Results**: Check accuracy and confidence scores
4. **Provide Feedback**: Note areas for improvement
5. **Plan Integration**: Consider SAP MDG integration strategy

---

**Ready to proceed?** Run `streamlit run app.py` and start exploring! 🚀