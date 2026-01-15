# Material Classification System - UNSPSC Code Prediction

## What This Project Does

I built this system to automatically classify materials into the correct UNSPSC (United Nations Standard Products and Services Code) categories. Instead of manually searching through thousands of codes, you just provide a material description and a Technical Data Sheet PDF, and the system predicts the most likely classifications.

The system uses machine learning to analyze both the text description and technical specifications extracted from PDFs to make accurate predictions.

## Why Both Inputs Matter

I designed this to require **both** a material description and a TDS PDF because:

1. **Better accuracy**: Text descriptions provide context, while PDFs contain precise technical specs
2. **More data points**: Extracting manufacturer, dimensions, weight, etc. gives the model more to work with
3. **Real-world workflow**: In practice, you usually have both documents available anyway

## What You'll Get

When you classify a material, the system returns:

- **Top 5 UNSPSC code predictions** ranked by confidence
- **Confidence scores** for each prediction (so you know how certain the model is)
- **Extracted attributes** from the PDF (weight, dimensions, manufacturer, model, etc.)
- **Explainability** showing which keywords influenced the classification

## Getting Started

### Quick Setup

```bash
cd material-classification-poc
chmod +x run_poc.sh
./run_poc.sh
```

This will create a virtual environment, install dependencies, generate training data, and train the model.

## How to Use It

Run `streamlit run app.py`

### Classifying a Material

1. **Enter the material description** - Be as specific as possible (e.g., "Industrial grade stainless steel pipe, 316L, for chemical processing")

2. **Upload or select a TDS PDF** - Either upload your own PDF or choose from the sample files

3. **Click "Classify Material"** - The system will:
   - Extract technical specs from the PDF
   - Combine them with your description
   - Predict the top 5 most likely UNSPSC codes
   - Show you which keywords were most influential

### Understanding the Results

The results show several pieces of information:

**Classification Details:**
- The top predicted UNSPSC code with a confidence score
- A dropdown to see all 5 predictions (in case the top one isn't quite right)
- Confidence badge (green = high confidence, yellow = medium, red = low)

**Extracted Attributes:**
- Technical specifications pulled from the PDF
- Each with its own confidence score
- Useful for verifying the extraction worked correctly

**Explainability:**
- Keywords that influenced the classification
- Shows why the model made its prediction
- Helps you understand and trust the results

## How It Works

### Classification Model

We are using a **TF-IDF vectorizer** combined with **Logistic Regression**:

- **TF-IDF** converts text descriptions into numerical features based on word importance
- **Logistic Regression** learns which features correspond to which UNSPSC codes
- The model considers both individual words and 2-word phrases (bigrams)
- I configured it to focus on the 500 most important features to reduce noise

### PDF Attribute Extraction

The PDF extractor uses **PyMuPDF** to:

- Extract all text from the PDF
- Apply regex patterns to identify specific attributes (weight, dimensions, etc.)
- Track where in the document each attribute was found
- Calculate confidence scores based on pattern match quality

### Combined Processing

When you provide both inputs, the system:

1. Extracts attributes from the PDF
2. Appends them to the material description
3. Uses the enhanced description for classification
4. Returns top 5 predictions instead of just one

This approach consistently gives better results than using either input alone.

## Training Your Own Model

If you want to compare different machine learning algorithms:

```bash
python train_model_comparison.py
```

This will train and compare:
- Logistic Regression (current default)
- Random Forest
- Naive Bayes
- Linear SVM
- XGBoost (if installed)

The script shows you a comparison table with accuracy, training time, and prediction speed for each model. You can then choose which one to use as your default.

## Current Performance

With the mock training data (500 samples):

- **Accuracy**: ~92% on test data
- **Extraction Rate**: 100% (all sample PDFs)
- **Average Confidence**: 89%
- **Processing Time**: <2 seconds per material

Keep in mind these numbers are based on synthetic data. Real-world performance will depend on your actual training data quality and diversity.

## Customization Ideas

Some things you might want to customize:

**Improving Classification:**
- Add more training data (more examples = better accuracy)
- Tune the TF-IDF parameters in `models/classifier.py`
- Try different ML algorithms using `train_model_comparison.py`
- Add domain-specific preprocessing rules

**Enhancing PDF Extraction:**
- Add more regex patterns in `models/pdf_extractor.py`
- Support additional attributes
- Implement OCR for scanned documents
- Handle tables and structured data better

**UI Improvements:**
- Customize the styling in `static/styles.css`
- Add batch processing capabilities
- Export results to different formats
- Integrate with your existing systems

## Troubleshooting

**"Model not found" error:**
```bash
python train_model.py
```

**"No module named X" error:**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

**Streamlit won't start:**
```bash
# Kill any process using port 8501
lsof -ti:8501 | xargs kill -9
streamlit run app.py
```

**PDF extraction not working:**
- Make sure the PDF contains selectable text (not a scanned image)
- Check that the PDF follows a standard TDS format
- Try one of the sample PDFs first to verify the system works

## Next Steps

This is a proof of concept that demonstrates the feasibility of automated material classification. To use it in production, you'd want to:

1. **Gather real training data** - Replace the mock data with actual material descriptions and their correct UNSPSC codes
2. **Retrain the model** - More diverse real-world data will improve accuracy significantly
3. **Expand PDF patterns** - Add support for different TDS formats you encounter
4. **Add validation** - Implement human-in-the-loop review for low-confidence predictions
5. **Integrate** - Connect to your SAP MDG system or other master data platforms

## Questions or Issues?

If something's not working as expected, check:
- The other documentation files (QUICK_START_GUIDE.md, PROJECT_SUMMARY.md, etc.)
- The code comments - I tried to explain the "why" not just the "what"
- The Streamlit app's "About" tab for usage tips

## License

MIT License - feel free to use and modify as needed.
