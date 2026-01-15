# Quick Start Guide

## Get Up and Running in 5 Minutes

This guide will help you set up and start using the material classification system as quickly as possible.

## Prerequisites

Make sure you have:
- Python 3.8 or higher installed (check with `python3 --version`)
- About 500 MB of free disk space
- 5 minutes of your time

## Option 1: Automated Setup (Easiest)

I created a script that does everything for you:

```bash
cd material-classification-poc
chmod +x run_poc.sh
./run_poc.sh
```

This script will:
1. Create a Python virtual environment
2. Install all required packages
3. Generate 500 sample materials with PDFs
4. Train the classification model
5. Tell you when it's ready to use

**Expected time**: 3-5 minutes depending on your computer

When you see "Setup complete!", you're ready to go.

## Option 2: Manual Setup (Step by Step)

If you prefer to see what's happening at each step:

### Step 1: Set Up Python Environment

```bash
# Navigate to the project
cd material-classification-poc

# Create a virtual environment
python3 -m venv venv

# Activate it (macOS/Linux)
source venv/bin/activate

# Or on Windows
venv\Scripts\activate
```

You'll know it worked when you see `(venv)` at the start of your command prompt.

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- scikit-learn (machine learning)
- streamlit (web interface)
- PyMuPDF (PDF processing)
- pandas (data handling)
- and a few other utilities

### Step 3: Generate Training Data

```bash
python utils/data_generator.py
```

This creates:
- 500 sample material descriptions
- 20 sample Technical Data Sheet PDFs
- All saved in the `data/` folder

You'll see progress messages as it generates each material and PDF.

### Step 4: Train the Model

```bash
python train_model.py
```

This:
- Loads the generated data
- Trains a TF-IDF + Logistic Regression model
- Tests its accuracy
- Saves the trained model to `trained_models/`

You should see accuracy metrics printed out (typically around 92% with the mock data).

### Step 5: Launch the Application

```bash
streamlit run app.py
```

Your browser should automatically open to `http://localhost:8501`.

If it doesn't, just copy that URL into your browser manually.

## Using the Application

### Your First Classification

Once the app opens, you'll see a clean interface. Here's how to use it:

1. **Enter a material description** in the text box
   - Try: "Industrial stainless steel pipe, 316L grade, for chemical processing applications"

2. **Select a sample PDF** from the dropdown
   - Or upload your own if you have one
   - The sample PDFs are named like "pump1.pdf", "Valve2.pdf", etc.

3. **Click "Classify Material"**

4. **Review the results**:
   - Top prediction with confidence score
   - Dropdown showing all 5 predictions
   - Extracted attributes from the PDF
   - Explanation of which keywords influenced the decision

### What You're Seeing

**Classification Details Section**:
- Shows the predicted UNSPSC code
- Confidence badge (green = high, yellow = medium, red = low)
- Dropdown to see alternative predictions

**Extracted Attributes Table**:
- Technical specs pulled from the PDF
- Each with its own confidence score
- These get combined with your description for better accuracy

**PDF Viewer**:
- Shows the actual PDF on the right side
- Helps you verify the extraction worked correctly

**Explainability** (click to expand):
- Shows which keywords were most important
- Helps you understand why the model made its prediction
- Useful for catching errors or verifying logic

## Tips for Best Results

### Writing Good Descriptions

The more specific you are, the better the predictions:

**Good examples**:
- "Industrial hydraulic pump, centrifugal type, 50 GPM flow rate, stainless steel construction"
- "Safety protective gloves, nitrile coating, cut-resistant level 5, for manufacturing applications"
- "Electronic pressure sensor, 0-100 PSI range, 4-20mA output, explosion-proof rated"

**Less helpful examples**:
- "pump" (too vague)
- "thing for factory" (no specific details)
- "metal part" (could be anything)

### Understanding Confidence Scores

- **80-100% (Green)**: Model is very confident, likely correct
- **60-79% (Yellow)**: Moderate confidence, worth reviewing
- **Below 60% (Red)**: Low confidence, definitely review manually

Low confidence isn't necessarily wrong - it might mean the material could fit multiple categories equally well.

### When the Top Prediction Isn't Right

That's why I included 5 predictions! Sometimes the material could fit multiple categories, or the description is ambiguous. Check the dropdown to see if one of the other predictions makes more sense.

## Common Issues and Solutions

### "Model not found" Error

You need to train the model first:
```bash
python train_model.py
```

### "No module named X" Error

Your virtual environment isn't activated or dependencies aren't installed:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Application Won't Start

Port 8501 might be in use:
```bash
# Kill any process using that port
lsof -ti:8501 | xargs kill -9

# Try again
streamlit run app.py
```

### PDF Extraction Not Working

Make sure the PDF:
- Contains selectable text (not a scanned image)
- Follows a somewhat standard TDS format
- Try one of the sample PDFs first to verify the system works

## Testing with Sample Data

I included several sample PDFs you can test with:

**Pumps**: pump1.pdf through pump8.pdf
- Various pump specifications
- Different manufacturers and models

**Valves**: Valve1.pdf through Valve4.pdf
- Different valve types
- Various pressure and temperature ratings

Try different combinations of descriptions and PDFs to see how the system handles different inputs.

## What to Expect

With the synthetic training data, you should see:
- Classifications complete in 1-2 seconds
- Around 90%+ confidence on most predictions
- All attributes successfully extracted from sample PDFs
- Clear explanations for each prediction

Remember, these numbers are based on mock data. With real training data, performance will improve significantly.

## Next Steps

Once you're comfortable with the basics:

1. **Try different descriptions** - See how the system handles various materials
2. **Upload your own PDFs** - Test with real Technical Data Sheets
3. **Check the explainability** - Understand what drives each prediction
4. **Read PROJECT_SUMMARY.md** - Learn more about how everything works
5. **Explore train_model_comparison.py** - Compare different ML algorithms

## Shutting Down

When you're done:

1. Go to your terminal
2. Press `Ctrl+C` (or `Cmd+C` on Mac)
3. The server will stop

To start again later:
```bash
cd material-classification-poc
source venv/bin/activate
streamlit run app.py
```

## Getting Help

If something's not working:

1. Check this guide first
2. Look at the error messages - they usually tell you what's wrong
3. Check PROJECT_SUMMARY.md for more details
4. Review the code comments - I documented everything thoroughly

## File Checklist

After setup, you should have:

```
✓ venv/ folder (virtual environment)
✓ data/mdg_multi_material_training_data_500.json (training data)
✓ data/tds_pdfs/ folder with sample PDFs
✓ trained_models/classifier.pkl (trained model)
✓ trained_models/vectorizer.pkl (TF-IDF vectorizer)
```

If any of these are missing, review the relevant setup step above.

## You're All Set!

That's it - you should now have a working material classification system. Start with simple examples and gradually try more complex materials to get a feel for how it works.

Remember: This is a proof of concept built with synthetic data. The real power comes when you train it on actual material descriptions from your organization.
