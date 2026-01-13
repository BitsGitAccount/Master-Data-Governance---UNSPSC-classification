# How to View the Application Output

## 🎯 The application is now running!

### Step 1: Access the Web Interface

The Streamlit application should automatically open in your browser. If not:

1. **Open your web browser** (Chrome, Firefox, Safari, etc.)
2. **Navigate to**: `http://localhost:8501`
3. You should see the Material Classification PoC interface

### Step 2: What You'll See

When the page loads, you'll see:

```
📦 Material Classification & Attribute Extraction PoC
Automated classification of materials into UNSPSC codes and extraction 
of attributes from Technical Data Sheets.
```

### Step 3: Explore the Application

The app has 4 tabs across the top:

#### Tab 1: 🔍 Classification
**Try this first!**

1. You'll see a text area that says "Enter material description"
2. Type something like: `Premium Steel Pipes for Industrial Use`
3. Click the blue button: **"🔍 Classify Material"**
4. Wait 1-2 seconds
5. **You'll see**:
   - Predicted UNSPSC Code
   - Confidence Score (with colored indicator)
   - Top 3 Predictions table
   - Explainability section showing influential keywords

#### Tab 2: 📄 PDF Extraction
**Try extracting from a sample PDF!**

1. Look for "Or select a sample TDS file:"
2. Open the dropdown menu
3. Select any PDF like: `MAT0001_TDS.pdf`
4. Click: **"🔍 Extract Attributes"**
5. **You'll see**:
   - Total Pages, Extraction Rate, Confidence metrics
   - Extracted attributes (Weight, Dimensions, Manufacturer, etc.)
   - Each attribute shows:
     - The extracted value
     - Confidence percentage
     - Source (which page and exact text)

#### Tab 3: ⚡ Batch Processing
**Process multiple materials at once!**

1. Use the slider to select how many samples (try 10)
2. Click: **"🚀 Process Batch"**
3. **You'll see**:
   - Total Processed, Correct Predictions, Accuracy metrics
   - A table showing all materials with predictions
   - Download button to export results as CSV

#### Tab 4: ℹ️ About
**Learn about the system**
- Overview of features
- Technical details
- Setup instructions

## 📸 Expected Screenshots

### Classification Result Example:
```
Predicted UNSPSC Code: 23456789
Confidence Score: 85%
[Green indicator: High confidence]

Top 3 Predictions:
1. 23456789 - 85%
2. 45678901 - 10%
3. 12345678 - 5%

Explainability:
Classification was influenced by keywords: 'steel', 'pipe', 'industrial'
```

### PDF Extraction Result Example:
```
Extraction Results:
- Total Pages: 1
- Extraction Rate: 100%
- Avg Confidence: 89%
- Quality Score: 89%
✓ Good quality extraction

WEIGHT: 4.28 kg (95% confidence)
DIMENSIONS: 122.7 x 102.9 x 69.1 cm (85% confidence)
MANUFACTURER: SteelCo Manufacturing (95% confidence)
```

## 🔧 If the browser doesn't open automatically:

1. Look at your terminal - you should see output like:
   ```
   You can now view your Streamlit app in your browser.
   
   Local URL: http://localhost:8501
   Network URL: http://192.168.x.x:8501
   ```

2. **Copy the Local URL** (`http://localhost:8501`)
3. **Paste it into your browser** address bar
4. Press Enter

## 💡 Interactive Features to Try

### Test Classification with Different Inputs:
```
1. "High-Quality Plastic Packaging for Industrial Use"
2. "Premium Steel Pipes - Grade 316L"
3. "Electronic Resistor Components"
4. "Industrial Hydraulic Oil"
5. "Safety Protective Gloves"
```

Each will show different UNSPSC codes and confidence levels!

### Test PDF Extraction:
- Try different sample PDFs (MAT0001 through MAT0020)
- Each PDF has different attributes
- Compare extraction confidence across files

### Test Batch Processing:
- Process 10 materials
- See which ones are predicted correctly
- Download the results CSV file

## 📊 What Success Looks Like

You should see:
✅ Application loads without errors
✅ Classification returns results in 1-2 seconds
✅ PDF extraction shows all 5 attributes
✅ Confidence scores are displayed
✅ Explainability shows keywords and sources
✅ Batch processing completes successfully

## 🛑 To Stop the Application

When you're done exploring:
1. Go back to your terminal
2. Press `Ctrl + C` (or `Cmd + C` on Mac)
3. The server will shut down

## 🔄 To Restart the Application

```bash
cd material-classification-poc
source venv/bin/activate
streamlit run app.py
```

Then go to `http://localhost:8501` again

## 📱 Taking Screenshots

To document your results:
1. Use your system's screenshot tool
2. Capture the browser window showing the results
3. Save for your documentation/presentation

## 🎥 Recording a Demo

If you want to record:
1. Use screen recording software (QuickTime, OBS, etc.)
2. Walk through each tab
3. Show the classification and extraction in action

---

**Enjoy exploring the Material Classification PoC!** 🚀

The application demonstrates:
- Real-time material classification
- PDF attribute extraction
- Confidence scoring
- Full explainability
- Batch processing capabilities