# Viewing Your Classification Results

## You're Ready to Go!

If you just ran `streamlit run app.py`, the application should now be running. Here's what to do next.

## Accessing the Interface

### If Your Browser Opened Automatically

Great! You should see the Material Classification interface. Skip to the "What You'll See" section below.

### If Nothing Happened

No worries - just open your browser manually:

1. Open any web browser (Chrome, Firefox, Safari, Edge, etc.)
2. Type in the address bar: `http://localhost:8501`
3. Press Enter

You should see the application load.

### If That Doesn't Work

Look at your terminal where you ran the command. You should see something like:

```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.1.100:8501
```

Copy the Local URL and paste it into your browser.

## What You'll See

### The Main Interface

When the page loads, you'll see:

**Top Banner**: 
- "SAP Material Classification"
- "Master Data Governance | Automated UNSPSC Classification"

**Input Section**:
- Left side: Text area for material description
- Right side: PDF upload area
- Blue "Classify Material" button below

That's it - nice and clean!

## Try Your First Classification

Let me walk you through a quick test:

### Step 1: Enter a Description

In the left text box, type something like:
```
Industrial centrifugal pump, stainless steel construction, 
50 GPM flow rate, 150 PSI max pressure, for chemical processing
```

### Step 2: Select a Sample PDF

Below the PDF upload area, you'll see "Or select a sample TDS file:"

Click the dropdown and choose any PDF (try "pump1.pdf" to match the pump description).

### Step 3: Click "Classify Material"

Hit the blue button and wait 1-2 seconds.

### Step 4: Review the Results

You'll see several sections appear:

**Request Information Banner** (blue/dark background):
- Shows your description and extracted info
- Displays manufacturer, model, material

**Classification Details** (left side):
- Dropdown showing top 5 UNSPSC code predictions
- Primary prediction highlighted with confidence badge
- Color coding: Green (high), Yellow (medium), Red (low)
- Table of extracted attributes from the PDF

**PDF Viewer** (right side):
- Shows the actual PDF you uploaded/selected
- Scroll through it to verify details

**Expandable Sections**:
- "View Classification Explainability" - shows which keywords mattered
- "View PDF Extraction Details" - shows extraction metrics and sources

## Understanding What You See

### Confidence Badges

The colored badges tell you how confident the model is:

- **Green (80-100%)**: High confidence - the prediction is likely correct
- **Yellow (60-79%)**: Medium confidence - worth reviewing
- **Red (below 60%)**: Low confidence - definitely review manually

### The Dropdown Selector

This shows all 5 predictions ranked by probability. The top one is selected by default, but you can choose a different one if it makes more sense for your material.

Why 5 predictions? Because sometimes materials could fit multiple categories, and I wanted to give you options rather than forcing a single choice.

### Extracted Attributes Table

This shows what the system pulled from the PDF:

- **Display Name**: Human-readable attribute name (e.g., "Maximum Operating Pressure")
- **Attribute Data**: The actual value extracted (e.g., "150 PSI")
- **Confidence**: How confident the extraction was (percentage)

These attributes get combined with your description to improve classification accuracy.

### Explainability Section

Click "View Classification Explainability" to see:

- A plain-language explanation
- Table of most influential keywords
- Their importance scores
- Why they matter for the prediction

This helps you verify the model's reasoning makes sense.

### PDF Extraction Details

Click "View PDF Extraction Details" to see:

- Overall extraction metrics (rate, confidence, quality)
- Where each attribute came from (page number, exact text)
- Context snippets showing the source

Useful for verifying the extraction worked correctly.

## Testing with Different Inputs

Try these example combinations:

**Example 1: Valve**
```
Description: "Industrial ball valve, 2-inch diameter, stainless steel, 
             high-pressure rated for gas applications"
PDF: Select "Valve1.pdf"
```

**Example 2: Another Pump**
```
Description: "Heavy-duty hydraulic pump for industrial equipment, 
             variable displacement design, high efficiency"
PDF: Select "pump2.pdf"
```

**Example 3: Custom Material**
```
Description: (Whatever material you want to classify)
PDF: Upload your own TDS document
```

## What Good Results Look Like

You should see:

✅ Classification completes in 1-2 seconds
✅ Confidence badge appears (any color is fine, but green is better)
✅ 5 predictions shown in the dropdown
✅ Extracted attributes table populated (4-8 attributes typically)
✅ PDF displays correctly on the right
✅ Explainability shows relevant keywords

## Common Questions

**Q: Why does it need both description and PDF?**
A: Combining both gives much better accuracy. The description provides context, the PDF provides precise specs.

**Q: What if my PDF doesn't have all attributes?**
A: That's okay. The system extracts what it can find and uses the description for the rest.

**Q: Can I use just a description or just a PDF?**
A: The system requires both. I found this approach works best for accurate classification.

**Q: Which prediction should I use?**
A: Usually the top one (it has the highest probability). But check the others if you're not sure - sometimes a lower-ranked prediction is more appropriate.

**Q: What if the confidence is low (red)?**
A: Review the prediction manually. Low confidence means the model is uncertain - it might be right, but you should verify.

## Tips for Best Results

### Write Better Descriptions

More specific = better predictions:

**Good**: "316L stainless steel centrifugal pump, 50 GPM, 150 PSI, for chemical processing"
**Poor**: "pump"

Include:
- Material/grade
- Type/model
- Specifications (flow rate, pressure, dimensions, etc.)
- Application/use case

### Use Quality PDFs

The system works best with:
- Text-based PDFs (not scanned images)
- Standard TDS format
- Clear specifications section
- Readable text

If extraction fails, the system will still work but with lower accuracy.

## Stopping the Application

When you're done:

1. Go to your terminal (where you ran `streamlit run app.py`)
2. Press `Ctrl+C` (or `Cmd+C` on Mac)
3. You'll see "Stopping..." and the server will shut down

## Starting Again Later

To use it again:

```bash
cd material-classification-poc
source venv/bin/activate  # Activate virtual environment
streamlit run app.py
```

Then go to `http://localhost:8501` in your browser.

## Taking Screenshots

Want to document your results?

**On Mac**: 
- `Cmd + Shift + 4` then drag to select area
- Or `Cmd + Shift + 3` for full screen

**On Windows**: 
- `Win + Shift + S` to open snipping tool
- Or `PrtScn` for full screen

**On Linux**: 
- Usually `PrtScn` or `Shift + PrtScn`

## Recording a Demo

If you want to show this to others:

**Mac**: Use QuickTime Player (File → New Screen Recording)
**Windows**: Use Xbox Game Bar (`Win + G`)
**Linux**: Use SimpleScreenRecorder or Kazam

Walk through:
1. Entering a description
2. Selecting/uploading a PDF
3. Clicking classify
4. Reviewing the results
5. Exploring the explainability

## What's Happening Behind the Scenes

When you click "Classify Material":

1. **PDF Processing** (~1 second)
   - Extract text from PDF
   - Apply pattern matching
   - Find attributes

2. **Description Enhancement** (<0.1 seconds)
   - Combine description + attributes
   - Preprocess text

3. **Classification** (<0.1 seconds)
   - Convert to TF-IDF features
   - Apply model
   - Get predictions

4. **Results Generation** (<0.1 seconds)
   - Rank predictions
   - Calculate confidence
   - Generate explanations

Total: Usually under 2 seconds

## Troubleshooting

**Page won't load**:
- Check terminal for errors
- Try restarting: `Ctrl+C` then `streamlit run app.py` again
- Verify port 8501 isn't blocked by firewall

**"Model not found" error**:
```bash
python train_model.py
```

**PDF won't upload**:
- Try a sample PDF first
- Check file size (should be under 10 MB)
- Verify it's actually a PDF file

**Results don't make sense**:
- Check the explainability to understand why
- Try being more specific in your description
- Verify the PDF uploaded correctly

## Next Steps

Now that you've seen the results:

1. **Experiment** - Try different materials and PDFs
2. **Read PROJECT_SUMMARY.md** - Understand how it works
3. **Review MODEL_COMPARISON_GUIDE.md** - Learn about different algorithms
4. **Think about improvements** - What would make this more useful?

## You're All Set!

That's everything you need to know about viewing and interpreting the results. The system is designed to be intuitive - enter a description, upload a PDF, get predictions.

If something's not clear or not working as expected, check the other documentation files or review the error messages in the terminal.
