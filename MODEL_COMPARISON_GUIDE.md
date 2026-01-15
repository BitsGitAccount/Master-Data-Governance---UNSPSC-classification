# Multi-Model Comparison Guide

This guide explains how to train and compare multiple machine learning models to find the best one for your material classification task.

## Why Compare Multiple Models?

Different ML algorithms have different strengths:
- **Logistic Regression**: Fast, interpretable, good baseline
- **Random Forest**: Handles non-linear patterns, robust to noise
- **Naive Bayes**: Works well with text, very fast
- **SVM (Support Vector Machine)**: Powerful for high-dimensional text data
- **XGBoost**: Often achieves highest accuracy (requires installation)

Comparing models helps you:
1. ✅ Find the most accurate model for your data
2. ✅ Balance accuracy vs. speed (important for production)
3. ✅ Understand which algorithms work best for your use case
4. ✅ Make data-driven decisions about model selection

## Quick Start

### Step 1: Run the Multi-Model Training Script

```bash
python train_model_comparison.py
```

This will:
1. Load your training data (JSON or CSV)
2. Train 4-5 different models
3. Compare their performance
4. Show you which model performs best
5. Save the best model as default

### Step 2: Review the Comparison Results

The script will output a comparison table like this:

```
MODEL COMPARISON SUMMARY
======================================================================

                 Model  Accuracy  Precision  Recall  F1-Score  Training Time (s)  Prediction Time (s)  Predictions per Second
       Random Forest    0.9200      0.9180   0.9200    0.9189               2.45                 0.08                    1250
  Logistic Regression    0.9100      0.9085   0.9100    0.9092               0.52                 0.01                   10000
              XGBoost    0.9150      0.9140   0.9150    0.9145               1.83                 0.03                    3333
          Naive Bayes    0.8800      0.8790   0.8800    0.8795               0.15                 0.01                   10000
           Linear SVM    0.9050      0.9040   0.9050    0.9045               0.78                 0.01                   10000

🏆 BEST MODEL: Random Forest
======================================================================
  Accuracy: 0.9200
  F1-Score: 0.9189
  Training Time: 2.45s
  Prediction Speed: 1250 predictions/sec
```

### Step 3: Choose Your Model

The script will ask what you want to save:

**Option 1: Save best model only (Recommended)**
- Saves the best performing model as default
- The web app will use this model automatically
- Simplest option for most users

**Option 2: Save all models**
- Saves all trained models for later comparison
- Useful if you want to experiment with different models
- Creates separate .pkl files for each model

**Option 3: Save specific model**
- Choose which model to save
- Useful if you prefer a specific model (e.g., faster but slightly less accurate)

## Understanding the Metrics

### Accuracy
- **What it means**: Percentage of correct predictions
- **Example**: 0.92 = 92% of predictions are correct
- **When to prioritize**: General-purpose metric, good for balanced datasets

### Precision
- **What it means**: Of all positive predictions, how many were correct?
- **Example**: 0.91 = 91% of predicted UNSPSC codes are actually correct
- **When to prioritize**: When false positives are costly

### Recall
- **What it means**: Of all actual positives, how many did we find?
- **Example**: 0.92 = We correctly identified 92% of materials
- **When to prioritize**: When missing items is costly

### F1-Score
- **What it means**: Harmonic mean of precision and recall
- **Example**: 0.92 = Balanced measure of model performance
- **When to prioritize**: When you want a single balanced metric

### Training Time
- **What it means**: How long it takes to train the model
- **Example**: 2.45s = Takes 2.45 seconds to train
- **When to prioritize**: If you need to retrain frequently

### Prediction Speed
- **What it means**: How many predictions per second
- **Example**: 1250 predictions/sec
- **When to prioritize**: For real-time or high-volume predictions

## Model Characteristics

### Logistic Regression (Current Default)
✅ **Pros:**
- Very fast training and prediction
- Highly interpretable (can see feature importance)
- Good baseline performance
- Works well with text data

❌ **Cons:**
- Assumes linear relationships
- May miss complex patterns

**Best for:** Fast predictions, explainability, baseline comparisons

### Random Forest
✅ **Pros:**
- Handles non-linear patterns
- Robust to noise and outliers
- Often achieves high accuracy
- Reduces overfitting

❌ **Cons:**
- Slower training time
- Less interpretable
- Larger model size

**Best for:** Maximum accuracy, complex datasets, production systems

### Naive Bayes
✅ **Pros:**
- Extremely fast training and prediction
- Works well with text
- Handles high-dimensional data
- Simple and efficient

❌ **Cons:**
- Assumes feature independence (often violated)
- May have lower accuracy than other models

**Best for:** Quick prototyping, large datasets, real-time applications

### Linear SVM
✅ **Pros:**
- Powerful for high-dimensional text
- Good generalization
- Handles sparse data well

❌ **Cons:**
- Doesn't provide probability scores by default
- Slower than some alternatives
- Sensitive to feature scaling

**Best for:** High-dimensional text classification, good accuracy/speed balance

### XGBoost (Optional)
✅ **Pros:**
- Often highest accuracy
- Handles complex patterns
- Built-in regularization
- Feature importance

❌ **Cons:**
- Requires separate installation (`pip install xgboost`)
- Slower training
- More complex to tune

**Best for:** Competition-level accuracy, production systems with complex data

## Installation Requirements

Basic models (included):
```bash
# Already installed with scikit-learn
- Logistic Regression
- Random Forest
- Naive Bayes
- Linear SVM
```

Optional model:
```bash
# Install XGBoost for advanced gradient boosting
pip install xgboost
```

## Advanced Usage

### Comparing Specific Models Only

You can modify `models/multi_model_classifier.py` to compare only specific models:

```python
# In MultiModelTrainer.__init__()
self.models = {
    'Logistic Regression': LogisticRegression(...),
    'Random Forest': RandomForestClassifier(...)
    # Comment out models you don't want to compare
}
```

### Custom Model Parameters

Tune model parameters for better performance:

```python
# Example: More trees in Random Forest
'Random Forest': RandomForestClassifier(
    n_estimators=200,  # More trees (default: 100)
    max_depth=15,      # Deeper trees
    random_state=42,
    n_jobs=-1
)
```

### Saving Comparison Results

All comparison results are automatically saved to:
```
trained_models/model_comparison_YYYYMMDD_HHMMSS.csv
```

You can open this CSV file to review results later.

## Workflow Recommendations

### For Development/Testing
1. Use `train_model_comparison.py` to find the best model
2. Save all models for experimentation
3. Test different models in the web app

### For Production
1. Run model comparison on your full dataset
2. Choose based on your priorities:
   - **Speed**: Logistic Regression or Naive Bayes
   - **Accuracy**: Random Forest or XGBoost
   - **Balance**: Linear SVM or Logistic Regression
3. Save only the best model as default
4. Monitor performance over time

### When to Retrain
- When you add new training data
- When classification accuracy drops
- When new UNSPSC categories are added
- Periodically (e.g., monthly) to ensure optimal performance

## Troubleshooting

### XGBoost Not Available
```bash
# Install XGBoost
pip install xgboost

# Or update requirements
echo "xgboost>=1.7.0" >> requirements.txt
pip install -r requirements.txt
```

### Model Training Fails
- Check you have enough training data (at least 100+ samples)
- Ensure data is properly formatted
- Try with just 2-3 models first

### Out of Memory
- Reduce `max_features` in TfidfVectorizer
- Use fewer estimators in Random Forest/XGBoost
- Train models one at a time

## Example Output

Here's what you'll see when running the comparison:

```
======================================================================
MULTI-MODEL MATERIAL CLASSIFICATION TRAINING
======================================================================

Loaded 500 material records

Data Statistics:
- Total samples: 500
- Number of UNSPSC categories: 5

Training samples: 400
Test samples: 100

======================================================================
INITIALIZING MULTI-MODEL TRAINER
======================================================================

Models to be trained and compared:
  1. Logistic Regression
  2. Random Forest
  3. Naive Bayes
  4. Linear SVM
  5. XGBoost

======================================================================
TRAINING AND COMPARING MULTIPLE MODELS
======================================================================

======================================================================
Training: Logistic Regression
======================================================================
✓ Training completed in 0.52s
  Accuracy: 0.9100
  Precision: 0.9085
  Recall: 0.9100
  F1-Score: 0.9092
  Prediction Speed: 10000 predictions/sec

[... similar output for other models ...]

======================================================================
MODEL COMPARISON SUMMARY
======================================================================

🏆 BEST MODEL: Random Forest
  Accuracy: 0.9200
  F1-Score: 0.9189

======================================================================
✓ Best model (Random Forest) saved as default classifier
  Location: trained_models/classifier.pkl
  The web app will now use this model for predictions
======================================================================
```

## Next Steps

1. ✅ Run `python train_model_comparison.py`
2. ✅ Review the comparison results
3. ✅ Choose and save the best model
4. ✅ Test with `streamlit run app.py`
5. ✅ Monitor performance and retrain as needed

For questions or issues, refer to the main README.md or IMPLEMENTATION_SUMMARY.md.
