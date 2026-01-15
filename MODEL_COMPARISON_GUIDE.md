# Comparing Different Machine Learning Models

## Why Compare Models?

When I first built this system, I chose Logistic Regression as the classification algorithm. It's fast, interpretable, and works well with text data. But I wanted to know: could other algorithms do better?

That's why I created the model comparison tool. It trains multiple different ML algorithms on the same data and shows you how they stack up against each other.

## Quick Start

Just run this command:

```bash
python train_model_comparison.py
```

The script will:
1. Load your training data
2. Train 4-5 different models (depending on what's installed)
3. Test each one on the same test set
4. Show you a comparison table
5. Let you choose which model to save as default

The whole process takes 2-5 minutes depending on your data size.

## Understanding the Models

Here are the models I included and why:

### Logistic Regression (Current Default)

**What it is**: A statistical model that learns linear relationships between features and categories.

**Strengths**:
- Very fast (trains in seconds, predicts instantly)
- Highly interpretable (I can show you exactly why it made each decision)
- Works great with text data converted to TF-IDF
- Reliable and well-understood

**Weaknesses**:
- Assumes linear relationships (might miss complex patterns)
- Can struggle if classes aren't linearly separable

**Best for**: Fast predictions, explainability, baseline comparisons

**My experience**: Consistently performs well (90-95% accuracy) and the explainability is excellent for building user trust.

### Random Forest

**What it is**: An ensemble of decision trees that vote on the final classification.

**Strengths**:
- Handles non-linear patterns well
- Robust to outliers and noisy data
- Often achieves highest accuracy
- Reduces overfitting through ensemble approach

**Weaknesses**:
- Slower training (2-5x longer than Logistic Regression)
- Larger model file size
- Less interpretable (harder to explain "why")
- Slower predictions

**Best for**: Maximum accuracy, production systems where speed is less critical

**My experience**: Usually gets 2-5% higher accuracy than Logistic Regression, but at the cost of speed and explainability.

### Naive Bayes

**What it is**: A probabilistic classifier based on Bayes' theorem with independence assumptions.

**Strengths**:
- Extremely fast (fastest of all models)
- Works surprisingly well with text
- Handles high-dimensional data efficiently
- Simple and requires little training data

**Weaknesses**:
- Assumes feature independence (rarely true in practice)
- Often has lower accuracy than other models
- Can be overconfident in predictions

**Best for**: Quick prototyping, real-time applications, very large datasets

**My experience**: Great for initial testing but usually 5-10% less accurate than Logistic Regression in my tests.

### Linear SVM (Support Vector Machine)

**What it is**: Finds the optimal hyperplane that separates different classes.

**Strengths**:
- Powerful for high-dimensional text data
- Good generalization
- Memory efficient with sparse data
- Handles large feature spaces well

**Weaknesses**:
- Longer training time
- Needs probability calibration (I handle this automatically)
- Sensitive to feature scaling
- Less interpretable than Logistic Regression

**Best for**: High-dimensional text classification, good balance of accuracy and speed

**My experience**: Performs similarly to Logistic Regression but takes longer to train. Good alternative if you need slightly different behavior.

### XGBoost (Optional)

**What it is**: An advanced gradient boosting algorithm that builds trees sequentially.

**Strengths**:
- Often achieves highest accuracy
- Built-in regularization prevents overfitting
- Can handle complex patterns
- Provides feature importance scores

**Weaknesses**:
- Requires separate installation (`pip install xgboost`)
- Slower training
- More hyperparameters to tune
- Less interpretable than simpler models

**Best for**: Competition-level accuracy, production systems with complex data

**My experience**: Usually ties with or slightly beats Random Forest for accuracy, but requires more setup.

## Reading the Results

When you run the comparison, you'll see a table like this:

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
```

### What Each Metric Means

**Accuracy**: Percentage of correct predictions
- Example: 0.92 = 92% correct
- Higher is better
- Good for getting a general sense of performance

**Precision**: Of all positive predictions, how many were actually correct?
- Answers: "When the model predicts UNSPSC code X, how often is it right?"
- Important when false positives are costly
- Higher is better

**Recall**: Of all actual positives, how many did we find?
- Answers: "Of all materials that should be code X, how many did we correctly identify?"
- Important when missing items is costly
- Higher is better

**F1-Score**: Harmonic mean of precision and recall
- Balances both precision and recall
- Good single metric for overall performance
- I usually focus on this one

**Training Time**: How long it took to train the model
- Important if you retrain frequently
- Not critical if you train once and use for months

**Prediction Speed**: How fast the model makes predictions
- Critical for real-time applications
- Important if processing large batches
- Measured in predictions per second

## Choosing the Right Model

Here's how I think about it:

### If You Want Maximum Accuracy
**Choose**: Random Forest or XGBoost
**Tradeoff**: Slower, less explainable
**Good for**: Production systems where accuracy is critical

### If You Want Speed
**Choose**: Logistic Regression or Naive Bayes
**Tradeoff**: Slightly lower accuracy
**Good for**: Real-time applications, batch processing

### If You Want Explainability
**Choose**: Logistic Regression
**Tradeoff**: Might miss complex patterns
**Good for**: When users need to understand why decisions were made

### If You Want Balance
**Choose**: Logistic Regression or Linear SVM
**Tradeoff**: Middle ground on everything
**Good for**: Most production use cases

## My Recommendation

For this material classification system, I recommend **Logistic Regression** (the current default) because:

1. **Explainability matters**: Users need to understand why a material was classified a certain way
2. **Speed is good**: Predictions in under 100ms keeps the UI responsive
3. **Accuracy is sufficient**: 90-95% accuracy is good enough for a decision support system
4. **Simple is better**: Easier to maintain and troubleshoot

However, if you find accuracy isn't high enough with your real data, try **Random Forest** - it often gets 2-5% better accuracy.

## Installing XGBoost

If you want to include XGBoost in the comparison:

```bash
# Activate your virtual environment first
source venv/bin/activate

# Install XGBoost
pip install xgboost

# Run the comparison
python train_model_comparison.py
```

XGBoost will now be included automatically.

## Customizing the Comparison

You can modify which models to compare by editing `models/multi_model_classifier.py`:

```python
# In the MultiModelTrainer class
self.models = {
    'Logistic Regression': LogisticRegression(...),
    'Random Forest': RandomForestClassifier(...),
    # Comment out models you don't want to test
    # 'Naive Bayes': MultinomialNB(...),
}
```

You can also tune the hyperparameters:

```python
'Random Forest': RandomForestClassifier(
    n_estimators=200,  # More trees (default: 100)
    max_depth=15,      # Deeper trees
    min_samples_split=5,  # Require more samples to split
    random_state=42,
    n_jobs=-1
)
```

## Saving Your Choice

After the comparison runs, you'll see options:

**Option 1: Save best model only**
- Automatically saves the highest-performing model
- Replaces the current default
- Simplest option - I recommend this

**Option 2: Save all models**
- Keeps all trained models
- Useful for later comparison
- Takes more disk space

**Option 3: Save specific model**
- You choose which one to save
- Good if you prefer speed over accuracy (or vice versa)

The saved model will be used automatically by the web app.

## When to Retrain

You should run the comparison and retrain when:

1. **You add new training data** - More data often changes which model works best
2. **Accuracy drops** - If you notice worse performance over time
3. **Requirements change** - If speed becomes more/less important
4. **New categories added** - When UNSPSC codes are added or changed
5. **Monthly/quarterly** - Regular retraining with accumulated data

## Troubleshooting

**"XGBoost not available" message**:
```bash
pip install xgboost
```

**Training takes too long**:
- Reduce `max_features` in TF-IDF vectorizer
- Use fewer estimators in Random Forest/XGBoost
- Train on a sample of data first

**Out of memory errors**:
- Reduce `max_features` in vectorizer
- Train models one at a time (edit the code)
- Use a machine with more RAM

**Models perform poorly**:
- Check training data quality
- Ensure you have enough examples (100+ per category minimum)
- Try different hyperparameters
- Verify data preprocessing is correct

## Example Workflow

Here's how I typically use this tool:

1. **Initial setup**: Train with default Logistic Regression
2. **Baseline**: Run comparison to see what's possible
3. **Choose**: Pick best model based on accuracy/speed needs
4. **Deploy**: Use in app for a week
5. **Evaluate**: Check real-world performance
6. **Iterate**: Adjust if needed

For this project, Logistic Regression was good enough, so I kept it as default. But knowing Random Forest could get 94% instead of 92% is useful information for future improvements.

## Results File

All comparison results are saved to:
```
trained_models/model_comparison_YYYYMMDD_HHMMSS.csv
```

You can open this in Excel to:
- Review historical comparisons
- Track performance over time
- Make charts and presentations
- Document model selection decisions

## Conclusion

Model comparison is about making informed decisions. Different models have different strengths, and the "best" one depends on your specific requirements.

For material classification:
- **Development**: Use Logistic Regression for speed and explainability
- **Production**: Consider Random Forest if you need that extra accuracy
- **Large scale**: Try Naive Bayes if processing millions of materials
- **Complex data**: XGBoost might help with very diverse materials

The comparison tool makes it easy to test all options and choose what works best for your situation.
