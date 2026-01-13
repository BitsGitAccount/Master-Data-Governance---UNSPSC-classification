"""
========================================
Material Classification Model
========================================

PURPOSE:
This module contains the ML model that classifies materials into UNSPSC codes.

HOW IT WORKS:
1. Uses TF-IDF (Term Frequency-Inverse Document Frequency) to convert text into numbers
2. Uses Logistic Regression to learn patterns between descriptions and UNSPSC codes
3. Returns top 5 predictions with confidence scores
4. Provides explainability showing which keywords influenced the decision

KEY CONCEPTS:
- TF-IDF: Converts text into numerical features based on word importance
- Logistic Regression: Statistical model that learns to classify based on features
- Confidence Score: Probability that the prediction is correct (0-100%)
- Explainability: Shows which words were most important for the classification

MAIN METHODS:
- train(): Train the model on material descriptions and their UNSPSC codes
- predict_with_confidence(): Classify materials and return top 5 predictions
- explain_prediction(): Show which keywords influenced the classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # Converts text to numbers
from sklearn.linear_model import LogisticRegression  # Classification algorithm
from sklearn.model_selection import train_test_split  # Split data into train/test
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  # Evaluation metrics
import joblib  # Save/load trained models
import os


# ============================================================================
# MATERIAL CLASSIFIER CLASS
# ============================================================================

class MaterialClassifier:
    """
    Material classification model with confidence scoring and explainability.
    
    This class handles the entire ML pipeline:
    - Text preprocessing
    - Feature extraction (TF-IDF)
    - Classification (Logistic Regression)
    - Confidence scoring
    - Explainability
    """
    
    def __init__(self):
        """
        Initialize the classifier with TF-IDF vectorizer and Logistic Regression.
        
        TF-IDF PARAMETERS EXPLAINED:
        - max_features=500: Only keep the 500 most important words (reduces noise)
        - ngram_range=(1,2): Consider both single words and 2-word phrases
        - stop_words='english': Remove common words like 'the', 'is', 'a'
        - min_df=2: Ignore words that appear in less than 2 documents
        
        LOGISTIC REGRESSION PARAMETERS EXPLAINED:
        - random_state=42: Makes results reproducible
        - max_iter=1000: Maximum training iterations (ensures convergence)
        - multi_class='multinomial': Handle multiple UNSPSC classes properly
        - solver='lbfgs': Optimization algorithm (good for multi-class)
        """
        # TF-IDF Vectorizer: Converts text descriptions into numerical features
        self.vectorizer = TfidfVectorizer(
            max_features=500,      # Limit to 500 most important words
            ngram_range=(1, 2),    # Use single words AND 2-word phrases
            stop_words='english',  # Remove common English words
            min_df=2               # Ignore rare words (appear in <2 documents)
        )
        
        # Logistic Regression Classifier: Learns to predict UNSPSC codes
        self.classifier = LogisticRegression(
            random_state=42,           # For reproducibility
            max_iter=1000,             # Max training iterations
            multi_class='multinomial', # Handle multiple classes
            solver='lbfgs'             # Optimization algorithm
        )
        
        # These will be populated during training/loading
        self.label_mapping = {}          # Maps UNSPSC codes to numbers (not currently used)
        self.reverse_label_mapping = {}  # Maps numbers back to UNSPSC codes (not currently used)
        self.feature_names = []          # List of words/phrases used as features
        
    def preprocess_text(self, text):
        """
        Clean and standardize text before processing.
        
        Steps:
        1. Handle missing/null values
        2. Convert to lowercase (so "Steel" and "steel" are treated the same)
        3. Remove extra whitespace
        
        Args:
            text: Raw material description
            
        Returns:
            Cleaned text string
        """
        # Handle missing values (NaN, None, etc.)
        if pd.isna(text):
            return ""
        
        # Convert to string, lowercase, and remove extra spaces
        return str(text).lower().strip()
    
    def train(self, X_train, y_train):
        """
        Train the classification model on material descriptions and UNSPSC codes.
        
        TRAINING PROCESS:
        1. Preprocess all text descriptions (lowercase, clean)
        2. Convert text to TF-IDF features (numbers)
        3. Train Logistic Regression on these features
        4. Save feature names for later explainability
        
        Args:
            X_train: List/array of material descriptions (text)
            y_train: List/array of corresponding UNSPSC codes (labels)
        """
        print("Preprocessing training data...")
        # Clean all text descriptions
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        
        print("Vectorizing text data...")
        # Convert text to TF-IDF numerical features
        # fit_transform: Learn vocabulary AND transform text
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        
        # Save feature names (words/phrases) for explainability later
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print("Training classifier...")
        # Train the Logistic Regression model
        # It learns: which features (words) predict which UNSPSC codes
        self.classifier.fit(X_train_vectorized, y_train)
        
        print("Training complete!")
        
    def predict(self, X_test):
        """
        Predict UNSPSC codes for new material descriptions.
        
        This is a simple version that just returns the predicted code.
        For more details (confidence, top predictions), use predict_with_confidence().
        
        Args:
            X_test: List/array of material descriptions to classify
            
        Returns:
            Array of predicted UNSPSC codes
        """
        # Preprocess text
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        
        # Convert to TF-IDF features using the SAME vocabulary learned during training
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        
        # Get predictions
        predictions = self.classifier.predict(X_test_vectorized)
        
        return predictions
    
    def predict_with_confidence(self, X_test):
        """
        Predict UNSPSC codes with confidence scores and top 5 predictions.
        
        This is the MAIN prediction method used in the app.
        
        WHAT IT RETURNS:
        For each input description, returns:
        - predicted_unspsc: The top predicted UNSPSC code
        - confidence: Probability of this prediction (0-1, where 1 = 100% confident)
        - top_predictions: List of top 5 UNSPSC codes with their probabilities
        
        Args:
            X_test: List/array of material descriptions to classify
            
        Returns:
            List of dictionaries, one per input description, containing:
            {
                'predicted_unspsc': '12345678',
                'confidence': 0.85,
                'top_predictions': [
                    {'unspsc_code': '12345678', 'probability': 0.85},
                    {'unspsc_code': '23456789', 'probability': 0.10},
                    ...
                ]
            }
        """
        # Preprocess and vectorize input text
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        
        # Get predictions (most likely UNSPSC code for each input)
        predictions = self.classifier.predict(X_test_vectorized)
        
        # Get probability scores for ALL possible UNSPSC codes
        # Shape: (num_inputs, num_classes)
        # Example: [[0.85, 0.10, 0.03, 0.02], ...] for 4 possible classes
        probabilities = self.classifier.predict_proba(X_test_vectorized)
        
        # Get confidence (highest probability for each prediction)
        confidences = np.max(probabilities, axis=1)
        
        # Build results for each input
        results = []
        for pred, conf, probs in zip(predictions, confidences, probabilities):
            # Get indices of top 5 predictions
            # argsort gives indices sorted by value, [::-1] reverses to descending, [:5] takes top 5
            top_indices = np.argsort(probs)[::-1][:5]
            
            # Create list of top 5 predictions with their probabilities
            top_predictions = [
                {
                    'unspsc_code': self.classifier.classes_[idx],  # Get UNSPSC code
                    'probability': float(probs[idx])                # Get probability
                }
                for idx in top_indices
            ]
            
            # Add result for this input
            results.append({
                'predicted_unspsc': pred,           # Top prediction
                'confidence': float(conf),          # Confidence (0-1)
                'top_predictions': top_predictions  # Top 5 predictions
            })
        
        return results
    
    def explain_prediction(self, text, prediction_result):
        """
        Explain WHY the model made a particular prediction.
        
        Shows which words/phrases in the description were most influential
        in predicting the UNSPSC code.
        
        HOW IT WORKS:
        1. Find which words from the description are in the model's vocabulary
        2. Get the model's "weight" for each word (how important it is for this UNSPSC code)
        3. Multiply weight × TF-IDF score to get importance
        4. Return top 5 most important words
        
        Args:
            text: Original material description
            prediction_result: Output from predict_with_confidence()
            
        Returns:
            Dictionary with:
            {
                'influential_words': [
                    {'word': 'plastic', 'importance': 0.45, 'weight': 2.3, 'tfidf': 0.19},
                    ...
                ],
                'explanation': "Classification was influenced by keywords: 'plastic', 'packaging'..."
            }
        """
        # Preprocess and vectorize the text
        text_processed = self.preprocess_text(text)
        text_vectorized = self.vectorizer.transform([text_processed])
        
        # Get the predicted UNSPSC code
        predicted_class = prediction_result['predicted_unspsc']
        
        # Find this class's index in the model
        class_index = np.where(self.classifier.classes_ == predicted_class)[0][0]
        
        # Get the model's learned weights for this UNSPSC code
        # These weights indicate how much each word/phrase influences this prediction
        coefficients = self.classifier.coef_[class_index]
        
        # Get indices of non-zero features (words that appear in the description)
        feature_indices = text_vectorized.nonzero()[1]
        
        # If no features found, return empty result
        if len(feature_indices) == 0:
            return {
                'influential_words': [],
                'explanation': "No significant features found in the text."
            }
        
        # Calculate importance for each word that appears in the description
        feature_weights = []
        for idx in feature_indices:
            feature_name = self.feature_names[idx]  # Get the actual word/phrase
            weight = coefficients[idx]               # Model's weight for this word
            tfidf_value = text_vectorized[0, idx]   # TF-IDF score in this description
            
            # Importance = weight × TF-IDF
            # High weight + high TF-IDF = very important for classification
            importance = weight * tfidf_value
            
            feature_weights.append({
                'word': feature_name,
                'importance': float(importance),  # Combined importance score
                'weight': float(weight),          # Model's learned weight
                'tfidf': float(tfidf_value)      # TF-IDF score in this text
            })
        
        # Sort by absolute importance (most influential words first)
        feature_weights.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        # Take top 5 most influential words
        top_features = feature_weights[:5]
        
        # Create human-readable explanation
        if top_features:
            # Get top 3 words for the explanation text
            top_words = [f"'{fw['word']}'" for fw in top_features[:3]]
            explanation = f"Classification was influenced by keywords: {', '.join(top_words)}. "
            explanation += f"These terms are strongly associated with the predicted UNSPSC category."
        else:
            explanation = "Classification based on general text patterns."
        
        return {
            'influential_words': top_features,  # Detailed list of influential words
            'explanation': explanation          # Simple text explanation
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Calculates accuracy and shows detailed classification report.
        
        Args:
            X_test: Test material descriptions
            y_test: True UNSPSC codes for test data
            
        Returns:
            Dictionary with accuracy and predictions
        """
        # Get predictions for test data
        predictions = self.predict(X_test)
        
        # Print evaluation metrics
        print("\n=== Model Evaluation ===")
        print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'predictions': predictions
        }
    
    def save_model(self, model_path='trained_models'):
        """
        Save the trained model to disk for later use.
        
        Saves two files:
        - vectorizer.pkl: TF-IDF vocabulary and parameters
        - classifier.pkl: Trained Logistic Regression model
        
        Args:
            model_path: Directory to save model files
        """
        # Create directory if it doesn't exist
        os.makedirs(model_path, exist_ok=True)
        
        # Save vectorizer (vocabulary + TF-IDF parameters)
        joblib.dump(self.vectorizer, os.path.join(model_path, 'vectorizer.pkl'))
        
        # Save classifier (trained model weights)
        joblib.dump(self.classifier, os.path.join(model_path, 'classifier.pkl'))
        
        print(f"Model saved to {model_path}/")
    
    def load_model(self, model_path='trained_models'):
        """
        Load a previously trained model from disk.
        
        Loads:
        - vectorizer.pkl: TF-IDF vocabulary and parameters
        - classifier.pkl: Trained Logistic Regression model
        
        Args:
            model_path: Directory containing model files
        """
        # Load vectorizer
        self.vectorizer = joblib.load(os.path.join(model_path, 'vectorizer.pkl'))
        
        # Load classifier
        self.classifier = joblib.load(os.path.join(model_path, 'classifier.pkl'))
        
        # Extract feature names for explainability
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"Model loaded from {model_path}/")
