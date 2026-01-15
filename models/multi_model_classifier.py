"""
========================================
Multi-Model Material Classification
========================================

PURPOSE:
This module trains and compares multiple ML models for material classification,
allowing you to choose the best performing model for your use case.

SUPPORTED MODELS:
1. Logistic Regression - Fast, interpretable (current default)
2. Random Forest - Ensemble method, handles non-linear patterns
3. Naive Bayes - Probabilistic, works well with text
4. SVM (Support Vector Machine) - Powerful for high-dimensional data
5. XGBoost - Advanced gradient boosting, often highest accuracy

COMPARISON METRICS:
- Accuracy: Overall correctness
- Precision: How many predicted positives are actually positive
- Recall: How many actual positives are correctly identified
- F1-Score: Harmonic mean of precision and recall
- Training Time: How long it takes to train the model
- Prediction Time: How fast the model makes predictions

USAGE:
    trainer = MultiModelTrainer()
    results = trainer.train_and_compare_models(X_train, y_train, X_test, y_test)
    best_model = trainer.get_best_model()
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import joblib
import os
import time
from datetime import datetime


class MultiModelTrainer:
    """
    Train and compare multiple classification models for material classification.
    """
    
    def __init__(self):
        """
        Initialize the multi-model trainer with various ML algorithms.
        """
        # TF-IDF Vectorizer (same for all models)
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=2
        )
        
        # Define multiple models to compare
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),
            'Naive Bayes': MultinomialNB(
                alpha=1.0
            ),
            'Linear SVM': LinearSVC(
                random_state=42,
                max_iter=1000,
                dual=False
            )
        }
        
        # Try to import XGBoost (optional dependency)
        try:
            from xgboost import XGBClassifier
            self.models['XGBoost'] = XGBClassifier(
                n_estimators=100,
                random_state=42,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        except ImportError:
            print("Note: XGBoost not installed. Install with: pip install xgboost")
        
        self.trained_models = {}
        self.comparison_results = {}
        self.feature_names = []
        self.best_model_name = None
        
    def preprocess_text(self, text):
        """Clean and standardize text."""
        if pd.isna(text):
            return ""
        return str(text).lower().strip()
    
    def train_and_compare_models(self, X_train, y_train, X_test, y_test):
        """
        Train all models and compare their performance.
        
        Args:
            X_train: Training descriptions
            y_train: Training labels (UNSPSC codes)
            X_test: Test descriptions
            y_test: Test labels (UNSPSC codes)
            
        Returns:
            DataFrame with comparison results for all models
        """
        print("="*70)
        print("TRAINING AND COMPARING MULTIPLE MODELS")
        print("="*70)
        
        # Preprocess text
        print("\nPreprocessing text data...")
        X_train_processed = [self.preprocess_text(text) for text in X_train]
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        
        # Vectorize text
        print("Vectorizing text data...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train_processed)
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Train and evaluate each model
        results = []
        
        for model_name, model in self.models.items():
            print(f"\n{'='*70}")
            print(f"Training: {model_name}")
            print(f"{'='*70}")
            
            try:
                # Measure training time
                train_start = time.time()
                model.fit(X_train_vectorized, y_train)
                train_time = time.time() - train_start
                
                # Measure prediction time
                pred_start = time.time()
                y_pred = model.predict(X_test_vectorized)
                pred_time = time.time() - pred_start
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Store results
                result = {
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Training Time (s)': train_time,
                    'Prediction Time (s)': pred_time,
                    'Predictions per Second': len(X_test) / pred_time if pred_time > 0 else 0
                }
                results.append(result)
                
                # Store trained model
                self.trained_models[model_name] = model
                
                # Print results
                print(f"✓ Training completed in {train_time:.2f}s")
                print(f"  Accuracy: {accuracy:.4f}")
                print(f"  Precision: {precision:.4f}")
                print(f"  Recall: {recall:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Prediction Speed: {len(X_test)/pred_time:.0f} predictions/sec")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {str(e)}")
                results.append({
                    'Model': model_name,
                    'Accuracy': 0,
                    'Precision': 0,
                    'Recall': 0,
                    'F1-Score': 0,
                    'Training Time (s)': 0,
                    'Prediction Time (s)': 0,
                    'Predictions per Second': 0,
                    'Error': str(e)
                })
        
        # Create comparison DataFrame
        self.comparison_results = pd.DataFrame(results)
        
        # Sort by accuracy (descending)
        self.comparison_results = self.comparison_results.sort_values(
            by='Accuracy', ascending=False
        ).reset_index(drop=True)
        
        # Determine best model
        if len(self.comparison_results) > 0 and self.comparison_results['Accuracy'].max() > 0:
            self.best_model_name = self.comparison_results.iloc[0]['Model']
        
        return self.comparison_results
    
    def print_detailed_comparison(self):
        """
        Print a detailed comparison of all models.
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON SUMMARY")
        print("="*70)
        
        if self.comparison_results.empty:
            print("No models trained yet.")
            return
        
        # Print comparison table
        print("\n" + self.comparison_results.to_string(index=False))
        
        # Highlight best model
        if self.best_model_name:
            print(f"\n{'='*70}")
            print(f"🏆 BEST MODEL: {self.best_model_name}")
            print(f"{'='*70}")
            best_row = self.comparison_results[
                self.comparison_results['Model'] == self.best_model_name
            ].iloc[0]
            print(f"  Accuracy: {best_row['Accuracy']:.4f}")
            print(f"  F1-Score: {best_row['F1-Score']:.4f}")
            print(f"  Training Time: {best_row['Training Time (s)']:.2f}s")
            print(f"  Prediction Speed: {best_row['Predictions per Second']:.0f} predictions/sec")
    
    def get_best_model(self):
        """
        Get the best performing model.
        
        Returns:
            Tuple of (model_name, model_object)
        """
        if self.best_model_name and self.best_model_name in self.trained_models:
            return self.best_model_name, self.trained_models[self.best_model_name]
        return None, None
    
    def save_model(self, model_name, model_path='trained_models'):
        """
        Save a specific trained model.
        
        Args:
            model_name: Name of the model to save
            model_path: Directory to save the model
        """
        if model_name not in self.trained_models:
            print(f"Error: Model '{model_name}' not found in trained models.")
            return
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(
            self.vectorizer,
            os.path.join(model_path, 'vectorizer.pkl')
        )
        
        # Save model with custom name
        model_filename = f"classifier_{model_name.lower().replace(' ', '_')}.pkl"
        joblib.dump(
            self.trained_models[model_name],
            os.path.join(model_path, model_filename)
        )
        
        print(f"✓ {model_name} saved to {model_path}/{model_filename}")
    
    def save_all_models(self, model_path='trained_models'):
        """
        Save all trained models.
        
        Args:
            model_path: Directory to save models
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Save vectorizer (shared by all models)
        joblib.dump(
            self.vectorizer,
            os.path.join(model_path, 'vectorizer.pkl')
        )
        print(f"✓ Vectorizer saved")
        
        # Save each model
        for model_name, model in self.trained_models.items():
            model_filename = f"classifier_{model_name.lower().replace(' ', '_')}.pkl"
            joblib.dump(
                model,
                os.path.join(model_path, model_filename)
            )
            print(f"✓ {model_name} saved")
        
        # Save comparison results
        if not self.comparison_results.empty:
            results_filename = os.path.join(
                model_path,
                f'model_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            )
            self.comparison_results.to_csv(results_filename, index=False)
            print(f"✓ Comparison results saved to {results_filename}")
    
    def save_best_model_as_default(self, model_path='trained_models'):
        """
        Save the best performing model as the default classifier.pkl.
        
        Args:
            model_path: Directory to save the model
        """
        if not self.best_model_name:
            print("Error: No best model determined yet. Train models first.")
            return
        
        os.makedirs(model_path, exist_ok=True)
        
        # Save vectorizer
        joblib.dump(
            self.vectorizer,
            os.path.join(model_path, 'vectorizer.pkl')
        )
        
        # Save best model as default classifier.pkl
        joblib.dump(
            self.trained_models[self.best_model_name],
            os.path.join(model_path, 'classifier.pkl')
        )
        
        print(f"\n{'='*70}")
        print(f"✓ Best model ({self.best_model_name}) saved as default classifier")
        print(f"  Location: {model_path}/classifier.pkl")
        print(f"  The web app will now use this model for predictions")
        print(f"{'='*70}")
    
    def predict_with_confidence(self, X_test, model_name=None):
        """
        Make predictions with confidence scores using a specific model.
        
        Args:
            X_test: List of material descriptions
            model_name: Name of model to use (default: best model)
            
        Returns:
            List of prediction results with confidence scores
        """
        # Use best model if not specified
        if model_name is None:
            model_name = self.best_model_name
        
        if model_name not in self.trained_models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.trained_models[model_name]
        
        # Preprocess and vectorize
        X_test_processed = [self.preprocess_text(text) for text in X_test]
        X_test_vectorized = self.vectorizer.transform(X_test_processed)
        
        # Get predictions
        predictions = model.predict(X_test_vectorized)
        
        # Get probabilities (if supported)
        try:
            probabilities = model.predict_proba(X_test_vectorized)
            confidences = np.max(probabilities, axis=1)
            
            # Build results
            results = []
            for pred, conf, probs in zip(predictions, confidences, probabilities):
                top_indices = np.argsort(probs)[::-1][:5]
                
                top_predictions = [
                    {
                        'unspsc_code': model.classes_[idx],
                        'probability': float(probs[idx])
                    }
                    for idx in top_indices
                ]
                
                results.append({
                    'predicted_unspsc': pred,
                    'confidence': float(conf),
                    'top_predictions': top_predictions,
                    'model_used': model_name
                })
            
            return results
            
        except AttributeError:
            # Model doesn't support predict_proba (e.g., LinearSVC)
            results = []
            for pred in predictions:
                results.append({
                    'predicted_unspsc': pred,
                    'confidence': None,
                    'top_predictions': [{'unspsc_code': pred, 'probability': None}],
                    'model_used': model_name
                })
            return results
