"""
Training script for Material Classification Model
Loads data, trains the model, and saves it for later use
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from models.classifier import MaterialClassifier
import os


def main():
    print("="*60)
    print("MATERIAL CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Check if data exists
    data_path = "data/mock_materials.csv"
    if not os.path.exists(data_path):
        print(f"\nError: Data file not found at {data_path}")
        print("Please run 'python utils/data_generator.py' first to generate mock data.")
        return
    
    # Load data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} material records")
    
    # Display data statistics
    print("\nData Statistics:")
    print(f"- Total samples: {len(df)}")
    print(f"- Number of UNSPSC categories: {df['UNSPSC_Code'].nunique()}")
    print("\nUNSPSC Code Distribution:")
    print(df['UNSPSC_Code'].value_counts())
    
    # Prepare features and labels
    X = df['Material_Description'].values
    y = df['UNSPSC_Code'].values
    
    # Split data
    print("\nSplitting data into train (80%) and test (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize and train model
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    classifier = MaterialClassifier()
    classifier.train(X_train, y_train)
    
    # Evaluate model
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    eval_results = classifier.evaluate(X_test, y_test)
    
    # Test prediction with confidence
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS WITH CONFIDENCE")
    print("="*60)
    
    # Test on a few examples
    sample_indices = [0, 1, 2]
    for idx in sample_indices:
        test_description = X_test[idx]
        actual_unspsc = y_test[idx]
        
        print(f"\n--- Sample {idx + 1} ---")
        print(f"Description: {test_description}")
        print(f"Actual UNSPSC: {actual_unspsc}")
        
        # Get prediction with confidence
        results = classifier.predict_with_confidence([test_description])
        result = results[0]
        
        print(f"Predicted UNSPSC: {result['predicted_unspsc']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        # Get explanation
        explanation = classifier.explain_prediction(test_description, result)
        print(f"Explanation: {explanation['explanation']}")
        
        # Show top predictions
        print("\nTop 3 Predictions:")
        for i, pred in enumerate(result['top_predictions'], 1):
            print(f"  {i}. {pred['unspsc_code']} (probability: {pred['probability']:.2%})")
        
        # Show influential words
        if explanation['influential_words']:
            print("\nTop Influential Words:")
            for word_data in explanation['influential_words'][:3]:
                print(f"  - '{word_data['word']}' (importance: {word_data['importance']:.4f})")
    
    # Save model
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    
    classifier.save_model('trained_models')
    
    print("\n✓ Training complete!")
    print("✓ Model saved to 'trained_models/' directory")
    print("\nYou can now run the application with: streamlit run app.py")


if __name__ == "__main__":
    main()