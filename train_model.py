"""
Training script for Material Classification Model
Loads data, trains the model, and saves it for later use

SUPPORTED DATA FORMATS:
1. CSV file: data/mock_materials.csv (generated data)
2. JSON file: data/mdg_multi_material_training_data_500.json (MDG data)
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from models.classifier import MaterialClassifier
import os


def load_training_data(data_source='csv'):
    """
    Load training data from either CSV or JSON source
    
    Args:
        data_source: 'csv' or 'json'
        
    Returns:
        pandas DataFrame with columns: Material_Description, UNSPSC_Code
    """
    if data_source == 'csv':
        data_path = "data/mock_materials.csv"
        if not os.path.exists(data_path):
            print(f"\nError: CSV file not found at {data_path}")
            print("Please run 'python utils/data_generator.py' first to generate mock data.")
            return None
        
        print(f"\nLoading data from {data_path}...")
        df = pd.read_csv(data_path)
        return df
    
    elif data_source == 'json':
        data_path = "data/mdg_multi_material_training_data_500.json"
        if not os.path.exists(data_path):
            print(f"\nError: JSON file not found at {data_path}")
            return None
        
        print(f"\nLoading data from {data_path}...")
        with open(data_path, 'r') as f:
            json_data = json.load(f)
        
        # Convert JSON to DataFrame format expected by classifier
        records = []
        for item in json_data:
            # Build enhanced description from material description and characteristics
            description_parts = [item['material_description']]
            
            # Add manufacturer and part number
            if item.get('manufacturer'):
                description_parts.append(f"Manufacturer: {item['manufacturer']}")
            if item.get('manufacturer_part_number'):
                description_parts.append(f"Part: {item['manufacturer_part_number']}")
            
            # Add characteristics
            if item.get('characteristics'):
                for char_name, char_value in item['characteristics'].items():
                    # Format characteristic name (remove prefixes and convert to readable format)
                    clean_name = char_name.replace('AAD375002_', '').replace('BAH609003_', '').replace('BAJ196003_', '').replace('BAI188003_', '')
                    clean_name = clean_name.replace('_', ' ').title()
                    description_parts.append(f"{clean_name}: {char_value}")
            
            records.append({
                'Material_Description': ' '.join(description_parts),
                'UNSPSC_Code': item['unspsc_class']
            })
        
        df = pd.DataFrame(records)
        return df
    
    else:
        print(f"\nError: Unknown data source '{data_source}'. Use 'csv' or 'json'")
        return None


def main():
    print("="*60)
    print("MATERIAL CLASSIFICATION MODEL TRAINING")
    print("="*60)
    
    # Ask user which data source to use
    print("\nAvailable data sources:")
    print("1. CSV - Generated mock data (data/mock_materials.csv)")
    print("2. JSON - MDG multi-material data (data/mdg_multi_material_training_data_500.json)")
    
    choice = input("\nSelect data source (1 or 2, default=2): ").strip()
    
    if choice == '1':
        data_source = 'csv'
    else:
        data_source = 'json'  # Default to JSON
    
    # Load data
    df = load_training_data(data_source)
    
    if df is None:
        return
    
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
