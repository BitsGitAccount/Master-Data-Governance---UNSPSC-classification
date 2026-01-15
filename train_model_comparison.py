"""
Multi-Model Training and Comparison Script
Trains multiple ML models and compares their performance

This script extends the original train_model.py to train and compare
multiple models, helping you choose the best one for your use case.
"""

import pandas as pd
import json
from sklearn.model_selection import train_test_split
from models.multi_model_classifier import MultiModelTrainer
import os


def load_training_data(data_source='json'):
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
                    # Format characteristic name
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
    print("="*70)
    print("MULTI-MODEL MATERIAL CLASSIFICATION TRAINING")
    print("="*70)
    
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
    
    # Initialize multi-model trainer
    print("\n" + "="*70)
    print("INITIALIZING MULTI-MODEL TRAINER")
    print("="*70)
    
    trainer = MultiModelTrainer()
    
    print("\nModels to be trained and compared:")
    for i, model_name in enumerate(trainer.models.keys(), 1):
        print(f"  {i}. {model_name}")
    
    # Train and compare all models
    comparison_results = trainer.train_and_compare_models(X_train, y_train, X_test, y_test)
    
    # Print detailed comparison
    trainer.print_detailed_comparison()
    
    # Test predictions with best model
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS WITH BEST MODEL")
    print("="*70)
    
    best_model_name, _ = trainer.get_best_model()
    
    if best_model_name:
        # Test on a few examples
        sample_indices = [0, 1, 2]
        for idx in sample_indices:
            test_description = X_test[idx]
            actual_unspsc = y_test[idx]
            
            print(f"\n--- Sample {idx + 1} ---")
            print(f"Description: {test_description[:100]}...")
            print(f"Actual UNSPSC: {actual_unspsc}")
            
            # Get prediction with confidence
            results = trainer.predict_with_confidence([test_description])
            result = results[0]
            
            print(f"Predicted UNSPSC: {result['predicted_unspsc']}")
            if result['confidence'] is not None:
                print(f"Confidence: {result['confidence']:.2%}")
            print(f"Model Used: {result['model_used']}")
            
            # Show top predictions
            if result['top_predictions']:
                print("\nTop 3 Predictions:")
                for i, pred in enumerate(result['top_predictions'][:3], 1):
                    if pred['probability'] is not None:
                        print(f"  {i}. {pred['unspsc_code']} (probability: {pred['probability']:.2%})")
                    else:
                        print(f"  {i}. {pred['unspsc_code']}")
    
    # Save models
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    print("\nOptions:")
    print("1. Save best model only as default (recommended)")
    print("2. Save all models for later comparison")
    print("3. Save specific model")
    
    save_choice = input("\nSelect option (1, 2, or 3, default=1): ").strip()
    
    if save_choice == '2':
        trainer.save_all_models('trained_models')
        print(f"\n✓ All models saved to 'trained_models/' directory")
    elif save_choice == '3':
        print("\nAvailable models:")
        for i, model_name in enumerate(trainer.trained_models.keys(), 1):
            print(f"  {i}. {model_name}")
        model_idx = input("\nSelect model number: ").strip()
        try:
            model_name = list(trainer.trained_models.keys())[int(model_idx) - 1]
            trainer.save_model(model_name, 'trained_models')
            
            # Ask if they want to set as default
            set_default = input(f"\nSet {model_name} as default for the app? (y/n, default=n): ").strip().lower()
            if set_default == 'y':
                trainer.save_best_model_as_default('trained_models')
        except (ValueError, IndexError):
            print("Invalid selection")
    else:
        # Default: Save best model as default
        trainer.save_best_model_as_default('trained_models')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review the model comparison results above")
    print("2. Run the application: streamlit run app.py")
    print("3. Test the predictions with real material descriptions")
    print("\nNote: The web app uses the default classifier.pkl file")


if __name__ == "__main__":
    main()
