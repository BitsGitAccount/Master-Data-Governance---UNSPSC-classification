"""
Enhanced Material Classification System
Combines exact matching, similarity search, and ML for accurate UNSPSC classification
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from models.classifier import MaterialClassifier
import json


class EnhancedMaterialClassifier:
    """
    Enhanced classifier that combines:
    1. Exact/near-exact matching for known materials
    2. Similarity-based search for close matches
    3. ML-based classification as fallback
    """
    
    def __init__(self):
        """Initialize the enhanced classifier"""
        self.ml_classifier = MaterialClassifier()
        self.training_data = []
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1  # Allow all terms for small datasets
        )
        self.training_vectors = None
        
    def load_training_data(self, json_path='data/mdg_multi_material_training_data_500.json'):
        """Load and store training data for similarity matching"""
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        self.training_data = []
        for item in json_data:
            # Build description same way as training
            description_parts = [item['material_description']]
            
            if item.get('manufacturer'):
                description_parts.append(f"Manufacturer: {item['manufacturer']}")
            if item.get('manufacturer_part_number'):
                description_parts.append(f"Part Number: {item['manufacturer_part_number']}")
            if item.get('additional_text'):
                description_parts.append(item['additional_text'])
            
            # Add attributes
            if item.get('labels') and item['labels'].get('attributes'):
                for attr in item['labels']['attributes']:
                    attr_name = attr.get('name', '')
                    attr_value = attr.get('value')
                    attr_unit = attr.get('unit', '')
                    
                    if attr_value is not None:
                        if isinstance(attr_value, bool):
                            description_parts.append(f"{attr_name}: {attr_value}")
                        elif attr_unit:
                            description_parts.append(f"{attr_name}: {attr_value} {attr_unit}")
                        else:
                            description_parts.append(f"{attr_name}: {attr_value}")
            
            # Get UNSPSC
            unspsc_code = None
            if item.get('labels') and item['labels'].get('unspsc_final'):
                unspsc_code = item['labels']['unspsc_final']
            elif item.get('unspsc_class'):
                unspsc_code = item['unspsc_class']
            
            if unspsc_code:
                full_description = ' '.join(description_parts)
                self.training_data.append({
                    'id': item.get('id'),
                    'material_description': item['material_description'],
                    'manufacturer': item.get('manufacturer'),
                    'part_number': item.get('manufacturer_part_number'),
                    'full_description': full_description,
                    'unspsc': unspsc_code
                })
        
        # Create TF-IDF vectors for all training samples
        descriptions = [item['full_description'] for item in self.training_data]
        self.training_vectors = self.vectorizer.fit_transform(descriptions)
        
    def find_similar_materials(self, input_description, top_k=5):
        """Find most similar materials from training data"""
        if not self.training_data or self.training_vectors is None:
            return []
        
        # Vectorize input
        input_vector = self.vectorizer.transform([input_description])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(input_vector, self.training_vectors)[0]
        
        # Get top k most similar
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'training_item': self.training_data[idx],
                'similarity': float(similarities[idx]),
                'unspsc': self.training_data[idx]['unspsc']
            })
        
        return results
    
    def predict_with_confidence(self, descriptions):
        """
        Predict UNSPSC with enhanced accuracy using similarity matching
        """
        results = []
        
        for description in descriptions:
            # Find similar materials
            similar_materials = self.find_similar_materials(description, top_k=5)
            
            if similar_materials and similar_materials[0]['similarity'] > 0.3:
                # Use similarity-based prediction
                # Weight predictions by similarity
                unspsc_scores = {}
                for match in similar_materials:
                    unspsc = match['unspsc']
                    sim = match['similarity']
                    if unspsc in unspsc_scores:
                        unspsc_scores[unspsc] += sim
                    else:
                        unspsc_scores[unspsc] = sim
                
                # Sort by score
                sorted_predictions = sorted(
                    unspsc_scores.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Normalize scores to probabilities
                total_score = sum(score for _, score in sorted_predictions)
                
                top_predictions = [
                    {
                        'unspsc_code': unspsc,
                        'probability': float(score / total_score),
                        'source': 'similarity_matching'
                    }
                    for unspsc, score in sorted_predictions[:5]
                ]
                
                # Add any missing slots from ML if needed
                if len(top_predictions) < 5:
                    ml_results = self.ml_classifier.predict_with_confidence([description])
                    ml_preds = ml_results[0]['top_predictions']
                    
                    existing_codes = {p['unspsc_code'] for p in top_predictions}
                    for ml_pred in ml_preds:
                        if ml_pred['unspsc_code'] not in existing_codes:
                            ml_pred['source'] = 'ml_fallback'
                            top_predictions.append(ml_pred)
                            if len(top_predictions) >= 5:
                                break
                
                result = {
                    'predicted_unspsc': top_predictions[0]['unspsc_code'],
                    'confidence': top_predictions[0]['probability'],
                    'top_predictions': top_predictions[:5],
                    'method': 'similarity_matching',
                    'best_match': similar_materials[0]['training_item'],
                    'match_similarity': similar_materials[0]['similarity']
                }
            else:
                # Fall back to ML classifier
                ml_results = self.ml_classifier.predict_with_confidence([description])
                result = ml_results[0]
                result['method'] = 'ml_classification'
                result['match_similarity'] = 0.0
            
            results.append(result)
        
        return results
    
    def explain_prediction(self, text, prediction_result):
        """Explain the prediction"""
        if prediction_result.get('method') == 'similarity_matching':
            best_match = prediction_result.get('best_match', {})
            similarity = prediction_result.get('match_similarity', 0)
            
            explanation = {
                'explanation': f"Found highly similar material in training data (similarity: {similarity:.1%}). "
                              f"Matched with: '{best_match.get('material_description', 'Unknown')}'",
                'method': 'similarity_matching',
                'match_details': best_match,
                'similarity_score': similarity,
                'influential_words': []
            }
        else:
            # Use ML classifier explanation
            explanation = self.ml_classifier.explain_prediction(text, prediction_result)
            explanation['method'] = 'ml_classification'
        
        return explanation
