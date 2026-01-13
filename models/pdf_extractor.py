"""
PDF Attribute Extractor
Extracts material attributes from Technical Data Sheets (TDS) PDFs
"""

import re
import fitz  # PyMuPDF
from typing import Dict, List, Tuple, Optional


class PDFAttributeExtractor:
    """Extract material attributes from PDF documents with explainability"""
    
    def __init__(self):
        # Regex patterns for different attributes
        self.patterns = {
            'weight': [
                r'weight[:\s]+(\d+\.?\d*)\s*(kg|g|lbs?|pounds?)',
                r'mass[:\s]+(\d+\.?\d*)\s*(kg|g|lbs?|pounds?)',
                r'(\d+\.?\d*)\s*(kg|g|lbs?|pounds?)\s+weight',
            ],
            'dimensions': [
                r'dimensions?[:\s]+(\d+\.?\d*)\s*(?:cm|mm|m|inch|in|")\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m|inch|in|")\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m|inch|in|")',
                r'size[:\s]+(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m|inch|in)',
                r'(\d+\.?\d*)\s*(?:cm|mm|m|inch|in)\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m|inch|in)\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m|inch|in)',
            ],
            'manufacturer': [
                r'manufacturer[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$|[,\.])',
                r'made by[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$|[,\.])',
                r'produced by[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$|[,\.])',
            ],
            'material_id': [
                r'material\s+id[:\s]+(MAT\d+)',
                r'product\s+id[:\s]+(MAT\d+)',
                r'item\s+id[:\s]+(MAT\d+)',
            ],
            'mpn': [
                r'part\s+number[:\s]+([A-Z]{3}\d{3})',
                r'MPN[:\s]+([A-Z]{3}\d{3})',
                r'model[:\s]+([A-Z]{3}\d{3})',
            ]
        }
    
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF with page and position information"""
        doc = fitz.open(pdf_path)
        pages_data = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            pages_data.append({
                'page_number': page_num + 1,
                'text': text,
                'full_text': text  # Keep full text for context
            })
        
        doc.close()
        return pages_data
    
    def find_attribute_in_text(self, text: str, attribute_name: str, patterns: List[str]) -> Optional[Dict]:
        """Find an attribute in text using regex patterns"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Extract the matched text and its context
                start_pos = max(0, match.start() - 50)
                end_pos = min(len(text), match.end() + 50)
                context = text[start_pos:end_pos].strip()
                
                # Format the extracted value
                if attribute_name == 'weight':
                    value = f"{match.group(1)} {match.group(2)}"
                elif attribute_name == 'dimensions':
                    value = f"{match.group(1)} x {match.group(2)} x {match.group(3)}"
                else:
                    value = match.group(1).strip()
                
                return {
                    'value': value,
                    'matched_text': match.group(0),
                    'context': context,
                    'pattern_used': pattern,
                    'confidence': self._calculate_confidence(match, pattern)
                }
        
        return None
    
    def _calculate_confidence(self, match, pattern: str) -> float:
        """Calculate confidence score based on match quality"""
        # Base confidence
        confidence = 0.7
        
        # Increase confidence if match is exact
        if ':' in match.group(0):
            confidence += 0.15
        
        # Increase confidence if pattern is more specific
        if 'manufacturer' in pattern.lower() or 'weight' in pattern.lower():
            confidence += 0.1
        
        # Cap at 0.95
        return min(confidence, 0.95)
    
    def extract_attributes(self, pdf_path: str) -> Dict:
        """Extract all attributes from a PDF"""
        print(f"Extracting attributes from: {pdf_path}")
        
        # Extract text from PDF
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        # Combine all text for searching
        full_text = "\n".join([page['text'] for page in pages_data])
        
        # Extract each attribute
        extracted_attributes = {}
        
        for attribute_name, patterns in self.patterns.items():
            result = self.find_attribute_in_text(full_text, attribute_name, patterns)
            
            if result:
                # Find which page this was on
                page_number = None
                for page in pages_data:
                    if result['matched_text'] in page['text']:
                        page_number = page['page_number']
                        break
                
                extracted_attributes[attribute_name] = {
                    'value': result['value'],
                    'confidence': result['confidence'],
                    'source': {
                        'page': page_number,
                        'matched_text': result['matched_text'],
                        'context': result['context']
                    },
                    'explanation': f"Found '{attribute_name}' on page {page_number}: '{result['matched_text']}'"
                }
            else:
                extracted_attributes[attribute_name] = {
                    'value': None,
                    'confidence': 0.0,
                    'source': None,
                    'explanation': f"Could not find '{attribute_name}' in the document"
                }
        
        return {
            'pdf_path': pdf_path,
            'attributes': extracted_attributes,
            'total_pages': len(pages_data)
        }
    
    def validate_extraction(self, extracted_data: Dict) -> Dict:
        """Validate extracted attributes and provide quality score"""
        attributes = extracted_data['attributes']
        
        # Count extracted attributes
        extracted_count = sum(1 for attr in attributes.values() if attr['value'] is not None)
        total_count = len(attributes)
        
        # Calculate average confidence
        confidences = [attr['confidence'] for attr in attributes.values() if attr['value'] is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        # Validation results
        validation = {
            'extraction_rate': extracted_count / total_count,
            'average_confidence': avg_confidence,
            'extracted_attributes': extracted_count,
            'total_attributes': total_count,
            'quality_score': (extracted_count / total_count) * avg_confidence,
            'status': 'good' if avg_confidence > 0.7 else 'needs_review'
        }
        
        return validation
    
    def extract_with_explainability(self, pdf_path: str) -> Dict:
        """Extract attributes with full explainability"""
        extraction_result = self.extract_attributes(pdf_path)
        validation_result = self.validate_extraction(extraction_result)
        
        return {
            'extraction': extraction_result,
            'validation': validation_result,
            'explainability': {
                'method': 'Regex pattern matching on extracted PDF text',
                'confidence_calculation': 'Based on match quality and pattern specificity',
                'attributes_found': [
                    name for name, data in extraction_result['attributes'].items()
                    if data['value'] is not None
                ],
                'attributes_missing': [
                    name for name, data in extraction_result['attributes'].items()
                    if data['value'] is None
                ]
            }
        }


def demo_extraction(pdf_path: str):
    """Demo function to show extraction capabilities"""
    extractor = PDFAttributeExtractor()
    result = extractor.extract_with_explainability(pdf_path)
    
    print("\n" + "="*60)
    print("PDF ATTRIBUTE EXTRACTION RESULTS")
    print("="*60)
    
    print(f"\nPDF: {result['extraction']['pdf_path']}")
    print(f"Total Pages: {result['extraction']['total_pages']}")
    
    print("\n--- Extracted Attributes ---")
    for attr_name, attr_data in result['extraction']['attributes'].items():
        print(f"\n{attr_name.upper()}:")
        if attr_data['value']:
            print(f"  Value: {attr_data['value']}")
            print(f"  Confidence: {attr_data['confidence']:.2%}")
            print(f"  Source: {attr_data['explanation']}")
        else:
            print(f"  Status: Not found")
    
    print("\n--- Validation Results ---")
    val = result['validation']
    print(f"Extraction Rate: {val['extraction_rate']:.1%}")
    print(f"Average Confidence: {val['average_confidence']:.2%}")
    print(f"Quality Score: {val['quality_score']:.2%}")
    print(f"Status: {val['status']}")
    
    print("\n--- Explainability ---")
    exp = result['explainability']
    print(f"Method: {exp['method']}")
    print(f"Attributes Found: {', '.join(exp['attributes_found'])}")
    if exp['attributes_missing']:
        print(f"Attributes Missing: {', '.join(exp['attributes_missing'])}")
    
    return result


if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        demo_extraction(pdf_path)
    else:
        print("Usage: python pdf_extractor.py <path_to_pdf>")