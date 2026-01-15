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
        """
        Initialize extractor with comprehensive patterns for SAP MDG integration.
        These characteristics map to standard SAP Material Master fields.
        """
        self.patterns = {
            # Technical Specifications
            'max_flow': [
                r'Max\.?\s*flow[:\s]+(\d+)\s*(m³?/h|m3/h|GPM|l/h|liters?/hour)',
                r'Maximum\s+flow[:\s]+(\d+)\s*(m³?/h|m3/h|GPM|l/h)',
                r'Flow\s+rate[:\s]+(\d+)\s*(m³?/h|m3/h|GPM|l/h)',
            ],
            'max_pressure': [
                r'Max\.?\s*(?:operating\s+)?pressure[:\s]+(\d+\.?\d*)\s*(bar|PSI|psi|kPa|MPa)',
                r'Maximum\s+pressure[:\s]+(\d+\.?\d*)\s*(bar|PSI|psi|kPa|MPa)',
                r'Operating\s+pressure[:\s]+(\d+\.?\d*)\s*(bar|PSI|psi|kPa|MPa)',
            ],
            'temperature_range': [
                r'(?:Max\.?\s*)?(?:working|operating)\s+temperature[:\s]+(-?\d+)\s*(?:º|°)?C?\s+to\s+[+]?(\d+)\s*(?:º|°)?C',
                r'Temperature\s+range[:\s]+(-?\d+)\s*(?:º|°)?C?\s+to\s+[+]?(\d+)\s*(?:º|°)?C',
                r'Working\s+temp[:\s]+(-?\d+)\s*(?:º|°)?C?\s+to\s+[+]?(\d+)\s*(?:º|°)?C',
            ],
            'max_speed': [
                r'Max\.?\s*speed[:\s]+(\d+)\s*(rpm|RPM)',
                r'Maximum\s+speed[:\s]+(\d+)\s*(rpm|RPM)',
                r'Rotation\s+speed[:\s]+(\d+)\s*(rpm|RPM)',
            ],
            
            # Vendor/Manufacturer Information (for SAP Vendor Master)
            'manufacturer': [
                r'(INOXPA|GRUNDFOS|KSB|WILO|Pentair|Xylem|Sulzer|Flowserve|ITT|Ebara)\s+S\.?A\.?U?\.?',
                r'Manufacturer[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|FT)',
                r'Made\s+by[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$)',
                r'Company[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$)',
            ],
            
            # Material Identification (for SAP Material Master)
            'model': [
                r'(DIN-FOOD|[A-Z]{2,}-[A-Z0-9\-]+)',
                r'Model[:\s]+([A-Z0-9\-]+)',
                r'Type[:\s]+([A-Z0-9\-]+)',
                r'Series[:\s]+([A-Z0-9\-]+)',
            ],
            'material': [
                r'Materials?\s+(?:in\s+contact[:\s]+)?([0-9\.]+\s*\([A-Z]+\s+[0-9A-Z]+\))',
                r'Construction[:\s]+([0-9\.]+\s*\([A-Z]+\s+[0-9A-Z]+\))',
                r'Body\s+material[:\s]+([A-Z]+\s+[0-9A-Z]+)',
                r'(1\.4404|AISI\s+316L?|Stainless\s+steel)',
            ],
            
            # Physical Characteristics (for SAP Material Master - Basic Data)
            'weight': [
                r'Weight[:\s]+(\d+\.?\d*)\s*(kg|g|lbs?|pounds?)',
                r'Mass[:\s]+(\d+\.?\d*)\s*(kg|g|lbs?|pounds?)',
                r'Net\s+weight[:\s]+(\d+\.?\d*)\s*(kg|g|lbs?|pounds?)',
            ],
            'dimensions': [
                r'Dimensions?[:\s]+(\d+\.?\d*)\s*(?:cm|mm|m|x)\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m|x)\s*[xX×]\s*(\d+\.?\d*)\s*(?:cm|mm|m)',
                r'Size[:\s]+(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)',
            ],
            
            # Procurement/Purchasing Data (for SAP Material Master - Purchasing)
            'lead_time': [
                r'Lead\s+time[:\s]+(\d+)\s*(days?|weeks?|months?)',
                r'Delivery\s+time[:\s]+(\d+)\s*(days?|weeks?)',
            ],
            
            # Classification Data (for SAP)
            'product_type': [
                r'Product\s+type[:\s]+([A-Za-z\s]+)',
                r'Category[:\s]+([A-Za-z\s]+)',
                r'Type[:\s]+(Pump|Valve|Motor|Sensor|Controller)',
            ],
            
            # Quality/Compliance (for SAP QM)
            'certification': [
                r'(CE|ISO\s+\d+|FDA|ATEX|EHEDG)',
                r'Certified[:\s]+([A-Z\s,]+)',
                r'Standards?[:\s]+([A-Z0-9\s,\-]+)',
            ],
            
            # Additional Technical Data
            'power': [
                r'Power[:\s]+(\d+\.?\d*)\s*(kW|W|HP)',
                r'Motor\s+power[:\s]+(\d+\.?\d*)\s*(kW|W|HP)',
            ],
            'voltage': [
                r'Voltage[:\s]+(\d+)\s*(V|VAC|VDC)',
                r'Supply[:\s]+(\d+)\s*(V|VAC)',
            ],
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
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                # Extract the matched text and its context
                start_pos = max(0, match.start() - 80)
                end_pos = min(len(text), match.end() + 80)
                context = text[start_pos:end_pos].strip()
                
                # Format the extracted value based on attribute type
                try:
                    if attribute_name == 'temperature_range':
                        value = f"{match.group(1)}°C to {match.group(2)}°C"
                    elif attribute_name in ['max_flow', 'max_pressure', 'max_speed']:
                        value = f"{match.group(1)} {match.group(2)}"
                    elif attribute_name == 'material':
                        value = match.group(1).strip()
                    elif attribute_name == 'manufacturer':
                        value = match.group(1).strip()
                    elif attribute_name == 'model':
                        value = match.group(1).strip()
                    else:
                        value = match.group(1).strip() if match.lastindex >= 1 else match.group(0).strip()
                except IndexError:
                    value = match.group(0).strip()
                
                return {
                    'value': value,
                    'matched_text': match.group(0),
                    'context': context,
                    'pattern_used': pattern,
                    'confidence': self._calculate_confidence(match, pattern, attribute_name)
                }
        
        return None
    
    def _calculate_confidence(self, match, pattern: str, attribute_name: str = '') -> float:
        """Calculate confidence score based on match quality"""
        # Base confidence
        confidence = 0.75
        
        # Increase confidence if match has clear labeling (colon or 'Max')
        matched_text = match.group(0)
        if ':' in matched_text or 'Max' in matched_text:
            confidence += 0.15
        
        # Increase confidence for specific attributes
        high_confidence_attrs = ['manufacturer', 'max_flow', 'max_pressure', 'temperature_range', 'max_speed']
        if attribute_name in high_confidence_attrs:
            confidence += 0.10
        
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
