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
        These patterns match the attributes in mdg_multi_material_training_data_500.json
        """
        self.patterns = {
            # Flow and Rate Specifications
            'Rated Flow': [
                r'Rated\s+[Ff]low[:\s]+(\d+\.?\d*)\s*(m³?/h|m3/h|cfm|CFM|L/min|l/min)',
                r'Flow\s+[Rr]ate[:\s]+(\d+\.?\d*)\s*(m³?/h|m3/h|cfm|CFM|L/min|l/min)',
                r'Max\.?\s*flow[:\s]+(\d+\.?\d*)\s*(m³?/h|m3/h|cfm|CFM|L/min|l/min)',
            ],
            'Average Air Flow': [
                r'Average\s+[Aa]ir\s+[Ff]low[:\s]+(\d+\.?\d*)\s*(cfm|CFM|m³?/h|m3/h)',
                r'Air\s+[Ff]low[:\s]+(\d+\.?\d*)\s*(cfm|CFM|m³?/h|m3/h)',
            ],
            'Airflow Capacity': [
                r'Airflow\s+[Cc]apacity[:\s]+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(m³?/h|m3/h)',
                r'Airflow[:\s]+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(cfm|CFM)',
            ],
            'Free Air Flow Rate': [
                r'Free\s+[Aa]ir\s+[Ff]low\s+[Rr]ate[:\s]+(\d+\.?\d*)\s*(cfm|CFM)',
            ],
            'Maximum Flow Rate': [
                r'Maximum\s+[Ff]low\s+[Rr]ate[:\s]+(\d+\.?\d*)\s*(gpm|GPM|L/min|l/min)',
            ],
            
            # Pressure Specifications
            'Working Pressure': [
                r'Working\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI|Pa|kPa|MPa)',
            ],
            'Maximum Working Pressure': [
                r'Maximum\s+[Ww]orking\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI|Pa|kPa|MPa)',
                r'Max\.?\s+[Ww]orking\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            'Maximum Pressure': [
                r'Maximum\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI|Pa|kPa|MPa)',
                r'Max\.?\s*[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            'Maximum Operating Pressure': [
                r'Maximum\s+[Oo]perating\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI|Pa)',
                r'Max\.?\s+[Oo]perating\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            'Operating Pressure': [
                r'Operating\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            'Bypass Setting': [
                r'Bypass\s+[Ss]etting[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            'Burst Pressure': [
                r'Burst\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            'Maximum Discharge Pressure': [
                r'Maximum\s+[Dd]ischarge\s+[Pp]ressure[:\s]+(\d+\.?\d*)\s*(bar|BAR|psi|PSI)',
            ],
            
            # Temperature Specifications
            'Maximum Inlet Air Temperature': [
                r'Maximum\s+[Ii]nlet\s+[Aa]ir\s+[Tt]emperature[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
                r'Max\.?\s+[Ii]nlet\s+[Tt]emp[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Maximum Water Temperature': [
                r'Maximum\s+[Ww]ater\s+[Tt]emperature[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Maximum Liquid Temperature': [
                r'Maximum\s+[Ll]iquid\s+[Tt]emperature[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Maximum Ambient Temperature': [
                r'Maximum\s+[Aa]mbient\s+[Tt]emperature[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Maximum Fluid Temperature': [
                r'Maximum\s+[Ff]luid\s+[Tt]emperature[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Minimum Fluid Temperature': [
                r'Minimum\s+[Ff]luid\s+[Tt]emperature[:\s]+(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Operating Temperature': [
                r'Operating\s+[Tt]emperature[:\s]+(-?\d+\.?\d*)\s+to\s+[+]?(\d+\.?\d*)\s*(?:º|°)?C',
                r'Operating\s+[Tt]emp[:\s]+(-?\d+\.?\d*)\s+to\s+[+]?(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            'Operating Temperature Range': [
                r'Operating\s+[Tt]emperature\s+[Rr]ange[:\s]+(-?\d+\.?\d*)\s+to\s+[+]?(\d+\.?\d*)\s*(?:º|°)?C',
            ],
            
            # Physical Properties
            'Weight': [
                r'Weight[:\s]+(\d+\.?\d*)\s*(kg|g|lb|lbs?|pounds?)',
                r'Net\s+[Ww]eight[:\s]+(\d+\.?\d*)\s*(kg|g|lb|lbs?)',
                r'Mass[:\s]+(\d+\.?\d*)\s*(kg|g)',
                r'Typical\s+[Ww]eight[:\s]+(\d+\.?\d*)\s*(kg|g)',
            ],
            'Air Connection Size': [
                r'Air\s+[Cc]onnection\s+[Ss]ize[:\s]+(\d+\.?\d*)\s*(inch|mm|cm)',
            ],
            'Pipe Connection Size': [
                r'Pipe\s+[Cc]onnection\s+[Ss]ize[:\s]+(\d+\.?\d*)\s*(inch|mm)',
            ],
            'Suction Nozzle Size': [
                r'Suction\s+[Nn]ozzle\s+[Ss]ize[:\s]+(DN\s*\d+)',
            ],
            'Discharge Nozzle Size': [
                r'Discharge\s+[Nn]ozzle\s+[Ss]ize[:\s]+(DN\s*\d+)',
            ],
            
            # Power and Electrical
            'Motor Power': [
                r'Motor\s+[Pp]ower[:\s]+(\d+\.?\d*)\s*(hp|HP|kW|W)',
                r'Rated\s+[Mm]otor\s+[Pp]ower[:\s]+(\d+\.?\d*)\s*(hp|HP|kW|W)',
            ],
            'Output Power': [
                r'Output\s+[Pp]ower[:\s]+(\d+\.?\d*)\s*(hp|HP|kW|W)',
            ],
            'Power Consumption': [
                r'Power\s+[Cc]onsumption[:\s]+(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*(W|kW)',
            ],
            'Supply Voltage': [
                r'Supply\s+[Vv]oltage[:\s]+(\d+\.?\d*)\s*(V|VAC|VDC)',
            ],
            'Input Voltage': [
                r'Input\s+[Vv]oltage[:\s]+(\d+\.?\d*)\s*(V|VAC)',
            ],
            'Motor Voltage': [
                r'Motor\s+[Vv]oltage[:\s]+(\d+\.?\d*)\s*(V|VAC)',
            ],
            'Operating Voltage': [
                r'Operating\s+[Vv]oltage[:\s]+(\d+\.?\d*)\s*(V|VAC)',
            ],
            'Motor Speed': [
                r'Motor\s+[Ss]peed[:\s]+(\d+\.?\d*)\s*(rpm|RPM)',
                r'Pump\s+[Ss]peed[:\s]+(\d+\.?\d*)\s*(rpm|RPM)',
            ],
            'Motor Frequency': [
                r'Motor\s+[Ff]requency[:\s]+(\d+\.?\d*)\s*(Hz|hz)',
            ],
            'Current Draw': [
                r'Current\s+[Dd]raw[:\s]+(\d+\.?\d*)\s*(A|amp|amps)',
            ],
            
            # System Capacity
            'System Capacity': [
                r'System\s+[Cc]apacity[:\s]+(\d+\.?\d*)\s*(gallon|gal|L|liter)',
            ],
            'Tank Capacity': [
                r'Tank\s+[Cc]apacity[:\s]+(\d+\.?\d*)\s*(L|liter|gallon|gal)',
            ],
            'Tank Size': [
                r'Tank\s+[Ss]ize[:\s]+(\d+\.?\d*)\s*(gallon|gal|L|liter)',
            ],
            
            # Performance Specifications
            'Maximum Head': [
                r'Maximum\s+[Hh]ead[:\s]+(\d+\.?\d*)\s*(ft|m|meter)',
            ],
            'Rated Head': [
                r'Rated\s+[Hh]ead[:\s]+(\d+\.?\d*)\s*(m|meter)',
            ],
            'Shut-Off Head': [
                r'Shut-?[Oo]ff\s+[Hh]ead[:\s]+(\d+\.?\d*)\s*(m|meter)',
            ],
            'Pump Efficiency': [
                r'Pump\s+[Ee]fficiency[:\s]+(\d+\.?\d*)\s*%',
            ],
            'NPSH Required': [
                r'NPSH\s+[Rr]equired[:\s]+(\d+\.?\d*)\s*(m|meter)',
            ],
            'Maximum Suction Lift': [
                r'Maximum\s+[Ss]uction\s+[Ll]ift[:\s]+(\d+\.?\d*)\s*(m|meter)',
            ],
            'Sound Level': [
                r'Sound\s+[Ll]evel[:\s]+(\d+\.?\d*)\s*(dBA|dB)',
                r'Noise\s+[Ll]evel[:\s]+(\d+\.?\d*)\s*(dBA|dB)',
            ],
            
            # Material and Construction
            'Pump Type': [
                r'Pump\s+[Tt]ype[:\s]+([A-Za-z\s\-]+?)(?:\n|$|,)',
            ],
            'Connector Type': [
                r'Connector\s+[Tt]ype[:\s]+([A-Za-z\s\-]+?)(?:\n|$|,)',
            ],
            'Casing Material': [
                r'Casing\s+[Mm]aterial[:\s]+([A-Za-z0-9\s\-/]+?)(?:\n|$)',
            ],
            'Impeller Material': [
                r'Impeller\s+[Mm]aterial[:\s]+([A-Za-z0-9\s\-/]+?)(?:\n|$)',
            ],
            'Shaft Material': [
                r'Shaft\s+[Mm]aterial[:\s]+([A-Za-z0-9\s\-]+?)(?:\n|$)',
            ],
            'Insulator Material': [
                r'Insulator\s+[Mm]aterial[:\s]+([A-Za-z0-9\s\-]+?)(?:\n|$)',
            ],
            
            # Manufacturer Information
            'manufacturer': [
                r'Manufacturer[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$)',
                r'Made\s+by[:\s]+([A-Z][A-Za-z\s&,\.]+?)(?:\n|$)',
            ],
            
            # Certifications and Standards
            'UL Certification': [
                r'(UL\s*\d+\s*[Ll]isted)',
                r'UL\s+[Cc]ertification[:\s]+(UL\s*\d+)',
            ],
            'Certification': [
                r'Certification[:\s]+([A-Z0-9\s,]+?)(?:\n|$)',
                r'(CE|ETL|ISO\s+\d+|FDA|ATEX|EHEDG|RoHS)',
            ],
            'Compliance': [
                r'Compliance[:\s]+([A-Za-z0-9\s,]+?)(?:\n|$)',
            ],
            
            # Additional Specifications
            'Contact Pitch': [
                r'Contact\s+[Pp]itch[:\s]+(\d+\.?\d*)\s*(mm|cm)',
            ],
            'Number of Contacts': [
                r'Number\s+of\s+[Cc]ontacts[:\s]+(\d+)',
            ],
            'Maximum Data Transfer Rate': [
                r'Maximum\s+[Dd]ata\s+[Tt]ransfer\s+[Rr]ate[:\s]+(\d+\.?\d*)\s*(Gbit/s|Gbps)',
            ],
            'Filtration Efficiency': [
                r'Filtration\s+[Ee]fficiency[:\s]+(\d+\.?\d*)\s*%',
            ],
            'Particle Size Rating': [
                r'Particle\s+[Ss]ize\s+[Rr]ating[:\s]+(\d+\.?\d*)\s*(µm|micron)',
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
