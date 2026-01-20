"""
Material UNSPSC Classification Web App

PURPOSE:
This is the main web application for classifying materials into UNSPSC codes.

IMPORTANT: This system requires BOTH inputs to work:
1. Material Description (text)
2. Technical Data Sheet PDF (file)

HOW IT WORKS:
1. User provides material description
2. User uploads/selects a TDS PDF
3. System extracts attributes from PDF (weight, dimensions, manufacturer, etc.)
4. System combines description + PDF attributes for better accuracy
5. System classifies and returns top 5 UNSPSC code predictions
6. System explains which keywords influenced the decision

MAIN FUNCTION:
- render_combined_classification_tab() - The core classification interface (currently in use)

KEY TECHNOLOGIES:
- Streamlit: Web interface framework
- TF-IDF + Logistic Regression: Classification algorithm
- PyMuPDF: PDF text extraction
"""

import streamlit as st  # Web interface framework
import pandas as pd      # Data manipulation and display
import os               # File system operations
from models.classifier import MaterialClassifier  # Our ML classification model
from models.enhanced_classifier import EnhancedMaterialClassifier  # Enhanced classifier with similarity matching
from models.pdf_extractor import PDFAttributeExtractor  # PDF attribute extraction
import tempfile  # For handling temporary uploaded files
import base64  # For encoding PDF to display in iframe


# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Material UNSPSC Classification",
    layout="wide"
)


# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
def inject_custom_css():
    """Load custom CSS for SAP Horizon theme from external file"""
    css_file = os.path.join(os.path.dirname(__file__), 'static', 'styles.css')
    
    if os.path.exists(css_file):
        with open(css_file, 'r') as f:
            css_content = f.read()
        st.markdown(f'<style>{css_content}</style>', unsafe_allow_html=True)
    else:
        st.warning("CSS file not found. Using default styling.")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_confidence_badge(confidence):
    """Generate HTML for confidence badge with color coding"""
    confidence_pct = confidence * 100
    
    if confidence >= 0.8:
        badge_class = "confidence-high"
    elif confidence >= 0.6:
        badge_class = "confidence-medium"
    else:
        badge_class = "confidence-low"
    
    return f'<span class="confidence-badge {badge_class}">{confidence_pct:.0f}%</span>'


def display_pdf(pdf_path):
    """Display PDF in an iframe using base64 encoding"""
    try:
        with open(pdf_path, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        
        pdf_display = f'''
        <iframe src="data:application/pdf;base64,{base64_pdf}" 
                width="100%" height="800px" type="application/pdf"
                style="border: 2px solid #e2e8f0; border-radius: 8px;">
        </iframe>
        '''
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_classifier():
    """Load the enhanced classification model"""
    # Load base ML classifier
    ml_classifier = MaterialClassifier()
    
    if os.path.exists('trained_models/classifier.pkl'):
        ml_classifier.load_model('trained_models')
    else:
        return None
    
    # Create enhanced classifier with similarity matching
    enhanced_classifier = EnhancedMaterialClassifier()
    enhanced_classifier.ml_classifier = ml_classifier
    
    # Load training data for similarity matching
    try:
        enhanced_classifier.load_training_data('data/mdg_multi_material_training_data_500.json')
    except Exception as e:
        st.warning(f"Could not load training data for similarity matching: {e}")
        return ml_classifier  # Fall back to basic classifier
    
    return enhanced_classifier


@st.cache_resource
def load_pdf_extractor():
    """Initialize the PDF attribute extractor"""
    # Force reload to get latest version
    import importlib
    import sys
    if 'models.pdf_extractor' in sys.modules:
        importlib.reload(sys.modules['models.pdf_extractor'])
        from models.pdf_extractor import PDFAttributeExtractor as PDFExtractorReloaded
        return PDFExtractorReloaded()
    return PDFAttributeExtractor()


# ============================================================================
# MAIN CLASSIFICATION INTERFACE
# ============================================================================

def render_combined_classification_tab():
    """
    Main classification interface with professional form-based design
    """
    
    # Inject custom CSS
    inject_custom_css()
    
    # === HEADER SECTION ===
    st.markdown('<h2 style="color: #000000; margin-top: 10px; margin-bottom: 4px;">Material UNSPSC Classification System</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #000000; font-size: 14px; margin-bottom: 12px;"><strong>SAP Master Data Governance</strong> | Automated classification using material descriptions and Technical Data Sheets</p>', unsafe_allow_html=True)
    
    # === LOAD ML MODELS ===
    classifier = load_classifier()
    extractor = load_pdf_extractor()
    
    if classifier is None:
        st.error(" Model not found. Please train the model first by running: `python train_model.py`")
        return
    
    st.success(" Models loaded successfully")
    
    # ========================================================================
    # INPUT SECTION
    # ========================================================================
    st.markdown('<h3 style="color: #000000; margin-top: 12px; margin-bottom: 12px; font-weight: 700;">Input Information</h3>', unsafe_allow_html=True)
    
    # Create two columns for Material Description and TDS Upload
    col1, col2 = st.columns(2)
    
    with col1:
        # Material Description
        material_description = st.text_input(
            "Material Description (Required):",
            max_chars=100,
            placeholder="e.g., High-Quality Plastic Packaging for Industrial Use",
            key="combined_desc"
        )
        
        # Material Number
        material_number = st.text_input(
            "Material Number:",
            max_chars=100,
            placeholder="e.g., MAT-12345",
            key="material_number"
        )
        
        # Manufacturer Name
        manufacturer_name = st.text_input(
            "Manufacturer Name:",
            max_chars=100,
            placeholder="e.g., ABC Manufacturing Co.",
            key="manufacturer_name"
        )
        
        # Manufacturer Part Number (MPN)
        manufacturer_part_number = st.text_input(
            "Manufacturer Part Number (MPN):",
            max_chars=100,
            placeholder="e.g., MPN-67890",
            key="manufacturer_mpn"
        )
    
    with col2:
        uploaded_file = st.file_uploader(
            "Technical Data Sheet (Required):",
            type=['pdf'],
            help="Upload a Technical Data Sheet in PDF format",
            key="combined_pdf"
        )
    
    # Classify Button
    st.markdown('<div style="margin-top: 15px; margin-bottom: 15px;"></div>', unsafe_allow_html=True)
    if st.button("Classify Material", type="primary", use_container_width=True):
        
        # Validation
        if not material_description.strip():
            st.error("❌ Material description is required")
            return
        
        pdf_path = None
        pdf_display_path = None
        
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name
                pdf_display_path = pdf_path
        
        if not pdf_path:
            st.error("❌ TDS PDF is required")
            return
        
        # ========================================================================
        # PROCESSING
        # ========================================================================
        
        with st.spinner("🔄 Extracting attributes from PDF..."):
            pdf_result = extractor.extract_with_explainability(pdf_path)
            pdf_attributes = pdf_result['extraction']['attributes']
        
        # Build enhanced description combining all input fields
        input_parts = [material_description]
        
        # Add optional fields if provided
        if material_number.strip():
            input_parts.append(f"Material Number: {material_number}")
        if manufacturer_name.strip():
            input_parts.append(f"Manufacturer: {manufacturer_name}")
        if manufacturer_part_number.strip():
            input_parts.append(f"MPN: {manufacturer_part_number}")
        
        # Add PDF extracted attributes
        attribute_text_parts = []
        for attr_name, attr_data in pdf_attributes.items():
            if attr_data['value']:
                attribute_text_parts.append(f"{attr_name.replace('_', ' ')}: {attr_data['value']}")
        
        # Combine all parts
        combined_input = " ".join(input_parts)
        if attribute_text_parts:
            enhanced_description = combined_input + " " + " ".join(attribute_text_parts)
        else:
            enhanced_description = combined_input
        
        with st.spinner("🔄 Classifying material..."):
            results = classifier.predict_with_confidence([enhanced_description])
            result = results[0]
            explanation = classifier.explain_prediction(enhanced_description, result)
        
        # Clean up temp file
        if uploaded_file and os.path.exists(pdf_path):
            pass  # Keep for display, will clean up later
        
        # ========================================================================
        # RESULTS DISPLAY - NEW PROFESSIONAL LAYOUT
        # ========================================================================
        
        st.markdown('<div style="margin-top: 20px; margin-bottom: 10px;"></div>', unsafe_allow_html=True)
        st.markdown('<h2 style="color: #000000; margin-top: 0; margin-bottom: 15px; font-weight: 700;">Classification Results</h2>', unsafe_allow_html=True)
        
        # ========================================================================
        # MATERIAL INFO BANNER (Top Section)
        # ========================================================================
        
        # Use user-provided values with fallback to PDF-extracted values
        display_manufacturer = manufacturer_name if manufacturer_name.strip() else pdf_attributes.get('manufacturer', {}).get('value', 'N/A')
        display_material_number = material_number if material_number.strip() else 'N/A'
        display_mpn = manufacturer_part_number if manufacturer_part_number.strip() else 'N/A'
        
        st.markdown(f"""
        <div class="material-info-banner">
            <h3 style="margin-top: 0; margin-bottom: 10px; color: #FFFFFF; font-size: 16px; font-weight: 700;">▶ Request Information</h3>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px 30px;">
                <div class="info-item">
                    <span class="info-label" style="color: #FFFFFF;">Material Description:</span>
                    <span style="color: #FFFFFF;">{material_description[:100]}{'...' if len(material_description) > 100 else ''}</span>
                </div>
                <div class="info-item">
                    <span class="info-label" style="color: #FFFFFF;">Material Number:</span>
                    <span style="color: #FFFFFF;">{display_material_number}</span>
                </div>
                <div class="info-item">
                    <span class="info-label" style="color: #FFFFFF;">Manufacturer Name:</span>
                    <span style="color: #FFFFFF;">{display_manufacturer}</span>
                </div>
                <div class="info-item">
                    <span class="info-label" style="color: #FFFFFF;">Manufacturer Part Number (MPN):</span>
                    <span style="color: #FFFFFF;">{display_mpn}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # ========================================================================
        # 2-COLUMN LAYOUT: Classification Details | PDF Viewer
        # ========================================================================
        
        col_left, col_right = st.columns([6, 4])
        
        with col_left:
            # ================================================================
            # CLASSIFICATION DETAILS SECTION
            # ================================================================
            
            st.markdown('<div class="section-title">■ Classification Details</div>', unsafe_allow_html=True)
            
            # UNSPSC Selection with Top 5 Predictions
            st.markdown('<div class="field-label">Select Classification:</div>', unsafe_allow_html=True)
            
            # Create options for selectbox with confidence scores
            unspsc_options = []
            for idx, pred in enumerate(result['top_predictions'][:5]):
                unspsc_options.append(
                    f"{pred['unspsc_code']} ({pred['probability']:.0%})"
                )
            
            selected_unspsc = st.selectbox(
                "UNSPSC Code:",
                options=unspsc_options,
                index=0,
                label_visibility="collapsed"
            )
            
            # Display primary prediction with confidence badge
            primary_confidence = result['confidence']
            st.markdown(f"""
            <div style="margin-top: 10px; margin-bottom: 20px; padding: 12px; background: #F5F6F7; border-radius: 4px;">
                <span style="color: #000000; font-weight: 700;">Primary Prediction:</span>
                <span style="color: #000000; font-size: 16px; font-weight: 700; margin-left: 10px;">
                    {result['predicted_unspsc']}
                </span>
                {get_confidence_badge(primary_confidence)}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-title" style="margin-top: 30px;">■ Extracted Attributes</div>', unsafe_allow_html=True)
            
            # ================================================================
            # EXTRACTED ATTRIBUTES AS TABLE
            # ================================================================
            
            # Define display names for SAP MDG integration
            # These map to standard SAP Material Master fields
            display_names = {
                # Technical Specifications
                'max_flow': 'Maximum Flow',
                'max_pressure': 'Maximum Operating Pressure',
                'temperature_range': 'Operating Temperature Range',
                'max_speed': 'Maximum Speed',
                
                # Vendor/Manufacturer (SAP Vendor Master)
                'manufacturer': 'Manufacturer',
                
                # Material Identification (SAP Material Master)
                'model': 'Model/Type',
                'material': 'Material Composition',
                
                # Physical Characteristics (SAP Basic Data)
                'weight': 'Net Weight',
                'dimensions': 'Dimensions (L×W×H)',
                
                # Procurement Data (SAP Purchasing)
                'lead_time': 'Lead Time',
                
                # Classification (SAP)
                'product_type': 'Product Type',
                
                # Quality/Compliance (SAP QM)
                'certification': 'Certifications',
                
                # Additional Technical Data
                'power': 'Power Rating',
                'voltage': 'Voltage Rating',
            }
            
            # Build table data with three columns
            table_data = []
            extracted_count = 0
            
            for attr_name, attr_data in pdf_attributes.items():
                # Only show fields that were successfully extracted
                if attr_data['value'] and attr_data['confidence'] > 0:
                    display_name = display_names.get(attr_name, attr_name.replace('_', ' ').title())
                    confidence = attr_data['confidence']
                    confidence_pct = f"{confidence * 100:.0f}%"
                    
                    table_data.append({
                        'Display Name': display_name,
                        'Attribute Data': attr_data['value'],
                        'Confidence': confidence_pct
                    })
                    extracted_count += 1
            
            # Display table if attributes were extracted
            if extracted_count > 0:
                # Create DataFrame for better table display
                df = pd.DataFrame(table_data)
                
                # Display using Streamlit's dataframe with custom styling
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Display Name": st.column_config.TextColumn(
                            "Display Name",
                            width="medium",
                        ),
                        "Attribute Data": st.column_config.TextColumn(
                            "Attribute Data",
                            width="large",
                        ),
                        "Confidence": st.column_config.TextColumn(
                            "Confidence",
                            width="small",
                        ),
                    }
                )
            else:
                st.info("ℹ️ No attributes could be extracted from the PDF. The document may not contain standard technical specifications.")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ================================================================
            # EXPLAINABILITY SECTION
            # ================================================================
            
            with st.expander("▶ View Classification Explainability", expanded=False):
                st.markdown(f"**Explanation:** {explanation['explanation']}")
                
                if explanation['influential_words']:
                    st.markdown("**Most Influential Keywords:**")
                    words_df = pd.DataFrame([
                        {
                            'Keyword': w['word'],
                            'Importance': f"{w['importance']:.4f}",
                            'Weight': f"{w['weight']:.4f}"
                        }
                        for w in explanation['influential_words'][:8]
                    ])
                    st.dataframe(words_df, use_container_width=True, hide_index=True)
            
            # ================================================================
            # PDF EXTRACTION METRICS
            # ================================================================
            
            with st.expander("▶ View PDF Extraction Details", expanded=False):
                val = pdf_result['validation']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Extraction Rate", f"{val['extraction_rate']:.0%}")
                with col2:
                    st.metric("Avg Confidence", f"{val['average_confidence']:.0%}")
                with col3:
                    st.metric("Quality Score", f"{val['quality_score']:.0%}")
                
                st.markdown("**Extraction Sources:**")
                for attr_name, attr_data in pdf_attributes.items():
                    if attr_data['value']:
                        st.markdown(f"**{attr_name.replace('_', ' ').title()}**")
                        st.markdown(f"- Source: {attr_data['explanation']}")
                        if attr_data['source'] and attr_data['source']['context']:
                            st.code(attr_data['source']['context'][:200], language=None)
        
        with col_right:
            # ================================================================
            # PDF VIEWER SECTION
            # ================================================================
            
            st.markdown('<div class="section-title">■ Technical Data Sheet</div>', unsafe_allow_html=True)
            
            if pdf_display_path and os.path.exists(pdf_display_path):
                display_pdf(pdf_display_path)
            else:
                st.info("PDF preview not available")
        
        # Clean up temp file after display
        if uploaded_file and pdf_path and os.path.exists(pdf_path):
            try:
                os.unlink(pdf_path)
            except:
                pass


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Main application entry point"""
    
    st.markdown("""
    <div style="text-align: center; padding: 10px 0 8px 0; background: #FFFFFF; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
        <h1 style="color: #000000; margin-bottom: 5px; font-weight: 700;">SAP Material Classification</h1>
        <p style="color: #000000; font-size: 16px; margin: 0;">
            Master Data Governance | Automated UNSPSC Classification
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    render_combined_classification_tab()


if __name__ == "__main__":
    main()
