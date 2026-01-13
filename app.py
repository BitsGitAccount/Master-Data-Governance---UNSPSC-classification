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
from models.pdf_extractor import PDFAttributeExtractor  # PDF attribute extraction
import tempfile  # For handling temporary uploaded files


# ============================================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================================
# This must be the first Streamlit command - configures the page layout/appearance
st.set_page_config(
    page_title="Material UNSPSC Classification",  # Browser tab title
    page_icon="📦",  # Browser tab icon
    layout="wide"  # Use full width of browser (vs centered)
)


# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================
# @st.cache_resource decorator ensures these models are loaded only ONCE
# and cached in memory for subsequent users/requests (improves performance)

@st.cache_resource
def load_classifier():
    """
    Load the trained ML classification model from disk.
    
    Returns:
        MaterialClassifier object if model exists, None otherwise
        
    NOTE: The model must be trained first using: python train_model.py
    This creates the .pkl files in the trained_models/ directory
    """
    # Initialize classifier object
    classifier = MaterialClassifier()
    
    # Check if trained model file exists
    if os.path.exists('trained_models/classifier.pkl'):
        # Load the pre-trained model weights
        classifier.load_model('trained_models')
        return classifier
    
    # Return None if model hasn't been trained yet
    return None


@st.cache_resource
def load_pdf_extractor():
    """
    Initialize the PDF attribute extractor.
    
    Returns:
        PDFAttributeExtractor object
        
    NOTE: This doesn't require training - it uses regex patterns
    to extract attributes like weight, dimensions, manufacturer, etc.
    """
    return PDFAttributeExtractor()


def render_classification_tab():
    """Render the material classification tab"""
    st.header("📋 Material Classification")
    st.markdown("Classify materials into UNSPSC codes based on their descriptions.")
    
    # Load classifier
    classifier = load_classifier()
    
    if classifier is None:
        st.error("❌ Model not found. Please train the model first by running: `python train_model.py`")
        return
    
    st.success("✓ Model loaded successfully")
    
    # Input section
    st.subheader("Input Material Description")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        material_description = st.text_area(
            "Enter material description:",
            height=100,
            placeholder="e.g., High-Quality Plastic Packaging for Industrial Use"
        )
    
    with col2:
        st.info("""
        **Tips:**
        - Include material type
        - Mention key features
        - Specify intended use
        """)
    
    # Classify button
    if st.button("🔍 Classify Material", type="primary"):
        if not material_description.strip():
            st.warning("Please enter a material description")
            return
        
        with st.spinner("Classifying material..."):
            # Get prediction with confidence
            results = classifier.predict_with_confidence([material_description])
            result = results[0]
            
            # Get explanation
            explanation = classifier.explain_prediction(material_description, result)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        # Main result
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted UNSPSC Code", result['predicted_unspsc'])
        
        with col2:
            confidence = result['confidence']
            st.metric("Confidence Score", f"{confidence:.1%}")
            
            if confidence >= 0.8:
                st.success("High confidence")
            elif confidence >= 0.6:
                st.warning("Medium confidence - review recommended")
            else:
                st.error("Low confidence - manual review required")
        
        # Top predictions
        st.subheader("🎯 Top 3 Predictions")
        
        pred_data = []
        for pred in result['top_predictions']:
            pred_data.append({
                'UNSPSC Code': pred['unspsc_code'],
                'Probability': f"{pred['probability']:.1%}"
            })
        
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)
        
        # Explainability
        st.subheader("💡 Explainability")
        
        st.info(explanation['explanation'])
        
        if explanation['influential_words']:
            st.markdown("**Most Influential Keywords:**")
            
            words_data = []
            for word_info in explanation['influential_words'][:5]:
                words_data.append({
                    'Keyword': word_info['word'],
                    'Importance Score': f"{word_info['importance']:.4f}",
                    'Weight': f"{word_info['weight']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(words_data), use_container_width=True, hide_index=True)


def render_pdf_extraction_tab():
    """Render the PDF attribute extraction tab"""
    st.header("📄 PDF Attribute Extraction")
    st.markdown("Extract material attributes from Technical Data Sheet (TDS) PDFs for testing purposes.")
    
    # Load extractor
    extractor = load_pdf_extractor()
    
    st.success("✓ PDF Extractor loaded successfully")
    
    # File upload
    st.subheader("Upload TDS PDF")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a Technical Data Sheet in PDF format"
    )
    
    # Sample files
    st.markdown("**Or select a sample TDS file:**")
    sample_files = []
    if os.path.exists('data/tds_pdfs'):
        sample_files = [f for f in os.listdir('data/tds_pdfs') if f.endswith('.pdf')]
    
    selected_sample = None
    if sample_files:
        selected_sample = st.selectbox(
            "Available sample PDFs:",
            options=[''] + sample_files,
            format_func=lambda x: "Select a sample..." if x == '' else x
        )
    
    # Extract button
    if st.button("🔍 Extract Attributes", type="primary"):
        pdf_path = None
        
        # Determine PDF source
        if uploaded_file:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name
        elif selected_sample and selected_sample != '':
            pdf_path = os.path.join('data/tds_pdfs', selected_sample)
        else:
            st.warning("Please upload a PDF or select a sample file")
            return
        
        with st.spinner("Extracting attributes from PDF..."):
            result = extractor.extract_with_explainability(pdf_path)
        
        # Clean up temp file
        if uploaded_file and os.path.exists(pdf_path):
            os.unlink(pdf_path)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Extraction Results")
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pages", result['extraction']['total_pages'])
        
        with col2:
            val = result['validation']
            st.metric("Extraction Rate", f"{val['extraction_rate']:.0%}")
        
        with col3:
            st.metric("Avg Confidence", f"{val['average_confidence']:.0%}")
        
        with col4:
            st.metric("Quality Score", f"{val['quality_score']:.0%}")
        
        # Status indicator
        if val['status'] == 'good':
            st.success("✓ Good quality extraction")
        else:
            st.warning("⚠️ Extraction needs review")
        
        # Extracted attributes
        st.subheader("📦 Extracted Attributes")
        
        attributes = result['extraction']['attributes']
        
        for attr_name, attr_data in attributes.items():
            with st.expander(f"**{attr_name.upper().replace('_', ' ')}**", expanded=True):
                if attr_data['value']:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown(f"**Value:** `{attr_data['value']}`")
                        st.markdown(f"**Confidence:** {attr_data['confidence']:.0%}")
                    
                    with col2:
                        st.markdown(f"**Source:** {attr_data['explanation']}")
                        if attr_data['source'] and attr_data['source']['context']:
                            st.code(attr_data['source']['context'], language=None)
                else:
                    st.warning(attr_data['explanation'])
        
        # Explainability section
        st.subheader("💡 Extraction Explainability")
        
        exp = result['explainability']
        
        st.markdown(f"**Method:** {exp['method']}")
        st.markdown(f"**Confidence Calculation:** {exp['confidence_calculation']}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if exp['attributes_found']:
                st.success(f"**Attributes Found:** {', '.join(exp['attributes_found'])}")
        
        with col2:
            if exp['attributes_missing']:
                st.error(f"**Attributes Missing:** {', '.join(exp['attributes_missing'])}")


def render_batch_processing_tab():
    """Render the batch processing tab"""
    st.header("⚡ Batch Processing")
    st.markdown("Process multiple materials at once for efficiency.")
    
    classifier = load_classifier()
    
    if classifier is None:
        st.error("❌ Model not found. Please train the model first.")
        return
    
    # Load sample data
    if os.path.exists('data/mock_materials.csv'):
        st.subheader("Sample Data")
        df = pd.read_csv('data/mock_materials.csv')
        
        st.markdown(f"**Total records available:** {len(df)}")
        
        num_samples = st.slider("Number of samples to process:", 5, 50, 10)
        
        if st.button("🚀 Process Batch", type="primary"):
            with st.spinner(f"Processing {num_samples} materials..."):
                sample_df = df.head(num_samples).copy()
                
                # Get predictions
                descriptions = sample_df['Material_Description'].values
                results = classifier.predict_with_confidence(descriptions)
                
                # Add predictions to dataframe
                sample_df['Predicted_UNSPSC'] = [r['predicted_unspsc'] for r in results]
                sample_df['Confidence'] = [f"{r['confidence']:.1%}" for r in results]
                sample_df['Match'] = sample_df['UNSPSC_Code'] == sample_df['Predicted_UNSPSC']
            
            st.success(f"✓ Processed {num_samples} materials")
            
            # Display metrics
            accuracy = (sample_df['Match'].sum() / len(sample_df)) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(sample_df))
            with col2:
                st.metric("Correct Predictions", sample_df['Match'].sum())
            with col3:
                st.metric("Accuracy", f"{accuracy:.1f}%")
            
            # Display results
            st.subheader("📊 Results")
            
            display_df = sample_df[[
                'Material_ID', 'Material_Description', 
                'UNSPSC_Code', 'Predicted_UNSPSC', 'Confidence', 'Match'
            ]]
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Match": st.column_config.CheckboxColumn("Match")
                }
            )
            
            # Download results
            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="classification_results.csv",
                mime="text/csv"
            )
    else:
        st.warning("No sample data found. Run `python utils/data_generator.py` first.")


# ============================================================================
# MAIN CLASSIFICATION INTERFACE
# ============================================================================
# This is the PRIMARY function that handles the complete classification workflow

def render_combined_classification_tab():
    """
    Main classification interface requiring BOTH material description AND PDF.
    
    WORKFLOW:
    1. Load ML models (classifier + PDF extractor)
    2. Get material description from user (REQUIRED)
    3. Get PDF file from user (REQUIRED)
    4. Validate both inputs are provided
    5. Extract attributes from PDF
    6. Combine description + PDF attributes
    7. Classify using enhanced description
    8. Display top 5 predictions with explainability
    """
    
    # === HEADER SECTION ===
    st.header("🎯 Material Classification")
    st.markdown("Classify materials using **both** material description and TDS PDF for accurate UNSPSC predictions.")
    # Warning box to emphasize both inputs are mandatory
    st.info("⚠️ **Both material description and PDF are required** for classification.")
    
    # === LOAD ML MODELS ===
    # These are cached, so they load only once per session
    classifier = load_classifier()  # ML model for classification
    extractor = load_pdf_extractor()  # Tool to extract attributes from PDF
    
    # Check if classifier was trained - if not, show error and stop
    if classifier is None:
        st.error("❌ Model not found. Please train the model first by running: `python train_model.py`")
        return
    
    # Models loaded successfully
    st.success("✓ Models loaded successfully")
    
    # ========================================================================
    # STEP 1: MATERIAL DESCRIPTION INPUT (REQUIRED)
    # ========================================================================
    st.subheader("📋 Step 1: Provide Material Description (Required)")
    
    # Create two-column layout: main input area + tips sidebar
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Text area for material description input
        material_description = st.text_area(
            "Enter material description:",
            height=100,
            placeholder="e.g., High-Quality Plastic Packaging for Industrial Use",
            key="combined_desc"  # Unique key to avoid conflicts
        )
    
    with col2:
        # Tips box to help users write better descriptions
        st.info("""
        **Tips:**
        - Include material type
        - Mention key features
        - Specify intended use
        """)
    
    # ========================================================================
    # STEP 2: PDF UPLOAD (REQUIRED)
    # ========================================================================
    st.subheader("📄 Step 2: Upload Technical Data Sheet (Required)")
    
    # File uploader widget - accepts only PDF files
    uploaded_file = st.file_uploader(
        "Choose a TDS PDF file",
        type=['pdf'],  # Restrict to PDF files only
        help="Upload a Technical Data Sheet in PDF format (Required)",
        key="combined_pdf"  # Unique key to avoid conflicts
    )
    
    # Alternative: Use sample files for testing/demo
    st.markdown("**Or select a sample TDS file:**")
    sample_files = []
    
    # Check if sample PDFs directory exists
    if os.path.exists('data/tds_pdfs'):
        # Get list of all PDF files in the directory
        sample_files = [f for f in os.listdir('data/tds_pdfs') if f.endswith('.pdf')]
    
    selected_sample = None
    if sample_files:
        # Dropdown to select from available sample PDFs
        selected_sample = st.selectbox(
            "Available sample PDFs:",
            options=[''] + sample_files,  # Empty string as first option (no selection)
            format_func=lambda x: "Select a sample TDS file..." if x == '' else x,
            key="combined_sample"
        )
    
    # ========================================================================
    # CLASSIFICATION BUTTON & PROCESSING
    # ========================================================================
    # When user clicks classify button, execute the following logic
    if st.button("🚀 Classify Material", type="primary"):
        
        # === VALIDATION STEP 1: Check Material Description ===
        if not material_description.strip():
            st.error("❌ Material description is required. Please enter a material description.")
            return  # Stop execution if validation fails
        
        # === VALIDATION STEP 2: Check PDF File ===
        pdf_path = None
        
        # Case 1: User uploaded a file
        if uploaded_file:
            # Save uploaded file to temporary location
            # (Streamlit uploads need to be saved to disk before processing)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                pdf_path = tmp_file.name
        
        # Case 2: User selected a sample file
        elif selected_sample and selected_sample != '':
            pdf_path = os.path.join('data/tds_pdfs', selected_sample)
        
        # If neither option was used, show error
        if not pdf_path:
            st.error("❌ TDS PDF is required. Please upload a PDF file or select a sample PDF.")
            return  # Stop execution if validation fails
        
        # === BOTH INPUTS PROVIDED - PROCEED WITH CLASSIFICATION ===
        
        # ========================================================================
        # STEP 3: EXTRACT ATTRIBUTES FROM PDF
        # ========================================================================
        pdf_attributes = {}  # Will store extracted attributes (weight, dimensions, etc.)
        
        # Show spinner while extracting (can take a few seconds)
        with st.spinner("Extracting attributes from PDF..."):
            # Call PDF extractor - returns extracted attributes with confidence scores
            pdf_result = extractor.extract_with_explainability(pdf_path)
            pdf_attributes = pdf_result['extraction']['attributes']
        
        # Clean up temporary file if it was an upload
        if uploaded_file and os.path.exists(pdf_path):
            os.unlink(pdf_path)  # Delete temporary file
        
        # ========================================================================
        # STEP 4: COMBINE DESCRIPTION + PDF ATTRIBUTES
        # ========================================================================
        # This is the KEY STEP that improves classification accuracy!
        # We append PDF attributes to the description to give more context to the classifier
        
        attribute_text_parts = []  # List to store formatted attribute text
        
        # Loop through each extracted attribute
        for attr_name, attr_data in pdf_attributes.items():
            if attr_data['value']:  # Only include if attribute was found
                # Format: "weight: 2.5 kg", "manufacturer: BASF", etc.
                attribute_text_parts.append(f"{attr_name.replace('_', ' ')}: {attr_data['value']}")
        
        # Create enhanced description
        if attribute_text_parts:
            # Combine: original description + " " + all PDF attributes
            enhanced_description = material_description + " " + " ".join(attribute_text_parts)
            st.success(f"✓ Enhanced description with {len(attribute_text_parts)} PDF attributes")
        else:
            # No attributes extracted - use original description only
            enhanced_description = material_description
            st.warning("⚠️ No attributes extracted from PDF. Classification will use description only.")
        
        # ========================================================================
        # STEP 5: CLASSIFY THE MATERIAL
        # ========================================================================
        # Show spinner while classifying
        with st.spinner("Classifying material with combined data..."):
            # Get predictions with confidence scores
            # Input: enhanced description (description + PDF attributes)
            # Output: predicted UNSPSC, confidence, top 5 predictions
            results = classifier.predict_with_confidence([enhanced_description])
            result = results[0]  # Get first result (single item)
            
            # Get explanation - which keywords influenced the classification?
            explanation = classifier.explain_prediction(enhanced_description, result)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Classification Results")
        
        # Main result
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Predicted UNSPSC Code", result['predicted_unspsc'])
        
        with col2:
            confidence = result['confidence']
            st.metric("Confidence Score", f"{confidence:.1%}")
            
            if confidence >= 0.8:
                st.success("High confidence")
            elif confidence >= 0.6:
                st.warning("Medium confidence - review recommended")
            else:
                st.error("Low confidence - manual review required")
        
        # Show what data was used
        st.subheader("📦 Data Sources Used")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**✓ Material Description**")
            st.text_area("Original description:", material_description, height=80, disabled=True, key="show_desc")
        
        with col2:
            st.markdown("**✓ PDF Attributes Extracted**")
            pdf_info = "\n".join([f"• {k.replace('_', ' ').title()}: {v['value']}" 
                                  for k, v in pdf_attributes.items() if v['value']])
            if pdf_info:
                st.text_area("Extracted attributes:", pdf_info, height=80, disabled=True, key="show_pdf")
            else:
                st.info("No attributes were successfully extracted from the PDF")
        
        # Top predictions (show top 5)
        st.subheader("🎯 Top 5 Predictions")
        
        pred_data = []
        for idx, pred in enumerate(result['top_predictions'][:5], 1):
            pred_data.append({
                'Rank': idx,
                'UNSPSC Code': pred['unspsc_code'],
                'Probability': f"{pred['probability']:.1%}"
            })
        
        st.dataframe(pd.DataFrame(pred_data), use_container_width=True, hide_index=True)
        
        # Explainability
        st.subheader("💡 Explainability")
        
        st.info(explanation['explanation'])
        
        if explanation['influential_words']:
            st.markdown("**Most Influential Keywords:**")
            
            words_data = []
            for word_info in explanation['influential_words'][:8]:
                words_data.append({
                    'Keyword': word_info['word'],
                    'Importance Score': f"{word_info['importance']:.4f}",
                    'Weight': f"{word_info['weight']:.4f}"
                })
            
            st.dataframe(pd.DataFrame(words_data), use_container_width=True, hide_index=True)
        
        # PDF Extraction Details
        st.subheader("📄 PDF Extraction Details")
        
        val = pdf_result['validation']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Extraction Rate", f"{val['extraction_rate']:.0%}")
        with col2:
            st.metric("Avg Confidence", f"{val['average_confidence']:.0%}")
        with col3:
            st.metric("Quality Score", f"{val['quality_score']:.0%}")
        
        with st.expander("View detailed PDF extraction"):
            for attr_name, attr_data in pdf_attributes.items():
                if attr_data['value']:
                    st.markdown(f"**{attr_name.upper().replace('_', ' ')}**")
                    st.markdown(f"- Value: `{attr_data['value']}`")
                    st.markdown(f"- Confidence: {attr_data['confidence']:.0%}")
                    st.markdown(f"- Source: {attr_data['explanation']}")
                    if attr_data['source'] and attr_data['source']['context']:
                        st.code(attr_data['source']['context'], language=None)
                    st.markdown("---")


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """
    Main application entry point.
    
    This function is called when the app starts.
    It displays the title and renders the main classification interface.
    """
    
    # === APPLICATION HEADER ===
    st.title("📦 Material UNSPSC Classification")
    st.markdown("""
    Automated classification of materials into UNSPSC codes using both material descriptions and Technical Data Sheets.
    """)
    
    # === RENDER MAIN INTERFACE ===
    # Call the main classification function
    # (This is the only tab/section currently displayed)
    render_combined_classification_tab()


# ============================================================================
# APPLICATION STARTUP
# ============================================================================
# This block runs when the script is executed directly (not imported)
if __name__ == "__main__":
    main()  # Start the Streamlit application
