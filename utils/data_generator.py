"""
Mock data generator for Material Classification PoC
Generates both structured data (CSV) and unstructured data (TDS PDFs)
"""

import os
import random
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer

# UNSPSC Categories with descriptions
UNSPSC_CATEGORIES = {
    "12345678": {
        "name": "Plastic Packaging Materials",
        "keywords": ["plastic", "packaging", "container", "wrapper", "film", "bottle"]
    },
    "23456789": {
        "name": "Steel Pipes and Tubes",
        "keywords": ["steel", "pipe", "tube", "conduit", "stainless", "metal"]
    },
    "34567890": {
        "name": "Electronic Components",
        "keywords": ["electronic", "component", "resistor", "capacitor", "circuit", "semiconductor"]
    },
    "45678901": {
        "name": "Industrial Chemicals",
        "keywords": ["chemical", "solvent", "reagent", "acid", "industrial", "compound"]
    },
    "56789012": {
        "name": "Mechanical Fasteners",
        "keywords": ["bolt", "screw", "nut", "fastener", "rivet", "anchor"]
    },
    "67890123": {
        "name": "Safety Equipment",
        "keywords": ["safety", "protective", "helmet", "glove", "goggles", "equipment"]
    },
    "78901234": {
        "name": "Measurement Instruments",
        "keywords": ["measurement", "gauge", "meter", "instrument", "sensor", "detector"]
    },
    "89012345": {
        "name": "Industrial Lubricants",
        "keywords": ["lubricant", "oil", "grease", "hydraulic", "industrial", "fluid"]
    }
}

# Mock manufacturers
MANUFACTURERS = [
    "Acme Corporation", "Global Industries", "TechnoMaterials Inc.",
    "SteelCo Manufacturing", "Precision Components Ltd.", "SafeGuard Equipment",
    "ChemPro Solutions", "Industrial Supplies Co.", "MeasureTech Systems"
]


def generate_material_description(unspsc_code):
    """Generate a realistic material description based on UNSPSC category"""
    category = UNSPSC_CATEGORIES[unspsc_code]
    keywords = category["keywords"]
    
    # Create varied descriptions
    templates = [
        f"High-Quality {random.choice(keywords).title()} for Industrial Use",
        f"Premium {random.choice(keywords).title()} - {random.choice(['Heavy-Duty', 'Standard', 'Professional Grade'])}",
        f"{random.choice(['Commercial', 'Industrial', 'Professional'])} {random.choice(keywords).title()}",
        f"{random.choice(keywords).title()} - {random.choice(['Type A', 'Type B', 'Grade 1', 'Grade 2'])}",
        f"Standard {random.choice(keywords).title()} {random.choice(['Assembly', 'Component', 'Unit'])}"
    ]
    
    return random.choice(templates)


def generate_structured_data(num_samples=100):
    """Generate mock structured material data"""
    data = []
    
    for i in range(num_samples):
        unspsc_code = random.choice(list(UNSPSC_CATEGORIES.keys()))
        manufacturer = random.choice(MANUFACTURERS)
        mpn = f"{random.choice(['ABC', 'XYZ', 'DEF', 'GHI'])}{random.randint(100, 999)}"
        description = generate_material_description(unspsc_code)
        
        data.append({
            "Material_ID": f"MAT{i+1:04d}",
            "Material_Description": description,
            "Manufacturer": manufacturer,
            "MPN": mpn,
            "UNSPSC_Code": unspsc_code,
            "Category_Name": UNSPSC_CATEGORIES[unspsc_code]["name"]
        })
    
    df = pd.DataFrame(data)
    return df


def generate_tds_pdf(material_data, output_path):
    """Generate a mock Technical Data Sheet PDF"""
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#003366'),
        spaceAfter=30,
    )
    
    # Add title
    title = Paragraph("TECHNICAL DATA SHEET", title_style)
    story.append(title)
    story.append(Spacer(1, 0.3*inch))
    
    # Product information
    product_info = [
        ["Product Information", ""],
        ["Material ID:", material_data["Material_ID"]],
        ["Description:", material_data["Material_Description"]],
        ["Manufacturer:", material_data["Manufacturer"]],
        ["Part Number:", material_data["MPN"]],
        ["UNSPSC Code:", material_data["UNSPSC_Code"]],
    ]
    
    t1 = Table(product_info, colWidths=[2*inch, 4*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t1)
    story.append(Spacer(1, 0.4*inch))
    
    # Physical properties
    weight = random.uniform(0.1, 50.0)
    length = random.uniform(5, 200)
    width = random.uniform(5, 150)
    height = random.uniform(2, 100)
    
    physical_props = [
        ["Physical Properties", ""],
        ["Weight:", f"{weight:.2f} kg"],
        ["Dimensions:", f"{length:.1f} cm x {width:.1f} cm x {height:.1f} cm"],
        ["Volume:", f"{(length * width * height / 1000):.2f} liters"],
        ["Density:", f"{(weight / (length * width * height / 1000000)):.2f} kg/m³"],
    ]
    
    t2 = Table(physical_props, colWidths=[2*inch, 4*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t2)
    story.append(Spacer(1, 0.4*inch))
    
    # Additional specifications
    specs = [
        ["Additional Specifications", ""],
        ["Material Grade:", random.choice(["Grade A", "Grade B", "Premium", "Standard"])],
        ["Operating Temperature:", f"{random.randint(-20, 150)}°C to {random.randint(200, 500)}°C"],
        ["Compliance:", random.choice(["ISO 9001", "CE Certified", "RoHS Compliant", "FDA Approved"])],
        ["Shelf Life:", f"{random.randint(12, 60)} months"],
    ]
    
    t3 = Table(specs, colWidths=[2*inch, 4*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(t3)
    
    # Build PDF
    doc.build(story)
    print(f"Generated PDF: {output_path}")


def main():
    """Main function to generate all mock data"""
    # Create directories
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/tds_pdfs", exist_ok=True)
    
    print("Generating mock structured data...")
    df = generate_structured_data(100)
    
    # Save CSV
    csv_path = "data/mock_materials.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved structured data to {csv_path}")
    print(f"Generated {len(df)} material records")
    
    # Generate PDFs for a subset of materials
    print("\nGenerating TDS PDFs...")
    num_pdfs = 20  # Generate PDFs for 20 materials
    for idx in range(num_pdfs):
        material = df.iloc[idx].to_dict()
        pdf_path = f"data/tds_pdfs/{material['Material_ID']}_TDS.pdf"
        generate_tds_pdf(material, pdf_path)
    
    print(f"\nData generation complete!")
    print(f"- Structured data: {csv_path}")
    print(f"- TDS PDFs: data/tds_pdfs/ ({num_pdfs} files)")
    
    # Display sample data
    print("\nSample of generated data:")
    print(df.head(10).to_string())
    
    # Display UNSPSC distribution
    print("\nUNSPSC Code Distribution:")
    print(df['UNSPSC_Code'].value_counts().to_string())


if __name__ == "__main__":
    main()