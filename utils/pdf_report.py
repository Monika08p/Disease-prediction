"""
utils/pdf_report.py

Generates a downloadable PDF prediction report using reportlab.
Includes patient inputs, prediction result, risk level, probability,
top contributing factors, and a disclaimer.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

REPORTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)


def generate_pdf_report(disease: str, patient_name: str, record: dict, friendly_names: dict,
                         result: dict, top_factors: list, explanation: str) -> str:
    """
    Builds a PDF report and saves it to the reports/ folder.
    Returns the absolute file path of the generated PDF.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = "".join(c if c.isalnum() else "_" for c in patient_name)[:40] or "patient"
    filename = f"{disease}_report_{safe_name}_{timestamp}.pdf"
    filepath = os.path.join(REPORTS_DIR, filename)

    doc = SimpleDocTemplate(filepath, pagesize=A4,
                             topMargin=20 * mm, bottomMargin=20 * mm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("TitleStyle", parent=styles["Title"], fontSize=18)
    heading_style = ParagraphStyle("HeadingStyle", parent=styles["Heading2"], spaceBefore=12)
    normal_style = styles["Normal"]

    elements = []

    disease_label = "Heart Disease" if disease == "heart" else "Diabetes"
    elements.append(Paragraph(f"{disease_label} Risk Prediction Report", title_style))
    elements.append(Paragraph(f"Patient: <b>{patient_name}</b>", normal_style))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%d %b %Y, %H:%M')}", normal_style))
    elements.append(Spacer(1, 10))

    # --- Result summary ---
    elements.append(Paragraph("Prediction Result", heading_style))
    risk_color = {
        "Low Risk": colors.green,
        "Medium Risk": colors.orange,
        "High Risk": colors.red,
    }.get(result["risk_label"], colors.black)

    result_table_data = [
        ["Prediction", result["verdict"]],
        ["Probability of Disease", f"{result['proba']}%"],
        ["Risk Level", result["risk_label"]],
    ]
    result_table = Table(result_table_data, colWidths=[180, 280])
    result_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("TEXTCOLOR", (1, 2), (1, 2), risk_color),
        ("FONTNAME", (1, 2), (1, 2), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(result_table)
    elements.append(Spacer(1, 14))

    # --- Patient input data ---
    elements.append(Paragraph("Patient Input Data", heading_style))
    input_rows = [["Field", "Value"]]
    for field, value in record.items():
        input_rows.append([friendly_names.get(field, field), str(value)])

    input_table = Table(input_rows, colWidths=[280, 180])
    input_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(input_table)
    elements.append(Spacer(1, 14))

    # --- Top contributing factors ---
    elements.append(Paragraph("Top Contributing Factors", heading_style))
    factor_rows = [["Factor", "Patient Value", "Population Avg", "Direction"]]
    for f in top_factors:
        factor_rows.append([
            f["friendly_name"], str(f["patient_value"]),
            str(f["population_mean"]), f["direction"]
        ])
    factor_table = Table(factor_rows, colWidths=[150, 100, 100, 110])
    factor_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f766e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(factor_table)
    elements.append(Spacer(1, 14))

    # --- Explanation ---
    elements.append(Paragraph("Explanation", heading_style))
    elements.append(Paragraph(explanation.replace("**", ""), normal_style))
    elements.append(Spacer(1, 14))

    # --- Disclaimer ---
    disclaimer_style = ParagraphStyle("Disclaimer", parent=normal_style,
                                       fontSize=8, textColor=colors.grey)
    elements.append(Paragraph(
        "Disclaimer: This report is generated by an educational machine learning "
        "model and is NOT a certified medical diagnosis. Please consult a qualified "
        "healthcare professional for any medical concerns.",
        disclaimer_style
    ))

    doc.build(elements)
    return filepath
