"""
PancrAI — PDF Report Export
Generates a professional PDF clinical report using reportlab.
Falls back to HTML if reportlab is not available.
"""

import os
import io
import base64
from datetime import datetime
from typing import Optional, Dict, Any


def generate_pdf_report(
    patient_info: Dict,
    result: Dict,
    staging_report: Optional[Dict] = None,
    radiomics: Optional[Dict] = None,
    report_html: Optional[str] = None,
) -> bytes:
    """
    Generate a PDF clinical report.

    Args:
        patient_info: Patient demographics dict
        result: Analysis result dict from run_inline_analysis
        staging_report: TNM staging report dict (optional)
        radiomics: Radiomics features dict (optional)
        report_html: Gemini-generated HTML report (optional)

    Returns:
        PDF bytes
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm, cm
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            HRFlowable, PageBreak
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
        from reportlab.platypus import Image as RLImage

        return _generate_pdf_reportlab(
            patient_info, result, staging_report, radiomics
        )

    except ImportError:
        # Fallback: return HTML as bytes
        html = _generate_html_fallback(patient_info, result, staging_report)
        return html.encode("utf-8")


def _generate_pdf_reportlab(patient_info, result, staging_report, radiomics):
    """Generate PDF using reportlab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak, KeepTogether
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=20*mm, bottomMargin=20*mm
    )

    # Colors
    BLUE      = colors.HexColor("#1F5C99")
    LIGHTBLUE = colors.HexColor("#E8F0FE")
    DARKGRAY  = colors.HexColor("#333333")
    MIDGRAY   = colors.HexColor("#666666")
    RED       = colors.HexColor("#D32F2F")
    ORANGE    = colors.HexColor("#F57C00")
    GREEN     = colors.HexColor("#388E3C")

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title",
        fontSize=22, textColor=BLUE, fontName="Helvetica-Bold",
        alignment=TA_CENTER, spaceAfter=6)
    subtitle_style = ParagraphStyle("Subtitle",
        fontSize=12, textColor=MIDGRAY, fontName="Helvetica",
        alignment=TA_CENTER, spaceAfter=4)
    h1_style = ParagraphStyle("H1",
        fontSize=14, textColor=BLUE, fontName="Helvetica-Bold",
        spaceBefore=14, spaceAfter=6, borderPad=4)
    h2_style = ParagraphStyle("H2",
        fontSize=11, textColor=DARKGRAY, fontName="Helvetica-Bold",
        spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle("Body",
        fontSize=9, textColor=DARKGRAY, fontName="Helvetica",
        alignment=TA_JUSTIFY, spaceAfter=4, leading=14)
    small_style = ParagraphStyle("Small",
        fontSize=8, textColor=MIDGRAY, fontName="Helvetica",
        spaceAfter=2)
    warn_style = ParagraphStyle("Warn",
        fontSize=9, textColor=RED, fontName="Helvetica-Bold",
        spaceAfter=4)

    def tbl_style(header_color=BLUE):
        return TableStyle([
            ("BACKGROUND",    (0,0), (-1,0), header_color),
            ("TEXTCOLOR",     (0,0), (-1,0), colors.white),
            ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",      (0,0), (-1,0), 9),
            ("FONTNAME",      (0,1), (-1,-1), "Helvetica"),
            ("FONTSIZE",      (0,1), (-1,-1), 8),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [colors.white, LIGHTBLUE]),
            ("GRID",          (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("TOPPADDING",    (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
        ])

    story = []
    W = A4[0] - 40*mm  # usable width

    # ── Header ────────────────────────────────────────────────────────────────
    story.append(Paragraph("PancrAI", title_style))
    story.append(Paragraph("Intelligent Pancreatic Tumor Detection", subtitle_style))
    story.append(Paragraph("AI-Assisted Clinical Diagnostic Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=BLUE, spaceAfter=10))

    # ── Patient Info ──────────────────────────────────────────────────────────
    story.append(Paragraph("PATIENT INFORMATION", h1_style))
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    pat_data = [
        ["Field", "Value", "Field", "Value"],
        ["Patient Name",  patient_info.get("name", "N/A"),
         "Report Date",   now],
        ["Age",           str(patient_info.get("age", "N/A")),
         "Sex",           patient_info.get("sex", "N/A")],
        ["Scan Type",     patient_info.get("scan_type", "CT"),
         "Risk Level",    result.get("risk_level", "N/A")],
        ["Symptoms",      patient_info.get("symptoms", "None provided")[:40],
         "Report ID",     f"PCR-{datetime.now().strftime('%Y%m%d%H%M')}"],
    ]
    t = Table(pat_data, colWidths=[W*0.18, W*0.32, W*0.18, W*0.32])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Spacer(1, 8))

    # ── Analysis Results ──────────────────────────────────────────────────────
    story.append(Paragraph("ANALYSIS RESULTS", h1_style))

    tumor_class = result.get("tumor_class", "Unknown")
    confidence  = result.get("primary_confidence", 0.0)
    uncertainty = result.get("uncertainty_score", 0.0)
    dice        = result.get("dice_score", 0.0)
    iou         = result.get("iou_score", 0.0)

    res_color = RED if "Malignant" in tumor_class else ORANGE if tumor_class == "Cystic (IPMN)" else GREEN
    res_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Detection Result",  tumor_class,
         "Primary Confidence",f"{confidence*100:.1f}%"],
        ["Risk Level",        result.get("risk_level", "N/A"),
         "Uncertainty Score", f"{uncertainty:.1f}%"],
        ["Dice Score",        f"{dice:.4f}" if dice > 0 else "N/A",
         "IoU Score",         f"{iou:.4f}" if iou > 0 else "N/A"],
        ["Tumor Detected",    "Yes" if result.get("tumor_detected") else "No",
         "TTA Augmentations", "8x (enabled)"],
    ]
    t = Table(res_data, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Spacer(1, 6))

    # Uncertainty warning
    if uncertainty > 60:
        story.append(Paragraph(
            f"⚠ HIGH UNCERTAINTY ({uncertainty:.1f}%) — Specialist radiologist review strongly recommended",
            warn_style))

    # ── Tumor Measurements ────────────────────────────────────────────────────
    measurements = result.get("measurements")
    if measurements:
        story.append(Paragraph("TUMOR MEASUREMENTS", h1_style))
        meas_data = [
            ["Measurement", "Value", "Measurement", "Value"],
            ["Area (cm²)",      f"{measurements.get('area_cm2', 0):.3f}",
             "Area (pixels)",   str(measurements.get("area_pixels", "N/A"))],
            ["Centroid X",      f"{measurements.get('centroid_x', 0):.0f} px",
             "Centroid Y",      f"{measurements.get('centroid_y', 0):.0f} px"],
            ["Bounding Box W",  f"{measurements.get('bbox_w', 0)} px",
             "Bounding Box H",  f"{measurements.get('bbox_h', 0)} px"],
            ["Aspect Ratio",    f"{measurements.get('aspect_ratio', 0):.3f}",
             "Area %",          f"{measurements.get('area_pct', 0):.2f}%"],
        ]
        t = Table(meas_data, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 6))

    # ── TNM Staging ───────────────────────────────────────────────────────────
    if staging_report:
        story.append(Paragraph("TNM STAGING ASSESSMENT", h1_style))

        ts   = staging_report.get("t_stage", {})
        os_  = staging_report.get("overall_stage", {})
        res  = staging_report.get("resectability", {})
        risk = staging_report.get("risk_score", {})

        stg_data = [
            ["Parameter", "Value"],
            ["T-Stage",              ts.get("t_stage", "N/A")],
            ["Size Estimate",        ts.get("size_estimate", "N/A")],
            ["Overall Stage",        os_.get("overall_stage", "N/A")],
            ["TNM Classification",   os_.get("tnm", "N/A")],
            ["5-Year Survival Est.", os_.get("five_year_survival", "N/A")],
            ["Resectability",        res.get("status", "N/A")],
            ["Suggested Procedure",  res.get("procedure", "MDT review required")],
            ["Composite Risk Score", f"{risk.get('score', 0)}/100 — {risk.get('category', 'N/A')}"],
        ]
        t = Table(stg_data, colWidths=[W*0.40, W*0.60])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 6))

        # Recommendations
        recs = risk.get("recommendations", [])
        if recs:
            story.append(Paragraph("Clinical Recommendations:", h2_style))
            for rec in recs:
                story.append(Paragraph(f"• {rec}", body_style))
        story.append(Spacer(1, 6))

    # ── Radiomics Summary ─────────────────────────────────────────────────────
    if radiomics:
        story.append(Paragraph("RADIOMICS FEATURE SUMMARY", h1_style))

        key_features = [
            ["Feature", "Value", "Feature", "Value"],
            ["Shape Circularity", f"{radiomics.get('shape_circularity', 0):.4f}",
             "Shape Solidity",    f"{radiomics.get('shape_solidity', 0):.4f}"],
            ["Intensity Mean",    f"{radiomics.get('intensity_mean', 0):.2f}",
             "Intensity Entropy", f"{radiomics.get('intensity_entropy', 0):.4f}"],
            ["GLCM Contrast",     f"{radiomics.get('glcm_contrast', 0):.4f}",
             "GLCM Homogeneity",  f"{radiomics.get('glcm_homogeneity', 0):.4f}"],
            ["LBP Entropy",       f"{radiomics.get('lbp_entropy', 0):.4f}",
             "Gradient Edge Den.",f"{radiomics.get('gradient_edge_density', 0):.4f}"],
            ["Wavelet LL Mean",   f"{radiomics.get('wavelet_ll_mean', 0):.3f}",
             "GLCM Correlation",  f"{radiomics.get('glcm_correlation', 0):.4f}"],
        ]
        t = Table(key_features, colWidths=[W*0.22, W*0.28, W*0.22, W*0.28])
        t.setStyle(tbl_style())
        story.append(t)
        story.append(Spacer(1, 6))

    # ── Confidence Scores ─────────────────────────────────────────────────────
    story.append(Paragraph("CLASSIFIER CONFIDENCE SCORES", h1_style))
    scores = result.get("confidence_scores", [0.25, 0.25, 0.25, 0.25])
    classes = ["No Tumor", "Benign", "Malignant (PDAC)", "Cystic (IPMN)"]
    conf_data = [["Class", "Confidence", "Visual"]]
    for cls, sc in zip(classes, scores):
        bar = "█" * int(sc * 20) + "░" * (20 - int(sc * 20))
        conf_data.append([cls, f"{sc*100:.1f}%", bar])
    t = Table(conf_data, colWidths=[W*0.30, W*0.15, W*0.55])
    t.setStyle(tbl_style())
    story.append(t)
    story.append(Spacer(1, 10))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=6))
    story.append(Paragraph(
        "⚠ DISCLAIMER: This report was generated by an AI-assisted system (PancrAI) "
        "for decision support purposes only. It must be reviewed and validated by a "
        "licensed radiologist or physician before any clinical decision-making. "
        "PancrAI does not replace professional medical judgment. All staging, "
        "resectability, and risk assessments are estimates requiring clinical correlation.",
        small_style))
    story.append(Paragraph(
        f"Generated: {now} | PancrAI v1.0 | For Research Use Only",
        small_style))

    doc.build(story)
    return buf.getvalue()


def _generate_html_fallback(patient_info, result, staging_report):
    """HTML fallback when reportlab is not available."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    tumor_class = result.get("tumor_class", "Unknown")
    confidence  = result.get("primary_confidence", 0.0)
    risk        = result.get("risk_level", "Unknown")
    unc         = result.get("uncertainty_score", 0.0)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PancrAI Report — {patient_info.get('name','Patient')}</title>
<style>
  body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 24px; color: #333; }}
  h1 {{ color: #1F5C99; border-bottom: 2px solid #1F5C99; padding-bottom: 8px; }}
  h2 {{ color: #2E75B6; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 16px; }}
  th {{ background: #1F5C99; color: white; padding: 8px; text-align: left; }}
  td {{ padding: 6px 8px; border: 1px solid #ddd; }}
  tr:nth-child(even) {{ background: #f5f8fc; }}
  .disclaimer {{ background: #FFF9C4; border: 1px solid #F9A825; padding: 12px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
<h1>PancrAI — AI-Assisted Clinical Report</h1>
<p><strong>Patient:</strong> {patient_info.get('name','N/A')} |
   <strong>Age:</strong> {patient_info.get('age','N/A')} |
   <strong>Date:</strong> {now}</p>
<h2>Analysis Results</h2>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Detection Result</td><td><strong>{tumor_class}</strong></td></tr>
  <tr><td>Confidence</td><td>{confidence*100:.1f}%</td></tr>
  <tr><td>Risk Level</td><td>{risk}</td></tr>
  <tr><td>Uncertainty</td><td>{unc:.1f}%</td></tr>
</table>
<div class="disclaimer">
⚠ This report was generated by an AI system for decision support only.
Must be reviewed by a licensed radiologist before clinical use.
</div>
</body>
</html>"""
    return html


if __name__ == "__main__":
    print("Testing PDF generation...")
    test_patient = {"name": "Test Patient", "age": 65, "sex": "Male",
                    "scan_type": "CT", "symptoms": "Epigastric pain"}
    test_result = {
        "tumor_class": "Malignant (PDAC)", "primary_confidence": 0.87,
        "risk_level": "High", "uncertainty_score": 22.0,
        "tumor_detected": True, "dice_score": 0.8204, "iou_score": 0.7671,
        "confidence_scores": [0.02, 0.08, 0.87, 0.03],
        "measurements": {"area_cm2": 3.5, "area_pixels": 700,
                         "centroid_x": 112, "centroid_y": 98,
                         "bbox_w": 35, "bbox_h": 30, "aspect_ratio": 1.17,
                         "area_pct": 1.39},
    }
    pdf_bytes = generate_pdf_report(test_patient, test_result)
    with open("/tmp/test_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    print(f"PDF generated: {len(pdf_bytes)} bytes")
    print("Check /tmp/test_report.pdf")
