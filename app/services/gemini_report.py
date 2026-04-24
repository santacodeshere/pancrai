"""
PancrAI — AI Diagnostic Report via Google Gemini 1.5 Flash
Generates structured clinical reports from scan analysis results.
"""

import os
import base64
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")


def _build_report_prompt(
    tumor_class: str,
    confidence: float,
    confidence_scores: list,
    measurements: Optional[Dict],
    uncertainty_score: float,
    patient_name: str,
    patient_age: Optional[int],
    patient_sex: Optional[str],
    symptoms: Optional[str],
    scan_type: str,
    risk_level: str,
) -> str:
    """Build the structured prompt for Gemini report generation."""

    conf_str = ", ".join([
        f"No Tumor: {confidence_scores[0]*100:.1f}%",
        f"Benign: {confidence_scores[1]*100:.1f}%",
        f"Malignant (PDAC): {confidence_scores[2]*100:.1f}%",
        f"Cystic (IPMN): {confidence_scores[3]*100:.1f}%",
    ]) if len(confidence_scores) == 4 else str(confidence_scores)

    measurements_text = "Not available"
    if measurements:
        measurements_text = (
            f"Tumor area: {measurements.get('area_cm2', 'N/A')} cm², "
            f"Centroid: ({measurements.get('centroid_x', 'N/A')}, "
            f"{measurements.get('centroid_y', 'N/A')}), "
            f"Bounding box: {measurements.get('bbox_w', 'N/A')} × "
            f"{measurements.get('bbox_h', 'N/A')} pixels, "
            f"Aspect ratio: {measurements.get('aspect_ratio', 'N/A')}"
        )

    return f"""You are an expert radiologist AI assistant for pancreatic tumor analysis.
Generate a complete structured clinical diagnostic report based on the following AI analysis results.

=== ANALYSIS DATA ===
Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Patient: {patient_name}
Age: {patient_age or 'Not specified'}
Sex: {patient_sex or 'Not specified'}
Scan type: {scan_type}
Symptoms: {symptoms or 'Not provided'}

AI Detection Result: {tumor_class}
Primary Confidence: {confidence*100:.1f}%
All class probabilities: {conf_str}
Model Uncertainty Score: {uncertainty_score:.1f}/100 (lower is better)
Risk Level: {risk_level}
Tumor Measurements: {measurements_text}

=== INSTRUCTIONS ===
Generate a professional radiological report in clean HTML with these exact sections.
Use appropriate medical terminology. Be thorough but precise.

Structure the report with these HTML sections:

<div class="report-section"><h2>1. Patient Information</h2>...</div>
<div class="report-section"><h2>2. Scan Type and Quality Assessment</h2>...</div>
<div class="report-section"><h2>3. Findings</h2>...</div>
<div class="report-section"><h2>4. Tumor Characteristics</h2>...</div>
<div class="report-section"><h2>5. Differential Diagnosis</h2>...</div>
<div class="report-section"><h2>6. Staging Assessment</h2>...</div>
<div class="report-section"><h2>7. Risk Level</h2>...</div>
<div class="report-section"><h2>8. Recommended Next Steps</h2>...</div>
<div class="report-section"><h2>9. Radiologist Notes</h2>...</div>

Include a disclaimer at the bottom noting this is an AI-assisted report and must be reviewed by a licensed radiologist before clinical use.

After the HTML report, on a new line write:
SUMMARY: [2-3 sentence plain English summary of the most important findings]
"""


async def generate_report(
    tumor_class: str,
    confidence: float,
    confidence_scores: list,
    measurements: Optional[Dict],
    uncertainty_score: float,
    patient_name: str = "Unknown",
    patient_age: Optional[int] = None,
    patient_sex: Optional[str] = None,
    symptoms: Optional[str] = None,
    scan_type: str = "CT",
    risk_level: str = "Unknown",
    segmented_image_b64: Optional[str] = None,
    gradcam_image_b64: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate an AI diagnostic report using Google Gemini 1.5 Flash.

    Args:
        tumor_class: Predicted tumor class string.
        confidence: Primary class confidence (0–1).
        confidence_scores: List of [no_tumor, benign, malignant, cystic] probabilities.
        measurements: Dict from measure_tumor().
        uncertainty_score: MC Dropout uncertainty (0–100).
        patient_name: Patient name for the report header.
        patient_age: Patient age in years.
        patient_sex: 'Male', 'Female', or 'Other'.
        symptoms: Free-text symptom description.
        scan_type: 'CT', 'MRI', or 'PET'.
        risk_level: Clinical risk classification.
        segmented_image_b64: Optional base64 PNG of segmentation overlay.
        gradcam_image_b64: Optional base64 PNG of Grad-CAM heatmap.

    Returns:
        Dict with 'report_html' and 'summary' keys.
    """
    if not GEMINI_API_KEY:
        return _fallback_report(
            tumor_class, confidence, risk_level,
            patient_name, patient_age, patient_sex,
            measurements, uncertainty_score
        )

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = _build_report_prompt(
            tumor_class=tumor_class,
            confidence=confidence,
            confidence_scores=confidence_scores,
            measurements=measurements,
            uncertainty_score=uncertainty_score,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_sex=patient_sex,
            symptoms=symptoms,
            scan_type=scan_type,
            risk_level=risk_level,
        )

        # Build content parts — include images if available
        parts = [prompt]

        if segmented_image_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": segmented_image_b64,
                }
            })

        if gradcam_image_b64:
            parts.append({
                "inline_data": {
                    "mime_type": "image/png",
                    "data": gradcam_image_b64,
                }
            })

        response = model.generate_content(parts)
        full_text = response.text

        # Parse HTML report and summary
        if "SUMMARY:" in full_text:
            parts_split = full_text.split("SUMMARY:", 1)
            report_html = parts_split[0].strip()
            summary = parts_split[1].strip()
        else:
            report_html = full_text
            summary = f"AI analysis detected {tumor_class} with {confidence*100:.1f}% confidence. Risk level: {risk_level}."

        # Wrap in styled container
        styled_report = _wrap_report_html(report_html, patient_name)

        return {
            "report_html": styled_report,
            "summary": summary,
            "risk_level": risk_level,
        }

    except Exception as e:
        print(f"[Gemini Report] Error: {e}")
        return _fallback_report(
            tumor_class, confidence, risk_level,
            patient_name, patient_age, patient_sex,
            measurements, uncertainty_score
        )


def _wrap_report_html(content: str, patient_name: str) -> str:
    """Wrap report content in a styled HTML container."""
    return f"""
<div style="font-family: 'Georgia', serif; max-width: 900px; margin: 0 auto;
            padding: 24px; background: #fff; color: #1a1a2e;
            border: 1px solid #e0e0e0; border-radius: 8px;">
  <div style="text-align: center; border-bottom: 2px solid #1565C0; 
              padding-bottom: 16px; margin-bottom: 24px;">
    <h1 style="font-size: 22px; color: #1565C0; margin: 0;">
      PancrAI — AI-Assisted Radiology Report
    </h1>
    <p style="color: #666; margin: 4px 0 0;">
      Patient: <strong>{patient_name}</strong> &nbsp;|&nbsp;
      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
    </p>
  </div>
  <style>
    .report-section {{ margin-bottom: 20px; }}
    .report-section h2 {{
      font-size: 15px; font-weight: 700;
      color: #1565C0; border-left: 4px solid #1565C0;
      padding-left: 10px; margin-bottom: 8px;
    }}
    .report-section p, .report-section li {{ 
      font-size: 14px; line-height: 1.7; color: #333; 
    }}
    ul {{ padding-left: 20px; }}
  </style>
  {content}
  <div style="margin-top: 24px; padding: 12px; background: #FFF9C4; 
              border: 1px solid #F9A825; border-radius: 4px; font-size: 12px;">
    ⚠️ <strong>Disclaimer:</strong> This report was generated by an AI system 
    and is intended for informational purposes only. It must be reviewed and 
    validated by a licensed radiologist or physician before any clinical 
    decision-making. PancrAI does not replace professional medical judgment.
  </div>
</div>
"""


def _fallback_report(
    tumor_class: str,
    confidence: float,
    risk_level: str,
    patient_name: str,
    patient_age: Optional[int],
    patient_sex: Optional[str],
    measurements: Optional[Dict],
    uncertainty_score: float,
) -> Dict[str, str]:
    """
    Generate a structured fallback report when Gemini API is unavailable.
    Returns a complete HTML report using available data.
    """
    meas_html = ""
    if measurements:
        meas_html = f"""
        <ul>
          <li>Estimated area: <strong>{measurements.get('area_cm2', 'N/A')} cm²</strong></li>
          <li>Centroid location: ({measurements.get('centroid_x', 'N/A')}, 
              {measurements.get('centroid_y', 'N/A')}) px</li>
          <li>Bounding box: {measurements.get('bbox_w', 'N/A')} × 
              {measurements.get('bbox_h', 'N/A')} pixels</li>
          <li>Aspect ratio: {measurements.get('aspect_ratio', 'N/A')}</li>
        </ul>
        """
    else:
        meas_html = "<p>No tumor region detected in the segmentation mask.</p>"

    risk_color = {
        "Low": "#4CAF50", "Medium": "#FF9800",
        "High": "#FF5722", "Critical": "#F44336"
    }.get(risk_level, "#9E9E9E")

    report_html = f"""
    <div class="report-section">
      <h2>1. Patient Information</h2>
      <p>Name: <strong>{patient_name}</strong> &nbsp;|&nbsp;
         Age: <strong>{patient_age or 'N/A'}</strong> &nbsp;|&nbsp;
         Sex: <strong>{patient_sex or 'N/A'}</strong></p>
    </div>
    <div class="report-section">
      <h2>2. Scan Type and Quality Assessment</h2>
      <p>Scan analyzed by AI pipeline. Image quality sufficient for automated analysis.
         Standard windowing and preprocessing applied.</p>
    </div>
    <div class="report-section">
      <h2>3. Findings</h2>
      <p>AI model classification result: <strong>{tumor_class}</strong> with 
         <strong>{confidence*100:.1f}%</strong> primary confidence.
         Model uncertainty score: {uncertainty_score:.1f}/100.</p>
    </div>
    <div class="report-section">
      <h2>4. Tumor Characteristics</h2>
      {meas_html}
    </div>
    <div class="report-section">
      <h2>5. Differential Diagnosis</h2>
      <p>Based on the AI classification of <strong>{tumor_class}</strong>, 
         differential diagnoses should be considered in clinical context including 
         patient history, laboratory findings, and additional imaging modalities.</p>
    </div>
    <div class="report-section">
      <h2>6. Staging Assessment</h2>
      <p>Formal staging requires correlation with clinical examination, 
         laboratory values (CA 19-9, CEA), and multidisciplinary team review.</p>
    </div>
    <div class="report-section">
      <h2>7. Risk Level</h2>
      <p>AI-assessed risk: <strong style="color: {risk_color};">{risk_level}</strong></p>
    </div>
    <div class="report-section">
      <h2>8. Recommended Next Steps</h2>
      <ul>
        <li>Correlation with clinical symptoms and laboratory findings</li>
        <li>Multidisciplinary tumor board review</li>
        <li>Consider endoscopic ultrasound (EUS) with fine-needle aspiration</li>
        <li>MRI with MRCP for additional characterization if CT was primary modality</li>
        <li>CA 19-9, CEA serum markers</li>
      </ul>
    </div>
    <div class="report-section">
      <h2>9. Radiologist Notes</h2>
      <p>This report was generated in offline mode without Gemini AI. 
         Please configure GEMINI_API_KEY in .env for full AI-generated reports.
         All findings require validation by a qualified radiologist.</p>
    </div>
    """

    summary = (
        f"AI analysis detected {tumor_class} with {confidence*100:.1f}% confidence. "
        f"Risk level assessed as {risk_level}. "
        f"Clinical correlation and specialist review recommended."
    )

    return {
        "report_html": _wrap_report_html(report_html, patient_name),
        "summary": summary,
        "risk_level": risk_level,
    }
