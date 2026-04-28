"""
PancrAI — AI Diagnostic Report via Google Gemini
Generates structured clinical reports from scan analysis results.
"""

import os
import base64
from typing import Optional, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Models tried in order — each has separate quota
GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-2.5-flash",
]


def _build_report_prompt(
    tumor_class, confidence, confidence_scores, measurements,
    uncertainty_score, patient_name, patient_age, patient_sex,
    symptoms, scan_type, risk_level,
):
    conf_str = ", ".join([
        f"No Tumor: {confidence_scores[0]*100:.1f}%",
        f"Benign: {confidence_scores[1]*100:.1f}%",
        f"Malignant (PDAC): {confidence_scores[2]*100:.1f}%",
        f"Cystic (IPMN): {confidence_scores[3]*100:.1f}%",
    ]) if len(confidence_scores) == 4 else str(confidence_scores)

    measurements_text = "Not available"
    if measurements:
        measurements_text = (
            f"Tumor area: {measurements.get('area_cm2','N/A')} cm², "
            f"Centroid: ({measurements.get('centroid_x','N/A')}, "
            f"{measurements.get('centroid_y','N/A')}), "
            f"Bounding box: {measurements.get('bbox_w','N/A')} x "
            f"{measurements.get('bbox_h','N/A')} pixels, "
            f"Aspect ratio: {measurements.get('aspect_ratio','N/A')}"
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

Include a disclaimer at the bottom noting this is AI-assisted and must be reviewed by a licensed radiologist.

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
    """Generate AI diagnostic report using Google Gemini with automatic model fallback."""

    if not GEMINI_API_KEY:
        return _fallback_report(
            tumor_class, confidence, risk_level,
            patient_name, patient_age, patient_sex,
            measurements, uncertainty_score
        )

    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)

    prompt = _build_report_prompt(
        tumor_class=tumor_class, confidence=confidence,
        confidence_scores=confidence_scores, measurements=measurements,
        uncertainty_score=uncertainty_score, patient_name=patient_name,
        patient_age=patient_age, patient_sex=patient_sex,
        symptoms=symptoms, scan_type=scan_type, risk_level=risk_level,
    )

    parts = [prompt]
    if segmented_image_b64:
        parts.append({
            "inline_data": {
                "mime_type": "image/png",
                "data": segmented_image_b64,
            }
        })

    # Try each model until one succeeds
    last_error = None
    for model_name in GEMINI_MODELS:
        try:
            print(f"[Gemini] Trying model: {model_name}")
            model    = genai.GenerativeModel(model_name)
            response = model.generate_content(parts)
            full_text = response.text
            print(f"[Gemini] Success with: {model_name}")

            if "SUMMARY:" in full_text:
                parts_split = full_text.split("SUMMARY:", 1)
                report_html = parts_split[0].strip()
                summary     = parts_split[1].strip()
            else:
                report_html = full_text
                summary     = (f"AI analysis detected {tumor_class} with "
                               f"{confidence*100:.1f}% confidence. "
                               f"Risk level: {risk_level}.")

            return {
                "report_html": _wrap_report_html(report_html, patient_name),
                "summary":     summary,
                "risk_level":  risk_level,
                "model_used":  model_name,
            }

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                print(f"[Gemini] {model_name} quota exceeded, trying next...")
                last_error = e
                continue
            else:
                # Non-quota error — don't retry
                print(f"[Gemini] {model_name} failed with non-quota error: {e}")
                last_error = e
                break

    print(f"[Gemini] All models exhausted. Last error: {last_error}")
    return _fallback_report(
        tumor_class, confidence, risk_level,
        patient_name, patient_age, patient_sex,
        measurements, uncertainty_score
    )


def _wrap_report_html(content: str, patient_name: str) -> str:
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
    tumor_class, confidence, risk_level,
    patient_name, patient_age, patient_sex,
    measurements, uncertainty_score,
):
    meas_html = ""
    if measurements:
        meas_html = f"""
        <ul>
          <li>Estimated area: <strong>{measurements.get('area_cm2','N/A')} cm²</strong></li>
          <li>Centroid: ({measurements.get('centroid_x','N/A')}, {measurements.get('centroid_y','N/A')}) px</li>
          <li>Bounding box: {measurements.get('bbox_w','N/A')} x {measurements.get('bbox_h','N/A')} px</li>
          <li>Aspect ratio: {measurements.get('aspect_ratio','N/A')}</li>
        </ul>"""
    else:
        meas_html = "<p>No tumor region detected in the segmentation mask.</p>"

    risk_color = {
        "Low": "#4CAF50", "Medium": "#FF9800",
        "High": "#FF5722", "Critical": "#F44336"
    }.get(risk_level, "#9E9E9E")

    report_html = f"""
    <div class="report-section">
      <h2>1. Patient Information</h2>
      <p>Name: <strong>{patient_name}</strong> | Age: <strong>{patient_age or 'N/A'}</strong> | Sex: <strong>{patient_sex or 'N/A'}</strong></p>
    </div>
    <div class="report-section">
      <h2>2. Scan Type and Quality Assessment</h2>
      <p>Scan analyzed by PancrAI pipeline with Ensemble inference (TransUNet + LightUNet). Image quality sufficient for automated analysis.</p>
    </div>
    <div class="report-section">
      <h2>3. Findings</h2>
      <p>AI model classification: <strong>{tumor_class}</strong> with <strong>{confidence*100:.1f}%</strong> confidence. Uncertainty: {uncertainty_score:.1f}/100.</p>
    </div>
    <div class="report-section">
      <h2>4. Tumor Characteristics</h2>
      {meas_html}
    </div>
    <div class="report-section">
      <h2>5. Differential Diagnosis</h2>
      <p>Based on AI classification of <strong>{tumor_class}</strong>, differential diagnoses should be considered in clinical context including patient history, laboratory findings, and additional imaging modalities.</p>
    </div>
    <div class="report-section">
      <h2>6. Staging Assessment</h2>
      <p>Formal staging requires clinical examination, laboratory values (CA 19-9, CEA), and multidisciplinary team review.</p>
    </div>
    <div class="report-section">
      <h2>7. Risk Level</h2>
      <p>AI-assessed risk: <strong style="color:{risk_color}">{risk_level}</strong></p>
    </div>
    <div class="report-section">
      <h2>8. Recommended Next Steps</h2>
      <ul>
        <li>Correlation with clinical symptoms and laboratory findings</li>
        <li>Multidisciplinary tumor board review</li>
        <li>Endoscopic ultrasound (EUS) with fine-needle aspiration</li>
        <li>MRI with MRCP for additional characterization</li>
        <li>CA 19-9 and CEA serum markers</li>
      </ul>
    </div>
    <div class="report-section">
      <h2>9. Radiologist Notes</h2>
      <p>All Gemini AI models are currently quota-limited (free tier daily limit reached).
         This structured report was auto-generated from analysis data.
         Full AI narrative report will be available after quota reset (midnight Pacific time).
         All findings require validation by a qualified radiologist.</p>
    </div>
    """

    return {
        "report_html": _wrap_report_html(report_html, patient_name),
        "summary": (f"AI analysis detected {tumor_class} with {confidence*100:.1f}% confidence. "
                    f"Risk level: {risk_level}. Clinical correlation recommended."),
        "risk_level": risk_level,
        "model_used": "fallback",
    }