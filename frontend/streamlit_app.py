"""
PancrAI — Streamlit Frontend v3.0
Integrates all advanced features:
- Ensemble inference (TransUNet + LightUNet)
- Differential diagnosis generator
- RECIST measurements
- CA 19-9 combined risk score
- SHAP feature importance
- Attention map visualization
- Calibration curve
- Survival curves
- DICOM metadata extraction
- Confusion matrix
- TNM staging, radiomics, 3D viz, PDF export, TTA
"""

import sys
import os
import io
import base64
import requests
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PancrAI — Pancreatic Tumor Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
CLASS_NAMES  = ["No Tumor", "Benign", "Malignant (PDAC)", "Cystic (IPMN)"]
CLASS_COLORS = {
    "No Tumor": "#4CAF50", "Benign": "#FF9800",
    "Malignant (PDAC)": "#F44336", "Cystic (IPMN)": "#2196F3",
    "Unknown": "#9E9E9E"
}
RISK_COLORS = {
    "Low": "#4CAF50", "Medium": "#FF9800",
    "High": "#FF5722", "Critical": "#F44336", "Unknown": "#9E9E9E"
}


def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .stApp { background: #0D1117; color: #E6EDF3; }
    section[data-testid="stSidebar"] { background: #161B22; border-right: 1px solid #21262D; }
    .metric-card { background: #161B22; border: 1px solid #21262D; border-radius: 10px; padding: 20px; text-align: center; }
    .metric-card .value { font-size: 2.2rem; font-weight: 700; color: #58A6FF; font-family: 'IBM Plex Mono', monospace; }
    .metric-card .label { font-size: 0.82rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 4px; }
    .scan-step-card { background: #161B22; border: 1px solid #21262D; border-radius: 8px; padding: 12px; text-align: center; }
    .scan-step-card .step-name { font-size: 0.78rem; font-weight: 600; color: #58A6FF; margin-bottom: 6px; text-transform: uppercase; }
    .scan-step-card .step-desc { font-size: 0.7rem; color: #8B949E; line-height: 1.4; }
    .chat-msg-user { background: #1C2128; border-left: 3px solid #58A6FF; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0; }
    .chat-msg-assistant { background: #161B22; border-left: 3px solid #3FB950; border-radius: 0 8px 8px 0; padding: 10px 14px; margin: 8px 0; line-height: 1.6; }
    .section-header { font-size: 1.4rem; font-weight: 700; color: #E6EDF3; border-bottom: 2px solid #21262D; padding-bottom: 8px; margin-bottom: 20px; }
    .warning-box { background: #2D1B00; border: 1px solid #9E6A03; border-radius: 8px; padding: 12px 16px; color: #E3B341; font-size: 0.88rem; }
    .info-box { background: #0D2136; border: 1px solid #388BFD; border-radius: 8px; padding: 12px 16px; color: #79C0FF; font-size: 0.88rem; }
    .success-box { background: #0D2B0D; border: 1px solid #3FB950; border-radius: 8px; padding: 12px 16px; color: #56D364; font-size: 0.88rem; }
    .stButton > button { background: #238636; color: white; border: none; border-radius: 6px; font-weight: 600; }
    .stButton > button:hover { background: #2EA043; }
    .pancrai-logo { font-size: 1.6rem; font-weight: 700; color: #58A6FF; }
    .pancrai-logo span { color: #3FB950; }
    .feature-badge { display: inline-block; background: #1F5C99; color: white; font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; margin: 2px; }
    .dicom-field { padding: 4px 0; border-bottom: 1px solid #21262D; }
    .dicom-label { color: #8B949E; font-size: 0.8rem; }
    .dicom-value { color: #E6EDF3; font-size: 0.85rem; font-family: monospace; }
    </style>
    """, unsafe_allow_html=True)


def api_get(endpoint, **params):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_post(endpoint, data=None, json_data=None, files=None):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", data=data,
                          json=json_data, files=files, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def b64_to_image(b64_str):
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64_str)))
    except Exception:
        return None


def _check_api():
    try:
        r = requests.get(f"{API_BASE.replace('/api/v1','')}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─── Inline Analysis ──────────────────────────────────────────────────────────

def run_inline_analysis(file_bytes, filename, patient_info,
                        use_tta=True, use_ensemble=True):
    try:
        from app.services.preprocessing import (
            load_from_bytes, run_full_pipeline, preprocess_to_tensor)
        from app.services.segmentation import run_segmentation
        from app.services.tta import run_tta_segmentation
        from app.services.ensemble import build_light_unet, run_ensemble_segmentation
        from app.models.transunet import build_transunet
        from app.models.classifier import build_classifier, classify_from_mask
        from app.services.radiomics import extract_all_radiomics, get_clinical_interpretation
        from app.services.staging import generate_staging_report
        from app.services.advanced_analytics import (
            generate_differential_diagnosis,
            calculate_recist,
            calculate_shap_importance,
        )
        import torch

        image = load_from_bytes(file_bytes, filename)
        preprocess_steps = run_full_pipeline(image)

        if "seg_model" not in st.session_state:
            seg_weights = os.getenv("MODEL_WEIGHTS_PATH", "./weights/transunet_best.pth")
            with st.spinner("Loading AI models..."):
                st.session_state.seg_model = build_transunet(weights_path=seg_weights)
                st.session_state.cls_model = build_classifier()
                st.session_state.light_unet = build_light_unet()

        seg_model  = st.session_state.seg_model
        light_unet = st.session_state.light_unet
        tensor     = preprocess_to_tensor(image)
        image_np   = np.array(image) if not isinstance(image, np.ndarray) else image

        # Segmentation
        if use_ensemble:
            seg_result = run_ensemble_segmentation(
                seg_model, light_unet, tensor, image_np, primary_weight=0.70)
        elif use_tta:
            seg_result = run_tta_segmentation(
                seg_model, tensor, image_np, n_aug=8, threshold=0.5)
        else:
            seg_result = run_segmentation(image, seg_model)

        # Classification
        seg_mask = seg_result.get("mask")
        cls_result = classify_from_mask(seg_mask) if seg_mask is not None \
                     else classify_from_mask(np.zeros((224, 224), dtype=np.float32))

        pred_idx    = cls_result["class_idx"]
        tumor_class = cls_result["class_name"]
        confidence  = cls_result["confidence"]
        probs       = cls_result["confidence_scores"]
        risk        = cls_result["risk_level"]

        # Radiomics
        radiomics_features = None
        radiomics_interp   = None
        try:
            img_for_radio = (image_np * 255).astype(np.uint8) \
                            if image_np.max() <= 1.0 else image_np.astype(np.uint8)
            radiomics_features = extract_all_radiomics(img_for_radio, seg_mask)
            radiomics_interp   = get_clinical_interpretation(radiomics_features)
        except Exception as e:
            st.warning(f"Radiomics: {e}")

        # Staging
        staging = None
        try:
            staging = generate_staging_report(
                tumor_class=tumor_class, confidence=confidence,
                measurements=seg_result.get("measurements"),
                uncertainty_score=0.0, patient_age=patient_info.get("age"),
                radiomics=radiomics_features)
        except Exception as e:
            st.warning(f"Staging: {e}")

        # Differential diagnosis
        differential = []
        try:
            differential = generate_differential_diagnosis(
                tumor_class=tumor_class, confidence=confidence,
                measurements=seg_result.get("measurements"),
                radiomics=radiomics_features,
                patient_age=patient_info.get("age"),
                patient_sex=patient_info.get("sex"))
        except Exception as e:
            st.warning(f"Differential Dx: {e}")

        # RECIST
        recist = None
        try:
            recist = calculate_recist(seg_result.get("measurements", {}))
        except Exception as e:
            st.warning(f"RECIST: {e}")

        # SHAP
        shap_result = None
        try:
            if radiomics_features:
                shap_result = calculate_shap_importance(
                    radiomics_features, tumor_class, confidence)
        except Exception as e:
            st.warning(f"SHAP: {e}")

        # Free GPU memory before uncertainty
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Uncertainty
        unc_score   = 0.0
        unc_details = {}
        try:
            from app.services.uncertainty import mc_dropout_inference
            unc         = mc_dropout_inference(seg_model, tensor.clone(), T=10)  # reduced from T=20
            unc_score   = unc["uncertainty_score"]
            unc_details = unc
        except Exception as e:
            st.warning(f"Uncertainty: {e}")

        return {
            "tumor_detected":      pred_idx > 0,
            "tumor_class":         tumor_class,
            "tumor_class_index":   pred_idx,
            "confidence_scores":   probs,
            "primary_confidence":  confidence,
            "risk_level":          risk,
            "dice_score":          seg_result.get("dice_score", 0.0),
            "iou_score":           seg_result.get("iou_score", 0.0),
            "uncertainty_score":   unc_score,
            "uncertainty_details": unc_details,
            "measurements":        seg_result.get("measurements"),
            "radiomics":           radiomics_features,
            "radiomics_interp":    radiomics_interp,
            "staging":             staging,
            "differential":        differential,
            "recist":              recist,
            "shap":                shap_result,
            "ensemble_enabled":    use_ensemble,
            "tta_enabled":         use_tta,
            "agreement_score":     seg_result.get("agreement_score"),
            "images": {
                "preprocessing_steps":  preprocess_steps,
                "segmentation_overlay": seg_result.get("overlay_b64"),
            },
        }

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ─── Report ───────────────────────────────────────────────────────────────────

def generate_report_and_store(result, patient_info):
    try:
        import asyncio
        from app.services.gemini_report import generate_report as _gen
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        report = loop.run_until_complete(_gen(
            tumor_class=result.get("tumor_class", "Unknown"),
            confidence=result.get("primary_confidence", 0.0),
            confidence_scores=result.get("confidence_scores", [0.25]*4),
            measurements=result.get("measurements"),
            uncertainty_score=result.get("uncertainty_score", 0.0),
            patient_name=patient_info.get("name", "Unknown"),
            patient_age=patient_info.get("age"),
            patient_sex=patient_info.get("sex"),
            symptoms=patient_info.get("symptoms"),
            scan_type=patient_info.get("scan_type", "CT"),
            risk_level=result.get("risk_level", "Unknown"),
            segmented_image_b64=result.get("images", {}).get("segmentation_overlay"),
            gradcam_image_b64=None,
        ))
        st.session_state["last_report"] = report
        return True
    except Exception as e:
        st.error(f"Report failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return False


def render_report(patient_info):
    report = st.session_state.get("last_report")
    if not report:
        return
    st.markdown("---")
    st.markdown("**📋 AI Diagnostic Report**")
    summary = report.get("summary", "")
    if summary:
        st.markdown(f'<div class="info-box">📌 <strong>Summary:</strong> {summary}</div>',
                    unsafe_allow_html=True)
    report_html = report.get("report_html", "")
    if report_html:
        st.components.v1.html(report_html, height=800, scrolling=True)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download HTML", report_html.encode("utf-8"),
                f"PancrAI_{patient_info.get('name','patient')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
                "text/html", key="dl_html")
        with col2:
            try:
                from utils.pdf_export import generate_pdf_report
                result    = st.session_state.get("last_result", {})
                pdf_bytes = generate_pdf_report(
                    patient_info, result,
                    result.get("staging"), result.get("radiomics"))
                st.download_button(
                    "⬇️ Download PDF", pdf_bytes,
                    f"PancrAI_{patient_info.get('name','patient')}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    "application/pdf", key="dl_pdf")
            except Exception as e:
                st.caption(f"PDF: {e}")
    if st.button("🔄 Regenerate", key="btn_regen"):
        del st.session_state["last_report"]
        st.rerun()


# ─── Dashboard ────────────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown('<div class="section-header">🏥 Clinical Dashboard</div>',
                unsafe_allow_html=True)

    stats = api_get("/dashboard/stats") or {
        "total_patients": 0, "total_scans": 0, "scans_today": 0,
        "avg_confidence": 0, "detection_rate": 0, "tumor_type_distribution": {}}

    cols = st.columns(5)
    for col, (icon, val, label) in zip(cols, [
        ("👤", stats["total_patients"],               "Total Patients"),
        ("🔬", stats["total_scans"],                  "Total Scans"),
        ("📅", stats["scans_today"],                  "Scans Today"),
        ("🎯", f"{stats['avg_confidence']*100:.1f}%", "Avg Confidence"),
        ("📊", f"{stats['detection_rate']*100:.1f}%", "Detection Rate"),
    ]):
        col.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem">{icon}</div>
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 Performance", "📈 Calibration", "🗂️ Confusion Matrix"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            dist = stats.get("tumor_type_distribution", {})
            if dist:
                from utils.visualization import tumor_type_pie
                st.plotly_chart(tumor_type_pie(dist), use_container_width=True)
            else:
                st.markdown('<div class="info-box">Analyze scans to populate chart.</div>',
                            unsafe_allow_html=True)
        with c2:
            import pandas as pd
            st.markdown("**Model Performance vs Benchmarks**")
            st.dataframe(pd.DataFrame({
                "Metric":     ["Dice", "IoU", "Sensitivity", "Hausdorff", "Params"],
                "PancrAI":    ["0.8204", "0.7671", "0.8754", "4.74px", "100.5M"],
                "U-Net":      ["0.743",  "0.591",  "0.820",  "—",      "31M"],
                "Attn U-Net": ["0.773",  "0.636",  "0.842",  "—",      "34M"],
                "TransUNet*": ["0.847",  "0.739",  "0.891",  "—",      "105M"],
            }), use_container_width=True, hide_index=True)
            st.caption("*Original paper, different dataset split")

    with tab2:
        from app.services.advanced_analytics import create_calibration_curve
        st.plotly_chart(create_calibration_curve(), use_container_width=True)

    with tab3:
        from app.services.advanced_analytics import create_confusion_matrix
        st.plotly_chart(create_confusion_matrix(), use_container_width=True)

    st.markdown("<br>**System Capabilities (v3.0)**")
    badges = [
        "TransUNet Segmentation", "Ensemble Inference", "8x TTA",
        "MC Dropout Uncertainty", "57 Radiomics Features",
        "TNM Staging", "RECIST Measurement", "CA 19-9 Integration",
        "Differential Diagnosis", "SHAP Importance", "Survival Curves",
        "3D Visualization", "Gemini AI Reports", "Groq Chat",
        "PDF Export", "DICOM Metadata", "Calibration Curve", "Confusion Matrix",
    ]
    st.markdown(" ".join(
        f'<span class="feature-badge">{b}</span>' for b in badges),
        unsafe_allow_html=True)


# ─── New Scan ─────────────────────────────────────────────────────────────────

def page_new_scan():
    st.markdown('<div class="section-header">🔬 New Scan Analysis</div>',
                unsafe_allow_html=True)

    with st.expander("📋 Patient Information", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            patient_name = st.text_input("Patient Name *", placeholder="John Doe")
        with c2:
            patient_age = st.number_input("Age", 0, 120, 55)
        with c3:
            patient_sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        with c4:
            scan_type = st.selectbox("Scan Type", ["CT", "MRI", "PET"])
        symptoms = st.text_area("Symptoms", height=60,
                                placeholder="Epigastric pain, weight loss, jaundice...")

    with st.expander("🧪 Laboratory Values (Optional)"):
        c1, c2, c3 = st.columns(3)
        with c1:
            ca199     = st.number_input("CA 19-9 (U/mL)", min_value=0.0,
                                        max_value=100000.0, value=0.0, step=1.0)
            ca199_val = ca199 if ca199 > 0 else None
        with c2:
            has_jaundice    = st.checkbox("Obstructive Jaundice")
            has_weight_loss = st.checkbox("Unexplained Weight Loss > 5kg")
        with c3:
            has_diabetes = st.checkbox("New-Onset Diabetes")

    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        use_ensemble = st.toggle("🤝 Ensemble Inference (TransUNet + LightUNet)", value=False)
    with col_opt2:
        use_tta = st.toggle("🔁 Test Time Augmentation (8x)", value=False)

    st.markdown("**Upload Medical Scan**")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["dcm", "png", "jpg", "jpeg", "nii"],
    )

    if uploaded_file and uploaded_file.name.lower().endswith(".dcm"):
        _render_dicom_metadata(uploaded_file.read())
        uploaded_file.seek(0)

    if not uploaded_file:
        st.markdown('<div class="info-box">📎 Upload a scan to begin.</div>',
                    unsafe_allow_html=True)
        if st.session_state.get("last_result") and st.session_state.get("last_patient"):
            _render_results_panel(
                st.session_state["last_result"],
                st.session_state["last_patient"],
                st.session_state.get("last_ca199"),
                st.session_state.get("last_symptoms_flags", {}),
            )
        return

    cp, ci = st.columns([1, 2])
    with cp:
        try:
            pil = Image.open(uploaded_file).convert("L")
            st.image(pil, caption="Preview", use_container_width=True)
            uploaded_file.seek(0)
        except Exception:
            st.info("Preview not available.")
            uploaded_file.seek(0)
    with ci:
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:16px">
            <div style="color:#8B949E;font-size:0.78rem;text-transform:uppercase">File Info</div>
            <div style="margin-top:10px">
                <div><span style="color:#8B949E">Name:</span>
                     <span style="color:#E6EDF3;font-family:monospace">{uploaded_file.name}</span></div>
                <div><span style="color:#8B949E">Size:</span>
                     <span style="color:#E6EDF3;font-family:monospace">{uploaded_file.size/1024:.1f} KB</span></div>
                <div><span style="color:#8B949E">Mode:</span>
                     <span style="color:#3FB950">{"Ensemble" if use_ensemble else "TTA 8x" if use_tta else "Standard"}</span></div>
                <div><span style="color:#8B949E">CA 19-9:</span>
                     <span style="color:#E6EDF3">{f"{ca199_val:.0f} U/mL" if ca199_val else "Not provided"}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    bcol1, _ = st.columns([1, 4])
    with bcol1:
        if st.button("🗑️ Clear", key="btn_clear"):
            for k in ["last_result", "last_patient", "last_report",
                      "last_ca199", "last_symptoms_flags"]:
                st.session_state.pop(k, None)
            st.rerun()

    if st.button("🚀 Analyze Scan", type="primary", key="btn_analyze"):
        if not patient_name:
            st.warning("Please enter a patient name.")
            return
        patient_info = {
            "name": patient_name, "age": patient_age,
            "sex": patient_sex, "scan_type": scan_type, "symptoms": symptoms
        }
        symptoms_flags = {
            "has_jaundice":    has_jaundice,
            "has_weight_loss": has_weight_loss,
            "has_diabetes":    has_diabetes,
        }
        file_bytes = uploaded_file.read()
        with st.spinner(f"🔄 Running {'Ensemble' if use_ensemble else 'TTA'} analysis..."):
            result = run_inline_analysis(
                file_bytes, uploaded_file.name,
                patient_info, use_tta=use_tta, use_ensemble=use_ensemble)
        if result:
            st.session_state["last_result"]         = result
            st.session_state["last_patient"]        = patient_info
            st.session_state["last_ca199"]          = ca199_val
            st.session_state["last_symptoms_flags"] = symptoms_flags
            st.session_state.pop("last_report", None)
            st.success("✅ Analysis complete!")

    if st.session_state.get("last_result") and st.session_state.get("last_patient"):
        _render_results_panel(
            st.session_state["last_result"],
            st.session_state["last_patient"],
            st.session_state.get("last_ca199"),
            st.session_state.get("last_symptoms_flags", {}),
        )


def _render_dicom_metadata(file_bytes):
    from app.services.advanced_analytics import extract_dicom_metadata
    meta = extract_dicom_metadata(file_bytes)
    if not meta.get("available"):
        return
    with st.expander("🏥 DICOM Metadata", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Patient**")
            for k, v in meta["patient"].items():
                if v != "N/A":
                    st.markdown(
                        f'<div class="dicom-field">'
                        f'<span class="dicom-label">{k.replace("_"," ").title()}:</span> '
                        f'<span class="dicom-value">{v}</span></div>',
                        unsafe_allow_html=True)
        with c2:
            st.markdown("**Study**")
            for k, v in meta["study"].items():
                if v != "N/A":
                    st.markdown(
                        f'<div class="dicom-field">'
                        f'<span class="dicom-label">{k.replace("_"," ").title()}:</span> '
                        f'<span class="dicom-value">{v}</span></div>',
                        unsafe_allow_html=True)
        with c3:
            st.markdown("**Acquisition**")
            for k, v in meta["acquisition"].items():
                if v != "N/A":
                    st.markdown(
                        f'<div class="dicom-field">'
                        f'<span class="dicom-label">{k.replace("_"," ").title()}:</span> '
                        f'<span class="dicom-value">{v}</span></div>',
                        unsafe_allow_html=True)


def _render_results_panel(result, patient_info, ca199_val=None, symptoms_flags=None):
    st.markdown("---")
    st.markdown('<div class="section-header">📊 Analysis Results</div>',
                unsafe_allow_html=True)

    tumor_class = result.get("tumor_class", "Unknown")
    confidence  = result.get("primary_confidence", 0.0)
    risk        = result.get("risk_level", "Unknown")
    unc         = result.get("uncertainty_score", 0.0)
    dice        = result.get("dice_score", 0.0)
    tc_color    = CLASS_COLORS.get(tumor_class, "#9E9E9E")
    risk_color  = RISK_COLORS.get(risk, "#9E9E9E")
    unc_color   = "#4CAF50" if unc < 30 else "#FF9800" if unc < 60 else "#F44336"

    if result.get("ensemble_enabled"):
        agr = result.get("agreement_score", 0)
        st.markdown(
            f'<div class="success-box">🤝 <strong>Ensemble Inference Active</strong> — '
            f'TransUNet (70%) + LightUNet (30%) | Model Agreement: {agr:.1f}%</div>',
            unsafe_allow_html=True)
    elif result.get("tta_enabled"):
        st.markdown(
            '<div class="success-box">🔁 <strong>TTA Active</strong> — 8x augmented predictions averaged</div>',
            unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Detection Result</div>
            <div style="font-size:1.1rem;font-weight:700;color:{tc_color};margin-top:8px">{tumor_class}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Confidence</div>
            <div class="value">{confidence*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Risk Level</div>
            <div style="font-size:1.3rem;font-weight:700;color:{risk_color};margin-top:8px">{risk}</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Uncertainty</div>
            <div class="value" style="color:{unc_color}">{unc:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    unc_det = result.get("uncertainty_details", {})
    if unc_det.get("high_uncertainty_warning"):
        st.markdown(
            f'<div class="warning-box">⚠️ {unc_det.get("warning_message","High uncertainty.")}</div>',
            unsafe_allow_html=True)

    tabs = st.tabs([
        "🔄 Pipeline", "🧬 Segmentation", "📊 Radiomics & SHAP",
        "🏥 Staging", "🔬 Differential Dx", "📐 RECIST",
        "📈 Survival", "🧪 CA 19-9", "📈 3D View",
    ])

    # ── Tab 0: Pipeline ──
    with tabs[0]:
        st.markdown("**Preprocessing Pipeline**")
        steps = result.get("images", {}).get("preprocessing_steps", [])
        if steps:
            cols = st.columns(min(len(steps), 8))
            for i, (col, step) in enumerate(zip(cols, steps)):
                with col:
                    pil = b64_to_image(step["image_b64"])
                    if pil:
                        st.markdown(
                            f'<div class="scan-step-card">'
                            f'<div class="step-name">{i+1}. {step["name"]}</div>',
                            unsafe_allow_html=True)
                        st.image(pil, use_container_width=True)
                        st.markdown(
                            f'<div class="step-desc">{step["description"]}</div></div>',
                            unsafe_allow_html=True)

    # ── Tab 1: Segmentation ──
    with tabs[1]:
        st.markdown("**Segmentation Overlay**")
        seg_b64 = result.get("images", {}).get("segmentation_overlay")
        if seg_b64:
            pil = b64_to_image(seg_b64)
            if pil:
                st.image(pil, use_container_width=True)
        st.caption("Red = tumor region | Yellow = bounding box | Green = centroid")

        scores   = result.get("confidence_scores", [0.25]*4)
        pred_idx = result.get("tumor_class_index", 0)
        if len(scores) == 4:
            from utils.visualization import confidence_bar_chart
            st.plotly_chart(
                confidence_bar_chart(scores, CLASS_NAMES, pred_idx),
                use_container_width=True)

        measurements = result.get("measurements")
        if measurements:
            st.markdown("**Tumor Measurements**")
            mc = st.columns(4)
            for col, (label, val) in zip(mc, [
                ("Area",         f"{measurements.get('area_cm2',0):.3f} cm²"),
                ("Centroid",     f"({measurements.get('centroid_x',0):.0f}, {measurements.get('centroid_y',0):.0f}) px"),
                ("Bounding Box", f"{measurements.get('bbox_w',0)} x {measurements.get('bbox_h',0)} px"),
                ("Aspect Ratio", f"{measurements.get('aspect_ratio',0):.3f}"),
            ]):
                col.markdown(f"""<div class="metric-card">
                    <div class="label">{label}</div>
                    <div style="font-size:1rem;font-weight:600;color:#58A6FF;
                    margin-top:6px;font-family:monospace">{val}</div>
                </div>""", unsafe_allow_html=True)

        cd, ci2 = st.columns(2)
        with cd:
            st.metric("Dice Score", f"{dice:.4f}" if dice > 0 else "N/A")
        with ci2:
            st.metric("IoU Score", f"{result.get('iou_score',0):.4f}"
                      if result.get('iou_score', 0) > 0 else "N/A")

    # ── Tab 2: Radiomics & SHAP ──
    with tabs[2]:
        radiomics = result.get("radiomics")
        shap      = result.get("shap")
        if radiomics:
            c1, c2 = st.columns(2)
            with c1:
                from app.services.visualization_3d import create_radiomics_radar_chart
                st.plotly_chart(create_radiomics_radar_chart(radiomics),
                                use_container_width=True)
            with c2:
                if shap:
                    from app.services.advanced_analytics import create_shap_waterfall_chart
                    st.plotly_chart(
                        create_shap_waterfall_chart(shap, tumor_class),
                        use_container_width=True)

            interp = result.get("radiomics_interp", {})
            if interp:
                st.markdown("**Clinical Interpretation**")
                for k, v in interp.items():
                    color = "#F44336" if "irregular" in v.lower() or "malignant" in v.lower() \
                            else "#FF9800" if "moderate" in v.lower() \
                            else "#4CAF50"
                    st.markdown(
                        f'<div style="background:#161B22;border-left:3px solid {color};'
                        f'padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0;">'
                        f'<strong style="color:{color}">{k}:</strong> '
                        f'<span style="color:#E6EDF3">{v}</span></div>',
                        unsafe_allow_html=True)

            with st.expander("📋 All 57 Radiomics Features"):
                import pandas as pd
                from app.services.radiomics import get_radiomics_summary
                summary = get_radiomics_summary(radiomics)
                for cat, feats in summary.items():
                    st.markdown(f"**{cat}**")
                    df = pd.DataFrame([{
                        "Feature": k.replace(
                            cat.lower().replace(" ", "_")+"_", ""
                        ).replace("_", " ").title(),
                        "Value": f"{v:.4f}",
                    } for k, v in feats.items()])
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("Radiomics unavailable — requires tumor detection")

    # ── Tab 3: Staging ──
    with tabs[3]:
        staging = result.get("staging")
        if staging:
            ts     = staging.get("t_stage", {})
            os_    = staging.get("overall_stage", {})
            res    = staging.get("resectability", {})
            risk_s = staging.get("risk_score", {})

            from app.services.visualization_3d import create_risk_gauge
            st.plotly_chart(
                create_risk_gauge(risk_s.get("score", 0),
                                  risk_s.get("category", "Unknown"),
                                  risk_s.get("color", "#FF5722")),
                use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                t_color = ts.get("color", "#FF9800")
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:16px">
                    <div style="font-size:2rem;font-weight:700;color:{t_color};text-align:center">{ts.get('t_stage','N/A')}</div>
                    <div style="color:#8B949E;text-align:center;margin-top:4px">{ts.get('description','')}</div>
                    <div style="margin-top:12px">
                        <div><span style="color:#8B949E">Stage:</span> <span style="color:#E6EDF3;font-weight:700">{os_.get('overall_stage','N/A')}</span></div>
                        <div><span style="color:#8B949E">TNM:</span> <span style="color:#E6EDF3">{os_.get('tnm','N/A')}</span></div>
                        <div><span style="color:#8B949E">5-Year Survival:</span> <span style="color:#E6EDF3">{os_.get('five_year_survival','N/A')}</span></div>
                    </div>
                </div>""", unsafe_allow_html=True)
            with col2:
                res_color = res.get("color", "#FF9800")
                st.markdown(f"""
                <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:16px">
                    <div style="font-size:1.1rem;font-weight:700;color:{res_color}">{res.get('status','N/A')}</div>
                    <div style="color:#8B949E;font-size:0.85rem;margin-top:8px">{res.get('description','')}</div>
                    <div style="margin-top:12px;color:#E6EDF3;font-size:0.85rem">
                        <strong>Procedure:</strong><br>{res.get('procedure','MDT review required')}
                    </div>
                </div>""", unsafe_allow_html=True)

            recs = risk_s.get("recommendations", [])
            if recs:
                st.markdown("**Clinical Recommendations**")
                for rec in recs:
                    color = "#F44336" if any(x in rec for x in ["🚨","🔴","⚠"]) \
                            else "#FF9800" if any(x in rec for x in ["⚕️","🔬"]) \
                            else "#4CAF50"
                    st.markdown(
                        f'<div style="background:#161B22;border-left:3px solid {color};'
                        f'padding:8px 12px;margin:4px 0;border-radius:0 4px 4px 0;'
                        f'color:#E6EDF3">{rec}</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Staging unavailable")

    # ── Tab 4: Differential Dx ──
    with tabs[4]:
        differential = result.get("differential", [])
        if differential:
            from app.services.advanced_analytics import create_differential_diagnosis_chart
            st.plotly_chart(
                create_differential_diagnosis_chart(differential),
                use_container_width=True)
            st.markdown("**Detailed Differential Diagnosis**")
            for i, dx in enumerate(differential[:4]):
                prob = dx.get("probability_pct", "N/A")
                with st.expander(f"#{i+1} {dx['diagnosis']} — {prob}", expanded=(i==0)):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**Supporting Features**")
                        for f in dx.get("supporting", []):
                            st.markdown(f"• {f}")
                        st.markdown("**Distinguishing Features**")
                        for f in dx.get("distinguishing", []):
                            st.markdown(f"• {f}")
                    with c2:
                        st.markdown("**Recommended Workup**")
                        for w in dx.get("workup", []):
                            st.markdown(f"• {w}")
                        st.markdown(f"**ICD-10:** `{dx.get('icd10','N/A')}`")
                        st.markdown(f"**Prognosis:** {dx.get('prognosis','N/A')}")
        else:
            st.info("No tumor detected — differential diagnosis not applicable")

    # ── Tab 5: RECIST ──
    with tabs[5]:
        recist = result.get("recist")
        if recist and recist.get("available"):
            st.markdown("**RECIST 1.1 Measurements**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Longest Diameter",
                      f"{recist['longest_diameter_mm']} mm",
                      f"({recist['longest_diameter_cm']} cm)")
            c2.metric("Shortest Diameter",
                      f"{recist['shortest_diameter_mm']} mm")
            c3.metric("Estimated Volume",
                      f"{recist['estimated_volume_cm3']} cm³")
            st.markdown(f"""
            <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:16px;margin-top:12px">
                <div><span style="color:#8B949E">Lesion Type:</span>
                     <span style="color:#E6EDF3;font-weight:700">{recist['lesion_type']}</span></div>
                <div><span style="color:#8B949E">RECIST Eligible:</span>
                     <span style="color:{'#4CAF50' if recist['recist_eligible'] else '#F44336'}">
                     {"✅ Yes" if recist['recist_eligible'] else "❌ No (< 10mm)"}</span></div>
                <div><span style="color:#8B949E">Standard:</span>
                     <span style="color:#E6EDF3">{recist['measurement_standard']}</span></div>
                <div><span style="color:#8B949E">Sphericity:</span>
                     <span style="color:#E6EDF3">{recist['sphericity']:.3f}</span></div>
                <div style="margin-top:8px;color:#8B949E;font-size:0.8rem">{recist['note']}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("RECIST measurements require tumor detection")

    # ── Tab 6: Survival ──
    with tabs[6]:
        staging = result.get("staging")
        if staging:
            from app.services.advanced_analytics import create_survival_curve
            stage = staging.get("overall_stage", {}).get("overall_stage", "Stage IIA")
            fig   = create_survival_curve(tumor_class, stage, patient_info.get("age"))
            st.plotly_chart(fig, use_container_width=True)
            surv = staging.get("overall_stage", {}).get("five_year_survival", "N/A")
            prog = staging.get("overall_stage", {}).get("prognosis", "N/A")
            st.markdown(
                f'<div class="info-box">📊 <strong>Estimated 5-Year Survival:</strong> {surv} — {prog}</div>',
                unsafe_allow_html=True)
        else:
            st.info("Survival curve requires staging assessment")

    # ── Tab 7: CA 19-9 ──
    with tabs[7]:
        staging = result.get("staging")
        if ca199_val is not None and staging:
            from app.services.advanced_analytics import calculate_combined_risk_score
            sflag    = symptoms_flags or {}
            combined = calculate_combined_risk_score(
                ca199_value=ca199_val,
                imaging_risk_score=staging.get("risk_score", {}).get("score", 50),
                tumor_class=tumor_class,
                confidence=confidence,
                measurements=result.get("measurements"),
                patient_age=patient_info.get("age"),
                patient_sex=patient_info.get("sex"),
                has_jaundice=sflag.get("has_jaundice", False),
                has_weight_loss=sflag.get("has_weight_loss", False),
                has_diabetes_new=sflag.get("has_diabetes", False),
            )
            score_color = combined["color"]
            st.markdown(f"""
            <div style="background:#161B22;border:2px solid {score_color};
                border-radius:12px;padding:20px;text-align:center;margin-bottom:16px">
                <div style="font-size:3rem;font-weight:700;color:{score_color}">
                    {combined['total_score']}/100</div>
                <div style="font-size:1.2rem;color:{score_color};margin-top:4px">
                    {combined['category']}</div>
                <div style="color:#8B949E;margin-top:8px">
                    {combined['recommended_action']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("**Score Breakdown**")
            for comp, data in combined["components"].items():
                if isinstance(data, dict):
                    score  = data.get("score", 0)
                    max_s  = data.get("max", 40)
                    interp = data.get("interpretation", "")
                    pct    = score / max_s if max_s > 0 else 0
                    color  = "#F44336" if pct > 0.7 else "#FF9800" if pct > 0.4 else "#4CAF50"
                    st.markdown(f"""
                    <div style="background:#161B22;border:1px solid #21262D;
                        border-radius:8px;padding:12px;margin:6px 0">
                        <div style="display:flex;justify-content:space-between">
                            <span style="color:#E6EDF3;font-weight:600">{comp}</span>
                            <span style="color:{color};font-weight:700">{score:.1f}/{max_s}</span>
                        </div>
                        <div style="color:#8B949E;font-size:0.82rem;margin-top:4px">{interp}</div>
                    </div>""", unsafe_allow_html=True)
            if combined.get("ca199_clinical_note"):
                st.markdown(
                    f'<div class="warning-box">{combined["ca199_clinical_note"]}</div>',
                    unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="info-box">Enter CA 19-9 value in Lab Values above '
                'and analyze a scan to see combined risk assessment.</div>',
                unsafe_allow_html=True)

    # ── Tab 8: 3D View ──
    with tabs[8]:
        measurements  = result.get("measurements")
        has_tumor     = (measurements is not None and
                         measurements.get("area_cm2", 0) > 0.01)
        if has_tumor:
            import cv2
            from app.services.visualization_3d import create_3d_tumor_surface
            h, w      = 224, 224
            fake_mask = np.zeros((h, w), dtype=np.uint8)
            cx = int(measurements.get("centroid_x", 112))
            cy = int(measurements.get("centroid_y", 112))
            bw = max(int(measurements.get("bbox_w", 20)), 5)
            bh = max(int(measurements.get("bbox_h", 20)), 5)
            cx = max(bw//2+1, min(w-bw//2-1, cx))
            cy = max(bh//2+1, min(h-bh//2-1, cy))
            cv2.ellipse(fake_mask, (cx, cy), (bw//2, bh//2), 0, 0, 360, 255, -1)
            fig = create_3d_tumor_surface(
                fake_mask,
                title=f"3D Tumor — {tumor_class} | {measurements.get('area_cm2',0):.3f} cm²",
                color="#FF4444",
            )
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Est. Diameter",
                      f"{2*(measurements.get('area_cm2',0)/3.14159)**0.5:.2f} cm")
            c2.metric("Cross-section Area",
                      f"{measurements.get('area_cm2',0):.3f} cm²")
            c3.metric("Shape",
                      "Round"    if measurements.get("aspect_ratio", 1) < 1.3
                      else "Oval" if measurements.get("aspect_ratio", 1) < 2.0
                      else "Elongated")
            st.caption("3D ellipsoid approximation. Upload NIfTI for true volumetric rendering.")
        else:
            st.info("No tumor detected — 3D visualization requires segmentation output")
            st.markdown(
                '<div class="info-box">Upload a real CT scan slice with tumor '
                'to see 3D visualization.</div>',
                unsafe_allow_html=True)

    # ── Report ──
    st.markdown("<br>---")
    st.markdown("**📋 AI Diagnostic Report**")
    if not st.session_state.get("last_report"):
        st.markdown(
            '<div class="info-box">Generate a full structured clinical report with Gemini AI.</div>',
            unsafe_allow_html=True)
        if st.button("📝 Generate AI Diagnostic Report", key="btn_report"):
            with st.spinner("🤖 Generating..."):
                ok = generate_report_and_store(result, patient_info)
            if ok:
                st.success("✅ Report ready!")
                st.rerun()
    if st.session_state.get("last_report"):
        render_report(patient_info)


# ─── AI Chat ──────────────────────────────────────────────────────────────────

def page_chat():
    st.markdown('<div class="section-header">💬 AI Clinical Assistant</div>',
                unsafe_allow_html=True)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    result  = st.session_state.get("last_result")
    patient = st.session_state.get("last_patient")

    if result:
        tc       = result.get("tumor_class", "Unknown")
        tc_color = CLASS_COLORS.get(tc, "#9E9E9E")
        conf     = result.get("primary_confidence", 0.0)
        risk     = result.get("risk_level", "Unknown")
        staging  = result.get("staging", {})
        stage    = staging.get("overall_stage", {}).get("overall_stage", "N/A") if staging else "N/A"
        name     = patient.get("name", "N/A") if patient else "N/A"
        st.markdown(f"""<div class="info-box">
            🔬 <span style="color:{tc_color};font-weight:700">{tc}</span> |
            Confidence: <strong>{conf*100:.1f}%</strong> |
            Risk: <strong>{risk}</strong> |
            Stage: <strong>{stage}</strong> |
            Patient: <strong>{name}</strong>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="warning-box">⚠️ Analyze a scan first for context-aware responses.</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div style="text-align:center;color:#8B949E;padding:40px 0">
            <div style="font-size:2rem">🤖</div>
            <div style="margin-top:8px">Ask anything about findings, staging, treatment, or guidelines.</div>
            <div style="margin-top:8px;font-size:0.85rem">
                "Explain Stage IIA PDAC" · "What is FOLFIRINOX?" ·
                "Should I order a PET-CT?" · "What does CA 19-9 elevation mean?"
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            css  = "chat-msg-user" if msg["role"] == "user" else "chat-msg-assistant"
            icon = "👤 <strong>You</strong>" if msg["role"] == "user" \
                   else "🤖 <strong>PancrAI</strong>"
            st.markdown(f'<div class="{css}">{icon}<br>{msg["content"]}</div>',
                        unsafe_allow_html=True)

    user_input = st.chat_input("Ask a clinical question...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ctx = None
        if result:
            staging = result.get("staging", {})
            ctx = {
                "tumor_class":        result.get("tumor_class"),
                "primary_confidence": result.get("primary_confidence"),
                "risk_level":         result.get("risk_level"),
                "uncertainty_score":  result.get("uncertainty_score"),
                "measurements":       result.get("measurements"),
                "staging":            staging,
                "differential":       result.get("differential", [])[:2],
                "recist":             result.get("recist"),
                "patient":            patient,
            }
        with st.spinner("Thinking..."):
            from app.services.groq_chat import chat as groq_chat
            response = groq_chat(message=user_input,
                                  history=st.session_state.chat_history[:-1],
                                  prediction_context=ctx)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─── Comparison ───────────────────────────────────────────────────────────────

def page_comparison():
    st.markdown('<div class="section-header">📈 Longitudinal Comparison</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="info-box">Upload two scans from different time points.</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scan 1 (Baseline)**")
        file1 = st.file_uploader("Upload Scan 1",
                                  type=["dcm","png","jpg","jpeg","nii"],
                                  key="cmp_file1")
        date1 = st.date_input("Scan 1 Date", key="cmp_date1")
    with col2:
        st.markdown("**Scan 2 (Follow-up)**")
        file2 = st.file_uploader("Upload Scan 2",
                                  type=["dcm","png","jpg","jpeg","nii"],
                                  key="cmp_file2")
        date2 = st.date_input("Scan 2 Date", key="cmp_date2")

    if not file1 or not file2:
        return

    if st.button("🔍 Compare Scans", type="primary"):
        with st.spinner("Analyzing both scans..."):
            from app.services.preprocessing import load_from_bytes
            from app.services.segmentation import run_segmentation
            from app.models.classifier import classify_from_mask
            from app.services.staging import generate_staging_report
            from app.services.advanced_analytics import calculate_recist

            if "seg_model" not in st.session_state:
                from app.models.transunet import build_transunet
                from app.models.classifier import build_classifier
                st.session_state.seg_model = build_transunet()
                st.session_state.cls_model = build_classifier()

            def analyze(fb, fn):
                img  = load_from_bytes(fb, fn)
                seg  = run_segmentation(img, st.session_state.seg_model)
                mask = seg.get("mask")
                cls  = classify_from_mask(mask) if mask is not None \
                       else classify_from_mask(np.zeros((224, 224)))
                stg  = generate_staging_report(
                    cls["class_name"], cls["confidence"],
                    seg.get("measurements"), 0.0)
                rec  = calculate_recist(seg.get("measurements", {}))
                return {"seg": seg, "class": cls["class_name"],
                        "confidence": cls["confidence"], "risk": cls["risk_level"],
                        "mask": mask, "staging": stg, "recist": rec}

            r1 = analyze(file1.read(), file1.name)
            r2 = analyze(file2.read(), file2.name)

        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"**Scan 1 — {date1}**")
            img1 = b64_to_image(r1["seg"]["overlay_b64"])
            if img1:
                st.image(img1, use_container_width=True)
            tc1_color = CLASS_COLORS.get(r1["class"], "#9E9E9E")
            st.markdown(
                f'<span style="color:{tc1_color};font-weight:700">{r1["class"]}</span>'
                f' — {r1["confidence"]*100:.1f}%',
                unsafe_allow_html=True)
            if r1.get("staging"):
                st.caption(f"Stage: {r1['staging'].get('overall_stage',{}).get('overall_stage','N/A')}")
            if r1.get("recist", {}).get("available"):
                st.caption(f"LD: {r1['recist']['longest_diameter_mm']}mm")
        with c2:
            st.markdown(f"**Scan 2 — {date2}**")
            img2 = b64_to_image(r2["seg"]["overlay_b64"])
            if img2:
                st.image(img2, use_container_width=True)
            tc2_color = CLASS_COLORS.get(r2["class"], "#9E9E9E")
            st.markdown(
                f'<span style="color:{tc2_color};font-weight:700">{r2["class"]}</span>'
                f' — {r2["confidence"]*100:.1f}%',
                unsafe_allow_html=True)
            if r2.get("staging"):
                st.caption(f"Stage: {r2['staging'].get('overall_stage',{}).get('overall_stage','N/A')}")
            if r2.get("recist", {}).get("available"):
                st.caption(f"LD: {r2['recist']['longest_diameter_mm']}mm")
        with c3:
            st.markdown("**Difference Map**")
            m1, m2 = r1.get("mask"), r2.get("mask")
            if m1 is not None and m2 is not None:
                from utils.image_utils import create_diff_map
                st.image(create_diff_map(m1, m2), use_container_width=True)
                st.caption("🟢 New | 🔴 Resolved | 🔵 Stable")

        a1    = r1["seg"].get("measurements") or {}
        a2    = r2["seg"].get("measurements") or {}
        area1 = a1.get("area_cm2", 0.0)
        area2 = a2.get("area_cm2", 0.0)
        ld1   = r1["recist"].get("longest_diameter_mm", 0) \
                if r1.get("recist", {}).get("available") else 0
        ld2   = r2["recist"].get("longest_diameter_mm", 0) \
                if r2.get("recist", {}).get("available") else 0
        change    = ((area2 - area1) / max(area1, 0.001)) * 100
        ld_change = ((ld2 - ld1) / max(ld1, 0.001)) * 100 if ld1 > 0 else 0

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Area Scan 1",      f"{area1:.3f} cm²")
        mc2.metric("Area Scan 2",      f"{area2:.3f} cm²",
                   delta=f"{change:+.1f}%", delta_color="inverse")
        mc3.metric("RECIST LD Scan 1", f"{ld1:.1f} mm")
        mc4.metric("RECIST LD Scan 2", f"{ld2:.1f} mm",
                   delta=f"{ld_change:+.1f}%", delta_color="inverse")

        from utils.visualization import tumor_growth_chart
        st.plotly_chart(
            tumor_growth_chart([str(date1), str(date2)], [area1, area2]),
            use_container_width=True)

        direction = "grown" if change > 5 else "shrunk" if change < -5 else "stable"
        box = "warning-box" if change > 5 else "success-box" if change < -5 else "info-box"
        st.markdown(
            f'<div class="{box}">📊 Tumor has <strong>{direction}</strong>: '
            f'{area1:.3f}→{area2:.3f} cm² ({change:+.1f}%) | '
            f'RECIST: {ld1:.1f}→{ld2:.1f} mm ({ld_change:+.1f}%) | '
            f'Stage: {r1["class"]} → {r2["class"]}</div>',
            unsafe_allow_html=True)


# ─── Patient Records ──────────────────────────────────────────────────────────

def page_records():
    st.markdown('<div class="section-header">👥 Patient Records</div>',
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔍 Search", "➕ New Patient"])

    with tab1:
        search   = st.text_input("Search by name", placeholder="Enter name...")
        patients = api_get("/patients", search=search, limit=100) or []
        if not patients:
            st.markdown('<div class="info-box">No patients found.</div>',
                        unsafe_allow_html=True)
            return
        import pandas as pd
        df = pd.DataFrame([{
            "ID": p["id"], "Name": p["name"],
            "Age": p.get("age", "N/A"), "Sex": p.get("sex", "N/A"),
            "Created": p["created_at"][:10]
        } for p in patients])
        sel = st.dataframe(df, use_container_width=True, hide_index=True,
                           on_select="rerun", selection_mode="single-row")
        if sel and sel.selection.rows:
            pid = df.iloc[sel.selection.rows[0]]["ID"]
            p   = api_get(f"/patients/{pid}")
            if p:
                st.markdown(f"---\n**{p['name']}** (ID: {p['id']})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Age", p.get("age", "N/A"))
                c2.metric("Sex", p.get("sex", "N/A"))
                c3.metric("Registered", p["created_at"][:10])
                if p.get("medical_history"):
                    st.markdown(f"**History:** {p['medical_history']}")

    with tab2:
        with st.form("new_patient_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name *")
                age  = st.number_input("Age", 0, 120, 50)
            with c2:
                sex     = st.selectbox("Sex", ["Male", "Female", "Other"])
                contact = st.text_input("Contact")
            history   = st.text_area("Medical History")
            submitted = st.form_submit_button("Create Patient")
        if submitted:
            if not name:
                st.error("Name required.")
            else:
                res = api_post("/patients", json_data={
                    "name": name, "age": age, "sex": sex,
                    "contact": contact, "medical_history": history})
                if res:
                    st.success(f"✅ Patient '{name}' created — ID: {res['id']}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown('<div class="pancrai-logo">Pancr<span>AI</span></div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#8B949E;font-size:0.75rem;margin-bottom:20px">'
            'Intelligent Pancreatic Tumor Detection v3.0</div>',
            unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio("Navigation",
                        ["🏥 Dashboard", "🔬 New Scan", "💬 AI Chat",
                         "📈 Comparison", "👥 Patient Records"],
                        label_visibility="collapsed")
        st.markdown("---")
        st.markdown(f"""<div style="font-size:0.75rem;color:#8B949E">
            <div>API: {"🟢 Online" if _check_api() else "🔴 Offline"}</div>
            <div>Gemini: {"✅" if os.getenv("GEMINI_API_KEY") else "⚠️ Not set"}</div>
            <div>Groq: {"✅" if os.getenv("GROQ_API_KEY") else "⚠️ Not set"}</div>
        </div>
        <div style="margin-top:10px;font-size:0.72rem;color:#3FB950">
            ✅ Ensemble (TransUNet+LightUNet)<br>
            ✅ TTA 8x augmentations<br>
            ✅ 57 Radiomics features<br>
            ✅ SHAP feature importance<br>
            ✅ TNM + RECIST staging<br>
            ✅ Differential diagnosis<br>
            ✅ CA 19-9 integration<br>
            ✅ Survival curves<br>
            ✅ 3D tumor visualization<br>
            ✅ Calibration curve<br>
            ✅ Confusion matrix<br>
            ✅ DICOM metadata<br>
            ✅ PDF + HTML export<br>
            ✅ Gemini AI reports<br>
            ✅ Groq clinical chat<br>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#4A4F58;text-align:center">'
            'PancrAI v3.0 | For research use only</div>',
            unsafe_allow_html=True)
        return page


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    from dotenv import load_dotenv
    load_dotenv()
    page = sidebar()
    {
        "🏥 Dashboard":       page_dashboard,
        "🔬 New Scan":        page_new_scan,
        "💬 AI Chat":         page_chat,
        "📈 Comparison":      page_comparison,
        "👥 Patient Records": page_records,
    }.get(page, page_dashboard)()


if __name__ == "__main__":
    main()
