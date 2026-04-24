"""
PancrAI — Streamlit Frontend
Complete clinical UI for pancreatic tumor detection and analysis.
"""

import sys
import os
import io
import json
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
CLASS_NAMES = ["No Tumor", "Benign", "Malignant (PDAC)", "Cystic (IPMN)"]
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
    .stButton > button { background: #238636; color: white; border: none; border-radius: 6px; font-weight: 600; }
    .stButton > button:hover { background: #2EA043; }
    .pancrai-logo { font-size: 1.6rem; font-weight: 700; color: #58A6FF; }
    .pancrai-logo span { color: #3FB950; }
    </style>
    """, unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

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
        r = requests.get(f"{API_BASE.replace('/api/v1', '')}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


# ─── Inline Analysis ──────────────────────────────────────────────────────────

def run_inline_analysis(file_bytes, filename, patient_info):
    try:
        from app.services.preprocessing import (
            load_from_bytes, run_full_pipeline, preprocess_to_tensor
        )
        from app.services.segmentation import run_segmentation
        from app.models.transunet import build_transunet
        from app.models.classifier import build_classifier, classify_from_mask

        image = load_from_bytes(file_bytes, filename)
        preprocess_steps = run_full_pipeline(image)

        # Load models once and cache in session state
        if "seg_model" not in st.session_state:
            seg_weights = os.getenv("MODEL_WEIGHTS_PATH", "./weights/transunet_best.pth")
            with st.spinner("Loading AI models..."):
                st.session_state.seg_model = build_transunet(weights_path=seg_weights)
                st.session_state.cls_model = build_classifier()

        # Run segmentation
        seg_result = run_segmentation(image, st.session_state.seg_model)

        # ── Rule-based classification from real segmentation mask ──
        seg_mask = seg_result.get("mask")
        if seg_mask is not None:
            cls_result = classify_from_mask(seg_mask)
        else:
            cls_result = classify_from_mask(np.zeros((224, 224), dtype=np.float32))

        pred_idx    = cls_result["class_idx"]
        tumor_class = cls_result["class_name"]
        confidence  = cls_result["confidence"]
        probs       = cls_result["confidence_scores"]
        risk        = cls_result["risk_level"]

        # Tensor for Grad-CAM and uncertainty only
        tensor = preprocess_to_tensor(image)

        # Grad-CAM
        gradcam_b64 = None
        try:
            from app.services.gradcam import generate_gradcam_results
            gcam = generate_gradcam_results(
                image, st.session_state.seg_model, seg_result.get("measurements")
            )
            gradcam_b64 = gcam.get("gradcam_b64")
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")

        # Uncertainty
        unc_score  = 0.0
        unc_b64    = None
        unc_details = {}
        try:
            from app.services.uncertainty import mc_dropout_inference
            unc = mc_dropout_inference(st.session_state.seg_model, tensor.clone(), T=20)
            unc_score   = unc["uncertainty_score"]
            unc_b64     = unc["uncertainty_heatmap_b64"]
            unc_details = unc
        except Exception as e:
            st.warning(f"Uncertainty unavailable: {e}")

        return {
            "tumor_detected":     pred_idx > 0,
            "tumor_class":        tumor_class,
            "tumor_class_index":  pred_idx,
            "confidence_scores":  probs,
            "primary_confidence": confidence,
            "risk_level":         risk,
            "dice_score":         seg_result["dice_score"],
            "iou_score":          seg_result["iou_score"],
            "uncertainty_score":  unc_score,
            "uncertainty_details": unc_details,
            "measurements":       seg_result.get("measurements"),
            "images": {
                "preprocessing_steps":  preprocess_steps,
                "segmentation_overlay": seg_result["overlay_b64"],
                "gradcam":              gradcam_b64,
                "uncertainty_heatmap":  unc_b64,
            },
        }

    except Exception as e:
        st.error(f"Analysis failed: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None


# ─── Report Generation ────────────────────────────────────────────────────────

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
            confidence_scores=result.get("confidence_scores", [0.25] * 4),
            measurements=result.get("measurements"),
            uncertainty_score=result.get("uncertainty_score", 0.0),
            patient_name=patient_info.get("name", "Unknown"),
            patient_age=patient_info.get("age"),
            patient_sex=patient_info.get("sex"),
            symptoms=patient_info.get("symptoms"),
            scan_type=patient_info.get("scan_type", "CT"),
            risk_level=result.get("risk_level", "Unknown"),
            segmented_image_b64=result.get("images", {}).get("segmentation_overlay"),
            gradcam_image_b64=result.get("images", {}).get("gradcam"),
        ))
        st.session_state["last_report"] = report
        return True

    except Exception as e:
        st.error(f"Report generation failed: {e}")
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
        st.markdown(
            f'<div class="info-box">📌 <strong>Summary:</strong> {summary}</div>',
            unsafe_allow_html=True,
        )

    report_html = report.get("report_html", "")
    if report_html:
        st.components.v1.html(report_html, height=800, scrolling=True)
        st.download_button(
            label="⬇️ Download Report (HTML)",
            data=report_html.encode("utf-8"),
            file_name=f"PancrAI_Report_{patient_info.get('name', 'patient')}_{datetime.now().strftime('%Y%m%d_%H%M')}.html",
            mime="text/html",
            key="dl_report_btn",
        )

    if st.button("🔄 Regenerate Report", key="btn_regen_report"):
        del st.session_state["last_report"]
        st.rerun()


# ─── Page 1: Dashboard ────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown('<div class="section-header">🏥 Clinical Dashboard</div>',
                unsafe_allow_html=True)

    stats = api_get("/dashboard/stats") or {
        "total_patients": 0, "total_scans": 0, "scans_today": 0,
        "avg_confidence": 0, "detection_rate": 0, "tumor_type_distribution": {}
    }

    cols = st.columns(5)
    for col, (icon, val, label) in zip(cols, [
        ("👤", stats["total_patients"],              "Total Patients"),
        ("🔬", stats["total_scans"],                 "Total Scans"),
        ("📅", stats["scans_today"],                 "Scans Today"),
        ("🎯", f"{stats['avg_confidence']*100:.1f}%","Avg Confidence"),
        ("📊", f"{stats['detection_rate']*100:.1f}%","Detection Rate"),
    ]):
        col.markdown(f"""<div class="metric-card">
            <div style="font-size:1.8rem">{icon}</div>
            <div class="value">{val}</div>
            <div class="label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        dist = stats.get("tumor_type_distribution", {})
        if dist:
            from utils.visualization import tumor_type_pie
            st.plotly_chart(tumor_type_pie(dist), use_container_width=True)
        else:
            st.markdown(
                '<div class="info-box">No scan data yet. Analyze a scan to see statistics.</div>',
                unsafe_allow_html=True)

    with col2:
        st.markdown("**Model Performance (Reference)**")
        import pandas as pd
        st.dataframe(pd.DataFrame({
            "Metric":    ["Dice Score", "IoU",   "Sensitivity", "Specificity"],
            "TransUNet": ["0.820",      "0.769", "0.876",       "0.998"],
            "Benchmark": ["0.780",      "0.670", "0.820",       "0.945"],
        }), use_container_width=True, hide_index=True)

    st.markdown("**Recent Patients**")
    patients = api_get("/patients", limit=5) or []
    if patients:
        import pandas as pd
        st.dataframe(pd.DataFrame([
            {"Patient": p["name"], "Age": p.get("age", "N/A"),
             "Date": p["created_at"][:10]}
            for p in patients
        ]), use_container_width=True, hide_index=True)
    else:
        st.markdown(
            '<div class="info-box">No patients yet. Create one in Patient Records.</div>',
            unsafe_allow_html=True)


# ─── Page 2: New Scan Analysis ────────────────────────────────────────────────

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
        symptoms = st.text_area(
            "Symptoms / Clinical Notes",
            placeholder="e.g., epigastric pain, weight loss...", height=80)

    st.markdown("**Upload Medical Scan**")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["dcm", "png", "jpg", "jpeg", "nii"],
        help="Supported: DICOM, PNG, JPEG, NIfTI",
    )

    if not uploaded_file:
        st.markdown(
            '<div class="info-box">📎 Upload a scan to begin analysis.</div>',
            unsafe_allow_html=True)
        if st.session_state.get("last_result") and st.session_state.get("last_patient"):
            _render_results_panel(
                st.session_state["last_result"],
                st.session_state["last_patient"]
            )
        return

    # File preview
    cp, ci = st.columns([1, 2])
    with cp:
        try:
            pil_img = Image.open(uploaded_file).convert("L")
            st.image(pil_img, caption="Preview", use_container_width=True)
            uploaded_file.seek(0)
        except Exception:
            st.info("Preview not available for this format.")
            uploaded_file.seek(0)
    with ci:
        st.markdown(f"""
        <div style="background:#161B22;border:1px solid #21262D;border-radius:8px;padding:16px">
            <div style="color:#8B949E;font-size:0.78rem;text-transform:uppercase">File Info</div>
            <div style="margin-top:10px">
                <div><span style="color:#8B949E">Name:</span>
                     <span style="color:#E6EDF3;font-family:monospace">{uploaded_file.name}</span></div>
                <div style="margin-top:4px"><span style="color:#8B949E">Size:</span>
                     <span style="color:#E6EDF3;font-family:monospace">{uploaded_file.size/1024:.1f} KB</span></div>
                <div style="margin-top:4px"><span style="color:#8B949E">Type:</span>
                     <span style="color:#E6EDF3">{scan_type}</span></div>
                <div style="margin-top:4px"><span style="color:#8B949E">Patient:</span>
                     <span style="color:#E6EDF3">{patient_name or 'Not specified'}</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Clear button ──
    if st.button("🗑️ Clear Previous Results", key="btn_clear"):
        st.session_state.pop("last_result", None)
        st.session_state.pop("last_patient", None)
        st.session_state.pop("last_report", None)
        st.rerun()

    # ── Analyze button ──
    if st.button("🚀 Analyze Scan", type="primary", key="btn_analyze"):
        if not patient_name:
            st.warning("Please enter a patient name.")
            return
        patient_info = {
            "name": patient_name, "age": patient_age,
            "sex": patient_sex, "scan_type": scan_type, "symptoms": symptoms
        }
        file_bytes = uploaded_file.read()
        with st.spinner("🔄 Running AI analysis pipeline..."):
            result = run_inline_analysis(file_bytes, uploaded_file.name, patient_info)
        if result:
            st.session_state["last_result"]  = result
            st.session_state["last_patient"] = patient_info
            st.session_state.pop("last_report", None)
            st.success("✅ Analysis complete! Results shown below.")

    # ── Always render results from session state ──
    if st.session_state.get("last_result") and st.session_state.get("last_patient"):
        _render_results_panel(
            st.session_state["last_result"],
            st.session_state["last_patient"]
        )


# ─── Results Panel ────────────────────────────────────────────────────────────

def _render_results_panel(result, patient_info):
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

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Detection Result</div>
            <div style="font-size:1.1rem;font-weight:700;color:{tc_color};margin-top:8px">
                {tumor_class}</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Confidence</div>
            <div class="value">{confidence*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="label">Risk Level</div>
            <div style="font-size:1.3rem;font-weight:700;color:{risk_color};margin-top:8px">
                {risk}</div>
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

    # Preprocessing pipeline
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**🔄 Preprocessing Pipeline**")
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

    # Segmentation + Grad-CAM
    st.markdown("<br>", unsafe_allow_html=True)
    cs, cg = st.columns(2)
    with cs:
        st.markdown("**Segmentation Overlay**")
        seg_b64 = result.get("images", {}).get("segmentation_overlay")
        if seg_b64:
            pil = b64_to_image(seg_b64)
            if pil:
                st.image(pil, use_container_width=True)
        st.caption("Red = tumor | Yellow = bounding box | Green = centroid")
    with cg:
        st.markdown("**Grad-CAM Explainability**")
        gc_b64 = result.get("images", {}).get("gradcam")
        if gc_b64:
            pil = b64_to_image(gc_b64)
            if pil:
                st.image(pil, use_container_width=True)
            st.caption("Blue = low attention | Red = high attention")
        else:
            st.info("Grad-CAM not available")

    # Confidence chart
    scores   = result.get("confidence_scores", [0.25] * 4)
    pred_idx = result.get("tumor_class_index", 0)
    if len(scores) == 4:
        from utils.visualization import confidence_bar_chart
        st.plotly_chart(
            confidence_bar_chart(scores, CLASS_NAMES, pred_idx),
            use_container_width=True)

    # Measurements
    measurements = result.get("measurements")
    if measurements:
        st.markdown("**Tumor Measurements**")
        mc = st.columns(4)
        for col, (label, val) in zip(mc, [
            ("Area",         f"{measurements['area_cm2']:.3f} cm²"),
            ("Centroid",     f"({measurements['centroid_x']:.0f}, {measurements['centroid_y']:.0f}) px"),
            ("Bounding Box", f"{measurements['bbox_w']} x {measurements['bbox_h']} px"),
            ("Aspect Ratio", f"{measurements['aspect_ratio']:.3f}"),
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
        st.metric("IoU Score",  f"{result.get('iou_score', 0.0):.4f}"
                  if result.get("iou_score", 0.0) > 0 else "N/A")

    # Report section
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**📋 AI Diagnostic Report**")

    if not st.session_state.get("last_report"):
        st.markdown(
            '<div class="info-box">Click below to generate a full clinical report using Gemini AI.</div>',
            unsafe_allow_html=True)
        if st.button("📝 Generate AI Diagnostic Report", key="btn_gen_report"):
            with st.spinner("🤖 Generating report... 10-20 seconds"):
                success = generate_report_and_store(result, patient_info)
            if success:
                st.success("✅ Report generated!")
                st.rerun()

    if st.session_state.get("last_report"):
        render_report(patient_info)


# ─── Page 3: AI Chat ──────────────────────────────────────────────────────────

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
        name     = patient.get("name", "N/A") if patient else "N/A"
        st.markdown(f"""<div class="info-box">
            🔬 <strong>Scan context loaded:</strong>
            Detection: <span style="color:{tc_color};font-weight:700">{tc}</span> |
            Confidence: <strong>{conf*100:.1f}%</strong> |
            Risk: <strong>{risk}</strong> |
            Patient: <strong>{name}</strong>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="warning-box">⚠️ No scan loaded. Analyze a scan first.</div>',
            unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""<div style="text-align:center;color:#8B949E;padding:40px 0;">
            <div style="font-size:2rem">🤖</div>
            <div style="margin-top:8px">Ask me anything about pancreatic tumors,
                treatment options, or clinical guidelines.</div>
            <div style="margin-top:8px;font-size:0.85rem">
                Try: "What does this result mean?" ·
                "What are the treatment options?" ·
                "What is the survival rate?"</div>
        </div>""", unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-msg-user">👤 <strong>You</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="chat-msg-assistant">🤖 <strong>PancrAI Assistant</strong>'
                    f'<br>{msg["content"]}</div>',
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    user_input = st.chat_input("Ask a clinical question...")

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        ctx = None
        if result:
            ctx = {
                "tumor_class":        result.get("tumor_class"),
                "primary_confidence": result.get("primary_confidence"),
                "risk_level":         result.get("risk_level"),
                "uncertainty_score":  result.get("uncertainty_score"),
                "dice_score":         result.get("dice_score"),
                "iou_score":          result.get("iou_score"),
                "confidence_scores":  result.get("confidence_scores"),
                "measurements":       result.get("measurements"),
                "patient":            patient,
            }
        with st.spinner("Thinking..."):
            from app.services.groq_chat import chat as groq_chat
            response = groq_chat(
                message=user_input,
                history=st.session_state.chat_history[:-1],
                prediction_context=ctx,
            )
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()


# ─── Page 4: Longitudinal Comparison ─────────────────────────────────────────

def page_comparison():
    st.markdown('<div class="section-header">📈 Longitudinal Comparison</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="info-box">Upload two scans from different time points '
        'to analyze tumor progression.</div>',
        unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Scan 1 (Earlier / Baseline)**")
        file1 = st.file_uploader("Upload Scan 1",
                                  type=["dcm","png","jpg","jpeg","nii"],
                                  key="cmp_file1")
        date1 = st.date_input("Scan 1 Date", key="cmp_date1")
    with col2:
        st.markdown("**Scan 2 (Later / Follow-up)**")
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

            if "seg_model" not in st.session_state:
                from app.models.transunet import build_transunet
                from app.models.classifier import build_classifier
                st.session_state.seg_model = build_transunet()
                st.session_state.cls_model = build_classifier()

            def analyze(fb, fn):
                img = load_from_bytes(fb, fn)
                seg = run_segmentation(img, st.session_state.seg_model)
                seg_mask = seg.get("mask")
                if seg_mask is not None:
                    cls_result = classify_from_mask(seg_mask)
                else:
                    cls_result = classify_from_mask(
                        np.zeros((224, 224), dtype=np.float32))
                return {
                    "seg":        seg,
                    "class":      cls_result["class_name"],
                    "confidence": cls_result["confidence"],
                    "risk":       cls_result["risk_level"],
                    "mask":       seg_mask,
                }

            r1 = analyze(file1.read(), file1.name)
            r2 = analyze(file2.read(), file2.name)

        st.markdown("---")
        st.markdown("**Side-by-Side Comparison**")
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

        with c3:
            st.markdown("**Difference Map**")
            m1, m2 = r1.get("mask"), r2.get("mask")
            if m1 is not None and m2 is not None:
                from utils.image_utils import create_diff_map
                st.image(create_diff_map(m1, m2), use_container_width=True)
                st.caption("🟢 New growth | 🔴 Resolved | 🔵 Stable")

        st.markdown("<br>**Quantitative Changes**")
        a1    = r1["seg"].get("measurements") or {}
        a2    = r2["seg"].get("measurements") or {}
        area1 = a1.get("area_cm2", 0.0)
        area2 = a2.get("area_cm2", 0.0)
        change = ((area2 - area1) / max(area1, 0.001)) * 100

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Tumor Area (Scan 1)", f"{area1:.3f} cm²")
        mc2.metric("Tumor Area (Scan 2)", f"{area2:.3f} cm²",
                   delta=f"{change:+.1f}%", delta_color="inverse")
        mc3.metric("Risk Change", r2["risk"])

        from utils.visualization import tumor_growth_chart
        st.plotly_chart(
            tumor_growth_chart([str(date1), str(date2)], [area1, area2]),
            use_container_width=True)

        direction = "grown" if change > 5 else "shrunk" if change < -5 else "remained stable"
        st.markdown(
            f'<div class="info-box">📊 Between {date1} and {date2}, tumor has '
            f'<strong>{direction}</strong> from {area1:.3f} to {area2:.3f} cm² '
            f'({change:+.1f}%). Class: <strong>{r1["class"]}</strong> → '
            f'<strong>{r2["class"]}</strong>.</div>',
            unsafe_allow_html=True)


# ─── Page 5: Patient Records ──────────────────────────────────────────────────

def page_records():
    st.markdown('<div class="section-header">👥 Patient Records</div>',
                unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔍 Search Patients", "➕ New Patient"])

    with tab1:
        search   = st.text_input("Search by name", placeholder="Enter patient name...")
        patients = api_get("/patients", search=search, limit=100) or []
        if not patients:
            st.markdown('<div class="info-box">No patients found.</div>',
                        unsafe_allow_html=True)
            return
        import pandas as pd
        df = pd.DataFrame([{
            "ID": p["id"], "Name": p["name"],
            "Age": p.get("age","N/A"), "Sex": p.get("sex","N/A"),
            "Created": p["created_at"][:10]
        } for p in patients])
        selected = st.dataframe(df, use_container_width=True, hide_index=True,
                                on_select="rerun", selection_mode="single-row")
        if selected and selected.selection.rows:
            pid = df.iloc[selected.selection.rows[0]]["ID"]
            p   = api_get(f"/patients/{pid}")
            if p:
                st.markdown(f"---\n**{p['name']}** (ID: {p['id']})")
                c1, c2, c3 = st.columns(3)
                c1.metric("Age",        p.get("age","N/A"))
                c2.metric("Sex",        p.get("sex","N/A"))
                c3.metric("Registered", p["created_at"][:10])
                if p.get("medical_history"):
                    st.markdown(f"**History:** {p['medical_history']}")

    with tab2:
        st.markdown("**Create New Patient Record**")
        with st.form("new_patient_form"):
            c1, c2 = st.columns(2)
            with c1:
                name = st.text_input("Full Name *")
                age  = st.number_input("Age", 0, 120, 50)
            with c2:
                sex     = st.selectbox("Sex", ["Male","Female","Other"])
                contact = st.text_input("Contact")
            history   = st.text_area("Medical History")
            submitted = st.form_submit_button("Create Patient")
        if submitted:
            if not name:
                st.error("Name is required.")
            else:
                res = api_post("/patients", json_data={
                    "name": name, "age": age, "sex": sex,
                    "contact": contact, "medical_history": history,
                })
                if res:
                    st.success(f"✅ Patient '{name}' created with ID {res['id']}")


# ─── Sidebar ──────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown('<div class="pancrai-logo">Pancr<span>AI</span></div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div style="color:#8B949E;font-size:0.75rem;margin-bottom:20px">'
            'Intelligent Pancreatic Tumor Detection</div>',
            unsafe_allow_html=True)
        st.markdown("---")
        page = st.radio(
            "Navigation",
            ["🏥 Dashboard","🔬 New Scan","💬 AI Chat",
             "📈 Comparison","👥 Patient Records"],
            label_visibility="collapsed")
        st.markdown("---")
        st.markdown(f"""<div style="font-size:0.75rem;color:#8B949E">
            <div>API: {"🟢 Online" if _check_api() else "🔴 Offline (inline mode)"}</div>
            <div>Gemini: {"✅ Configured" if os.getenv("GEMINI_API_KEY") else "⚠️ Not set"}</div>
            <div>Groq: {"✅ Configured" if os.getenv("GROQ_API_KEY") else "⚠️ Not set"}</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            '<div style="font-size:0.7rem;color:#4A4F58;text-align:center">'
            'PancrAI v1.0.0<br>For research use only</div>',
            unsafe_allow_html=True)
        return page


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    inject_css()
    from dotenv import load_dotenv
    load_dotenv()
    page = sidebar()
    {
        "🏥 Dashboard":      page_dashboard,
        "🔬 New Scan":       page_new_scan,
        "💬 AI Chat":        page_chat,
        "📈 Comparison":     page_comparison,
        "👥 Patient Records": page_records,
    }.get(page, page_dashboard)()


if __name__ == "__main__":
    main()