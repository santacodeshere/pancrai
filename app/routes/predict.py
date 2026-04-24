"""
PancrAI — Prediction Routes
Runs the full AI inference pipeline on uploaded scans.
"""

import os
import json
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Scan, Prediction, Patient
from app.models.schemas import PredictionResponse

router = APIRouter()

# Lazy-loaded model instances (initialized on first request)
_segmentation_model = None
_classifier_model = None
_device = "cpu"


def _load_models():
    """Load TransUNet and EfficientNetB4 models (lazy initialization)."""
    global _segmentation_model, _classifier_model

    if _segmentation_model is None or _classifier_model is None:
        seg_weights = os.getenv("MODEL_WEIGHTS_PATH", "./weights/transunet_best.pth")
        cls_weights = os.getenv("CLASSIFIER_WEIGHTS_PATH", "./weights/efficientnet_best.pth")

        from app.models.transunet import build_transunet
        from app.models.classifier import build_classifier

        _segmentation_model = build_transunet(pretrained=True, weights_path=seg_weights)
        _classifier_model = build_classifier(pretrained=True, weights_path=cls_weights)

    return _segmentation_model, _classifier_model


@router.post("/predict/{scan_id}")
async def run_prediction(
    scan_id: int,
    run_gradcam: bool = Query(True, description="Generate Grad-CAM heatmaps"),
    run_uncertainty: bool = Query(True, description="Run MC Dropout uncertainty estimation"),
    db: Session = Depends(get_db),
):
    """
    Run the full AI inference pipeline on a scan.

    Steps:
    1. Load and preprocess the image
    2. Run TransUNet segmentation
    3. Run EfficientNetB4 classification
    4. Generate Grad-CAM heatmaps (optional)
    5. MC Dropout uncertainty estimation (optional)
    6. Save results to database

    Returns full prediction results including images as base64 strings.
    """
    # Fetch scan
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail=f"Scan {scan_id} not found")
    if not os.path.exists(scan.file_path):
        raise HTTPException(status_code=404, detail="Scan file not found on disk")

    # Load models
    try:
        seg_model, cls_model = _load_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

    # Load and preprocess image
    from app.services.preprocessing import (
        load_from_bytes, run_full_pipeline, preprocess_to_tensor
    )
    with open(scan.file_path, "rb") as f:
        raw_bytes = f.read()

    image = load_from_bytes(raw_bytes, os.path.basename(scan.file_path))
    preprocess_steps = run_full_pipeline(image)

    # Segmentation
    from app.services.segmentation import run_segmentation
    seg_result = run_segmentation(image, seg_model, device=_device)

    # Classification — use preprocessed tensor
    import torch
    import torch.nn.functional as F
    tensor = preprocess_to_tensor(image).to(_device)

    cls_model.eval()
    with torch.no_grad():
        logits = cls_model(tensor)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

    pred_class_idx = int(probs.index(max(probs)))
    from app.models.classifier import CLASS_NAMES, get_risk_level
    tumor_class = CLASS_NAMES[pred_class_idx]
    primary_confidence = probs[pred_class_idx]
    risk_level = get_risk_level(pred_class_idx, primary_confidence)
    tumor_detected = pred_class_idx > 0

    measurements = seg_result.get("measurements")

    # Grad-CAM
    gradcam_b64 = None
    gradcampp_b64 = None
    if run_gradcam:
        try:
            from app.services.gradcam import generate_gradcam_results
            gcam_results = generate_gradcam_results(
                image, seg_model, measurements, device=_device
            )
            gradcam_b64 = gcam_results.get("gradcam_b64")
            gradcampp_b64 = gcam_results.get("gradcampp_b64")
        except Exception as e:
            print(f"[Predict] Grad-CAM failed: {e}")

    # Uncertainty
    uncertainty_score = 0.0
    uncertainty_b64 = None
    unc_details = {}
    if run_uncertainty:
        try:
            from app.services.uncertainty import mc_dropout_inference
            unc = mc_dropout_inference(
                seg_model, tensor.clone(), T=20, device=_device, is_segmentation=True
            )
            uncertainty_score = unc["uncertainty_score"]
            uncertainty_b64 = unc["uncertainty_heatmap_b64"]
            unc_details = {
                "confidence": unc["confidence"],
                "confidence_interval": unc["confidence_interval"],
                "high_uncertainty_warning": unc["high_uncertainty_warning"],
                "warning_message": unc.get("warning_message"),
            }
        except Exception as e:
            print(f"[Predict] Uncertainty failed: {e}")

    # Save prediction to DB
    pred_db = Prediction(
        scan_id=scan_id,
        tumor_detected=tumor_detected,
        tumor_class=tumor_class,
        tumor_class_index=pred_class_idx,
        confidence_scores=json.dumps(probs),
        primary_confidence=primary_confidence,
        dice_score=seg_result["dice_score"],
        iou_score=seg_result["iou_score"],
        tumor_area_pixels=measurements["area_pixels"] if measurements else None,
        tumor_area_cm2=measurements["area_cm2"] if measurements else None,
        tumor_centroid_x=measurements["centroid_x"] if measurements else None,
        tumor_centroid_y=measurements["centroid_y"] if measurements else None,
        uncertainty_score=uncertainty_score,
        risk_level=risk_level,
    )
    db.add(pred_db)
    db.commit()
    db.refresh(pred_db)

    return {
        "prediction_id": pred_db.id,
        "scan_id": scan_id,
        "tumor_detected": tumor_detected,
        "tumor_class": tumor_class,
        "tumor_class_index": pred_class_idx,
        "confidence_scores": probs,
        "primary_confidence": primary_confidence,
        "risk_level": risk_level,
        "dice_score": seg_result["dice_score"],
        "iou_score": seg_result["iou_score"],
        "uncertainty_score": uncertainty_score,
        "uncertainty_details": unc_details,
        "measurements": measurements,
        "images": {
            "preprocessing_steps": preprocess_steps,
            "segmentation_overlay": seg_result["overlay_b64"],
            "gradcam": gradcam_b64,
            "gradcam_plus_plus": gradcampp_b64,
            "uncertainty_heatmap": uncertainty_b64,
        }
    }


@router.get("/predictions/{scan_id}", response_model=List[PredictionResponse])
async def get_predictions_for_scan(scan_id: int, db: Session = Depends(get_db)):
    """Get all predictions for a scan."""
    return db.query(Prediction).filter(
        Prediction.scan_id == scan_id
    ).order_by(Prediction.created_at.desc()).all()


@router.get("/dashboard/stats")
async def get_dashboard_stats(db: Session = Depends(get_db)):
    """Get aggregate statistics for the dashboard."""
    from datetime import date
    from sqlalchemy import func

    total_patients = db.query(Patient).count()
    total_scans = db.query(Scan).count()
    scans_today = db.query(Scan).filter(
        func.date(Scan.upload_date) == date.today()
    ).count()

    preds = db.query(Prediction).all()
    total_preds = len(preds)
    avg_confidence = (
        sum(p.primary_confidence for p in preds if p.primary_confidence) / max(total_preds, 1)
    )
    detection_rate = (
        sum(1 for p in preds if p.tumor_detected) / max(total_preds, 1)
    )

    # Tumor type distribution
    tumor_dist = {}
    for p in preds:
        key = p.tumor_class or "Unknown"
        tumor_dist[key] = tumor_dist.get(key, 0) + 1

    return {
        "total_patients": total_patients,
        "total_scans": total_scans,
        "scans_today": scans_today,
        "avg_confidence": round(avg_confidence, 4),
        "detection_rate": round(detection_rate, 4),
        "tumor_type_distribution": tumor_dist,
    }


@router.post("/compare")
async def compare_scans(
    scan_id_1: int,
    scan_id_2: int,
    db: Session = Depends(get_db),
):
    """
    Compare two scans for longitudinal tumor progression analysis.
    Returns growth metrics and trend summary.
    """
    # Get latest predictions for each scan
    p1 = db.query(Prediction).filter(
        Prediction.scan_id == scan_id_1
    ).order_by(Prediction.created_at.desc()).first()
    p2 = db.query(Prediction).filter(
        Prediction.scan_id == scan_id_2
    ).order_by(Prediction.created_at.desc()).first()

    if not p1 or not p2:
        raise HTTPException(
            status_code=404,
            detail="Predictions not found for one or both scans. Run predict first."
        )

    area1 = p1.tumor_area_cm2 or 0.0
    area2 = p2.tumor_area_cm2 or 0.0
    conf1 = p1.primary_confidence or 0.0
    conf2 = p2.primary_confidence or 0.0

    if area1 > 0:
        area_change = ((area2 - area1) / area1) * 100
    else:
        area_change = 0.0

    direction = "stable"
    if area_change > 5:
        direction = "increased"
    elif area_change < -5:
        direction = "decreased"

    trend = (
        f"Tumor area changed from {area1:.3f} cm² to {area2:.3f} cm² "
        f"({area_change:+.1f}%). "
        f"Classification changed from '{p1.tumor_class}' to '{p2.tumor_class}'. "
        f"Malignancy confidence: {conf1*100:.1f}% → {conf2*100:.1f}%."
    )

    risk_prog = "stable"
    risk_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    r1 = risk_order.get(p1.risk_level or "Low", 0)
    r2 = risk_order.get(p2.risk_level or "Low", 0)
    if r2 > r1:
        risk_prog = "worsening"
    elif r2 < r1:
        risk_prog = "improving"

    return {
        "scan1": {"id": scan_id_1, "class": p1.tumor_class,
                  "confidence": conf1, "area_cm2": area1,
                  "risk": p1.risk_level},
        "scan2": {"id": scan_id_2, "class": p2.tumor_class,
                  "confidence": conf2, "area_cm2": area2,
                  "risk": p2.risk_level},
        "area_change_percent": round(area_change, 2),
        "area_change_direction": direction,
        "trend_summary": trend,
        "risk_progression": risk_prog,
    }
