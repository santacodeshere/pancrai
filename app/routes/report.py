"""
PancrAI — Report Generation Routes
Triggers Gemini AI report generation for completed predictions.
"""

import json
from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Prediction, Scan, Patient
from app.models.schemas import ReportRequest, ReportResponse

router = APIRouter()


@router.post("/report/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest, db: Session = Depends(get_db)):
    """
    Generate an AI diagnostic report for a completed prediction.
    Uses Gemini 1.5 Flash to produce a structured HTML report.
    """
    pred = db.query(Prediction).filter(Prediction.id == request.prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")

    scan = db.query(Scan).filter(Scan.id == pred.scan_id).first()
    patient = db.query(Patient).filter(Patient.id == scan.patient_id).first() if scan else None

    # Parse confidence scores
    try:
        conf_scores = json.loads(pred.confidence_scores or "[]")
    except Exception:
        conf_scores = [0.25, 0.25, 0.25, 0.25]

    from app.services.gemini_report import generate_report as gemini_generate

    result = await gemini_generate(
        tumor_class=pred.tumor_class or "Unknown",
        confidence=pred.primary_confidence or 0.0,
        confidence_scores=conf_scores,
        measurements={
            "area_cm2": pred.tumor_area_cm2,
            "centroid_x": pred.tumor_centroid_x,
            "centroid_y": pred.tumor_centroid_y,
        } if pred.tumor_area_cm2 else None,
        uncertainty_score=pred.uncertainty_score or 0.0,
        patient_name=request.patient_name or (patient.name if patient else "Unknown"),
        patient_age=request.patient_age or (patient.age if patient else None),
        patient_sex=request.patient_sex or (patient.sex if patient else None),
        symptoms=request.symptoms,
        scan_type=scan.scan_type if scan else "CT",
        risk_level=pred.risk_level or "Unknown",
    )

    # Save report to prediction record
    pred.report_text = result["report_html"]
    pred.report_summary = result["summary"]
    db.commit()

    return ReportResponse(
        report_html=result["report_html"],
        summary=result["summary"],
        risk_level=result["risk_level"],
    )


@router.get("/report/{prediction_id}")
async def get_report(prediction_id: int, db: Session = Depends(get_db)):
    """Retrieve a previously generated report."""
    pred = db.query(Prediction).filter(Prediction.id == prediction_id).first()
    if not pred:
        raise HTTPException(status_code=404, detail="Prediction not found")
    if not pred.report_text:
        raise HTTPException(
            status_code=404,
            detail="Report not yet generated. Call POST /report/generate first."
        )
    return {
        "prediction_id": prediction_id,
        "report_html": pred.report_text,
        "summary": pred.report_summary,
        "risk_level": pred.risk_level,
    }
