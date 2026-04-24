"""
PancrAI — Pydantic Schemas
Request/response models for API endpoints.
"""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field


# ─── Patient Schemas ────────────────────────────────────────────────────────

class PatientCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    age: Optional[int] = Field(None, ge=0, le=150)
    sex: Optional[str] = Field(None, pattern="^(Male|Female|Other)$")
    contact: Optional[str] = None
    medical_history: Optional[str] = None


class PatientResponse(BaseModel):
    id: int
    name: str
    age: Optional[int]
    sex: Optional[str]
    contact: Optional[str]
    medical_history: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Scan Schemas ────────────────────────────────────────────────────────────

class ScanCreate(BaseModel):
    patient_id: int
    scan_type: str = Field(..., pattern="^(CT|MRI|PET)$")
    notes: Optional[str] = None


class ScanResponse(BaseModel):
    id: int
    patient_id: int
    scan_type: str
    file_path: str
    file_format: Optional[str]
    upload_date: datetime
    notes: Optional[str]

    class Config:
        from_attributes = True


# ─── Prediction Schemas ──────────────────────────────────────────────────────

class TumorMeasurements(BaseModel):
    area_pixels: float
    area_cm2: float
    centroid_x: float
    centroid_y: float
    bbox_x: int
    bbox_y: int
    bbox_w: int
    bbox_h: int
    aspect_ratio: float


class PredictionResult(BaseModel):
    tumor_detected: bool
    tumor_class: str
    tumor_class_index: int
    confidence_scores: List[float]  # [no_tumor, benign, malignant, cystic]
    primary_confidence: float
    dice_score: float
    iou_score: float
    uncertainty_score: float
    measurements: Optional[TumorMeasurements]
    risk_level: str


class PredictionResponse(BaseModel):
    id: int
    scan_id: int
    tumor_detected: bool
    tumor_class: Optional[str]
    primary_confidence: Optional[float]
    dice_score: Optional[float]
    uncertainty_score: Optional[float]
    report_summary: Optional[str]
    risk_level: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True


# ─── Preprocessing Schemas ───────────────────────────────────────────────────

class PreprocessingStep(BaseModel):
    name: str
    description: str
    image_b64: str  # base64 encoded image


class PreprocessingResult(BaseModel):
    steps: List[PreprocessingStep]
    image_size: tuple
    scan_type: str


# ─── Report Schemas ──────────────────────────────────────────────────────────

class ReportRequest(BaseModel):
    scan_id: int
    prediction_id: int
    patient_name: Optional[str] = "Unknown"
    patient_age: Optional[int] = None
    patient_sex: Optional[str] = None
    symptoms: Optional[str] = None


class ReportResponse(BaseModel):
    report_html: str
    summary: str
    risk_level: str


# ─── Chat Schemas ────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    message: str
    prediction_context: Optional[dict] = None
    history: Optional[List[ChatMessage]] = []


class ChatResponse(BaseModel):
    response: str
    role: str = "assistant"


# ─── Comparison Schemas ──────────────────────────────────────────────────────

class ComparisonResult(BaseModel):
    scan1_class: str
    scan2_class: str
    scan1_confidence: float
    scan2_confidence: float
    area_change_percent: float
    area_change_direction: str   # "increased" / "decreased" / "stable"
    trend_summary: str
    risk_progression: str


# ─── Dashboard Schemas ───────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_patients: int
    total_scans: int
    scans_today: int
    avg_confidence: float
    detection_rate: float
    tumor_type_distribution: dict
