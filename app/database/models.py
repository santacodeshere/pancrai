"""
PancrAI — SQLAlchemy ORM Models
Defines Patient, Scan, and Prediction tables.
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    ForeignKey, Boolean
)
from sqlalchemy.orm import relationship
from app.database.db import Base


class Patient(Base):
    """Patient demographic and contact information."""
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name = Column(String(200), nullable=False, index=True)
    age = Column(Integer, nullable=True)
    sex = Column(String(10), nullable=True)           # Male / Female / Other
    contact = Column(String(100), nullable=True)
    medical_history = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    scans = relationship("Scan", back_populates="patient", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Patient id={self.id} name='{self.name}'>"


class Scan(Base):
    """Medical scan file metadata linked to a patient."""
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False, index=True)
    scan_type = Column(String(20), nullable=False)    # CT / MRI / PET
    file_path = Column(String(500), nullable=False)
    file_format = Column(String(20), nullable=True)   # DICOM / PNG / NIfTI
    upload_date = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)

    # Relationships
    patient = relationship("Patient", back_populates="scans")
    predictions = relationship("Prediction", back_populates="scan", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Scan id={self.id} patient_id={self.patient_id} type='{self.scan_type}'>"


class Prediction(Base):
    """Model prediction results for a scan."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    scan_id = Column(Integer, ForeignKey("scans.id"), nullable=False, index=True)

    # Detection results
    tumor_detected = Column(Boolean, default=False)
    tumor_class = Column(String(50), nullable=True)   # No Tumor / Benign / Malignant / Cystic
    tumor_class_index = Column(Integer, nullable=True)  # 0 / 1 / 2 / 3

    # Confidence scores (JSON string of [c0, c1, c2, c3])
    confidence_scores = Column(Text, nullable=True)
    primary_confidence = Column(Float, nullable=True)

    # Segmentation metrics
    dice_score = Column(Float, nullable=True)
    iou_score = Column(Float, nullable=True)

    # Tumor measurements
    tumor_area_pixels = Column(Float, nullable=True)
    tumor_area_cm2 = Column(Float, nullable=True)
    tumor_centroid_x = Column(Float, nullable=True)
    tumor_centroid_y = Column(Float, nullable=True)

    # Uncertainty
    uncertainty_score = Column(Float, nullable=True)  # 0–100

    # AI report
    report_text = Column(Text, nullable=True)
    report_summary = Column(Text, nullable=True)

    # Risk
    risk_level = Column(String(20), nullable=True)   # Low / Medium / High / Critical

    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    scan = relationship("Scan", back_populates="predictions")

    def __repr__(self):
        return f"<Prediction id={self.id} scan_id={self.scan_id} class='{self.tumor_class}'>"
