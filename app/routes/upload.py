"""
PancrAI — Upload Routes
Handles scan file upload and patient record creation.
"""

import os
import shutil
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from sqlalchemy.orm import Session

from app.database.db import get_db
from app.database.models import Patient, Scan
from app.models.schemas import PatientCreate, PatientResponse, ScanResponse

router = APIRouter()

UPLOAD_DIR = "./uploads"
ALLOWED_EXTENSIONS = {".dcm", ".png", ".jpg", ".jpeg", ".nii", ".gz", ".bmp", ".tiff"}
MAX_SIZE_BYTES = 50 * 1024 * 1024   # 50 MB


@router.post("/patients", response_model=PatientResponse, status_code=201)
async def create_patient(patient: PatientCreate, db: Session = Depends(get_db)):
    """Create a new patient record."""
    db_patient = Patient(**patient.model_dump())
    db.add(db_patient)
    db.commit()
    db.refresh(db_patient)
    return db_patient


@router.get("/patients", response_model=list[PatientResponse])
async def list_patients(
    search: str = "",
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """List patients with optional name search."""
    q = db.query(Patient)
    if search:
        q = q.filter(Patient.name.ilike(f"%{search}%"))
    return q.order_by(Patient.created_at.desc()).offset(skip).limit(limit).all()


@router.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int, db: Session = Depends(get_db)):
    """Get a specific patient by ID."""
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return p


@router.post("/upload", response_model=ScanResponse, status_code=201)
async def upload_scan(
    file: UploadFile = File(...),
    patient_id: int = Form(...),
    scan_type: str = Form("CT"),
    notes: str = Form(""),
    db: Session = Depends(get_db),
):
    """
    Upload a medical scan file (DICOM, PNG, NIfTI).

    Validates:
    - File extension is supported
    - File size is within limits
    - Patient exists

    Returns the created Scan record.
    """
    # Validate patient
    patient = db.query(Patient).filter(Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")

    # Validate extension
    filename = file.filename or "upload.png"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # Read and validate size
    content = await file.read()
    if len(content) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)/1e6:.1f} MB). Max 50 MB."
        )
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    # Save to disk
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"p{patient_id}_{timestamp}{ext}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        f.write(content)

    # Detect format
    format_map = {
        ".dcm": "DICOM", ".nii": "NIfTI", ".gz": "NIfTI",
        ".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG",
    }
    file_format = format_map.get(ext, "Standard")

    # Create scan record
    scan = Scan(
        patient_id=patient_id,
        scan_type=scan_type.upper(),
        file_path=file_path,
        file_format=file_format,
        notes=notes or None,
    )
    db.add(scan)
    db.commit()
    db.refresh(scan)

    return scan


@router.delete("/scans/{scan_id}", status_code=204)
async def delete_scan(scan_id: int, db: Session = Depends(get_db)):
    """Delete a scan record and its file."""
    scan = db.query(Scan).filter(Scan.id == scan_id).first()
    if not scan:
        raise HTTPException(status_code=404, detail="Scan not found")

    if os.path.exists(scan.file_path):
        os.remove(scan.file_path)

    db.delete(scan)
    db.commit()
