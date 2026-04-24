"""
PancrAI — API Integration Tests
Tests for FastAPI endpoints using TestClient.
"""

import pytest
import io
import numpy as np
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture(scope="module")
def client():
    """Create a test client with an in-memory database."""
    import os
    os.environ["DATABASE_URL"] = "sqlite:///./test_pancrai.db"
    from app.main import app
    from app.database.db import init_db
    init_db()
    with TestClient(app) as c:
        yield c
    # Cleanup
    if os.path.exists("./test_pancrai.db"):
        os.remove("./test_pancrai.db")


@pytest.fixture
def sample_png_bytes():
    """Create a small synthetic PNG for testing."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    arr[20:44, 20:44] = 200   # bright square simulating organ
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def created_patient(client):
    """Create a test patient and return its data."""
    response = client.post("/api/v1/patients", json={
        "name": "Test Patient",
        "age": 55,
        "sex": "Male",
    })
    assert response.status_code == 201
    return response.json()


# ─── Health Check ─────────────────────────────────────────────────────────────

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["service"] == "PancrAI"
    assert data["status"] == "operational"


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


# ─── Patient CRUD ─────────────────────────────────────────────────────────────

def test_create_patient(client):
    r = client.post("/api/v1/patients", json={
        "name": "Jane Doe",
        "age": 62,
        "sex": "Female",
        "medical_history": "Type 2 diabetes",
    })
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "Jane Doe"
    assert data["id"] is not None


def test_list_patients(client, created_patient):
    r = client.get("/api/v1/patients")
    assert r.status_code == 200
    patients = r.json()
    assert isinstance(patients, list)
    assert len(patients) >= 1


def test_search_patients(client, created_patient):
    r = client.get("/api/v1/patients", params={"search": "Test"})
    assert r.status_code == 200
    patients = r.json()
    names = [p["name"] for p in patients]
    assert any("Test" in n for n in names)


def test_get_patient_not_found(client):
    r = client.get("/api/v1/patients/99999")
    assert r.status_code == 404


def test_create_patient_missing_name(client):
    r = client.post("/api/v1/patients", json={"age": 40})
    assert r.status_code == 422  # Validation error


# ─── Scan Upload ──────────────────────────────────────────────────────────────

def test_upload_scan_valid(client, created_patient, sample_png_bytes):
    files = {"file": ("scan.png", sample_png_bytes, "image/png")}
    data = {
        "patient_id": str(created_patient["id"]),
        "scan_type": "CT",
        "notes": "Test upload",
    }
    r = client.post("/api/v1/upload", data=data, files=files)
    assert r.status_code == 201
    scan = r.json()
    assert scan["patient_id"] == created_patient["id"]
    assert scan["scan_type"] == "CT"
    assert scan["file_format"] == "PNG"


def test_upload_invalid_extension(client, created_patient):
    files = {"file": ("scan.xyz", b"test content", "application/octet-stream")}
    data = {
        "patient_id": str(created_patient["id"]),
        "scan_type": "CT",
    }
    r = client.post("/api/v1/upload", data=data, files=files)
    assert r.status_code == 400


def test_upload_patient_not_found(client, sample_png_bytes):
    files = {"file": ("scan.png", sample_png_bytes, "image/png")}
    data = {"patient_id": "99999", "scan_type": "CT"}
    r = client.post("/api/v1/upload", data=data, files=files)
    assert r.status_code == 404


def test_upload_empty_file(client, created_patient):
    files = {"file": ("empty.png", b"", "image/png")}
    data = {"patient_id": str(created_patient["id"]), "scan_type": "CT"}
    r = client.post("/api/v1/upload", data=data, files=files)
    assert r.status_code == 400


# ─── Chat Endpoint ────────────────────────────────────────────────────────────

def test_chat_basic(client):
    r = client.post("/api/v1/chat", json={
        "message": "What is pancreatic cancer?",
        "history": [],
    })
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert len(data["response"]) > 0


def test_chat_empty_message(client):
    r = client.post("/api/v1/chat", json={
        "message": "  ",
        "history": [],
    })
    assert r.status_code == 400


def test_chat_with_context(client):
    r = client.post("/api/v1/chat", json={
        "message": "What does this result mean?",
        "history": [],
        "prediction_context": {
            "tumor_class": "Malignant (PDAC)",
            "primary_confidence": 0.87,
            "risk_level": "Critical",
        },
    })
    assert r.status_code == 200
    assert "response" in r.json()


# ─── Dashboard Stats ──────────────────────────────────────────────────────────

def test_dashboard_stats(client):
    r = client.get("/api/v1/dashboard/stats")
    assert r.status_code == 200
    data = r.json()
    assert "total_patients" in data
    assert "total_scans" in data
    assert "tumor_type_distribution" in data
    assert data["total_patients"] >= 0
