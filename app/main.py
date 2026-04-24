"""
PancrAI — FastAPI Application Entry Point
Registers all routes and initializes the database on startup.
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.database.db import init_db
from app.routes import upload, predict, report, chat

load_dotenv()

# ─── App Instance ──────────────────────────────────────────────────────────

app = FastAPI(
    title="PancrAI — Pancreatic Tumor Detection API",
    description=(
        "Intelligent pancreatic tumor detection and clinical decision support "
        "system using TransUNet segmentation, EfficientNetB4 classification, "
        "Grad-CAM explainability, and AI-generated diagnostic reports."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─── CORS ──────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routes ───────────────────────────────────────────────────────────────

app.include_router(upload.router, prefix="/api/v1", tags=["Upload"])
app.include_router(predict.router, prefix="/api/v1", tags=["Predict"])
app.include_router(report.router, prefix="/api/v1", tags=["Report"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])

# ─── Lifecycle ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Initialize database and ensure upload directory exists."""
    init_db()
    os.makedirs("./uploads", exist_ok=True)
    os.makedirs("./weights", exist_ok=True)
    print("[PancrAI] Database initialized.")
    print("[PancrAI] API ready at http://localhost:8000")
    print("[PancrAI] Docs available at http://localhost:8000/docs")


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "service": "PancrAI",
        "status": "operational",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "upload": "/api/v1/upload",
            "predict": "/api/v1/predict",
            "report": "/api/v1/report",
            "chat": "/api/v1/chat",
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
