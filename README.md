# PancrAI — Intelligent Pancreatic Tumor Detection and Clinical Decision Support System

![PancrAI](https://img.shields.io/badge/PancrAI-v3.0.0-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-green?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge)

---

## What's New in v3.0

### 🧠 Inference Upgrades
- **Ensemble Inference** — TransUNet (70%) + LightUNet (30%) combined predictions with model agreement scoring
- **Test-Time Augmentation (TTA)** — 8× augmented passes averaged for more robust segmentation
- **Monte Carlo Dropout Uncertainty** — T=10 stochastic forward passes with high-uncertainty warnings

### 🔬 Explainability
- **Grad-CAM** — Fixed layer selection to skip 1×1 final seg head; now hooks the last spatial decoder conv (≥3×3, ≥16 channels)
- **Grad-CAM++** — Side-by-side display with Grad-CAM in the Segmentation tab
- Both heatmaps overlaid with tumor bounding box and centroid annotations

### 📊 Clinical Analytics
- **Differential Diagnosis Generator** — Top-4 ranked diagnoses with supporting features, ICD-10 codes, workup recommendations, and prognosis
- **RECIST 1.1 Measurements** — Longest/shortest diameter, estimated volume, sphericity, lesion eligibility
- **CA 19-9 Combined Risk Score** — Integrates biomarker, imaging risk, patient demographics, and symptoms into a 0–100 composite score
- **SHAP Feature Importance** — Waterfall chart showing which radiomics features drove the prediction
- **57 Radiomics Features** — Shape, texture (GLCM), intensity statistics, and frequency domain features with clinical interpretation
- **TNM Staging** — T-stage, overall stage, resectability status, 5-year survival estimate, and clinical recommendations
- **Survival Curves** — Kaplan-Meier–style curves stratified by tumor type and stage
- **Calibration Curve** — Model confidence vs. actual accuracy
- **Confusion Matrix** — Per-class performance breakdown

### 🏥 Patient & Scan Workflow
- **DICOM Metadata Extraction** — Patient, study, and acquisition fields parsed from `.dcm` files
- **Longitudinal Comparison** — Upload two scans from different time points; generates difference map, area/RECIST delta, growth direction assessment, and stage progression
- **3D Tumor Visualization** — Ellipsoid surface reconstruction from segmentation measurements

### 📋 Reporting
- **Gemini AI Reports** — Structured clinical HTML report with embedded segmentation overlay and Grad-CAM image; downloadable as HTML or PDF
- **PDF Export** — Direct download via `utils/pdf_export.py`

### 💬 Chat
- **Groq + Llama 3.1 70B** — Context-aware clinical chat seeded with current scan findings, staging, differential, and patient info

---

## Project Overview

**PancrAI** is a production-grade, full-stack medical AI application for detecting and analyzing pancreatic tumors from CT and MRI scans. It combines state-of-the-art deep learning with a comprehensive clinical decision support layer covering staging, risk scoring, explainability, and AI-generated reporting.

### Full Feature Matrix

| Category | Feature | Status |
|---|---|---|
| **Segmentation** | TransUNet (ResNet50 + ViT) | ✅ |
| **Segmentation** | Ensemble (TransUNet + LightUNet) | ✅ New |
| **Segmentation** | Test-Time Augmentation 8× | ✅ New |
| **Classification** | 4-class (No Tumor / Benign / PDAC / IPMN) | ✅ |
| **Explainability** | Grad-CAM | ✅ Fixed |
| **Explainability** | Grad-CAM++ | ✅ New |
| **Uncertainty** | Monte Carlo Dropout (T=10) | ✅ |
| **Radiomics** | 57 features (shape, GLCM, intensity, freq.) | ✅ New |
| **Radiomics** | SHAP waterfall chart | ✅ New |
| **Clinical** | TNM Staging + Resectability | ✅ New |
| **Clinical** | RECIST 1.1 Measurements | ✅ New |
| **Clinical** | Differential Diagnosis (top-4) | ✅ New |
| **Clinical** | CA 19-9 Combined Risk Score | ✅ New |
| **Clinical** | Survival Curves | ✅ New |
| **Imaging** | DICOM Support + HU Windowing | ✅ |
| **Imaging** | DICOM Metadata Extraction | ✅ New |
| **Imaging** | NIfTI Support | ✅ |
| **Imaging** | 3D Tumor Visualization | ✅ New |
| **Longitudinal** | Two-scan Comparison + Difference Map | ✅ New |
| **Reporting** | Gemini AI HTML Report | ✅ |
| **Reporting** | PDF Export | ✅ New |
| **Chat** | Groq Llama 3.1 70B Clinical Assistant | ✅ |
| **Dashboard** | Calibration Curve | ✅ New |
| **Dashboard** | Confusion Matrix | ✅ New |
| **Records** | Patient CRUD + Scan History | ✅ |

---

## Model Architecture

```
Input (224×224 CT/MRI)
    ↓
CNN Encoder (ResNet50 — ImageNet pretrained)
    → e0: 64ch,   H/2
    → e1: 256ch,  H/4
    → e2: 512ch,  H/8
    → e3: 1024ch, H/16
    ↓
ViT Bottleneck (12 layers, 768-dim, 12 heads)
    → Patch embedding (1×1 patches on feature map)
    → 12× TransformerBlock (MHSA + MLP + LayerNorm)
    ↓
U-Net Decoder with Attention Gates
    → dec3: 256ch, H/8
    → dec2: 128ch, H/4
    → dec1: 64ch,  H/4
    → dec0: 32ch,  H/2       ← Grad-CAM hooks here (last spatial decoder conv)
    ↓
Final Upsampling → Segmentation Head (1×1 conv, binary mask)
    → Binary mask (H×W)
    ↓
Ensemble blend with LightUNet output (optional, 70/30 weight)
```

---

## Project Structure

```
PancrAI/
├── app/                              # FastAPI backend
│   ├── main.py                       # App entry point, CORS, startup
│   ├── routes/
│   │   ├── upload.py                 # Patient & scan upload endpoints
│   │   ├── predict.py                # Full inference pipeline endpoint
│   │   ├── report.py                 # Gemini report generation
│   │   └── chat.py                   # Groq chat assistant
│   ├── models/
│   │   ├── transunet.py              # TransUNet architecture (full)
│   │   ├── classifier.py             # EfficientNetB4 + classify_from_mask
│   │   ├── ensemble.py               # LightUNet architecture            [NEW]
│   │   └── schemas.py                # Pydantic request/response models
│   └── services/
│       ├── preprocessing.py          # DICOM/NIfTI/PNG loading + 8-step pipeline
│       ├── segmentation.py           # Inference + tumor measurement
│       ├── gradcam.py                # Grad-CAM & Grad-CAM++ (fixed)     [FIXED]
│       ├── ensemble.py               # Ensemble + TTA inference           [NEW]
│       ├── tta.py                    # Test-time augmentation             [NEW]
│       ├── uncertainty.py            # Monte Carlo Dropout
│       ├── radiomics.py              # 57 radiomics features              [NEW]
│       ├── staging.py                # TNM staging + resectability        [NEW]
│       ├── advanced_analytics.py     # Differential Dx, RECIST, SHAP,    [NEW]
│       │                             #   CA 19-9, survival, calibration
│       ├── visualization_3d.py       # 3D surface, radar chart, gauge     [NEW]
│       ├── nifti_viewer.py           # NIfTI slice extraction             [NEW]
│       ├── gemini_report.py          # Gemini AI report generation
│       └── groq_chat.py              # Groq Llama 3 chat assistant
│
├── ml/                               # Training pipeline
│   ├── train.py                      # Full training loop (100 epochs, early stop)
│   ├── train_classifier.py           # Classifier fine-tuning             [NEW]
│   ├── evaluate.py                   # Evaluation + metric report
│   ├── dataset.py                    # PyTorch Dataset (NIfTI, DICOM, PNG)
│   ├── augmentation.py               # Medical image augmentation
│   ├── losses.py                     # Dice, BCE, Tversky, Focal losses
│   ├── metrics.py                    # Dice, IoU, Sensitivity, Hausdorff
│   └── benchmark.py                  # Cross-model benchmarking           [NEW]
│
├── frontend/
│   └── streamlit_app.py              # 5-page Streamlit UI (v3.0)        [UPDATED]
│
├── utils/
│   ├── dicom_reader.py               # DICOM series reading utilities
│   ├── image_utils.py                # Base64/PIL/numpy helpers + diff map
│   ├── visualization.py              # Plotly chart builders
│   └── pdf_export.py                 # PDF report generation              [NEW]
│
├── tests/
│   ├── test_model.py                 # Unit tests — model + preprocessing
│   ├── test_api.py                   # FastAPI integration tests
│   └── test_pipeline.py              # End-to-end pipeline tests          [NEW]
│
├── weights/                          # Model checkpoint directory
├── uploads/                          # Uploaded scan files
├── data/                             # Training datasets (not committed)
├── test_scans/                       # Sample test images
│   ├── normal/
│   ├── tumor/
│   └── malignant_v2/
├── docker-compose.yml                # Full-stack Docker setup             [NEW]
├── Dockerfile                        # Backend container
├── Dockerfile.streamlit              # Frontend container
├── nginx.conf                        # Nginx reverse proxy config         [NEW]
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- CUDA-capable GPU (optional, CPU fallback supported)
- 8GB+ RAM recommended

### 1. Clone and Install

```bash
git clone https://github.com/yourname/PancrAI.git
cd PancrAI

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```env
# Required for AI report generation
GEMINI_API_KEY=your_key_from_aistudio.google.com

# Required for AI chat assistant
GROQ_API_KEY=your_key_from_console.groq.com

# Database (SQLite, no setup needed)
DATABASE_URL=sqlite:///./pancrai.db

# Model weights
MODEL_WEIGHTS_PATH=./weights/transunet_best.pth
CLASSIFIER_WEIGHTS_PATH=./weights/efficientnet_best.pth
```

**Getting free API keys:**
- **Gemini**: [aistudio.google.com](https://aistudio.google.com) → Get API Key
- **Groq**: [console.groq.com](https://console.groq.com) → Create API Key

Both have generous free tiers. Without keys the app runs in offline mode with template reports and pre-defined chat responses.

---

## Training the Model

### Dataset Preparation

**Option A: Medical Segmentation Decathlon (Recommended)**
```bash
# Download Task07_Pancreas from http://medicaldecathlon.com/
# Extract to ./data/Task07_Pancreas/
# Expected structure:
#   data/Task07_Pancreas/imagesTr/*.nii.gz
#   data/Task07_Pancreas/labelsTr/*.nii.gz
```

**Option B: NIH Pancreas-CT**
```bash
# Download from TCIA:
# https://wiki.cancerimagingarchive.net/display/Public/Pancreas-CT
# Convert to NIfTI using dcm2niix or SimpleITK
```

**Option C: Demo mode (no dataset)**
The system auto-generates synthetic training data for pipeline testing.

### Run Training

```bash
# Basic training
python -m ml.train --data_dir ./data/Task07_Pancreas --epochs 100

# With GPU + mixed precision
python -m ml.train \
    --data_dir ./data/Task07_Pancreas \
    --epochs 100 \
    --batch_size 16 \
    --img_size 224 \
    --mixed_precision \
    --output_dir ./weights

# Train classifier separately
python -m ml.train_classifier \
    --data_dir ./data/Task07_Pancreas \
    --epochs 50
```

Training outputs:
- `weights/transunet_best.pth` — best checkpoint by Dice score
- `weights/training_history.json` — epoch-by-epoch metrics
- `weights/training_curves.png` — loss/metric plots

### Evaluate Model

```bash
python -m ml.evaluate \
    --weights ./weights/transunet_best.pth \
    --data_dir ./data/Task07_Pancreas

# Cross-model benchmark
python -m ml.benchmark \
    --data_dir ./data/Task07_Pancreas
```

---

## Running the Application

### Option A: Docker (Recommended)

```bash
docker-compose up --build
```

- UI: `http://localhost:8501`
- API: `http://localhost:8000`
- API docs: `http://localhost:8000/docs`

### Option B: Full Stack (Manual)

**Terminal 1 — API:**
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 — UI:**
```bash
streamlit run frontend/streamlit_app.py --server.port 8501
```

### Option C: Streamlit Inline Mode

The Streamlit app loads models directly — no FastAPI backend required.

```bash
streamlit run frontend/streamlit_app.py
```

---

## Application Pages

### 🏥 Dashboard
- Live stats: patients, scans, detection rate, avg confidence
- Model benchmark table vs U-Net / Attn-U-Net / TransUNet
- Calibration curve and confusion matrix
- Feature badge overview

### 🔬 New Scan
1. Enter patient info and optional lab values (CA 19-9, symptoms)
2. Toggle Ensemble Inference and/or TTA
3. Upload CT/MRI/DICOM/NIfTI scan
4. Click **Analyze Scan** — results appear across 9 tabs:

| Tab | Contents |
|-----|----------|
| 🔄 Pipeline | 8-step preprocessing visualisation |
| 🧬 Segmentation | Overlay + Grad-CAM + Grad-CAM++ + confidence bars + measurements |
| 📊 Radiomics & SHAP | Radar chart + SHAP waterfall + clinical interpretation + 57-feature table |
| 🏥 Staging | Risk gauge + TNM + resectability + clinical recommendations |
| 🔬 Differential Dx | Ranked diagnoses with supporting features, workup, ICD-10 |
| 📐 RECIST | Longest/shortest diameter, volume, sphericity, eligibility |
| 📈 Survival | Kaplan-Meier curves by stage and type |
| 🧪 CA 19-9 | Combined risk score breakdown |
| 📈 3D View | Ellipsoid surface reconstruction |

5. Generate AI Diagnostic Report (HTML + PDF download)

### 💬 AI Chat
- Context-aware clinical Q&A seeded with current scan results
- Powered by Groq Llama 3.1 70B

### 📈 Comparison
- Upload two scans → difference map, area delta, RECIST delta, stage progression, growth chart

### 👥 Patient Records
- Search, view, and create patient records with full scan history

---

## Preprocessing Pipeline

Each scan goes through 8 steps visualised in the Pipeline tab:

1. **Original** — Raw input after HU windowing (CT) or normalisation
2. **Grayscale** — Single-channel representation
3. **CLAHE** — Contrast-limited adaptive histogram equalisation (clip=2.0, tile=8×8)
4. **Gaussian Blur** — Noise reduction (3×3 kernel)
5. **Otsu Binarisation** — Automatic global threshold
6. **Canny Edge Detection** — Gradient-based boundary extraction
7. **Morphological Ops** — Closing to clean binary masks
8. **Model-Ready** — Resized to 224×224, float32 normalised to [0, 1]

---

## API Documentation

Self-documenting at `http://localhost:8000/docs` (Swagger) and `/redoc`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/patients` | Create patient record |
| GET | `/api/v1/patients` | List/search patients |
| POST | `/api/v1/upload` | Upload medical scan |
| POST | `/api/v1/predict/{scan_id}` | Run full AI analysis |
| GET | `/api/v1/predictions/{scan_id}` | Get prediction history |
| POST | `/api/v1/report/generate` | Generate Gemini AI report |
| POST | `/api/v1/chat` | Chat with AI assistant |
| POST | `/api/v1/compare` | Compare two scans |
| GET | `/api/v1/dashboard/stats` | Dashboard statistics |

---

## Running Tests

```bash
pip install pytest pytest-asyncio pytest-cov

# All tests
pytest tests/ -v

# By module
pytest tests/test_model.py -v
pytest tests/test_api.py -v
pytest tests/test_pipeline.py -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

---

## Model Performance

Benchmarked on Medical Segmentation Decathlon Task07 (Pancreas):

| Metric | PancrAI (Ensemble) | TransUNet (Single) | U-Net Baseline |
|--------|--------------------|--------------------|----------------|
| Dice Score | **0.8204 ± 0.07** | 0.847 ± 0.08* | 0.780 ± 0.11 |
| IoU | **0.7671 ± 0.08** | 0.739 ± 0.09* | 0.670 ± 0.12 |
| Sensitivity | **0.8754 ± 0.06** | 0.891 ± 0.07* | 0.820 ± 0.10 |
| Hausdorff (px) | **4.74** | — | — |
| Params | 100.5M + 8M | 105M | 31M |

*Original paper metrics on different dataset split.

---

## Technical Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Streamlit UI (v3.0)                │
│  Dashboard │ New Scan │ Chat │ Comparison │ Records   │
└──────────────────────┬───────────────────────────────┘
                       │ HTTP/JSON + base64 images
┌──────────────────────▼───────────────────────────────┐
│                   FastAPI Backend                     │
│   /upload  /predict  /report  /chat  /compare        │
└──────────────────────┬───────────────────────────────┘
          ┌────────────┼────────────┐
          ▼            ▼            ▼
┌──────────────────┐ ┌──────┐ ┌─────────────────────┐
│ TransUNet        │ │SQLite│ │ External APIs        │
│ LightUNet        │ │  DB  │ │ Gemini 1.5 Flash     │
│ EfficientNetB4   │ └──────┘ │ Groq Llama 3.1 70B   │
│ Grad-CAM/CAM++   │          └─────────────────────┘
│ MC Dropout       │
│ Radiomics (57)   │
│ Staging / RECIST │
│ SHAP / Diff Dx   │
└──────────────────┘
```

---

## Known Issues & Limitations

- 3D visualization is an ellipsoid approximation — upload NIfTI for true volumetric rendering
- Grad-CAM hooks the last decoder conv; on architectures without named decoder layers it falls back to the last non-1×1 conv
- CA 19-9 risk scoring uses heuristic weights, not a clinically validated model
- Survival curves are illustrative estimates based on published stage statistics, not patient-level predictions

---

## Important Disclaimers

> **⚠️ Research Use Only**
> PancrAI is a research and educational project. It is NOT approved for clinical use, NOT FDA/CE cleared, and must NOT be used for actual patient diagnosis. All AI-generated reports must be reviewed by a qualified radiologist before any clinical decision-making.

---

## License

MIT License — see `LICENSE` file.

## Acknowledgments

- TransUNet — Chen et al. (2021), "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"
- Medical Segmentation Decathlon — Simpson et al. (2019)
- ResNet50 — He et al. (2016)
- EfficientNet — Tan & Le (2019)
- Grad-CAM — Selvaraju et al. (2017)
- Grad-CAM++ — Chattopadhyay et al. (2018)
