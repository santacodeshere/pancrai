"""
PancrAI — Preprocessing Pipeline
Handles DICOM, NIfTI, PNG, JPEG inputs.
Returns all intermediate processing steps as base64-encoded images.
"""

import io
import base64
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image


# ─── Format Detection ────────────────────────────────────────────────────────

def detect_format(file_path: str) -> str:
    """Detect image format from file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".dcm":
        return "DICOM"
    elif ext in (".nii", ".gz"):
        return "NIfTI"
    elif ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"):
        return "Standard"
    return "Unknown"


# ─── DICOM Loading ────────────────────────────────────────────────────────────

def load_dicom(file_path: str) -> np.ndarray:
    """
    Load a DICOM file and apply CT HU windowing for soft tissue.

    Window center: 40 HU, Window width: 400 HU
    Returns uint8 image array normalized to [0, 255].
    """
    try:
        import pydicom
    except ImportError:
        raise ImportError("pydicom is required for DICOM support. pip install pydicom")

    ds = pydicom.dcmread(file_path)
    pixel_array = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present (converts to HU)
    if hasattr(ds, "RescaleSlope"):
        pixel_array = pixel_array * float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    # HU windowing — soft tissue: WC=40, WW=400
    wc, ww = 40.0, 400.0
    lower = wc - ww / 2   # -160 HU
    upper = wc + ww / 2   #  240 HU
    pixel_array = np.clip(pixel_array, lower, upper)

    # Normalize to [0, 255] uint8
    pixel_array = ((pixel_array - lower) / (upper - lower) * 255.0)
    return pixel_array.astype(np.uint8)


# ─── NIfTI Loading ────────────────────────────────────────────────────────────

def load_nifti(file_path: str, slice_axis: int = 2,
               slice_index: Optional[int] = None) -> np.ndarray:
    """
    Load a NIfTI file and extract the middle (or specified) axial slice.

    Returns uint8 array.
    """
    try:
        import nibabel as nib
    except ImportError:
        raise ImportError("nibabel is required for NIfTI support. pip install nibabel")

    img = nib.load(file_path)
    data = img.get_fdata()

    # Select middle slice along specified axis if no index given
    if slice_index is None:
        slice_index = data.shape[slice_axis] // 2

    if slice_axis == 0:
        sl = data[slice_index, :, :]
    elif slice_axis == 1:
        sl = data[:, slice_index, :]
    else:
        sl = data[:, :, slice_index]

    # Normalize to [0, 255]
    sl = sl.astype(np.float32)
    sl_min, sl_max = sl.min(), sl.max()
    if sl_max > sl_min:
        sl = (sl - sl_min) / (sl_max - sl_min) * 255.0
    return sl.astype(np.uint8)


# ─── Standard Image Loading ──────────────────────────────────────────────────

def load_standard(file_path: str) -> np.ndarray:
    """Load PNG/JPEG as grayscale uint8 array."""
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Try via PIL if OpenCV fails
        pil = Image.open(file_path).convert("L")
        img = np.array(pil)
    return img


def load_from_bytes(file_bytes: bytes,
                    filename: str = "upload.png") -> np.ndarray:
    """
    Load image from raw bytes (for FastAPI UploadFile handling).
    Auto-detects format from filename extension.
    """
    ext = Path(filename).suffix.lower()
    if ext == ".dcm":
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            img = load_dicom(tmp_path)
        finally:
            os.unlink(tmp_path)
        return img
    elif ext in (".nii", ".gz"):
        import tempfile, os
        suffix = ".nii.gz" if filename.endswith(".gz") else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            img = load_nifti(tmp_path)
        finally:
            os.unlink(tmp_path)
        return img
    else:
        # PNG / JPEG via numpy
        arr = np.frombuffer(file_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            pil = Image.open(io.BytesIO(file_bytes)).convert("L")
            img = np.array(pil)
        return img


# ─── Individual Processing Steps ─────────────────────────────────────────────

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply Contrast-Limited Adaptive Histogram Equalization."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def apply_gaussian_blur(gray: np.ndarray,
                        ksize: int = 5) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def apply_otsu(gray: np.ndarray) -> np.ndarray:
    """Apply Otsu's automatic thresholding → binary mask."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_canny(gray: np.ndarray) -> np.ndarray:
    """Apply Canny edge detection."""
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)
    return cv2.Canny(blurred, 30, 100)


def apply_morphology(binary: np.ndarray) -> np.ndarray:
    """Apply morphological opening then closing to clean binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def prepare_for_model(gray: np.ndarray,
                      target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Final preprocessing step for model input:
    CLAHE → Gaussian → resize → normalize [0,1].
    Returns float32 array (H, W).
    """
    enhanced = apply_clahe(gray)
    denoised = apply_gaussian_blur(enhanced, ksize=3)
    resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


# ─── Base64 Encoding ─────────────────────────────────────────────────────────

def array_to_b64(arr: np.ndarray, cmap: bool = False) -> str:
    """
    Convert numpy array (H, W) or (H, W, 3) to base64 PNG string.

    Args:
        arr: Input array. If 2D, converts to RGB via colormap or gray.
        cmap: If True, apply JET colormap (for heatmaps).
    """
    if arr.dtype != np.uint8:
        # Normalize to [0, 255] if float
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
        else:
            arr = np.zeros_like(arr, dtype=np.uint8)

    if arr.ndim == 2:
        if cmap:
            arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        else:
            arr_rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr_rgb = arr if arr.shape[2] == 3 else cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(arr_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_full_pipeline(image: np.ndarray,
                      target_size: Tuple[int, int] = (224, 224)
                      ) -> List[Dict[str, str]]:
    """
    Run the complete preprocessing pipeline on a grayscale image.

    Returns a list of dicts, each with:
      - name: Step name
      - description: What this step does
      - image_b64: Base64-encoded PNG
    """
    steps = []

    # Resize original to target for display consistency
    display = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # ── Step 1: Original ──
    steps.append({
        "name": "Original",
        "description": "Raw input after HU windowing / format conversion",
        "image_b64": array_to_b64(display),
    })

    # ── Step 2: Grayscale (already gray, show as info step) ──
    steps.append({
        "name": "Grayscale",
        "description": "Single-channel grayscale representation",
        "image_b64": array_to_b64(display),
    })

    # ── Step 3: CLAHE ──
    clahe_img = apply_clahe(display)
    steps.append({
        "name": "CLAHE Enhancement",
        "description": "Contrast-Limited Adaptive Histogram Equalization "
                       "(clip limit=2.0, tile 8×8)",
        "image_b64": array_to_b64(clahe_img),
    })

    # ── Step 4: Gaussian Blur ──
    blurred = apply_gaussian_blur(clahe_img, ksize=5)
    steps.append({
        "name": "Gaussian Blur",
        "description": "Gaussian noise reduction (σ=5×5 kernel)",
        "image_b64": array_to_b64(blurred),
    })

    # ── Step 5: Otsu Binarization ──
    binary = apply_otsu(clahe_img)
    steps.append({
        "name": "Otsu Binarization",
        "description": "Automatic global thresholding via Otsu's method",
        "image_b64": array_to_b64(binary),
    })

    # ── Step 6: Canny Edges ──
    edges = apply_canny(clahe_img)
    steps.append({
        "name": "Canny Edge Detection",
        "description": "Gradient-based edge extraction (low=30, high=100)",
        "image_b64": array_to_b64(edges),
    })

    # ── Step 7: Morphological Operations ──
    morph = apply_morphology(binary)
    steps.append({
        "name": "Morphological Ops",
        "description": "Opening (noise removal) + Closing (hole filling) "
                       "with 7×7 elliptical kernel",
        "image_b64": array_to_b64(morph),
    })

    # ── Step 8: Model-Ready ──
    model_ready = prepare_for_model(image, target_size)
    steps.append({
        "name": "Model-Ready",
        "description": "Normalized [0,1] float image resized to model "
                       f"input ({target_size[0]}×{target_size[1]})",
        "image_b64": array_to_b64(model_ready),
    })

    return steps


def preprocess_to_tensor(image: np.ndarray,
                          target_size: Tuple[int, int] = (224, 224)):
    """
    Full preprocessing → PyTorch tensor ready for model inference.

    Returns:
        torch.Tensor of shape (1, 3, H, W), float32, normalized
        with ImageNet mean/std.
    """
    import torch

    model_img = prepare_for_model(image, target_size)  # (H, W) float32

    # Convert to 3-channel
    rgb = np.stack([model_img] * 3, axis=0)  # (3, H, W)

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = (rgb - mean) / std

    tensor = torch.from_numpy(rgb).unsqueeze(0)  # (1, 3, H, W)
    return tensor
