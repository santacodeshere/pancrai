"""
PancrAI — Preprocessing Service
Handles image loading, enhancement, and pipeline visualization.
"""

import io
import base64
import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Safe resize helper  (OpenCV expects (W, H), not (H, W))
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cv2_size(target_size) -> Tuple[int, int]:
    """Return a valid (W, H) tuple for cv2.resize."""
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        return (224, 224)
    h, w = int(target_size[0]), int(target_size[1])
    return (w, h)


# ─────────────────────────────────────────────────────────────────────────────
# Encoding helper
# ─────────────────────────────────────────────────────────────────────────────

def array_to_b64(arr: np.ndarray) -> str:
    """Convert a numpy array (uint8 or float [0,1]) to a base-64 PNG string."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Image loading
# ─────────────────────────────────────────────────────────────────────────────

def load_from_bytes(file_bytes: bytes, filename: str) -> np.ndarray:
    """
    Load any supported scan format (PNG/JPG/DCM/NIfTI) and return a
    2-D float32 grayscale array normalised to [0, 1].
    """
    fname = filename.lower()

    # ── DICOM ──
    if fname.endswith(".dcm"):
        try:
            import pydicom
            ds    = pydicom.dcmread(io.BytesIO(file_bytes))
            arr   = ds.pixel_array.astype(np.float32)
            lo, hi = arr.min(), arr.max()
            if hi > lo:
                arr = (arr - lo) / (hi - lo)
            if arr.ndim == 3:
                arr = arr.mean(axis=-1)
            return arr
        except Exception as e:
            print(f"[Preprocessing] DICOM load failed: {e}")

    # ── NIfTI ──
    if fname.endswith((".nii", ".nii.gz")):
        try:
            import nibabel as nib
            nib_img = nib.load(io.BytesIO(file_bytes))
            data    = nib_img.get_fdata().astype(np.float32)
            # pick the middle axial slice
            if data.ndim == 3:
                data = data[:, :, data.shape[2] // 2]
            lo, hi = data.min(), data.max()
            if hi > lo:
                data = (data - lo) / (hi - lo)
            return data
        except Exception as e:
            print(f"[Preprocessing] NIfTI load failed: {e}")

    # ── PNG / JPG / standard image ──
    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("L")
        arr = np.array(pil, dtype=np.float32) / 255.0
        return arr
    except Exception as e:
        print(f"[Preprocessing] Image load failed: {e}")

    # fallback blank
    return np.zeros((224, 224), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Individual processing steps
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(gray: np.ndarray) -> np.ndarray:
    """Apply CLAHE contrast enhancement."""
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def apply_gaussian_blur(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Apply Gaussian blur for noise reduction."""
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), 0)


def apply_otsu(gray: np.ndarray) -> np.ndarray:
    """Apply Otsu's thresholding."""
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def apply_canny(gray: np.ndarray,
                low: int = 50, high: int = 150) -> np.ndarray:
    """Apply Canny edge detection."""
    if gray.dtype != np.uint8:
        gray = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    return cv2.Canny(gray, low, high)


def apply_morphology(binary: np.ndarray) -> np.ndarray:
    """Apply morphological closing to clean up a binary mask."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


# ─────────────────────────────────────────────────────────────────────────────
# Model input preparation
# ─────────────────────────────────────────────────────────────────────────────

def prepare_for_model(gray: np.ndarray,
                      target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Full preprocessing chain that produces a float32 [0,1] image
    ready to be converted to a tensor.
    """
    enhanced = apply_clahe(gray)
    denoised = apply_gaussian_blur(enhanced, ksize=3)

    # FIX: use _safe_cv2_size so OpenCV gets (W, H)
    resized    = cv2.resize(denoised, _safe_cv2_size(target_size),
                            interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def preprocess_to_tensor(image: np.ndarray,
                         target_size: Tuple[int, int] = (224, 224)):
    """
    Convert a grayscale numpy image to a (1, 3, H, W) float32 torch tensor.
    Gradients are NOT enabled here — call .requires_grad_(True) after .to(device).
    """
    import torch

    processed = prepare_for_model(image, target_size)

    # Stack to 3 channels (model expects RGB-like input)
    rgb = np.stack([processed, processed, processed], axis=0)   # (3, H, W)
    tensor = torch.from_numpy(rgb).unsqueeze(0).float()          # (1, 3, H, W)
    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (returns visualisation steps for the UI)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(image: np.ndarray,
                      target_size: Tuple[int, int] = (224, 224)
                      ) -> List[Dict[str, str]]:
    """
    Run every preprocessing step and return a list of dicts:
        [{"name": str, "description": str, "image_b64": str}, ...]
    """
    # FIX: use _safe_cv2_size for the display resize
    display = cv2.resize(image, _safe_cv2_size(target_size),
                         interpolation=cv2.INTER_LINEAR)

    # Ensure display is uint8 for visualisation
    if display.dtype != np.uint8:
        display = (np.clip(display, 0, 1) * 255).astype(np.uint8)

    steps: List[Dict[str, str]] = []

    steps.append({
        "name":        "Original",
        "description": "Raw input scan",
        "image_b64":   array_to_b64(display),
    })

    steps.append({
        "name":        "Grayscale",
        "description": "Single-channel grayscale",
        "image_b64":   array_to_b64(display),
    })

    clahe_img = apply_clahe(display)
    steps.append({
        "name":        "CLAHE",
        "description": "Contrast-limited adaptive histogram equalisation",
        "image_b64":   array_to_b64(clahe_img),
    })

    blurred = apply_gaussian_blur(clahe_img, ksize=3)
    steps.append({
        "name":        "Blur",
        "description": "Gaussian noise reduction",
        "image_b64":   array_to_b64(blurred),
    })

    binary = apply_otsu(clahe_img)
    steps.append({
        "name":        "Otsu",
        "description": "Otsu binarisation",
        "image_b64":   array_to_b64(binary),
    })

    edges = apply_canny(clahe_img)
    steps.append({
        "name":        "Edges",
        "description": "Canny edge detection",
        "image_b64":   array_to_b64(edges),
    })

    morph = apply_morphology(binary)
    steps.append({
        "name":        "Morphology",
        "description": "Morphological closing cleanup",
        "image_b64":   array_to_b64(morph),
    })

    model_ready = prepare_for_model(image, target_size)
    steps.append({
        "name":        "Model Ready",
        "description": "Normalised float32 model input",
        "image_b64":   array_to_b64(model_ready),
    })

    return steps
