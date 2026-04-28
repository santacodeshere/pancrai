"""
PancrAI — Interactive NIfTI Slice Viewer
Allows users to browse through slices of a NIfTI volume interactively.
Runs segmentation on the selected slice in real time.
"""

import numpy as np
import cv2
import base64
import io
from PIL import Image as PILImage
from typing import Optional, Tuple


def load_nifti_volume(file_bytes: bytes) -> Optional[np.ndarray]:
    """
    Load a NIfTI volume from bytes.

    Args:
        file_bytes: Raw bytes of .nii or .nii.gz file

    Returns:
        3D numpy array (H, W, N_slices) or None on failure
    """
    try:
        import nibabel as nib
        import tempfile, os

        # Write to temp file (nibabel needs file path)
        suffix = ".nii.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        img = nib.load(tmp_path)
        vol = img.get_fdata().astype(np.float32)
        os.unlink(tmp_path)

        # Ensure 3D
        if vol.ndim == 4:
            vol = vol[:, :, :, 0]
        if vol.ndim == 2:
            vol = vol[:, :, np.newaxis]

        return vol

    except Exception as e:
        print(f"[NIfTI Viewer] Error loading volume: {e}")
        return None


def normalize_ct_slice(sl: np.ndarray,
                        window_center: int = 40,
                        window_width: int = 400) -> np.ndarray:
    """Apply CT soft tissue windowing and normalize to uint8."""
    lo = window_center - window_width / 2
    hi = window_center + window_width / 2
    sl_clipped = np.clip(sl, lo, hi)
    sl_norm = (sl_clipped - lo) / (hi - lo)
    return (sl_norm * 255).astype(np.uint8)


def slice_to_b64(sl: np.ndarray) -> str:
    """Convert a 2D numpy array to base64 PNG string."""
    if sl.max() <= 1.0:
        sl_uint8 = (sl * 255).astype(np.uint8)
    else:
        sl_uint8 = sl.astype(np.uint8)

    sl_resized = cv2.resize(sl_uint8, (512, 512), interpolation=cv2.INTER_LINEAR)
    pil = PILImage.fromarray(sl_resized)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_slice_info(volume: np.ndarray, slice_idx: int) -> dict:
    """
    Get metadata about a specific slice.

    Returns:
        Dict with slice statistics
    """
    sl = volume[:, :, slice_idx]
    return {
        "slice_idx":    slice_idx,
        "n_slices":     volume.shape[2],
        "mean_hu":      round(float(sl.mean()), 1),
        "std_hu":       round(float(sl.std()), 1),
        "min_hu":       round(float(sl.min()), 1),
        "max_hu":       round(float(sl.max()), 1),
        "shape":        f"{sl.shape[0]} x {sl.shape[1]}",
    }


def render_nifti_viewer(
    volume: np.ndarray,
    slice_idx: int,
    seg_model=None,
    axis: int = 2,
) -> dict:
    """
    Render a single slice from a NIfTI volume with optional segmentation.

    Args:
        volume: 3D numpy array (H, W, N)
        slice_idx: Index of slice to render
        seg_model: Optional TransUNet model for real-time segmentation
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)

    Returns:
        Dict with slice_b64, overlay_b64, has_tumor, measurements
    """
    # Extract slice
    if axis == 2:
        sl = volume[:, :, slice_idx]
    elif axis == 1:
        sl = volume[:, slice_idx, :]
    else:
        sl = volume[slice_idx, :, :]

    # Normalize
    sl_norm = normalize_ct_slice(sl)
    slice_b64 = slice_to_b64(sl_norm)

    result = {
        "slice_b64":   slice_b64,
        "overlay_b64": slice_b64,  # default: no segmentation
        "has_tumor":   False,
        "measurements": None,
        "tumor_class": "N/A",
        "confidence":  0.0,
    }

    if seg_model is None:
        return result

    # Run segmentation on this slice
    try:
        import torch
        from app.services.preprocessing import preprocess_to_tensor
        from app.services.segmentation import run_segmentation
        from app.models.classifier import classify_from_mask

        # Convert slice to format expected by preprocessing
        sl_3ch = cv2.cvtColor(
            cv2.resize(sl_norm, (224, 224)),
            cv2.COLOR_GRAY2BGR
        )

        seg = run_segmentation(sl_norm, seg_model)
        mask = seg.get("mask")

        if mask is not None:
            cls_result = classify_from_mask(mask)
            result.update({
                "overlay_b64": seg.get("overlay_b64", slice_b64),
                "has_tumor":   cls_result["class_idx"] > 0,
                "measurements": seg.get("measurements"),
                "tumor_class": cls_result["class_name"],
                "confidence":  cls_result["confidence"],
                "dice_score":  seg.get("dice_score", 0.0),
            })

    except Exception as e:
        print(f"[NIfTI Viewer] Segmentation error on slice {slice_idx}: {e}")

    return result


def find_tumor_slices(volume: np.ndarray, seg_model,
                       axis: int = 2,
                       sample_rate: int = 5) -> list:
    """
    Scan volume to find slices likely containing tumor.
    Samples every `sample_rate` slices for efficiency.

    Args:
        volume: 3D numpy array
        seg_model: TransUNet model
        axis: Slicing axis
        sample_rate: Check every Nth slice

    Returns:
        List of slice indices where tumor was detected
    """
    n_slices = volume.shape[axis]
    tumor_slices = []

    try:
        from app.services.segmentation import run_segmentation
        from app.models.classifier import classify_from_mask

        for i in range(0, n_slices, sample_rate):
            if axis == 2:
                sl = volume[:, :, i]
            elif axis == 1:
                sl = volume[:, i, :]
            else:
                sl = volume[i, :, :]

            sl_norm = normalize_ct_slice(sl)
            seg = run_segmentation(sl_norm, seg_model)
            mask = seg.get("mask")

            if mask is not None:
                cls = classify_from_mask(mask)
                if cls["class_idx"] > 0 and cls["confidence"] > 0.55:
                    tumor_slices.append(i)

    except Exception as e:
        print(f"[NIfTI Viewer] Scan error: {e}")

    return tumor_slices


def render_slice_thumbnail_grid(
    volume: np.ndarray,
    slice_indices: list,
    axis: int = 2,
) -> list:
    """
    Generate thumbnail base64 images for a list of slice indices.

    Args:
        volume: 3D numpy array
        slice_indices: List of slice indices to render
        axis: Slicing axis

    Returns:
        List of dicts with {idx, b64, mean_intensity}
    """
    thumbnails = []
    for i in slice_indices:
        if axis == 2:
            sl = volume[:, :, i]
        elif axis == 1:
            sl = volume[:, i, :]
        else:
            sl = volume[i, :, :]

        sl_norm = normalize_ct_slice(sl)
        sl_small = cv2.resize(sl_norm, (128, 128))
        pil = PILImage.fromarray(sl_small)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()

        thumbnails.append({
            "idx":            i,
            "b64":            b64,
            "mean_intensity": round(float(sl.mean()), 1),
        })

    return thumbnails


if __name__ == "__main__":
    print("Testing NIfTI viewer utilities...")

    # Synthetic volume test
    vol = np.random.randn(256, 256, 50).astype(np.float32) * 100
    vol[100:150, 100:150, 20:30] += 200  # simulated tumor region

    info = get_slice_info(vol, 25)
    print(f"Slice info: {info}")

    sl_norm = normalize_ct_slice(vol[:, :, 25])
    b64 = slice_to_b64(sl_norm)
    print(f"Slice b64 length: {len(b64)}")
    print("NIfTI viewer utilities OK!")
