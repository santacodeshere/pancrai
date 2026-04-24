"""
PancrAI — Image Utility Functions
Conversion helpers used across the pipeline.
"""

import io
import base64
import numpy as np
import cv2
from PIL import Image
from typing import Union, Tuple


def b64_to_pil(b64_str: str) -> Image.Image:
    """Decode base64 PNG string to PIL Image."""
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes))


def pil_to_b64(pil_img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def b64_to_numpy(b64_str: str, grayscale: bool = False) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    img_bytes = base64.b64decode(b64_str)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imdecode(arr, flag)


def numpy_to_b64(arr: np.ndarray, fmt: str = ".png") -> str:
    """Encode numpy image array to base64 string."""
    success, buf = cv2.imencode(fmt, arr)
    if not success:
        raise ValueError("Failed to encode image to buffer")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def resize_keep_aspect(
    img: np.ndarray,
    max_size: int = 512,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Resize image keeping aspect ratio, with longest side = max_size."""
    h, w = img.shape[:2]
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def normalize_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalize float array to uint8 [0, 255]."""
    arr = arr.astype(np.float32)
    a_min, a_max = arr.min(), arr.max()
    if a_max > a_min:
        arr = (arr - a_min) / (a_max - a_min) * 255
    return arr.clip(0, 255).astype(np.uint8)


def create_side_by_side(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str = "Before",
    label2: str = "After",
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Create a side-by-side comparison image with labels.
    Both images are resized to `size` before concatenation.
    Returns RGB numpy array.
    """
    def prep(img):
        resized = cv2.resize(img, size)
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        return resized

    left = prep(img1)
    right = prep(img2)

    # Add separator line
    sep = np.ones((size[1], 4, 3), dtype=np.uint8) * 128

    combined = np.concatenate([left, sep, right], axis=1)

    # Add text labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(combined, label1, (10, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(combined, label2, (size[0] + 14, 20), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return combined


def create_diff_map(
    mask1: np.ndarray,
    mask2: np.ndarray,
    size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Create a visual difference map between two binary masks.

    Green: regions in mask2 but not mask1 (new tumor areas)
    Red: regions in mask1 but not mask2 (resolved/shrunk areas)
    Blue: common regions (stable tumor)
    """
    m1 = cv2.resize(mask1, size, interpolation=cv2.INTER_NEAREST) > 0
    m2 = cv2.resize(mask2, size, interpolation=cv2.INTER_NEAREST) > 0

    diff_img = np.zeros((*size, 3), dtype=np.uint8)

    # Stable (both)
    both = m1 & m2
    diff_img[both] = [100, 100, 255]   # blue

    # New growth
    new_areas = m2 & ~m1
    diff_img[new_areas] = [50, 220, 50]  # green

    # Resolved
    resolved = m1 & ~m2
    diff_img[resolved] = [220, 50, 50]   # red

    return diff_img
