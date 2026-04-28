"""
PancrAI — Test Time Augmentation (TTA) + Post-processing
Improves segmentation quality without retraining.

TTA: Run inference on 8 augmented versions of the scan,
average predictions for more robust segmentation.

Post-processing:
- Remove small false positive components
- Fill holes in segmentation mask
- Smooth boundaries with morphological operations
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional


# ── Test Time Augmentation ─────────────────────────────────────────────────────

def _augment(tensor: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Apply one of 8 deterministic augmentations to input tensor."""
    t = tensor.clone()
    # Flips
    if aug_idx in [1, 3, 5, 7]:
        t = torch.flip(t, dims=[3])   # horizontal flip
    if aug_idx in [2, 3, 6, 7]:
        t = torch.flip(t, dims=[2])   # vertical flip
    # 90-degree rotations
    if aug_idx in [4, 5, 6, 7]:
        t = torch.rot90(t, k=1, dims=[2, 3])
    return t


def _deaugment(pred: torch.Tensor, aug_idx: int) -> torch.Tensor:
    """Reverse the augmentation on the prediction."""
    p = pred.clone()
    # Reverse rotation first
    if aug_idx in [4, 5, 6, 7]:
        p = torch.rot90(p, k=-1, dims=[2, 3])
    # Reverse flips
    if aug_idx in [2, 3, 6, 7]:
        p = torch.flip(p, dims=[2])
    if aug_idx in [1, 3, 5, 7]:
        p = torch.flip(p, dims=[3])
    return p


@torch.no_grad()
def tta_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    n_aug: int = 8,
    threshold: float = 0.5,
) -> dict:
    """
    Run Test Time Augmentation inference.

    Args:
        model: TransUNet segmentation model
        tensor: Input tensor (1, 3, H, W)
        n_aug: Number of augmentations (1-8)
        threshold: Segmentation threshold

    Returns:
        Dict with mean_prob, binary_mask, improvement_note
    """
    model.eval()
    device = next(model.parameters()).device
    tensor = tensor.to(device)

    prob_sum = None

    for i in range(n_aug):
        aug_tensor = _augment(tensor, i)
        logits = model(aug_tensor)
        prob = torch.sigmoid(logits)
        prob = _deaugment(prob, i)

        if prob_sum is None:
            prob_sum = prob
        else:
            prob_sum = prob_sum + prob

    mean_prob = prob_sum / n_aug

    # Binary mask from averaged probabilities
    binary = (mean_prob > threshold).squeeze().cpu().numpy().astype(np.uint8)

    return {
        "mean_prob": mean_prob.squeeze().cpu().numpy(),
        "binary_mask": binary,
        "n_augmentations": n_aug,
    }


# ── Post-processing ────────────────────────────────────────────────────────────

def postprocess_mask(
    mask: np.ndarray,
    min_area_pct: float = 0.1,
    fill_holes: bool = True,
    smooth: bool = True,
) -> np.ndarray:
    """
    Clean up segmentation mask with post-processing.

    Steps:
        1. Remove connected components smaller than min_area_pct of image
        2. Fill holes within tumor region
        3. Smooth boundaries with morphological closing

    Args:
        mask: Binary uint8 mask (H, W), values 0 or 1
        min_area_pct: Minimum component area as % of image (default 0.1%)
        fill_holes: Fill internal holes in mask
        smooth: Apply morphological smoothing

    Returns:
        Cleaned binary mask (H, W), uint8
    """
    if mask.max() > 1:
        binary = (mask > 127).astype(np.uint8)
    else:
        binary = mask.astype(np.uint8)

    h, w = binary.shape
    total_pixels = h * w
    min_pixels = int(total_pixels * min_area_pct / 100.0)

    # Step 1: Remove small connected components
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    cleaned = np.zeros_like(binary)
    for label_idx in range(1, n_labels):  # skip background (0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_pixels:
            cleaned[labels == label_idx] = 1

    # Step 2: Fill holes
    if fill_holes and cleaned.sum() > 0:
        # Flood fill from corners to find background
        flood = cleaned.copy()
        mask_ff = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, mask_ff, (0, 0), 1)
        # Holes are where flood didn't reach but original was 0
        holes = (flood == 0) & (cleaned == 0)
        cleaned[holes] = 1

    # Step 3: Smooth boundaries
    if smooth and cleaned.sum() > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    return cleaned.astype(np.uint8)


# ── Combined TTA + Postprocess pipeline ───────────────────────────────────────

def run_tta_segmentation(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    image_np: np.ndarray,
    n_aug: int = 8,
    threshold: float = 0.5,
    min_area_pct: float = 0.1,
) -> dict:
    """
    Full TTA + post-processing segmentation pipeline.

    Args:
        model: TransUNet model
        tensor: Input tensor (1, 3, H, W)
        image_np: Original image as numpy array for overlay generation
        n_aug: Number of TTA augmentations
        threshold: Segmentation probability threshold
        min_area_pct: Minimum tumor area to keep (removes false positives)

    Returns:
        Dict with all segmentation outputs
    """
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    # Run TTA
    tta_result = tta_inference(model, tensor, n_aug=n_aug, threshold=threshold)
    raw_mask = tta_result["binary_mask"]

    # Post-process
    clean_mask = postprocess_mask(
        raw_mask,
        min_area_pct=min_area_pct,
        fill_holes=True,
        smooth=True,
    )

    # Resize to display size
    h, w = image_np.shape[:2]
    display_mask = cv2.resize(
        (clean_mask * 255).astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # Generate overlay
    if len(image_np.shape) == 2 or image_np.shape[2] == 1:
        display_img = cv2.cvtColor(
            (image_np * 255).astype(np.uint8)
            if image_np.max() <= 1 else image_np.astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    else:
        display_img = image_np.copy()

    overlay = display_img.copy()
    tumor_region = display_mask > 127

    if tumor_region.sum() > 0:
        overlay[tumor_region] = [0, 0, 200]  # Red tumor
        result_img = cv2.addWeighted(display_img, 0.6, overlay, 0.4, 0)

        # Bounding box
        ys, xs = np.where(tumor_region)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 200, 200), 2)

        # Centroid
        cx, cy = int(xs.mean()), int(ys.mean())
        cv2.drawMarker(result_img, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)
    else:
        result_img = display_img.copy()

    # Encode overlay to base64
    pil_img = PILImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    # Compute metrics
    tumor_pixels = int(clean_mask.sum())
    total_pixels = clean_mask.shape[0] * clean_mask.shape[1]
    area_pct = (tumor_pixels / total_pixels) * 100

    mm_per_pixel = 300.0 / clean_mask.shape[0]
    cm_per_pixel_sq = (mm_per_pixel / 10.0) ** 2
    area_cm2 = round(tumor_pixels * cm_per_pixel_sq, 3)

    measurements = None
    if tumor_pixels > 5:
        ys, xs = np.where(clean_mask > 0)
        measurements = {
            "area_pixels": tumor_pixels,
            "area_pct":    round(area_pct, 2),
            "area_cm2":    area_cm2,
            "centroid_x":  round(float(xs.mean()), 1),
            "centroid_y":  round(float(ys.mean()), 1),
            "bbox_x":      int(xs.min()),
            "bbox_y":      int(ys.min()),
            "bbox_w":      int(xs.max() - xs.min()),
            "bbox_h":      int(ys.max() - ys.min()),
            "aspect_ratio": round(
                float(xs.max()-xs.min()) / max(float(ys.max()-ys.min()), 1), 3
            ),
        }

    return {
        "mask":          (display_mask).astype(np.uint8),
        "prob_map":      tta_result["mean_prob"],
        "overlay_b64":   overlay_b64,
        "measurements":  measurements,
        "dice_score":    0.0,   # No ground truth at inference
        "iou_score":     0.0,
        "n_augmentations": n_aug,
        "tta_enabled":   True,
    }


if __name__ == "__main__":
    print("TTA + Post-processing module loaded successfully.")
    print("Functions available:")
    print("  tta_inference()       — Run TTA on a tensor")
    print("  postprocess_mask()    — Clean up segmentation mask")
    print("  run_tta_segmentation() — Full TTA + postprocess pipeline")
