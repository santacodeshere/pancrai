"""
PancrAI — Segmentation Evaluation Metrics
Dice, IoU, Sensitivity, Specificity, Hausdorff Distance.
"""

import torch
import numpy as np


def dice_score(preds: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Dice Similarity Coefficient (DSC).
    DSC = 2|X∩Y| / (|X| + |Y|)

    Args:
        preds: Sigmoid-activated predictions (B, 1, H, W) or (B, H, W).
        targets: Binary ground truth masks, same shape.
        threshold: Binarization threshold.

    Returns:
        Scalar Dice score averaged over batch.
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    intersection = (preds_bin * targets_bin).sum(dim=(-2, -1))
    union = preds_bin.sum(dim=(-2, -1)) + targets_bin.sum(dim=(-2, -1))
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice.mean().item()


def iou_score(preds: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Intersection over Union (IoU / Jaccard Index).
    IoU = |X∩Y| / |X∪Y|
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    intersection = (preds_bin * targets_bin).sum(dim=(-2, -1))
    union = (preds_bin + targets_bin).clamp(0, 1).sum(dim=(-2, -1))
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()


def sensitivity(preds: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Sensitivity (Recall / True Positive Rate).
    Sensitivity = TP / (TP + FN)

    Measures ability to detect all tumor pixels.
    Critical metric for medical imaging — missing tumor is dangerous.
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    tp = (preds_bin * targets_bin).sum(dim=(-2, -1))
    fn = ((1 - preds_bin) * targets_bin).sum(dim=(-2, -1))
    sens = (tp + smooth) / (tp + fn + smooth)
    return sens.mean().item()


def specificity(preds: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """
    Specificity (True Negative Rate).
    Specificity = TN / (TN + FP)
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    tn = ((1 - preds_bin) * (1 - targets_bin)).sum(dim=(-2, -1))
    fp = (preds_bin * (1 - targets_bin)).sum(dim=(-2, -1))
    spec = (tn + smooth) / (tn + fp + smooth)
    return spec.mean().item()


def precision(preds: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1e-5) -> float:
    """Precision = TP / (TP + FP)."""
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    tp = (preds_bin * targets_bin).sum(dim=(-2, -1))
    fp = (preds_bin * (1 - targets_bin)).sum(dim=(-2, -1))
    prec = (tp + smooth) / (tp + fp + smooth)
    return prec.mean().item()


def hausdorff_distance(preds: torch.Tensor, targets: torch.Tensor,
                       threshold: float = 0.5) -> float:
    """
    Hausdorff Distance — maximum surface distance between prediction
    and ground truth boundaries. Lower is better.

    Computed on CPU as numpy operation (scipy required).
    Falls back to 0 if scipy not available or masks are empty.
    """
    try:
        from scipy.ndimage import distance_transform_edt

        preds_np = (preds > threshold).cpu().numpy().squeeze()
        targets_np = (targets > 0.5).cpu().numpy().squeeze()

        if preds_np.ndim > 2:
            # Batch mode — average over batch
            scores = []
            for p, t in zip(preds_np, targets_np):
                scores.append(_hd_single(p, t))
            return float(np.mean(scores))
        else:
            return _hd_single(preds_np, targets_np)

    except Exception:
        return 0.0


def _hd_single(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Hausdorff distance for a single 2D mask pair."""
    from scipy.ndimage import distance_transform_edt

    if pred.sum() == 0 or target.sum() == 0:
        return 0.0

    pred_border = _get_border(pred)
    target_border = _get_border(target)

    if pred_border.sum() == 0 or target_border.sum() == 0:
        return 0.0

    # Distance from target border to prediction
    dt_pred = distance_transform_edt(~pred_border)
    dt_target = distance_transform_edt(~target_border)

    hd_pred_to_target = dt_target[pred_border].max()
    hd_target_to_pred = dt_pred[target_border].max()

    return float(max(hd_pred_to_target, hd_target_to_pred))


def _get_border(mask: np.ndarray) -> np.ndarray:
    """Extract border pixels of a binary mask using erosion."""
    import cv2
    mask_u8 = mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(mask_u8, kernel, iterations=1)
    border = mask_u8 - eroded
    return border.astype(bool)


def compute_all_metrics(preds: torch.Tensor,
                        targets: torch.Tensor,
                        threshold: float = 0.5) -> dict:
    """Compute all metrics at once and return as dict."""
    return {
        "dice": dice_score(preds, targets, threshold),
        "iou": iou_score(preds, targets, threshold),
        "sensitivity": sensitivity(preds, targets, threshold),
        "specificity": specificity(preds, targets, threshold),
        "precision": precision(preds, targets, threshold),
        "hausdorff": hausdorff_distance(preds, targets, threshold),
    }
