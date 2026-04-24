"""
PancrAI — Uncertainty Quantification via Monte Carlo Dropout
Runs T=20 stochastic forward passes with dropout enabled.
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple


def enable_dropout(model: nn.Module):
    """Enable all Dropout layers for MC inference."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()   # train() mode enables dropout even during eval


def disable_dropout(model: nn.Module):
    """Restore all layers to eval mode."""
    model.eval()


def mc_dropout_inference(
    model: nn.Module,
    input_tensor: torch.Tensor,
    T: int = 20,
    device: str = "cpu",
    is_segmentation: bool = True,
) -> Dict[str, Any]:
    """
    Monte Carlo Dropout inference.

    Performs T stochastic forward passes with dropout enabled,
    then computes mean prediction and variance (uncertainty).

    Args:
        model: TransUNet or classifier model.
        input_tensor: (1, C, H, W) preprocessed input tensor.
        T: Number of stochastic forward passes.
        device: 'cpu' or 'cuda'.
        is_segmentation: True for segmentation, False for classification.

    Returns:
        Dict with:
          - mean_prediction: averaged output (H, W) or (num_classes,)
          - uncertainty_map: variance per pixel / class (same shape as mean)
          - uncertainty_score: scalar 0–100 representing overall uncertainty
          - confidence_interval: string like "Model is 87.3% confident"
          - high_uncertainty_warning: bool
          - uncertainty_heatmap_b64: base64 PNG of uncertainty map
    """
    model.to(device)
    model.eval()
    enable_dropout(model)   # Keep dropout active

    input_tensor = input_tensor.to(device)
    predictions = []

    with torch.no_grad():
        for _ in range(T):
            out = model(input_tensor)  # (1, C, H, W) or (1, num_classes)
            if is_segmentation:
                out = torch.sigmoid(out)
            else:
                out = torch.softmax(out, dim=-1)
            predictions.append(out.squeeze(0).cpu().numpy())

    disable_dropout(model)

    preds = np.stack(predictions, axis=0)     # (T, ...) 

    mean_pred = preds.mean(axis=0)            # mean over T passes
    variance = preds.var(axis=0)              # variance = epistemic uncertainty

    # Scalar uncertainty score (0–100)
    uncertainty_score = float(np.clip(variance.mean() * 400, 0.0, 100.0))

    # For segmentation: take spatial uncertainty map
    if is_segmentation:
        # variance shape: (1, H, W) or (H, W)
        if mean_pred.ndim == 3:
            unc_map = variance.squeeze(0)     # (H, W)
            mean_out = mean_pred.squeeze(0)   # (H, W)
        else:
            unc_map = variance
            mean_out = mean_pred
    else:
        unc_map = variance                    # (num_classes,)
        mean_out = mean_pred                  # (num_classes,)

    uncertainty_heatmap_b64 = _uncertainty_heatmap(unc_map)

    high_uncertainty = uncertainty_score > 60.0
    confidence = round(100.0 - uncertainty_score, 1)
    confidence_msg = (
        f"Model is {confidence:.1f}% confident in this prediction."
    )
    if high_uncertainty:
        warning_msg = (
            "⚠️ High uncertainty detected — recommend specialist review."
        )
    else:
        warning_msg = None

    return {
        "mean_prediction": mean_out,
        "uncertainty_map": unc_map,
        "uncertainty_score": round(uncertainty_score, 2),
        "confidence": confidence,
        "confidence_interval": confidence_msg,
        "high_uncertainty_warning": high_uncertainty,
        "warning_message": warning_msg,
        "uncertainty_heatmap_b64": uncertainty_heatmap_b64,
    }


def _uncertainty_heatmap(uncertainty_map: np.ndarray) -> str:
    """
    Encode uncertainty map as a JET colormap PNG base64 string.

    High variance regions appear red, low variance appear blue.
    """
    import io, base64
    from PIL import Image

    unc = uncertainty_map.astype(np.float32)

    # Normalize to [0, 255]
    u_min, u_max = unc.min(), unc.max()
    if u_max > u_min:
        unc_norm = ((unc - u_min) / (u_max - u_min) * 255).astype(np.uint8)
    else:
        unc_norm = np.zeros_like(unc, dtype=np.uint8)

    if unc_norm.ndim == 1:
        # Classification uncertainty — create bar-style image
        h, w = 40, max(200, len(unc_norm) * 50)
        unc_img = np.zeros((h, w), dtype=np.uint8)
        bar_w = w // len(unc_norm)
        for i, val in enumerate(unc_norm):
            unc_img[:, i * bar_w: (i + 1) * bar_w] = int(val)
    else:
        unc_img = unc_norm

    heatmap = cv2.applyColorMap(unc_img, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(heatmap_rgb)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_uncertainty_color(score: float) -> str:
    """Return a hex color for the uncertainty score badge in the UI."""
    if score < 25:
        return "#4CAF50"   # green — low uncertainty
    elif score < 50:
        return "#FF9800"   # orange — moderate
    elif score < 75:
        return "#FF5722"   # deep orange — high
    else:
        return "#F44336"   # red — very high
