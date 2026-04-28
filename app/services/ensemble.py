"""
PancrAI — Ensemble Inference
Combines TransUNet (primary) with a lightweight U-Net (secondary model)
for more robust segmentation predictions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── Lightweight U-Net (secondary model) ───────────────────────────────────────

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class LightUNet(nn.Module):
    """
    Lightweight U-Net with 1/4 the parameters of standard U-Net.
    Used as secondary model in ensemble.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[16,32,64,128]):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools    = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.upsamples= nn.ModuleList()

        # Encoder
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            self.pools.append(nn.MaxPool2d(2))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for f in reversed(features):
            self.upsamples.append(nn.ConvTranspose2d(f*2, f, 2, stride=2))
            self.decoders.append(DoubleConv(f*2, f))

        self.final = nn.Conv2d(features[0], out_channels, 1)
        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        skips = skips[::-1]
        for up, dec, skip in zip(self.upsamples, self.decoders, skips):
            x = up(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.final(x)


def build_light_unet(weights_path: Optional[str] = None,
                      device: Optional[torch.device] = None) -> LightUNet:
    """Build and optionally load lightweight U-Net."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LightUNet(in_channels=3, out_channels=1)
    model.to(device)

    if weights_path and __import__("pathlib").Path(weights_path).exists():
        try:
            ckpt = torch.load(weights_path, map_location=device)
            state = ckpt.get("model", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"[LightUNet] Loaded weights from {weights_path}")
        except Exception as e:
            print(f"[LightUNet] Could not load weights: {e}. Using random init.")
    else:
        print("[LightUNet] No weights found — using randomly initialized LightUNet")

    model.eval()
    return model


# ── Ensemble Inference ────────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_inference(
    primary_model: nn.Module,
    secondary_model: nn.Module,
    tensor: torch.Tensor,
    primary_weight: float = 0.70,
    secondary_weight: float = 0.30,
    threshold: float = 0.5,
) -> dict:
    """
    Run ensemble inference combining TransUNet + LightUNet.

    Args:
        primary_model: TransUNet (main model)
        secondary_model: LightUNet (secondary)
        tensor: Input tensor (1, 3, H, W)
        primary_weight: Weight for primary model output (0-1)
        secondary_weight: Weight for secondary model output (0-1)
        threshold: Segmentation threshold

    Returns:
        Dict with ensemble prediction, individual predictions, agreement map
    """
    device = next(primary_model.parameters()).device
    tensor = tensor.to(device)

    primary_model.eval()
    secondary_model.eval()

    # Primary model prediction
    primary_logits = primary_model(tensor)
    primary_prob   = torch.sigmoid(primary_logits).squeeze().cpu().numpy()

    # Secondary model prediction
    try:
        secondary_logits = secondary_model(tensor)
        secondary_prob   = torch.sigmoid(secondary_logits).squeeze().cpu().numpy()
    except Exception as e:
        print(f"[Ensemble] Secondary model failed: {e}. Using primary only.")
        secondary_prob = primary_prob.copy()
        primary_weight = 1.0
        secondary_weight = 0.0

    # Weighted ensemble
    ensemble_prob = (primary_weight * primary_prob +
                     secondary_weight * secondary_prob)

    # Binary masks
    primary_mask   = (primary_prob   > threshold).astype(np.uint8)
    secondary_mask = (secondary_prob > threshold).astype(np.uint8)
    ensemble_mask  = (ensemble_prob  > threshold).astype(np.uint8)

    # Agreement map: where both models agree
    agreement = (primary_mask == secondary_mask).astype(np.float32)
    agreement_score = float(agreement.mean() * 100)

    # Disagreement regions (where models differ)
    disagreement = (primary_mask != secondary_mask).astype(np.uint8)

    # Ensemble improves over primary alone?
    primary_pixels   = int(primary_mask.sum())
    secondary_pixels = int(secondary_mask.sum())
    ensemble_pixels  = int(ensemble_mask.sum())

    return {
        "ensemble_prob":     ensemble_prob,
        "ensemble_mask":     ensemble_mask,
        "primary_prob":      primary_prob,
        "primary_mask":      primary_mask,
        "secondary_prob":    secondary_prob,
        "secondary_mask":    secondary_mask,
        "agreement_map":     agreement,
        "disagreement_map":  disagreement,
        "agreement_score":   agreement_score,
        "primary_pixels":    primary_pixels,
        "secondary_pixels":  secondary_pixels,
        "ensemble_pixels":   ensemble_pixels,
        "primary_weight":    primary_weight,
        "secondary_weight":  secondary_weight,
    }


def run_ensemble_segmentation(
    primary_model: nn.Module,
    secondary_model: nn.Module,
    tensor: torch.Tensor,
    image_np: np.ndarray,
    primary_weight: float = 0.70,
) -> dict:
    """
    Full ensemble segmentation pipeline with overlay generation.

    Returns complete segmentation result dict compatible with
    the existing segmentation service interface.
    """
    import cv2
    import base64
    from io import BytesIO
    from PIL import Image as PILImage

    ens = ensemble_inference(
        primary_model, secondary_model, tensor,
        primary_weight=primary_weight,
        secondary_weight=1.0 - primary_weight,
    )

    mask = (ens["ensemble_mask"] * 255).astype(np.uint8)
    prob = ens["ensemble_prob"]

    # Generate overlay
    if len(image_np.shape) == 2:
        display = cv2.cvtColor(
            (np.clip(image_np, 0, 1) * 255).astype(np.uint8)
            if image_np.max() <= 1 else image_np.astype(np.uint8),
            cv2.COLOR_GRAY2BGR
        )
    else:
        display = image_np.copy()

    h, w = display.shape[:2]
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = display.copy()
    tumor_region = mask_resized > 127

    measurements = None
    if tumor_region.sum() > 0:
        overlay[tumor_region] = [0, 0, 200]
        result_img = cv2.addWeighted(display, 0.6, overlay, 0.4, 0)

        ys, xs = np.where(tumor_region)
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        cx, cy = int(xs.mean()), int(ys.mean())

        cv2.rectangle(result_img, (x1, y1), (x2, y2), (0, 200, 200), 2)
        cv2.drawMarker(result_img, (cx, cy), (0, 255, 0),
                       cv2.MARKER_CROSS, 20, 2)

        tumor_pixels = int(tumor_region.sum())
        total_pixels = h * w
        mm_per_pixel = 300.0 / mask.shape[0]
        cm_per_pixel_sq = (mm_per_pixel / 10.0) ** 2

        measurements = {
            "area_pixels": tumor_pixels,
            "area_pct":    round(tumor_pixels / total_pixels * 100, 2),
            "area_cm2":    round(tumor_pixels * cm_per_pixel_sq, 3),
            "centroid_x":  cx,
            "centroid_y":  cy,
            "bbox_x": x1, "bbox_y": y1,
            "bbox_w": int(x2-x1), "bbox_h": int(y2-y1),
            "aspect_ratio": round(float(x2-x1)/max(float(y2-y1),1), 3),
        }
    else:
        result_img = display.copy()

    pil = PILImage.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    buf = BytesIO()
    pil.save(buf, format="PNG")
    overlay_b64 = base64.b64encode(buf.getvalue()).decode()

    # Agreement heatmap
    agr_map = (ens["agreement_map"] * 255).astype(np.uint8)
    agr_colored = cv2.applyColorMap(agr_map, cv2.COLORMAP_SUMMER)
    pil_agr = PILImage.fromarray(cv2.cvtColor(agr_colored, cv2.COLOR_BGR2RGB))
    buf2 = BytesIO()
    pil_agr.save(buf2, format="PNG")
    agreement_b64 = base64.b64encode(buf2.getvalue()).decode()

    return {
        "mask":             mask,
        "prob_map":         prob,
        "overlay_b64":      overlay_b64,
        "agreement_b64":    agreement_b64,
        "agreement_score":  ens["agreement_score"],
        "measurements":     measurements,
        "dice_score":       0.0,
        "iou_score":        0.0,
        "ensemble_enabled": True,
        "primary_pixels":   ens["primary_pixels"],
        "secondary_pixels": ens["secondary_pixels"],
        "ensemble_pixels":  ens["ensemble_pixels"],
    }


if __name__ == "__main__":
    print("Testing ensemble inference...")
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    light = build_light_unet(device=device)
    n_params = sum(p.numel() for p in light.parameters())
    print(f"LightUNet parameters: {n_params:,}")

    dummy = torch.randn(1, 3, 224, 224).to(device)
    out = light(dummy)
    print(f"LightUNet output shape: {out.shape}")
    print("Ensemble module OK!")
