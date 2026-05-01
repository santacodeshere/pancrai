"""
PancrAI — Grad-CAM Explainability Service
Generates Grad-CAM and Grad-CAM++ heatmaps for model predictions.
"""

import traceback
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Any, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _safe_cv2_size(target_size) -> Tuple[int, int]:
    """Always return (W, H) int tuple for cv2.resize."""
    if not isinstance(target_size, (tuple, list)) or len(target_size) != 2:
        return (224, 224)
    # target_size is (H, W) throughout this file — flip for cv2
    h, w = int(target_size[0]), int(target_size[1])
    return (w, h)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor,
                 target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        self.model.eval()
        # FIX: enable grad AFTER tensor is on device
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        # Works for both segmentation (4-D) and classification (2-D) outputs
        if output.dim() == 4:
            prob  = torch.sigmoid(output)
            flat  = prob.view(-1)
            k     = max(1, int(flat.numel() * 0.10))
            score = torch.topk(flat, k)[0].mean()
        else:
            score = output.max()

        self.model.zero_grad()
        score.backward()

        if self.gradients is None or self.activations is None:
            print("[GradCAM] WARNING: hooks did not fire — returning blank map")
            return np.zeros((target_size[0], target_size[1]), dtype=np.float32)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1)
        cam     = F.relu(cam).squeeze().cpu().numpy()

        # FIX: 1x1 spatial map squeezes to scalar — catch it
        if cam.ndim < 2:
            print(f"[GradCAM] WARNING: cam.shape={cam.shape} (1x1 spatial). "
                  "Hook an earlier decoder layer.")
            return np.zeros((target_size[0], target_size[1]), dtype=np.float32)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM++
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self.activations  = None
        self.gradients    = None
        self._hooks       = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor,
                 target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if output.dim() == 4:
            prob  = torch.sigmoid(output)
            flat  = prob.view(-1)
            k     = max(1, int(flat.numel() * 0.10))
            score = torch.topk(flat, k)[0].mean()
        else:
            score = output.max()

        self.model.zero_grad()
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            print("[GradCAM++] WARNING: hooks did not fire — returning blank map")
            return np.zeros((target_size[0], target_size[1]), dtype=np.float32)

        grads      = self.gradients
        acts       = self.activations
        grads_sq   = grads ** 2
        grads_cube = grads ** 3
        sum_acts   = acts.sum(dim=(2, 3), keepdim=True)

        denom   = 2 * grads_sq + sum_acts * grads_cube
        denom   = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha   = grads_sq / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1)
        cam = F.relu(cam).squeeze().detach().cpu().numpy()

        if cam.ndim < 2:
            print(f"[GradCAM++] WARNING: cam.shape={cam.shape} (1x1 spatial).")
            return np.zeros((target_size[0], target_size[1]), dtype=np.float32)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Layer Selection
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Pick the best Conv2d layer for Grad-CAM.

    Priority:
      1. Last Conv2d in a decoder/up block with kernel > 1×1 and out_channels >= 16
      2. Last Conv2d anywhere with kernel > 1×1
      3. Absolute last Conv2d (last resort)

    Skipping 1×1 convs is critical: TransUNet's final seg head is a 1×1 conv,
    producing a 1×1 spatial CAM that squeezes to a scalar → blank heatmap.
    """
    decoder_candidates = []
    general_candidates = []
    absolute_last      = None

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Conv2d):
            continue

        absolute_last = module
        kernel = module.kernel_size
        is_1x1 = (kernel == (1, 1)) or (kernel == 1)

        if not is_1x1 and module.out_channels >= 16:
            name_lower = name.lower()
            if any(k in name_lower for k in ("decoder", "dec", "up", "expand")):
                decoder_candidates.append((name, module))
            else:
                general_candidates.append((name, module))

    if decoder_candidates:
        chosen_name, chosen = decoder_candidates[-1]
        print(f"[GradCAM] Hooking decoder layer: {chosen_name}  "
              f"(out_ch={chosen.out_channels}, kernel={chosen.kernel_size})")
        return chosen

    if general_candidates:
        chosen_name, chosen = general_candidates[-1]
        print(f"[GradCAM] Hooking general layer: {chosen_name}  "
              f"(out_ch={chosen.out_channels}, kernel={chosen.kernel_size})")
        return chosen

    if absolute_last is not None:
        print("[GradCAM] WARNING: falling back to absolute last Conv2d (may be 1×1 seg head)")
        return absolute_last

    raise ValueError("[GradCAM] No Conv2d layer found in model.")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def cam_to_heatmap(cam: np.ndarray,
                   target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    cam_resized = cv2.resize(cam, _safe_cv2_size(target_size))
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    return cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)


def overlay_heatmap(original: np.ndarray,
                    cam: np.ndarray,
                    alpha: float = 0.55,
                    target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Overlay Grad-CAM heatmap on a grayscale image."""
    cv2_size = _safe_cv2_size(target_size)

    orig = cv2.resize(original, cv2_size)
    if orig.max() <= 1.0:
        orig = (orig * 255).astype(np.uint8)
    orig_bgr = cv2.cvtColor(orig.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # Enhance contrast of CAM before colouring
    cam_enh = cam.copy()
    lo, hi  = cam_enh.min(), cam_enh.max()
    if hi > lo:
        cam_enh = (cam_enh - lo) / (hi - lo)
    cam_enh = np.power(cam_enh, 0.5)   # gamma brighten mid-range

    heatmap = cam_to_heatmap(cam_enh, target_size)

    # Only paint where CAM is in top 30 %
    threshold    = np.percentile(cam_enh, 70)
    mask         = (cam_enh > threshold).astype(np.float32)
    mask_resized = cv2.resize(mask, cv2_size)[:, :, np.newaxis]

    blended = (
        orig_bgr.astype(np.float32) * (1 - alpha * mask_resized)
        + heatmap.astype(np.float32) * (alpha * mask_resized)
    ).astype(np.uint8)

    return blended


def draw_tumor_annotations(img: np.ndarray, measurements: dict) -> np.ndarray:
    if not measurements:
        return img
    out = img.copy()
    bx  = int(measurements.get("bbox_x", 0))
    by  = int(measurements.get("bbox_y", 0))
    bw  = int(measurements.get("bbox_w", 0))
    bh  = int(measurements.get("bbox_h", 0))
    cx  = int(measurements.get("centroid_x", 0))
    cy  = int(measurements.get("centroid_y", 0))
    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (255, 215, 0), 2)
    cv2.drawMarker(out, (cx, cy), (0, 255, 100), cv2.MARKER_CROSS, 20, 2)
    return out


def _ndarray_to_b64(img: np.ndarray) -> str:
    import base64
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _gray_fallback(image: np.ndarray,
                   target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    src = image.astype(np.uint8) if image.max() > 1.0 else (image * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(src, _safe_cv2_size(target_size)), cv2.COLOR_GRAY2RGB)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam_results(
    image: np.ndarray,
    model: torch.nn.Module,
    measurements=None,
    target_size: Tuple[int, int] = (224, 224),
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Returns dict with keys:
        "gradcam_b64"   — base64 PNG of Grad-CAM overlay
        "gradcampp_b64" — base64 PNG of Grad-CAM++ overlay
    """
    from app.services.preprocessing import preprocess_to_tensor

    model.eval().to(device)

    # FIX: pick a spatial decoder layer, not the 1×1 final seg head
    target_layer = _get_target_layer(model)

    # FIX: requires_grad_(True) must be called after .to(device)
    tensor = preprocess_to_tensor(image, target_size).to(device).requires_grad_(True)

    results: Dict[str, Any] = {}

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    try:
        gcam = GradCAM(model, target_layer)
        cam  = gcam.generate(tensor.clone(), target_size=target_size)
        gcam.remove_hooks()

        overlay = overlay_heatmap(image, cam, target_size=target_size)
        if measurements:
            overlay = draw_tumor_annotations(overlay, measurements)
        results["gradcam_b64"] = _ndarray_to_b64(overlay)

    except Exception:
        print("[GradCAM] Error generating heatmap:")
        traceback.print_exc()
        results["gradcam_b64"] = _ndarray_to_b64(_gray_fallback(image, target_size))

    # ── Grad-CAM++ ────────────────────────────────────────────────────────────
    try:
        gcampp = GradCAMPlusPlus(model, target_layer)
        cam    = gcampp.generate(tensor.clone(), target_size=target_size)
        gcampp.remove_hooks()

        overlay = overlay_heatmap(image, cam, target_size=target_size)
        if measurements:
            overlay = draw_tumor_annotations(overlay, measurements)
        results["gradcampp_b64"] = _ndarray_to_b64(overlay)

    except Exception:
        print("[GradCAM++] Error generating heatmap:")
        traceback.print_exc()
        results["gradcampp_b64"] = results.get("gradcam_b64", "")

    return results
