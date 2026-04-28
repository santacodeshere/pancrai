"""
PancrAI — Grad-CAM Explainability Service
Generates Grad-CAM and Grad-CAM++ heatmaps for model predictions.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []
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

    def generate(self, input_tensor: torch.Tensor, class_idx=None):
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        # ✅ FIXED: focus on tumor regions only
        if output.dim() == 4:
            prob = torch.sigmoid(output)
            score = (prob * (prob > 0.3).float()).sum()
        else:
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]

        self.model.zero_grad()
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

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
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._hooks = []
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

    def generate(self, input_tensor: torch.Tensor, class_idx=None):
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        # ✅ SAME FIX HERE
        if output.dim() == 4:
            prob = torch.sigmoid(output)
            score = (prob * (prob > 0.3).float()).sum()
        else:
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        grads_sq = grads ** 2
        grads_cube = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        denom = 2 * grads_sq + sum_acts * grads_cube
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))

        alpha = grads_sq / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().detach().cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Layer Selection (🔥 KEY FIX)
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Select best layer for Grad-CAM in TransUNet:
    Avoids final 1x1 conv, picks last meaningful decoder conv.
    """
    decoder_convs = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            if module.out_channels > 1:  # skip output layer
                decoder_convs.append((name, module))

    if decoder_convs:
        name, layer = decoder_convs[-1]
        print(f"[GradCAM] Using layer: {name} (out_ch={layer.out_channels})")
        return layer

    # fallback
    last = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last = module

    if last is None:
        raise ValueError("No Conv2d layer found.")

    return last


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def cam_to_heatmap(cam, target_size=(224, 224)):
    cam = cv2.resize(cam, target_size)
    cam = (cam * 255).astype(np.uint8)
    return cv2.applyColorMap(cam, cv2.COLORMAP_JET)


def overlay_heatmap(original, cam, alpha=0.4, target_size=(224, 224)):
    orig = cv2.resize(original, target_size)
    orig = cv2.cvtColor(orig, cv2.COLOR_GRAY2BGR)

    heatmap = cam_to_heatmap(cam, target_size)
    return cv2.addWeighted(orig, 1 - alpha, heatmap, alpha, 0)


def draw_tumor_annotations(img, measurements):
    if not measurements:
        return img

    out = img.copy()

    bx = measurements.get("bbox_x", 0)
    by = measurements.get("bbox_y", 0)
    bw = measurements.get("bbox_w", 0)
    bh = measurements.get("bbox_h", 0)

    cx = int(measurements.get("centroid_x", 0))
    cy = int(measurements.get("centroid_y", 0))

    cv2.rectangle(out, (bx, by), (bx + bw, by + bh), (255, 215, 0), 2)
    cv2.drawMarker(out, (cx, cy), (0, 255, 100), cv2.MARKER_CROSS, 20, 2)

    return out


def _ndarray_to_b64(img):
    import base64, io
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
# Main API
# ─────────────────────────────────────────────────────────────────────────────

def generate_gradcam_results(
    image: np.ndarray,
    model: torch.nn.Module,
    measurements=None,
    target_size=(224, 224),
    device="cpu",
):
    from app.services.preprocessing import preprocess_to_tensor

    model.eval().to(device)

    # ✅ FIXED HERE
    target_layer = _get_target_layer(model)

    tensor = preprocess_to_tensor(image, target_size).to(device)

    results = {}

    try:
        gcam = GradCAM(model, target_layer)
        cam = gcam.generate(tensor.clone())
        gcam.remove_hooks()

        overlay = overlay_heatmap(image, cam, target_size=target_size)

        if measurements:
            overlay = draw_tumor_annotations(overlay, measurements)

        results["gradcam_b64"] = _ndarray_to_b64(overlay)

    except Exception:
        fallback = cv2.cvtColor(cv2.resize(image, target_size), cv2.COLOR_GRAY2RGB)
        results["gradcam_b64"] = _ndarray_to_b64(fallback)

    try:
        gcampp = GradCAMPlusPlus(model, target_layer)
        cam = gcampp.generate(tensor.clone())
        gcampp.remove_hooks()

        overlay = overlay_heatmap(image, cam, target_size=target_size)

        if measurements:
            overlay = draw_tumor_annotations(overlay, measurements)

        results["gradcampp_b64"] = _ndarray_to_b64(overlay)

    except Exception:
        results["gradcampp_b64"] = results["gradcam_b64"]

    return results