"""
PancrAI — Grad-CAM Explainability Service
Generates Grad-CAM and Grad-CAM++ heatmaps for model predictions.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional


class GradCAM:
    """
    Grad-CAM implementation for CNN-based models.

    Hooks into the target convolutional layer to capture activations
    and gradients, then generates a class activation map.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: (1, C, H, W) input to model.
            class_idx: Target class for gradient calculation.
                       If None, uses argmax of output.

        Returns:
            Heatmap as float32 numpy array (H, W), values in [0, 1].
        """
        self.model.eval()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)  # Forward pass

        # For segmentation models, sum output to get scalar
        if output.dim() == 4:
            score = output.sum()
        else:
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]

        self.model.zero_grad()
        score.backward()

        # Global average pooling of gradients
        gradients = self.gradients   # (1, C, H, W)
        activations = self.activations  # (1, C, H, W)

        weights = gradients.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
        cam = (weights * activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32)


class GradCAMPlusPlus:
    """
    Grad-CAM++ for sharper, more precise localization.
    Weights gradients with their second-order derivative.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks: List = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self._hooks.append(
            self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor: torch.Tensor,
                 class_idx: Optional[int] = None) -> np.ndarray:
        """Generate Grad-CAM++ heatmap."""
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if output.dim() == 4:
            score = output.sum()
        else:
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients       # (1, C, H, W)
        acts = self.activations      # (1, C, H, W)

        # Second-order gradient approximation
        grads_sq = grads ** 2
        grads_cube = grads ** 3
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        denom = 2 * grads_sq + sum_acts * grads_cube
        denom = torch.where(denom != 0, denom,
                            torch.ones_like(denom))
        alpha = grads_sq / denom                        # (1, C, H, W)

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


# ─── Overlay Utilities ────────────────────────────────────────────────────────

def cam_to_heatmap(cam: np.ndarray,
                   target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Resize CAM and apply JET colormap.
    Returns uint8 BGR image (H, W, 3).
    """
    cam_resized = cv2.resize(cam, target_size, interpolation=cv2.INTER_LINEAR)
    cam_uint8 = (cam_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return heatmap


def overlay_heatmap(
    original: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.40,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Overlay Grad-CAM heatmap on the original image with 40% transparency.

    Args:
        original: Grayscale image (H, W) uint8.
        cam: Normalized CAM float32 (H, W).
        alpha: Heatmap opacity.
        target_size: Output size.

    Returns:
        RGB overlay image (H, W, 3) uint8.
    """
    orig_resized = cv2.resize(original, target_size)
    orig_rgb = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
    heatmap = cam_to_heatmap(cam, target_size)
    blended = cv2.addWeighted(orig_rgb, 1 - alpha, heatmap, alpha, 0)
    return blended


def draw_tumor_annotations(
    img: np.ndarray,
    measurements: Dict[str, Any],
) -> np.ndarray:
    """
    Draw bounding box, centroid marker, and measurement labels on image.

    Args:
        img: RGB image (H, W, 3) uint8.
        measurements: Dict from measure_tumor().

    Returns:
        Annotated image.
    """
    if not measurements:
        return img

    annotated = img.copy()
    bx = measurements.get("bbox_x", 0)
    by = measurements.get("bbox_y", 0)
    bw = measurements.get("bbox_w", 0)
    bh = measurements.get("bbox_h", 0)
    cx = int(measurements.get("centroid_x", 0))
    cy = int(measurements.get("centroid_y", 0))
    area = measurements.get("area_cm2", 0.0)

    # Bounding box (yellow)
    cv2.rectangle(annotated, (bx, by), (bx + bw, by + bh),
                  (255, 215, 0), 2, cv2.LINE_AA)

    # Centroid cross (bright green)
    cv2.drawMarker(annotated, (cx, cy), (0, 255, 100),
                   cv2.MARKER_CROSS, markerSize=20, thickness=2)

    # Labels
    labels = [
        f"Area: {area:.2f} cm2",
        f"Centroid: ({cx},{cy})",
        f"Size: {bw}x{bh} px",
        f"AR: {measurements.get('aspect_ratio', 0):.2f}",
    ]
    for i, lbl in enumerate(labels):
        y_pos = max(by - 10 + i * 16, 12 + i * 16)
        cv2.putText(annotated, lbl, (bx + 2, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (255, 255, 100), 1, cv2.LINE_AA)

    return annotated


# ─── High-level Interface ─────────────────────────────────────────────────────

def generate_gradcam_results(
    image: np.ndarray,
    model: torch.nn.Module,
    measurements: Optional[Dict[str, Any]] = None,
    target_size: tuple = (224, 224),
    device: str = "cpu",
) -> Dict[str, str]:
    """
    Generate both Grad-CAM and Grad-CAM++ heatmaps.

    Args:
        image: Input grayscale image (H, W).
        model: TransUNet or classifier model.
        measurements: Tumor measurement dict for annotation.
        target_size: Output resolution.
        device: Inference device.

    Returns:
        Dict with base64-encoded PNG strings:
          - gradcam_b64: Grad-CAM overlay
          - gradcampp_b64: Grad-CAM++ overlay
    """
    import io, base64
    from PIL import Image

    model.eval()
    model.to(device)

    # Get the last CNN conv layer for hooking
    target_layer = _get_last_conv_layer(model)

    from app.services.preprocessing import preprocess_to_tensor
    tensor = preprocess_to_tensor(image, target_size).to(device)

    results = {}

    # ── Grad-CAM ──
    try:
        gcam = GradCAM(model, target_layer)
        cam = gcam.generate(tensor.clone())
        gcam.remove_hooks()
        overlay = overlay_heatmap(image, cam, alpha=0.4, target_size=target_size)
        if measurements:
            overlay = draw_tumor_annotations(overlay, measurements)
        results["gradcam_b64"] = _ndarray_to_b64(overlay)
    except Exception as e:
        # Fallback: return plain image if hooking fails
        import traceback
        traceback.print_exc()
        fallback = cv2.cvtColor(
            cv2.resize(image, target_size), cv2.COLOR_GRAY2RGB)
        results["gradcam_b64"] = _ndarray_to_b64(fallback)

    # ── Grad-CAM++ ──
    try:
        gcampp = GradCAMPlusPlus(model, target_layer)
        cam_pp = gcampp.generate(tensor.clone())
        gcampp.remove_hooks()
        overlay_pp = overlay_heatmap(image, cam_pp, alpha=0.4, target_size=target_size)
        if measurements:
            overlay_pp = draw_tumor_annotations(overlay_pp, measurements)
        results["gradcampp_b64"] = _ndarray_to_b64(overlay_pp)
    except Exception:
        results["gradcampp_b64"] = results["gradcam_b64"]

    return results


def _get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Find the last Conv2d layer in the model for Grad-CAM hooking."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in model.")
    return last_conv


def _ndarray_to_b64(img: np.ndarray) -> str:
    """Convert RGB numpy array to base64 PNG string."""
    import io, base64
    from PIL import Image
    pil = Image.fromarray(img.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")
