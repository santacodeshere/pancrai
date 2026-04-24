"""
PancrAI — Test Suite
Tests for model architecture and preprocessing pipeline.
"""

import pytest
import numpy as np
import torch
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ─── Model Tests ──────────────────────────────────────────────────────────────

class TestTransUNet:
    """Tests for TransUNet architecture."""

    @pytest.fixture(scope="class")
    def model(self):
        from app.models.transunet import TransUNet
        return TransUNet(img_size=224, pretrained=False)

    def test_output_shape(self, model):
        """Model should output (B, 1, H, W) for binary segmentation."""
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 1, 224, 224), f"Expected (2,1,224,224), got {out.shape}"

    def test_predict_range(self, model):
        """Sigmoid output should be in [0, 1]."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            pred = model.predict(x)
        assert pred.min() >= 0.0 and pred.max() <= 1.0, \
            "Predictions outside [0, 1] range"

    def test_parameter_count(self, model):
        """Model should have a reasonable number of parameters (>10M)."""
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 10_000_000, f"Too few parameters: {n_params:,}"

    def test_gradient_flow(self, model):
        """Gradients should flow back through the entire model."""
        model.train()
        x = torch.randn(1, 3, 224, 224, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradients on input tensor"


class TestClassifier:
    """Tests for EfficientNetB4 classifier."""

    @pytest.fixture(scope="class")
    def model(self):
        from app.models.classifier import PancreasTumorClassifier
        return PancreasTumorClassifier(pretrained=False)

    def test_output_shape(self, model):
        """Should output (B, 4) logits."""
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 4), f"Expected (4,4), got {out.shape}"

    def test_predict_proba(self, model):
        """predict_proba should return valid class index and probabilities."""
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        cls_idx, probs = model.predict_proba(x)
        assert 0 <= cls_idx <= 3
        assert len(probs) == 4
        assert abs(sum(probs) - 1.0) < 1e-4, "Probabilities should sum to 1"


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessing:
    """Tests for image preprocessing pipeline."""

    @pytest.fixture
    def sample_image(self):
        """Create a synthetic grayscale test image."""
        rng = np.random.default_rng(42)
        return (rng.random((256, 256)) * 255).astype(np.uint8)

    def test_pipeline_returns_steps(self, sample_image):
        """Pipeline should return 8 steps."""
        from app.services.preprocessing import run_full_pipeline
        steps = run_full_pipeline(sample_image)
        assert len(steps) == 8, f"Expected 8 steps, got {len(steps)}"

    def test_pipeline_step_keys(self, sample_image):
        """Each step should have name, description, image_b64."""
        from app.services.preprocessing import run_full_pipeline
        steps = run_full_pipeline(sample_image)
        for step in steps:
            assert "name" in step
            assert "description" in step
            assert "image_b64" in step
            assert len(step["image_b64"]) > 100  # should be non-empty base64

    def test_preprocess_to_tensor_shape(self, sample_image):
        """Should return (1, 3, 224, 224) tensor."""
        from app.services.preprocessing import preprocess_to_tensor
        tensor = preprocess_to_tensor(sample_image)
        assert tensor.shape == (1, 3, 224, 224)
        assert tensor.dtype == torch.float32

    def test_clahe(self, sample_image):
        """CLAHE output should be same shape as input."""
        from app.services.preprocessing import apply_clahe
        result = apply_clahe(sample_image)
        assert result.shape == sample_image.shape

    def test_load_from_bytes_png(self):
        """Should load PNG bytes correctly."""
        from app.services.preprocessing import load_from_bytes
        from PIL import Image
        import io

        # Create a small PNG in memory
        pil = Image.fromarray(np.zeros((64, 64), dtype=np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        loaded = load_from_bytes(img_bytes, "test.png")
        assert loaded is not None
        assert loaded.shape == (64, 64)


# ─── Segmentation Tests ───────────────────────────────────────────────────────

class TestSegmentation:
    """Tests for segmentation service."""

    def test_measure_tumor_empty(self):
        """measure_tumor should return None for empty mask."""
        from app.services.segmentation import measure_tumor
        empty_mask = np.zeros((224, 224), dtype=np.uint8)
        result = measure_tumor(empty_mask, (224, 224))
        assert result is None

    def test_measure_tumor_with_blob(self):
        """measure_tumor should return measurements for a mask with a blob."""
        from app.services.segmentation import measure_tumor
        mask = np.zeros((224, 224), dtype=np.uint8)
        # Add a circular blob
        import cv2
        cv2.circle(mask, (112, 112), 30, 255, -1)

        result = measure_tumor(mask, (224, 224))
        assert result is not None
        assert result["area_pixels"] > 0
        assert result["area_cm2"] > 0
        assert 80 < result["centroid_x"] < 140
        assert 80 < result["centroid_y"] < 140

    def test_compute_mask_metrics_empty(self):
        """Empty mask should give (0, 0) metrics."""
        from app.services.segmentation import compute_mask_metrics
        empty = np.zeros((224, 224), dtype=np.uint8)
        dice, iou = compute_mask_metrics(empty)
        assert dice == 0.0
        assert iou == 0.0


# ─── Loss Function Tests ──────────────────────────────────────────────────────

class TestLosses:
    """Tests for loss functions."""

    def test_dice_loss_perfect(self):
        """Perfect prediction should give ~0 Dice loss."""
        from ml.losses import DiceLoss
        criterion = DiceLoss()
        # Simulate perfect prediction with high logits where mask=1
        target = torch.ones(2, 1, 32, 32)
        logits = torch.ones(2, 1, 32, 32) * 10  # sigmoid ≈ 1
        loss = criterion(logits, target)
        assert loss.item() < 0.05, f"Perfect prediction loss too high: {loss.item()}"

    def test_combined_loss_runs(self):
        """Combined loss should run without error."""
        from ml.losses import CombinedDiceBCELoss
        criterion = CombinedDiceBCELoss()
        logits = torch.randn(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        loss = criterion(logits, target)
        assert loss.item() > 0
        assert not torch.isnan(loss)


# ─── Metrics Tests ────────────────────────────────────────────────────────────

class TestMetrics:
    """Tests for segmentation metrics."""

    def test_dice_perfect(self):
        """Identical pred and target → Dice ≈ 1.0."""
        from ml.metrics import dice_score
        target = torch.ones(1, 1, 64, 64) * 0.9
        d = dice_score(target, target)
        assert d > 0.99

    def test_dice_no_overlap(self):
        """Non-overlapping masks → Dice = 0."""
        from ml.metrics import dice_score
        pred = torch.zeros(1, 1, 64, 64)
        pred[:, :, :32, :] = 0.9
        target = torch.zeros(1, 1, 64, 64)
        target[:, :, 32:, :] = 0.9
        d = dice_score(pred, target)
        assert d < 0.05

    def test_iou_bounds(self):
        """IoU should be in [0, 1]."""
        from ml.metrics import iou_score
        pred = torch.rand(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()
        iou = iou_score(pred, target)
        assert 0.0 <= iou <= 1.0

    def test_sensitivity_all_positive(self):
        """Predicting all positive with positive target → sensitivity = 1."""
        from ml.metrics import sensitivity
        pred = torch.ones(1, 1, 32, 32)
        target = torch.ones(1, 1, 32, 32)
        s = sensitivity(pred, target)
        assert s > 0.99
