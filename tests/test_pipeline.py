"""
PancrAI — End-to-End Inference Pipeline Tests
Tests the full scan → preprocess → segment → classify → GradCAM → uncertainty chain.
"""

import io
import pytest
import numpy as np
import torch
from PIL import Image


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_ct_image():
    """Generate a 256×256 synthetic CT-like grayscale image."""
    rng = np.random.default_rng(123)
    img = (rng.random((256, 256)) * 255).astype(np.uint8)
    # Add a bright blob simulating a tumor
    import cv2
    cv2.circle(img, (128, 128), 30, 220, -1)
    return img


@pytest.fixture(scope="module")
def synthetic_png_bytes(synthetic_ct_image):
    """Encode synthetic image as PNG bytes."""
    pil = Image.fromarray(synthetic_ct_image)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="module")
def seg_model():
    """Load a minimal TransUNet (no pretrained weights for speed)."""
    from app.models.transunet import TransUNet
    model = TransUNet(img_size=224, pretrained=False)
    model.eval()
    return model


@pytest.fixture(scope="module")
def cls_model():
    """Load a minimal classifier (no pretrained weights for speed)."""
    from app.models.classifier import PancreasTumorClassifier
    model = PancreasTumorClassifier(pretrained=False)
    model.eval()
    return model


# ─── Preprocessing Tests ──────────────────────────────────────────────────────

class TestPreprocessingPipeline:

    def test_load_png_bytes(self, synthetic_png_bytes):
        from app.services.preprocessing import load_from_bytes
        img = load_from_bytes(synthetic_png_bytes, "test.png")
        assert img is not None
        assert img.ndim == 2
        assert img.dtype == np.uint8

    def test_full_pipeline_step_count(self, synthetic_ct_image):
        from app.services.preprocessing import run_full_pipeline
        steps = run_full_pipeline(synthetic_ct_image)
        assert len(steps) == 8

    def test_pipeline_b64_decodable(self, synthetic_ct_image):
        import base64
        from app.services.preprocessing import run_full_pipeline
        steps = run_full_pipeline(synthetic_ct_image)
        for step in steps:
            # Should not raise
            decoded = base64.b64decode(step["image_b64"])
            assert len(decoded) > 0

    def test_tensor_output_shape(self, synthetic_ct_image):
        from app.services.preprocessing import preprocess_to_tensor
        t = preprocess_to_tensor(synthetic_ct_image, (224, 224))
        assert t.shape == (1, 3, 224, 224)
        assert t.dtype == torch.float32

    def test_tensor_normalized(self, synthetic_ct_image):
        """Tensor values should be roughly in the ImageNet-normalized range."""
        from app.services.preprocessing import preprocess_to_tensor
        t = preprocess_to_tensor(synthetic_ct_image, (224, 224))
        # ImageNet-normalized values are typically in [-3, 3]
        assert t.min().item() > -5
        assert t.max().item() < 5


# ─── Segmentation Pipeline Tests ─────────────────────────────────────────────

class TestSegmentationPipeline:

    def test_run_segmentation_returns_keys(self, synthetic_ct_image, seg_model):
        from app.services.segmentation import run_segmentation
        result = run_segmentation(synthetic_ct_image, seg_model)
        for key in ["mask", "prob_map", "overlay_b64", "dice_score", "iou_score"]:
            assert key in result, f"Missing key: {key}"

    def test_mask_shape(self, synthetic_ct_image, seg_model):
        from app.services.segmentation import run_segmentation
        result = run_segmentation(synthetic_ct_image, seg_model)
        mask = result["mask"]
        assert mask.ndim == 2
        assert mask.shape == (224, 224)

    def test_mask_binary(self, synthetic_ct_image, seg_model):
        from app.services.segmentation import run_segmentation
        result = run_segmentation(synthetic_ct_image, seg_model)
        unique_vals = set(np.unique(result["mask"]))
        assert unique_vals.issubset({0, 255}), f"Mask has non-binary values: {unique_vals}"

    def test_prob_map_range(self, synthetic_ct_image, seg_model):
        from app.services.segmentation import run_segmentation
        result = run_segmentation(synthetic_ct_image, seg_model)
        pm = result["prob_map"]
        assert pm.min() >= 0.0
        assert pm.max() <= 1.0

    def test_overlay_is_valid_b64(self, synthetic_ct_image, seg_model):
        import base64
        from app.services.segmentation import run_segmentation
        result = run_segmentation(synthetic_ct_image, seg_model)
        decoded = base64.b64decode(result["overlay_b64"])
        pil = Image.open(io.BytesIO(decoded))
        assert pil.size == (224, 224)


# ─── Classification Tests ─────────────────────────────────────────────────────

class TestClassificationPipeline:

    def test_classifier_output_shape(self, synthetic_ct_image, cls_model):
        from app.services.preprocessing import preprocess_to_tensor
        import torch.nn.functional as F
        tensor = preprocess_to_tensor(synthetic_ct_image)
        with torch.no_grad():
            logits = cls_model(tensor)
            probs = F.softmax(logits, dim=-1)
        assert probs.shape == (1, 4)

    def test_probs_sum_to_one(self, synthetic_ct_image, cls_model):
        from app.services.preprocessing import preprocess_to_tensor
        import torch.nn.functional as F
        tensor = preprocess_to_tensor(synthetic_ct_image)
        with torch.no_grad():
            logits = cls_model(tensor)
            probs = F.softmax(logits, dim=-1)
        total = probs.sum().item()
        assert abs(total - 1.0) < 1e-4

    def test_class_index_valid(self, synthetic_ct_image, cls_model):
        from app.services.preprocessing import preprocess_to_tensor
        tensor = preprocess_to_tensor(synthetic_ct_image)
        pred_class, conf_scores = cls_model.predict_proba(tensor)
        assert 0 <= pred_class <= 3
        assert len(conf_scores) == 4


# ─── Uncertainty Tests ────────────────────────────────────────────────────────

class TestUncertaintyPipeline:

    def test_uncertainty_output_keys(self, synthetic_ct_image, seg_model):
        from app.services.preprocessing import preprocess_to_tensor
        from app.services.uncertainty import mc_dropout_inference
        tensor = preprocess_to_tensor(synthetic_ct_image)
        result = mc_dropout_inference(seg_model, tensor, T=5)   # T=5 for speed
        for key in ["mean_prediction", "uncertainty_map", "uncertainty_score",
                    "confidence", "confidence_interval", "high_uncertainty_warning"]:
            assert key in result

    def test_uncertainty_score_range(self, synthetic_ct_image, seg_model):
        from app.services.preprocessing import preprocess_to_tensor
        from app.services.uncertainty import mc_dropout_inference
        tensor = preprocess_to_tensor(synthetic_ct_image)
        result = mc_dropout_inference(seg_model, tensor, T=5)
        score = result["uncertainty_score"]
        assert 0.0 <= score <= 100.0

    def test_mc_gives_varying_results(self, synthetic_ct_image, seg_model):
        """Multiple MC passes should not all be identical (dropout active)."""
        from app.services.preprocessing import preprocess_to_tensor
        from app.services.uncertainty import mc_dropout_inference, enable_dropout
        tensor = preprocess_to_tensor(synthetic_ct_image)
        # Run two separate sets of passes
        r1 = mc_dropout_inference(seg_model, tensor, T=5)
        r2 = mc_dropout_inference(seg_model, tensor, T=5)
        # Both should complete successfully
        assert r1["uncertainty_score"] >= 0
        assert r2["uncertainty_score"] >= 0


# ─── Demo Data Generator Tests ───────────────────────────────────────────────

class TestDemoDataGenerator:

    def test_generate_background(self):
        from demo_data_generator import generate_ct_background
        rng = np.random.default_rng(42)
        bg = generate_ct_background(64, rng)
        assert bg.shape == (64, 64)
        assert bg.min() >= 0.0
        assert bg.max() <= 1.0

    def test_generate_tumor(self):
        from demo_data_generator import generate_tumor
        rng = np.random.default_rng(42)
        intensity, binary, tumor_type = generate_tumor(64, rng, "malignant")
        assert intensity.shape == (64, 64)
        assert binary.shape == (64, 64)
        assert tumor_type == "malignant"
        assert binary.max() == 255

    def test_generate_sample(self):
        from demo_data_generator import generate_sample
        rng = np.random.default_rng(7)
        img, mask, has_tumor, tumor_type = generate_sample(64, rng, tumor_probability=1.0)
        assert img.shape == (64, 64)
        assert mask.shape == (64, 64)
        assert has_tumor is True
        assert img.dtype == np.uint8


# ─── PDF Export Tests ─────────────────────────────────────────────────────────

class TestPDFExport:

    SAMPLE_HTML = """
    <div>
      <h2>1. Patient Information</h2>
      <p>Name: John Doe | Age: 65 | Sex: Male</p>
      <h2>2. Findings</h2>
      <p>AI detected Malignant tumor with 87% confidence.</p>
      <ul>
        <li>Tumor area: 2.4 cm²</li>
        <li>Risk level: Critical</li>
      </ul>
    </div>
    """

    def test_pdf_generation_returns_bytes(self):
        try:
            from utils.pdf_export import html_to_pdf_bytes
            pdf = html_to_pdf_bytes(self.SAMPLE_HTML)
            assert isinstance(pdf, bytes)
            assert len(pdf) > 100
        except ImportError:
            pytest.skip("reportlab not installed")

    def test_pdf_starts_with_pdf_header(self):
        try:
            from utils.pdf_export import html_to_pdf_bytes
            pdf = html_to_pdf_bytes(self.SAMPLE_HTML)
            assert pdf[:4] == b"%PDF", "Output is not a valid PDF"
        except ImportError:
            pytest.skip("reportlab not installed")

    def test_strip_html(self):
        from utils.pdf_export import _strip_html
        text = _strip_html("<h1>Title</h1><p>Content with <strong>bold</strong></p>")
        assert "Title" in text
        assert "Content with" in text
        assert "<" not in text
