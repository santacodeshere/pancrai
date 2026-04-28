"""
PancrAI — Radiomics Feature Extraction
Extracts 50+ quantitative imaging features from tumor region.

Feature categories:
    1. Shape features (10)      — geometric properties of tumor mask
    2. Intensity features (10)  — statistical properties of tumor pixel values
    3. Texture / GLCM (16)     — Gray Level Co-occurrence Matrix features
    4. LBP features (8)        — Local Binary Pattern texture features
    5. Wavelet features (8)    — Frequency-domain texture features
    6. Gradient features (5)   — Edge and boundary features

Total: ~57 features
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _to_binary(mask: np.ndarray) -> np.ndarray:
    if mask.max() > 1:
        return (mask > 127).astype(np.uint8)
    return (mask > 0.5).astype(np.uint8)


def _get_tumor_pixels(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Extract pixel intensity values within tumor region."""
    binary = _to_binary(mask)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)

    # Resize mask to match image
    if binary.shape != gray.shape:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    pixels = gray[binary > 0].astype(np.float64)
    return pixels


# ── 1. Shape Features ──────────────────────────────────────────────────────────

def extract_shape_features(mask: np.ndarray) -> Dict[str, float]:
    """Extract 10 geometric shape features from segmentation mask."""
    binary = _to_binary(mask)
    h, w = binary.shape
    total = h * w

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    if not contours or binary.sum() == 0:
        return {f"shape_{k}": 0.0 for k in [
            "area_pct", "perimeter", "circularity", "solidity",
            "aspect_ratio", "extent", "compactness", "sphericity",
            "convexity", "elongation"
        ]}

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    hull_perimeter = cv2.arcLength(hull, True)

    x, y, bw, bh = cv2.boundingRect(cnt)

    # Fit ellipse for elongation
    if len(cnt) >= 5:
        (ex, ey), (ma, mi), angle = cv2.fitEllipse(cnt)
        elongation = float(mi / max(ma, 1e-6))
    else:
        elongation = 1.0

    circularity  = float(4 * np.pi * area / max(perimeter**2, 1e-6))
    solidity     = float(area / max(hull_area, 1e-6))
    aspect_ratio = float(max(bw, bh) / max(min(bw, bh), 1))
    extent       = float(area / max(bw * bh, 1))
    compactness  = float(perimeter**2 / max(4 * np.pi * area, 1e-6))
    sphericity   = float(np.sqrt(4 * np.pi * area) / max(perimeter, 1e-6))
    convexity    = float(hull_perimeter / max(perimeter, 1e-6))

    return {
        "shape_area_pct":    round(binary.sum() / total * 100, 3),
        "shape_perimeter":   round(float(perimeter), 2),
        "shape_circularity": round(np.clip(circularity, 0, 1), 4),
        "shape_solidity":    round(np.clip(solidity, 0, 1), 4),
        "shape_aspect_ratio":round(aspect_ratio, 4),
        "shape_extent":      round(np.clip(extent, 0, 1), 4),
        "shape_compactness": round(compactness, 4),
        "shape_sphericity":  round(np.clip(sphericity, 0, 1), 4),
        "shape_convexity":   round(np.clip(convexity, 0, 1), 4),
        "shape_elongation":  round(np.clip(elongation, 0, 1), 4),
    }


# ── 2. Intensity Features ──────────────────────────────────────────────────────

def extract_intensity_features(image: np.ndarray,
                                mask: np.ndarray) -> Dict[str, float]:
    """Extract 10 first-order intensity statistics from tumor ROI."""
    pixels = _get_tumor_pixels(image, mask)

    if len(pixels) == 0:
        return {f"intensity_{k}": 0.0 for k in [
            "mean", "std", "min", "max", "range",
            "skewness", "kurtosis", "entropy", "energy", "median"
        ]}

    mean     = float(np.mean(pixels))
    std      = float(np.std(pixels))
    mn       = float(np.min(pixels))
    mx       = float(np.max(pixels))
    rng      = mx - mn
    median   = float(np.median(pixels))

    # Skewness
    if std > 0:
        skewness = float(np.mean(((pixels - mean) / std) ** 3))
        kurtosis = float(np.mean(((pixels - mean) / std) ** 4) - 3)
    else:
        skewness = 0.0
        kurtosis = 0.0

    # Entropy and energy from normalized histogram
    hist, _ = np.histogram(pixels, bins=64, range=(0, 255))
    hist_norm = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-9)))
    energy  = float(np.sum(hist_norm ** 2))

    return {
        "intensity_mean":     round(mean, 3),
        "intensity_std":      round(std, 3),
        "intensity_min":      round(mn, 3),
        "intensity_max":      round(mx, 3),
        "intensity_range":    round(rng, 3),
        "intensity_skewness": round(skewness, 4),
        "intensity_kurtosis": round(kurtosis, 4),
        "intensity_entropy":  round(entropy, 4),
        "intensity_energy":   round(energy, 6),
        "intensity_median":   round(median, 3),
    }


# ── 3. GLCM Texture Features ───────────────────────────────────────────────────

def _compute_glcm(gray_roi: np.ndarray, distance: int = 1,
                   levels: int = 32) -> np.ndarray:
    """Compute Gray Level Co-occurrence Matrix."""
    # Quantize to fewer levels for efficiency
    quantized = (gray_roi / 255.0 * (levels - 1)).astype(np.int32)
    quantized = np.clip(quantized, 0, levels - 1)

    glcm = np.zeros((levels, levels), dtype=np.float64)

    # 4 directions: 0°, 45°, 90°, 135°
    directions = [(0, distance), (-distance, distance),
                  (-distance, 0), (-distance, -distance)]

    h, w = quantized.shape
    for dr, dc in directions:
        for r in range(max(0, -dr), min(h, h - dr)):
            for c in range(max(0, -dc), min(w, w - dc)):
                i = quantized[r, c]
                j = quantized[r + dr, c + dc]
                glcm[i, j] += 1
                glcm[j, i] += 1

    total = glcm.sum()
    if total > 0:
        glcm /= total
    return glcm


def extract_glcm_features(image: np.ndarray,
                           mask: np.ndarray) -> Dict[str, float]:
    """Extract 16 GLCM texture features from tumor ROI."""
    binary = _to_binary(mask)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    if binary.shape != gray.shape:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    # Extract ROI bounding box
    ys, xs = np.where(binary > 0)
    if len(ys) < 10:
        return {f"glcm_{k}": 0.0 for k in [
            "contrast", "dissimilarity", "homogeneity", "energy",
            "correlation", "asm", "entropy", "variance",
            "sum_average", "sum_variance", "sum_entropy",
            "diff_average", "diff_variance", "diff_entropy",
            "max_prob", "cluster_shade"
        ]}

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    roi = gray[y1:y2, x1:x2]

    # Use fast approximation for large ROIs
    if roi.size > 10000:
        step = max(1, int(np.sqrt(roi.size / 5000)))
        roi = roi[::step, ::step]

    levels = 16
    glcm = _compute_glcm(roi, distance=1, levels=levels)

    i_idx, j_idx = np.mgrid[0:levels, 0:levels].astype(float)
    mu_i = np.sum(i_idx * glcm)
    mu_j = np.sum(j_idx * glcm)
    std_i = np.sqrt(np.sum(glcm * (i_idx - mu_i)**2) + 1e-9)
    std_j = np.sqrt(np.sum(glcm * (j_idx - mu_j)**2) + 1e-9)

    contrast      = float(np.sum(glcm * (i_idx - j_idx)**2))
    dissimilarity = float(np.sum(glcm * np.abs(i_idx - j_idx)))
    homogeneity   = float(np.sum(glcm / (1 + (i_idx - j_idx)**2)))
    energy        = float(np.sum(glcm**2))
    asm           = energy
    correlation   = float(np.sum(glcm * (i_idx - mu_i) * (j_idx - mu_j)) / (std_i * std_j))
    entropy       = float(-np.sum(glcm * np.log2(glcm + 1e-9)))
    variance      = float(np.sum(glcm * (i_idx - mu_i)**2))
    max_prob      = float(glcm.max())

    # Sum and difference features
    sum_diag  = np.array([np.sum(np.diag(glcm, k) + np.diag(glcm, -k))
                           for k in range(levels)])
    diff_diag = np.array([np.abs(np.sum(glcm * (np.abs(i_idx - j_idx) == k)))
                           for k in range(levels)])
    sum_diag  = sum_diag / (sum_diag.sum() + 1e-9)
    diff_diag = diff_diag / (diff_diag.sum() + 1e-9)

    k_vals = np.arange(len(sum_diag), dtype=float)
    sum_average  = float(np.sum(k_vals * sum_diag))
    sum_variance = float(np.sum((k_vals - sum_average)**2 * sum_diag))
    sum_entropy  = float(-np.sum(sum_diag * np.log2(sum_diag + 1e-9)))

    k_vals2 = np.arange(len(diff_diag), dtype=float)
    diff_average  = float(np.sum(k_vals2 * diff_diag))
    diff_variance = float(np.sum((k_vals2 - diff_average)**2 * diff_diag))
    diff_entropy  = float(-np.sum(diff_diag * np.log2(diff_diag + 1e-9)))

    cluster_shade = float(np.sum(glcm * (i_idx + j_idx - mu_i - mu_j)**3))

    return {
        "glcm_contrast":      round(contrast, 4),
        "glcm_dissimilarity": round(dissimilarity, 4),
        "glcm_homogeneity":   round(homogeneity, 4),
        "glcm_energy":        round(energy, 6),
        "glcm_correlation":   round(np.clip(correlation, -1, 1), 4),
        "glcm_asm":           round(asm, 6),
        "glcm_entropy":       round(entropy, 4),
        "glcm_variance":      round(variance, 4),
        "glcm_sum_average":   round(sum_average, 4),
        "glcm_sum_variance":  round(sum_variance, 4),
        "glcm_sum_entropy":   round(sum_entropy, 4),
        "glcm_diff_average":  round(diff_average, 4),
        "glcm_diff_variance": round(diff_variance, 4),
        "glcm_diff_entropy":  round(diff_entropy, 4),
        "glcm_max_prob":      round(max_prob, 6),
        "glcm_cluster_shade": round(cluster_shade, 4),
    }


# ── 4. LBP Features ───────────────────────────────────────────────────────────

def extract_lbp_features(image: np.ndarray,
                          mask: np.ndarray) -> Dict[str, float]:
    """Extract 8 Local Binary Pattern features from tumor ROI."""
    binary = _to_binary(mask)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    if binary.shape != gray.shape:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(binary > 0)
    if len(ys) < 10:
        return {f"lbp_{k}": 0.0 for k in [
            "mean", "std", "energy", "entropy",
            "uniformity", "contrast", "smoothness", "third_moment"
        ]}

    # Manual LBP computation (radius=1, 8 neighbors)
    h, w = gray.shape
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]

    for idx, (nr, nc) in enumerate(neighbors):
        r_idx = np.clip(np.arange(h)[:, None] + nr, 0, h-1)
        c_idx = np.clip(np.arange(w)[None, :] + nc, 0, w-1)
        lbp += ((gray[r_idx, c_idx] >= gray).astype(np.uint8) << idx)

    # Extract LBP values in tumor region
    lbp_vals = lbp[binary > 0].astype(np.float64)

    if len(lbp_vals) == 0:
        return {f"lbp_{k}": 0.0 for k in [
            "mean", "std", "energy", "entropy",
            "uniformity", "contrast", "smoothness", "third_moment"
        ]}

    hist, _ = np.histogram(lbp_vals, bins=32, range=(0, 255))
    hist_norm = hist / (hist.sum() + 1e-9)

    mean         = float(np.mean(lbp_vals))
    std          = float(np.std(lbp_vals))
    energy       = float(np.sum(hist_norm**2))
    entropy      = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-9)))
    uniformity   = float(np.sum(hist_norm**2))
    contrast     = float(np.var(lbp_vals))
    smoothness   = float(1 - 1 / (1 + std**2 / 255**2))
    third_moment = float(np.mean(((lbp_vals - mean) / (std + 1e-9))**3))

    return {
        "lbp_mean":         round(mean, 3),
        "lbp_std":          round(std, 3),
        "lbp_energy":       round(energy, 6),
        "lbp_entropy":      round(entropy, 4),
        "lbp_uniformity":   round(uniformity, 6),
        "lbp_contrast":     round(contrast, 3),
        "lbp_smoothness":   round(smoothness, 4),
        "lbp_third_moment": round(third_moment, 4),
    }


# ── 5. Wavelet Features ────────────────────────────────────────────────────────

def extract_wavelet_features(image: np.ndarray,
                              mask: np.ndarray) -> Dict[str, float]:
    """Extract 8 wavelet-based texture features using Haar decomposition."""
    binary = _to_binary(mask)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    if binary.shape != gray.shape:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(binary > 0)
    if len(ys) < 16:
        return {f"wavelet_{k}": 0.0 for k in [
            "ll_mean", "ll_std", "lh_energy", "hl_energy",
            "hh_energy", "lh_entropy", "hl_entropy", "hh_entropy"
        ]}

    # Haar wavelet decomposition (manual, no pywavelets needed)
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    roi = gray[y1:y2, x1:x2].astype(np.float64)

    # Ensure even dimensions
    if roi.shape[0] % 2 != 0:
        roi = roi[:-1, :]
    if roi.shape[1] % 2 != 0:
        roi = roi[:, :-1]

    if roi.shape[0] < 4 or roi.shape[1] < 4:
        return {f"wavelet_{k}": 0.0 for k in [
            "ll_mean", "ll_std", "lh_energy", "hl_energy",
            "hh_energy", "lh_entropy", "hl_entropy", "hh_entropy"
        ]}

    # Haar transform
    h, w = roi.shape
    h2, w2 = h // 2, w // 2

    # Row-wise
    lo_row = (roi[:, 0::2] + roi[:, 1::2]) / 2
    hi_row = (roi[:, 0::2] - roi[:, 1::2]) / 2

    # Column-wise
    LL = (lo_row[0::2, :] + lo_row[1::2, :]) / 2
    LH = (lo_row[0::2, :] - lo_row[1::2, :]) / 2
    HL = (hi_row[0::2, :] + hi_row[1::2, :]) / 2
    HH = (hi_row[0::2, :] - hi_row[1::2, :]) / 2

    def _energy(arr):
        return float(np.sum(arr**2) / max(arr.size, 1))

    def _entropy(arr):
        hist, _ = np.histogram(arr, bins=32)
        p = hist / (hist.sum() + 1e-9)
        return float(-np.sum(p * np.log2(p + 1e-9)))

    return {
        "wavelet_ll_mean":    round(float(np.mean(LL)), 3),
        "wavelet_ll_std":     round(float(np.std(LL)), 3),
        "wavelet_lh_energy":  round(_energy(LH), 4),
        "wavelet_hl_energy":  round(_energy(HL), 4),
        "wavelet_hh_energy":  round(_energy(HH), 4),
        "wavelet_lh_entropy": round(_entropy(LH), 4),
        "wavelet_hl_entropy": round(_entropy(HL), 4),
        "wavelet_hh_entropy": round(_entropy(HH), 4),
    }


# ── 6. Gradient / Edge Features ───────────────────────────────────────────────

def extract_gradient_features(image: np.ndarray,
                               mask: np.ndarray) -> Dict[str, float]:
    """Extract 5 gradient-based boundary and edge features."""
    binary = _to_binary(mask)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    if gray.max() <= 1.0:
        gray = (gray * 255).astype(np.uint8)
    if binary.shape != gray.shape:
        binary = cv2.resize(binary, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

    ys, xs = np.where(binary > 0)
    if len(ys) < 5:
        return {f"gradient_{k}": 0.0 for k in [
            "mean_magnitude", "std_magnitude",
            "boundary_contrast", "edge_density", "gradient_entropy"
        ]}

    # Sobel gradients
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(gx**2 + gy**2)

    # Tumor region gradients
    tumor_grads = magnitude[binary > 0]

    # Boundary contrast (gradient at tumor boundary vs interior)
    kernel = np.ones((3, 3), np.uint8)
    boundary = cv2.dilate(binary, kernel) - binary
    boundary_grads = magnitude[boundary > 0] if boundary.sum() > 0 else tumor_grads

    hist, _ = np.histogram(tumor_grads, bins=32)
    p = hist / (hist.sum() + 1e-9)
    entropy = float(-np.sum(p * np.log2(p + 1e-9)))

    edge_map = cv2.Canny(gray, 30, 100)
    edge_density = float((edge_map[binary > 0] > 0).mean())

    return {
        "gradient_mean_magnitude":   round(float(np.mean(tumor_grads)), 3),
        "gradient_std_magnitude":    round(float(np.std(tumor_grads)), 3),
        "gradient_boundary_contrast":round(float(np.mean(boundary_grads)), 3),
        "gradient_edge_density":     round(edge_density, 4),
        "gradient_entropy":          round(entropy, 4),
    }


# ── Master extraction function ─────────────────────────────────────────────────

def extract_all_radiomics(
    image: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    """
    Extract all 57 radiomics features from tumor region.

    Args:
        image: Original CT scan image (H, W) or (H, W, 3), uint8 or float
        mask: Segmentation mask (H, W), binary or uint8

    Returns:
        Dict of 57 named radiomics features
    """
    features = {}
    features.update(extract_shape_features(mask))
    features.update(extract_intensity_features(image, mask))
    features.update(extract_glcm_features(image, mask))
    features.update(extract_lbp_features(image, mask))
    features.update(extract_wavelet_features(image, mask))
    features.update(extract_gradient_features(image, mask))
    return features


def get_radiomics_summary(features: Dict[str, float]) -> Dict[str, Dict]:
    """
    Group features by category for display.

    Returns:
        Dict of category -> {feature_name: value}
    """
    categories = {
        "Shape":     {k: v for k, v in features.items() if k.startswith("shape_")},
        "Intensity": {k: v for k, v in features.items() if k.startswith("intensity_")},
        "GLCM Texture": {k: v for k, v in features.items() if k.startswith("glcm_")},
        "LBP Texture":  {k: v for k, v in features.items() if k.startswith("lbp_")},
        "Wavelet":   {k: v for k, v in features.items() if k.startswith("wavelet_")},
        "Gradient":  {k: v for k, v in features.items() if k.startswith("gradient_")},
    }
    return categories


def get_clinical_interpretation(features: Dict[str, float]) -> Dict[str, str]:
    """
    Provide clinical interpretation of key radiomics features.

    Returns:
        Dict of feature -> clinical meaning
    """
    interpretations = {}

    circularity = features.get("shape_circularity", 0)
    if circularity > 0.85:
        interpretations["Morphology"] = "Highly circular — consistent with cystic lesion"
    elif circularity > 0.65:
        interpretations["Morphology"] = "Moderately round — consistent with well-defined benign tumor"
    else:
        interpretations["Morphology"] = "Irregular shape — may indicate malignant infiltration"

    solidity = features.get("shape_solidity", 0)
    if solidity < 0.75:
        interpretations["Border Regularity"] = "Low solidity — irregular borders, possible malignancy indicator"
    elif solidity > 0.90:
        interpretations["Border Regularity"] = "High solidity — smooth, well-defined borders"
    else:
        interpretations["Border Regularity"] = "Moderate solidity — borderline border regularity"

    entropy = features.get("intensity_entropy", 0)
    if entropy > 5.5:
        interpretations["Heterogeneity"] = "High entropy — heterogeneous tumor texture"
    elif entropy > 4.0:
        interpretations["Heterogeneity"] = "Moderate entropy — mixed texture pattern"
    else:
        interpretations["Heterogeneity"] = "Low entropy — homogeneous tumor texture"

    contrast = features.get("glcm_contrast", 0)
    if contrast > 50:
        interpretations["Texture Contrast"] = "High contrast texture — sharp internal boundaries"
    elif contrast > 20:
        interpretations["Texture Contrast"] = "Moderate texture contrast"
    else:
        interpretations["Texture Contrast"] = "Low contrast texture — smooth internal pattern"

    edge_density = features.get("gradient_edge_density", 0)
    if edge_density > 0.3:
        interpretations["Edge Complexity"] = "High edge density — complex internal structure"
    else:
        interpretations["Edge Complexity"] = "Low edge density — smooth internal structure"

    return interpretations


if __name__ == "__main__":
    print("Testing radiomics extraction...\n")

    # Synthetic test
    test_image = np.random.randint(80, 180, (224, 224), dtype=np.uint8)
    test_mask  = np.zeros((224, 224), dtype=np.uint8)
    import cv2 as _cv2
    _cv2.ellipse(test_mask, (112, 112), (40, 30), 0, 0, 360, 255, -1)

    features = extract_all_radiomics(test_image, test_mask)
    print(f"Total features extracted: {len(features)}")

    summary = get_radiomics_summary(features)
    for cat, feats in summary.items():
        print(f"\n{cat} ({len(feats)} features):")
        for k, v in list(feats.items())[:3]:
            print(f"  {k}: {v}")
        if len(feats) > 3:
            print(f"  ... and {len(feats)-3} more")

    print("\nClinical interpretations:")
    interp = get_clinical_interpretation(features)
    for k, v in interp.items():
        print(f"  {k}: {v}")
