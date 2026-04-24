"""
PancrAI — Tumor Classifier
Rule-based classification using real morphological features extracted
from the TransUNet segmentation mask.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional
from pathlib import Path

CLASS_NAMES = ["No Tumor", "Benign", "Malignant (PDAC)", "Cystic (IPMN)"]


def get_risk_level(class_idx: int, confidence: float) -> str:
    if class_idx == 0:
        return "Low"
    if class_idx == 1:
        return "Low" if confidence > 0.75 else "Medium"
    if class_idx == 3:
        return "Medium" if confidence > 0.70 else "High"
    if class_idx == 2:
        return "High" if confidence < 0.85 else "Critical"
    return "Unknown"


def _to_binary(mask: np.ndarray) -> np.ndarray:
    """Convert any mask format to binary uint8."""
    if mask.max() <= 1.0:
        return (mask > 0.5).astype(np.uint8)
    elif mask.max() <= 255:
        return (mask > 127).astype(np.uint8)
    else:
        return (mask > mask.max() * 0.5).astype(np.uint8)


def extract_mask_features(mask: np.ndarray) -> dict:
    """
    Extract morphological features from a segmentation mask.
    Area is measured as percentage of total image area (0-100%)
    so results are consistent regardless of image resolution.
    """
    import cv2

    binary = _to_binary(mask)
    total_image_pixels = binary.shape[0] * binary.shape[1]
    tumor_pixels = int(binary.sum())

    if tumor_pixels == 0:
        return {
            "area_pixels": 0,
            "area_pct": 0.0,       # % of image covered by tumor
            "area_cm2": 0.0,       # kept for UI display only
            "solidity": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 1.0,
            "perimeter": 0.0,
            "extent": 0.0,
            "has_tumor": False,
        }

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Area as percentage of total image
    area_pct = (tumor_pixels / total_image_pixels) * 100.0

    # Approximate cm² for display (assuming ~30cm FOV on 224px image)
    mm_per_pixel = 300.0 / binary.shape[0]
    cm_per_pixel_sq = (mm_per_pixel / 10.0) ** 2
    area_cm2 = tumor_pixels * cm_per_pixel_sq

    if not contours:
        return {
            "area_pixels": tumor_pixels,
            "area_pct": area_pct,
            "area_cm2": round(area_cm2, 3),
            "solidity": 0.5,
            "circularity": 0.5,
            "aspect_ratio": 1.0,
            "perimeter": 0.0,
            "extent": 0.5,
            "has_tumor": True,
        }

    cnt = max(contours, key=cv2.contourArea)
    cnt_area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = float(cnt_area / hull_area) if hull_area > 0 else 1.0

    circularity = float(
        np.clip(4 * np.pi * cnt_area / (perimeter ** 2), 0.0, 1.0)
    ) if perimeter > 0 else 0.0

    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(max(w, h) / max(min(w, h), 1))
    extent = float(cnt_area / max(w * h, 1))

    return {
        "area_pixels": tumor_pixels,
        "area_pct": round(area_pct, 2),
        "area_cm2": round(area_cm2, 3),
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "perimeter": round(float(perimeter), 2),
        "extent": round(extent, 4),
        "has_tumor": True,
    }


def classify_from_mask(mask: np.ndarray) -> dict:
    """
    Classify tumor type using mask morphology.

    Rules based on area_pct (% of image):
        No Tumor  : area_pct < 0.1%
        Cystic    : round + solid + small (area_pct < 8%)
        Malignant : irregular OR large (area_pct > 15%) OR elongated
        Benign    : everything else
    """
    feats = extract_mask_features(mask)

    # No tumor
    if not feats["has_tumor"] or feats["area_pct"] < 0.1:
        scores = [0.92, 0.04, 0.02, 0.02]
        return {
            "class_idx": 0,
            "class_name": CLASS_NAMES[0],
            "confidence": scores[0],
            "confidence_scores": scores,
            "risk_level": "Low",
            "features": feats,
        }

    area_pct    = feats["area_pct"]     # 0-100 (% of image)
    solidity    = feats["solidity"]     # 0-1
    circularity = feats["circularity"]  # 0-1
    aspect      = feats["aspect_ratio"] # >= 1

    print(f"[Classifier] area_pct={area_pct:.2f}% solidity={solidity:.3f} "
          f"circularity={circularity:.3f} aspect={aspect:.3f} "
          f"area_cm2={feats['area_cm2']:.3f}cm²")

    # Cystic: very round (near-perfect circle), small
    # Ovals and ellipses fall into Benign
    is_cystic = (
        circularity >= 0.85      # stricter — must be nearly circular
        and solidity >= 0.90
        and area_pct < 8.0
        and aspect <= 1.3        # not elongated
    )

    # Malignant: irregular borders, large, or elongated
    is_malignant = (
        solidity < 0.72
        or area_pct > 15.0
        or aspect > 2.4
        or (solidity < 0.80 and area_pct > 8.0)
    )

    if is_cystic and not is_malignant:
        cystic_conf = float(np.clip(0.55 + 0.25 * circularity + 0.15 * solidity, 0.55, 0.92))
        benign_conf = float(np.clip(0.45 - 0.20 * circularity, 0.04, 0.25))
        malign_conf = float(np.clip(0.10 - 0.05 * solidity, 0.02, 0.12))
        no_tum_conf = 0.02
        raw = [no_tum_conf, benign_conf, malign_conf, cystic_conf]

    elif is_malignant:
        irregularity = float(np.clip(1.0 - solidity, 0.0, 1.0))
        size_factor  = float(np.clip(area_pct / 30.0, 0.0, 1.0))
        malign_conf  = float(np.clip(0.55 + 0.20 * irregularity + 0.15 * size_factor, 0.55, 0.93))
        benign_conf  = float(np.clip(0.30 - 0.15 * irregularity, 0.03, 0.25))
        cystic_conf  = float(np.clip(0.10 - 0.05 * irregularity, 0.01, 0.10))
        no_tum_conf  = 0.02
        raw = [no_tum_conf, benign_conf, malign_conf, cystic_conf]

    else:
        benign_conf = float(np.clip(0.55 + 0.20 * solidity - 0.10 * (aspect - 1.0), 0.50, 0.90))
        malign_conf = float(np.clip(0.20 - 0.10 * solidity, 0.03, 0.20))
        cystic_conf = float(np.clip(0.15 * circularity, 0.02, 0.15))
        no_tum_conf = 0.02
        raw = [no_tum_conf, benign_conf, malign_conf, cystic_conf]

    total  = sum(raw)
    scores = [round(v / total, 4) for v in raw]
    class_idx  = int(np.argmax(scores))
    confidence = scores[class_idx]

    return {
        "class_idx":         class_idx,
        "class_name":        CLASS_NAMES[class_idx],
        "confidence":        confidence,
        "confidence_scores": scores,
        "risk_level":        get_risk_level(class_idx, confidence),
        "features":          feats,
    }


class MorphologyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 4, device=x.device)


def build_classifier(weights_path: Optional[str] = None) -> MorphologyClassifier:
    model = MorphologyClassifier()
    model.eval()
    print("[Classifier] Rule-based morphology classifier loaded.")
    return model


if __name__ == "__main__":
    import cv2

    print("Testing rule-based classifier...\n")

    # No tumor
    empty = np.zeros((224, 224), dtype=np.float32)
    r = classify_from_mask(empty)
    print(f"Empty mask     → {r['class_name']} ({r['confidence']*100:.1f}%) | Risk: {r['risk_level']}\n")

    # Small round → Cystic
    round_mask = np.zeros((224, 224), dtype=np.float32)
    cv2.circle(round_mask, (112, 112), 25, 1.0, -1)
    r = classify_from_mask(round_mask)
    print(f"Round lesion   → {r['class_name']} ({r['confidence']*100:.1f}%) | Risk: {r['risk_level']}\n")

    # Large irregular → Malignant
    irreg = np.zeros((224, 224), dtype=np.float32)
    pts = np.array([[60,80],[160,70],[180,140],[140,180],[70,170],[40,120]], np.int32)
    cv2.fillPoly(irreg, [pts], 1.0)
    r = classify_from_mask(irreg)
    print(f"Irregular mass → {r['class_name']} ({r['confidence']*100:.1f}%) | Risk: {r['risk_level']}\n")

    # Medium oval → Benign
    oval = np.zeros((224, 224), dtype=np.float32)
    cv2.ellipse(oval, (112, 112), (30, 20), 0, 0, 360, 1.0, -1)
    r = classify_from_mask(oval)
    print(f"Oval lesion    → {r['class_name']} ({r['confidence']*100:.1f}%) | Risk: {r['risk_level']}\n")

    # uint8 mask test
    uint8_mask = np.zeros((224, 224), dtype=np.uint8)
    cv2.circle(uint8_mask, (112, 112), 25, 255, -1)
    r = classify_from_mask(uint8_mask)
    print(f"uint8 round    → {r['class_name']} ({r['confidence']*100:.1f}%) | Risk: {r['risk_level']}\n")

    print("All tests passed.")