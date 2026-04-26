"""
Check what morphological features TransUNet actually produces
on the malignant test slices, so we can tune classifier thresholds.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
from pathlib import Path

from app.models.transunet import build_transunet
from app.services.preprocessing import load_from_bytes
from app.services.segmentation import run_segmentation
from app.models.classifier import classify_from_mask, extract_mask_features

print("Loading model...")
model = build_transunet(weights_path='./weights/transunet_best.pth')

malignant_dir = Path('test_scans/malignant')
pngs = sorted(malignant_dir.glob('*.png'))[:5]

print(f"\nAnalyzing {len(pngs)} malignant test slices...\n")

for png in pngs:
    with open(png, 'rb') as f:
        fb = f.read()

    from app.services.preprocessing import load_from_bytes
    img = load_from_bytes(fb, png.name)
    seg = run_segmentation(img, model)
    mask = seg.get('mask')

    feats = extract_mask_features(mask)
    result = classify_from_mask(mask)

    print(f"File: {png.name}")
    print(f"  Predicted mask: {int((mask>127).sum())} tumor pixels")
    print(f"  area_pct   : {feats['area_pct']:.2f}%")
    print(f"  solidity   : {feats['solidity']:.4f}")
    print(f"  circularity: {feats['circularity']:.4f}")
    print(f"  aspect     : {feats['aspect_ratio']:.4f}")
    print(f"  → {result['class_name']} ({result['confidence']*100:.1f}%)")
    print()
