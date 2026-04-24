import sys
sys.path.insert(0, '.')
import numpy as np

print("=== Step 1: Test classifier directly ===")
from app.models.classifier import classify_from_mask

# Simulate the exact mask from debug_seg.py
# mask was uint8, max=255, sum=201705 (very large — whole image almost)
# That's why it showed malignant before

# Let's test with the actual segmentation values we saw
mask_large = np.ones((224, 224), dtype=np.uint8) * 255  # worst case: fully filled
r = classify_from_mask(mask_large)
print(f"Full mask (100%):  {r['class_name']} ({r['confidence']*100:.1f}%) area_pct={r['features']['area_pct']:.1f}%")

mask_small = np.zeros((224, 224), dtype=np.uint8)
mask_small[80:120, 90:130] = 255  # small region
r = classify_from_mask(mask_small)
print(f"Small mask (~3%):  {r['class_name']} ({r['confidence']*100:.1f}%) area_pct={r['features']['area_pct']:.1f}%")

print("\n=== Step 2: Test full pipeline ===")
from app.models.transunet import build_transunet
from app.services.preprocessing import load_from_bytes
from app.services.segmentation import run_segmentation

print("Loading model...")
model = build_transunet(weights_path='./weights/transunet_best.pth')

print("Loading test image...")
with open('data/Task07_Pancreas/imagesTr/pancreas_001.nii.gz', 'rb') as f:
    b = f.read()
img = load_from_bytes(b, 'pancreas_001.nii.gz')

print("Running segmentation...")
seg = run_segmentation(img, model)

mask = seg.get('mask')
print(f"\nMask dtype: {mask.dtype}, max: {mask.max()}, sum: {mask.sum()}")
print(f"Tumor pixels: {(mask > 127).sum()} out of {mask.shape[0]*mask.shape[1]}")

print("\nRunning classification...")
result = classify_from_mask(mask)
print(f"\nFinal result: {result['class_name']} ({result['confidence']*100:.1f}%)")
print(f"Risk: {result['risk_level']}")
print(f"Features: area_pct={result['features']['area_pct']:.2f}% solidity={result['features']['solidity']:.3f} circularity={result['features']['circularity']:.3f}")
