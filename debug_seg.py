import sys
sys.path.insert(0, '.')
import numpy as np

print("Loading model...")
from app.models.transunet import build_transunet
model = build_transunet(weights_path='./weights/transunet_best.pth')
print("Model loaded.")

print("Loading image...")
from app.services.preprocessing import load_from_bytes
with open('data/Task07_Pancreas/imagesTr/pancreas_001.nii.gz', 'rb') as f:
    b = f.read()
img = load_from_bytes(b, 'pancreas_001.nii.gz')
print("Image loaded. Shape:", img.shape if hasattr(img, 'shape') else type(img))

print("Running segmentation...")
from app.services.segmentation import run_segmentation
result = run_segmentation(img, model)

print("\n=== Segmentation Result Keys ===")
for k, v in result.items():
    if hasattr(v, 'shape'):
        print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype} min={v.min():.4f} max={v.max():.4f} sum={v.sum():.1f}")
    elif isinstance(v, str):
        print(f"  {k}: string length={len(v)}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

mask = result.get('mask')
if mask is None:
    print("\nWARNING: 'mask' key not found!")
    print("Available keys:", list(result.keys()))
else:
    print(f"\nMask found!")
    print(f"  Shape : {mask.shape}")
    print(f"  Dtype : {mask.dtype}")
    print(f"  Min   : {mask.min():.4f}")
    print(f"  Max   : {mask.max():.4f}")
    print(f"  Sum   : {mask.sum():.1f}  (tumor pixels)")
    print(f"  Tumor detected: {mask.sum() > 5}")
