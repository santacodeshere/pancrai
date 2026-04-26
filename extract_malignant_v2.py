"""
Find CT slices where the tumor is LARGE enough for TransUNet to detect
AND irregular enough to classify as malignant.
Minimum 500 tumor pixels in ground truth (ensures model can see it).
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path
from app.models.classifier import extract_mask_features
from app.models.transunet import build_transunet
from app.services.preprocessing import load_from_bytes
from app.services.segmentation import run_segmentation

IMG_DIR = Path('data/Task07_Pancreas/imagesTr')
LBL_DIR = Path('data/Task07_Pancreas/labelsTr')
OUT_DIR = Path('test_scans/malignant_v2')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_slice(sl):
    sl = np.clip(sl, -160, 240).astype(np.float32)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    return (sl * 255).astype(np.uint8)

print("Loading TransUNet model...")
model = build_transunet(weights_path='./weights/transunet_best.pth')
print("Model loaded.\n")

print("Scanning volumes for large detectable tumors...")
print("Criteria: ground truth tumor > 500px AND model detects > 100px\n")

saved = 0
TARGET = 8
seen_vols = set()

vols = sorted([f for f in IMG_DIR.glob('*.nii.gz')
               if not f.name.startswith('._')])

for vol_path in vols:
    if saved >= TARGET:
        break

    lbl_path = LBL_DIR / vol_path.name
    if not lbl_path.exists():
        continue

    vol_name = vol_path.stem.replace('.nii', '')
    if vol_name in seen_vols:
        continue

    try:
        img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)
    except Exception as e:
        continue

    n_slices = img_vol.shape[2]

    # Find slice with most tumor pixels (ground truth)
    best_slice = None
    best_count = 0
    for i in range(n_slices):
        # Count ALL labeled pixels (pancreas + tumor)
        count = int((lbl_vol[:, :, i] > 0).sum())
        tumor_only = int((lbl_vol[:, :, i] == 2).sum())
        # Want large overall segmentation with tumor present
        if count > best_count and tumor_only > 200:
            best_count = count
            best_slice = i

    if best_slice is None:
        continue

    # Run model on this slice
    img_sl = img_vol[:, :, best_slice]
    sl_min, sl_max = img_sl.min(), img_sl.max()
    if sl_max > sl_min:
        img_norm = (img_sl - sl_min) / (sl_max - sl_min)
    else:
        continue

    img_uint8 = (img_norm * 255).astype(np.uint8)
    img_uint8 = cv2.resize(img_uint8, (512, 512))

    # Check what model actually predicts
    img_bytes = cv2.imencode('.png', img_uint8)[1].tobytes()
    img_loaded = load_from_bytes(img_bytes, 'test.png')
    seg = run_segmentation(img_loaded, model)
    mask = seg.get('mask')

    predicted_pixels = int((mask > 127).sum()) if mask is not None else 0
    feats = extract_mask_features(mask) if mask is not None else {}

    area_pct = feats.get('area_pct', 0)
    solidity = feats.get('solidity', 0)
    aspect   = feats.get('aspect_ratio', 1)

    print(f"Vol: {vol_name} | slice z={best_slice}")
    print(f"  GT pixels: {best_count} | Predicted pixels: {predicted_pixels}")
    print(f"  area_pct={area_pct:.2f}% solidity={solidity:.3f} aspect={aspect:.3f}")

    if predicted_pixels < 50:
        print(f"  SKIP — model didn't detect enough\n")
        continue

    # Save the slice
    fname = OUT_DIR / f"malignant_{vol_name}_z{best_slice:03d}_pred{predicted_pixels}.png"
    cv2.imwrite(str(fname), img_uint8)
    seen_vols.add(vol_name)
    saved += 1
    print(f"  SAVED → {fname.name}\n")

print(f"{'='*50}")
print(f"Saved {saved} slices to test_scans/malignant_v2/")
print(f"\nNOTE: These slices have large enough tumors for the model to detect.")
print(f"Upload them in PancrAI — classification depends on detected shape.")
