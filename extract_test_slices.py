"""
PancrAI — Extract Normal (No Tumor) Test Slices
Saves PNG slices from real pancreatic CT scans where NO tumor is present.
These are slices where the pancreas is visible but label == 0 (background only).

Usage:
    python extract_normal_slices.py
"""

import sys
sys.path.insert(0, '.')

import numpy as np
import cv2
import nibabel as nib
from pathlib import Path

print("Extracting normal (no tumor) slices from real pancreatic CT scans...")
print("These slices show the pancreas region WITHOUT any tumor.\n")

img_dir = Path('data/Task07_Pancreas/imagesTr')
lbl_dir = Path('data/Task07_Pancreas/labelsTr')

if not img_dir.exists():
    print("ERROR: data/Task07_Pancreas/imagesTr not found.")
    print("Make sure you are running this from the PancrAI folder.")
    sys.exit(1)

saved = 0
target = 5  # how many normal slices to save

for vol_path in sorted(img_dir.glob('pancreas_0*.nii.gz'))[:20]:
    if vol_path.name.startswith('._'):
        continue
    lbl_path = lbl_dir / vol_path.name
    if not lbl_path.exists():
        continue

    try:
        img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)
    except Exception as e:
        print(f"Skipping {vol_path.name}: {e}")
        continue

    n_slices = img_vol.shape[2]

    # Look for slices that:
    # 1. Have zero tumor pixels (label == 0 everywhere)
    # 2. Have some organ content (not empty/air slices)
    # 3. Come from the middle third of the volume (where pancreas usually is)
    mid_start = n_slices // 3
    mid_end   = 2 * n_slices // 3

    for i in range(mid_start, mid_end):
        lbl_sl = lbl_vol[:, :, i]
        img_sl = img_vol[:, :, i]

        has_tumor   = (lbl_sl > 0).sum() > 0
        has_content = (img_sl > img_sl.mean()).sum() > 500  # not empty

        if not has_tumor and has_content:
            # Normalize with soft-tissue CT windowing
            img_sl_clipped = np.clip(img_sl, -160, 240)
            img_norm = (img_sl_clipped + 160) / 400.0
            img_uint8 = (img_norm * 255).astype(np.uint8)
            img_uint8 = cv2.resize(img_uint8, (512, 512))

            name = vol_path.stem.replace('.nii', '')
            out_path = f'test_normal_{name}_z{i}.png'
            cv2.imwrite(out_path, img_uint8)

            print(f"Saved: {out_path}")
            print(f"  Volume: {vol_path.name}")
            print(f"  Slice:  {i}/{n_slices}")
            print(f"  Label:  All background (no tumor) ✓")
            print()

            saved += 1
            break  # one slice per volume

    if saved >= target:
        break

print(f"Done! {saved} normal slices saved in your PancrAI folder.")
print()
print("Files saved:")
for f in sorted(Path('.').glob('test_normal_*.png')):
    print(f"  {f.name}")
print()
print("Upload these in the app — the model should return 'No Tumor'.")
print("Compare with the tumor slices (test_slice_*.png) to see the difference.")