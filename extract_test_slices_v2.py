"""
Extract real CT slices from Task07 dataset for testing PancrAI.
Saves both tumor slices and normal (no-tumor) slices as PNG files.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path

IMG_DIR = Path('data/Task07_Pancreas/imagesTr')
LBL_DIR = Path('data/Task07_Pancreas/labelsTr')
OUT_DIR = Path('test_scans')
OUT_DIR.mkdir(exist_ok=True)

(OUT_DIR / 'tumor').mkdir(exist_ok=True)
(OUT_DIR / 'normal').mkdir(exist_ok=True)

def normalize_slice(sl):
    """Normalize a CT slice to 0-255 uint8."""
    # Apply soft tissue HU windowing
    sl = np.clip(sl, -160, 240).astype(np.float32)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    return (sl * 255).astype(np.uint8)

def save_slice(arr, path):
    """Resize to 512x512 and save as PNG."""
    arr_resized = cv2.resize(arr, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(str(path), arr_resized)

tumor_saved  = 0
normal_saved = 0

TUMOR_TARGET  = 10
NORMAL_TARGET = 10

print(f"Extracting slices from Task07 dataset...")
print(f"Target: {TUMOR_TARGET} tumor + {NORMAL_TARGET} normal slices\n")

vols = sorted([
    f for f in IMG_DIR.glob('*.nii.gz')
    if not f.name.startswith('._')
])

for vol_path in vols:
    if tumor_saved >= TUMOR_TARGET and normal_saved >= NORMAL_TARGET:
        break

    lbl_path = LBL_DIR / vol_path.name
    if not lbl_path.exists():
        continue

    try:
        img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)
    except Exception as e:
        print(f"  Skip {vol_path.name}: {e}")
        continue

    vol_name = vol_path.stem.replace('.nii', '')
    n_slices = img_vol.shape[2]

    # ── Tumor slices (label == 2 means tumor) ──
    if tumor_saved < TUMOR_TARGET:
        # Find slices with most tumor pixels
        tumor_counts = []
        for i in range(n_slices):
            count = int((lbl_vol[:, :, i] == 2).sum())
            if count > 100:
                tumor_counts.append((count, i))

        # Sort by tumor size, pick the best slice
        tumor_counts.sort(reverse=True)
        for count, i in tumor_counts[:2]:
            if tumor_saved >= TUMOR_TARGET:
                break
            sl  = img_vol[:, :, i]
            img = normalize_slice(sl)
            fname = OUT_DIR / 'tumor' / f'tumor_{vol_name}_z{i:03d}_px{count}.png'
            save_slice(img, fname)
            print(f"  [TUMOR]  {fname.name}  (tumor pixels: {count})")
            tumor_saved += 1

    # ── Normal slices (no pancreas at all) ──
    if normal_saved < NORMAL_TARGET:
        # Find slices with zero label (away from pancreas region)
        normal_idxs = []
        for i in range(n_slices):
            if lbl_vol[:, :, i].sum() == 0:
                normal_idxs.append(i)

        if normal_idxs:
            # Pick one from the middle of the normal range
            pick = normal_idxs[len(normal_idxs) // 2]
            sl  = img_vol[:, :, pick]
            img = normalize_slice(sl)
            fname = OUT_DIR / 'normal' / f'normal_{vol_name}_z{pick:03d}.png'
            save_slice(img, fname)
            print(f"  [NORMAL] {fname.name}")
            normal_saved += 1

print(f"\n{'='*50}")
print(f"Done!")
print(f"  Tumor slices saved : {tumor_saved}  → test_scans/tumor/")
print(f"  Normal slices saved: {normal_saved}  → test_scans/normal/")
print(f"\nUpload any of these PNG files in PancrAI to test detection.")
print(f"Tumor slices should show segmentation overlay + Benign/Malignant.")
print(f"Normal slices should show No Tumor (92% confidence).")
