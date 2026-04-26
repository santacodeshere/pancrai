"""
Extract CT slices that will classify as Malignant in PancrAI.
Malignant triggers: low solidity (<0.72) OR large area (>15% of image)
OR high aspect ratio (>2.4) OR (solidity<0.80 and area>8%)

Strategy: find slices where tumor is large OR irregular shaped.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import cv2
import nibabel as nib
from pathlib import Path

# Import the actual classifier to pre-screen slices
from app.models.classifier import classify_from_mask, extract_mask_features

IMG_DIR = Path('data/Task07_Pancreas/imagesTr')
LBL_DIR = Path('data/Task07_Pancreas/labelsTr')
OUT_DIR = Path('test_scans/malignant')
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_slice(sl):
    sl = np.clip(sl, -160, 240).astype(np.float32)
    sl = (sl - sl.min()) / (sl.max() - sl.min() + 1e-8)
    return (sl * 255).astype(np.uint8)


def mask_to_binary(lbl_sl):
    """Convert label slice to binary tumor mask (label==2 is tumor)."""
    tumor = (lbl_sl == 2).astype(np.uint8) * 255
    return cv2.resize(tumor, (224, 224), interpolation=cv2.INTER_NEAREST)


saved = 0
TARGET = 10
candidates = []

print("Scanning all volumes for malignant-looking tumor slices...")
print("(Looking for large area >8% OR irregular solidity <0.75)\n")

vols = sorted([
    f for f in IMG_DIR.glob('*.nii.gz')
    if not f.name.startswith('._')
])

for vol_path in vols:
    lbl_path = LBL_DIR / vol_path.name
    if not lbl_path.exists():
        continue

    try:
        img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
        lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)
    except Exception as e:
        continue

    vol_name = vol_path.stem.replace('.nii', '')
    n_slices = img_vol.shape[2]

    for i in range(n_slices):
        lbl_sl = lbl_vol[:, :, i]
        tumor_pixels = int((lbl_sl == 2).sum())

        if tumor_pixels < 50:
            continue

        # Get the mask at 224x224 (same as model input)
        mask = mask_to_binary(lbl_sl)
        feats = extract_mask_features(mask)

        area_pct    = feats['area_pct']
        solidity    = feats['solidity']
        circularity = feats['circularity']
        aspect      = feats['aspect_ratio']

        # Check if this would classify as malignant
        is_malignant = (
            solidity < 0.72
            or area_pct > 15.0
            or aspect > 2.4
            or (solidity < 0.80 and area_pct > 8.0)
        )

        if is_malignant:
            # Score: higher = more clearly malignant
            malignant_score = (
                (1.0 - solidity) * 40 +
                min(area_pct, 30) * 1.5 +
                max(aspect - 1.0, 0) * 10
            )
            candidates.append({
                'vol_name': vol_name,
                'slice_idx': i,
                'img_vol': img_vol,
                'score': malignant_score,
                'area_pct': area_pct,
                'solidity': solidity,
                'circularity': circularity,
                'aspect': aspect,
            })

# Sort by most clearly malignant
candidates.sort(key=lambda x: x['score'], reverse=True)

# Deduplicate — one slice per volume
seen_vols = set()
unique_candidates = []
for c in candidates:
    if c['vol_name'] not in seen_vols:
        seen_vols.add(c['vol_name'])
        unique_candidates.append(c)

print(f"Found {len(unique_candidates)} unique malignant-looking slices\n")

for c in unique_candidates[:TARGET]:
    img_sl = c['img_vol'][:, :, c['slice_idx']]
    img = normalize_slice(img_sl)
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)

    fname = (OUT_DIR /
             f"malignant_{c['vol_name']}_z{c['slice_idx']:03d}"
             f"_area{c['area_pct']:.1f}_sol{c['solidity']:.2f}.png")
    cv2.imwrite(str(fname), img_resized)

    print(f"Saved: {fname.name}")
    print(f"       area={c['area_pct']:.1f}% solidity={c['solidity']:.3f} "
          f"circularity={c['circularity']:.3f} aspect={c['aspect']:.2f}")

    # Verify classification
    mask = mask_to_binary(
        nib.load(str(LBL_DIR / f"{c['vol_name']}.nii.gz")).get_fdata()[:, :, c['slice_idx']]
    )
    result = classify_from_mask(mask)
    print(f"       → Classifier says: {result['class_name']} "
          f"({result['confidence']*100:.1f}%) | Risk: {result['risk_level']}\n")
    saved += 1

print(f"{'='*50}")
print(f"Saved {saved} malignant test slices → test_scans/malignant/")
print(f"\nUpload these in PancrAI — they should classify as Malignant (PDAC)")
