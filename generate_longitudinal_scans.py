"""
PancrAI — Generate Longitudinal Test Scans
Creates two PNG scans simulating the SAME patient at two timepoints:
  - scan_patient_A_month0.png  (baseline — smaller tumor)
  - scan_patient_A_month6.png  (follow-up — larger tumor, slight position shift)

Also tries to extract two real slices from Task07 if available.

Usage:
    python generate_longitudinal_scans.py
"""

import sys
import os
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, '.')


# ─── Synthetic scan generator ─────────────────────────────────────────────────

def make_ct_background(size, rng, seed_offset=0):
    """Generate realistic soft-tissue CT background."""
    img = np.zeros((size, size), dtype=np.float32)
    img += rng.uniform(0.28, 0.42)

    # Organ layers
    for scale in [8, 16, 32, 64]:
        noise = rng.random((size // scale + 1, size // scale + 1)).astype(np.float32)
        noise = cv2.resize(noise, (size, size))
        img += noise * 0.035

    # Pancreatic region — central horizontal band
    cx, cy = int(size * 0.45), int(size * 0.52)
    rx, ry = int(size * 0.22), int(size * 0.09)
    yy, xx = np.ogrid[:size, :size]
    pancreas = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    img[pancreas] += 0.08

    # Spine (bright oval, lower center)
    spine_y, spine_x = int(size * 0.70), int(size * 0.50)
    spine_mask = ((xx - spine_x) / 18) ** 2 + ((yy - spine_y) / 22) ** 2 <= 1.0
    img[spine_mask] += 0.35

    # Liver (large region, upper right)
    liver_x, liver_y = int(size * 0.65), int(size * 0.38)
    liver = ((xx - liver_x) / 60) ** 2 + ((yy - liver_y) / 45) ** 2 <= 1.0
    img[liver] += 0.12

    # Fine noise
    img += rng.normal(0, 0.018, (size, size)).astype(np.float32)
    return np.clip(img, 0, 1)


def add_tumor(img, cx, cy, rx, ry, intensity_boost, irregularity=0.0, rng=None):
    """
    Add a tumor blob to the image.

    Args:
        cx, cy: Tumor center
        rx, ry: Semi-axes
        intensity_boost: How bright the tumor is
        irregularity: 0=smooth ellipse, 1=very spiculated (malignant)
    """
    size = img.shape[0]
    yy, xx = np.ogrid[:size, :size]
    tumor_mask = np.zeros((size, size), dtype=np.uint8)

    # Base ellipse
    ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
    tumor_mask[ellipse] = 255

    # Add spiculations for malignant tumors
    if irregularity > 0 and rng is not None:
        n_spicules = int(irregularity * 10)
        for _ in range(n_spicules):
            angle = rng.uniform(0, 2 * np.pi)
            ext = rng.uniform(1.1, 1.0 + irregularity)
            spx = int(cx + rx * ext * np.cos(angle))
            spy = int(cy + ry * ext * np.sin(angle))
            spx = np.clip(spx, 0, size - 1)
            spy = np.clip(spy, 0, size - 1)
            cv2.line(tumor_mask, (cx, cy), (spx, spy),
                     255, max(1, int(irregularity * 3)))

    # Smooth the tumor boundary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tumor_mask = cv2.morphologyEx(tumor_mask, cv2.MORPH_CLOSE, kernel)

    # Apply intensity
    tumor_region = tumor_mask > 0
    dist_transform = cv2.distanceTransform(tumor_mask, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist > 0:
        gradient = dist_transform / max_dist
        img[tumor_region] += intensity_boost * (0.5 + 0.5 * gradient[tumor_region])

    return img, tumor_mask


def generate_scan_pair(size=512, seed=42):
    """
    Generate a pair of CT scans simulating tumor growth over 6 months.

    Returns:
        scan1 (uint8): Baseline scan — smaller tumor
        scan2 (uint8): Follow-up scan — larger tumor
        mask1 (uint8): Tumor mask for scan 1
        mask2 (uint8): Tumor mask for scan 2
        info (dict): Tumor parameters for both timepoints
    """
    rng = np.random.default_rng(seed)

    # Tumor parameters
    # Timepoint 1 — baseline
    t1_cx, t1_cy = int(size * 0.42), int(size * 0.50)
    t1_rx, t1_ry = int(size * 0.055), int(size * 0.045)  # ~28x23 pixels at 512
    t1_intensity = 0.22
    t1_irregularity = 0.3

    # Timepoint 2 — 6 months later
    # Tumor has grown ~35% and shifted slightly
    growth_factor = 1.35
    t2_cx = t1_cx + int(size * 0.012)   # slight position shift
    t2_cy = t1_cy + int(size * 0.008)
    t2_rx = int(t1_rx * growth_factor)
    t2_ry = int(t1_ry * growth_factor)
    t2_intensity = 0.26                  # slightly brighter (more vascular)
    t2_irregularity = 0.55               # more irregular margins

    # Generate backgrounds (same patient, slightly different slice)
    bg1 = make_ct_background(size, rng, seed_offset=0)
    bg2 = make_ct_background(size, rng, seed_offset=1)

    # Add tumors
    img1, mask1 = add_tumor(bg1, t1_cx, t1_cy, t1_rx, t1_ry,
                             t1_intensity, t1_irregularity, rng)
    img2, mask2 = add_tumor(bg2, t2_cx, t2_cy, t2_rx, t2_ry,
                             t2_intensity, t2_irregularity, rng)

    # Convert to uint8
    scan1 = (np.clip(img1, 0, 1) * 255).astype(np.uint8)
    scan2 = (np.clip(img2, 0, 1) * 255).astype(np.uint8)

    # Calculate tumor areas
    px_to_cm2 = (0.7 / 10) ** 2   # 0.7mm/pixel → cm²
    area1_cm2 = round(float((mask1 > 0).sum()) * px_to_cm2, 3)
    area2_cm2 = round(float((mask2 > 0).sum()) * px_to_cm2, 3)
    growth_pct = round((area2_cm2 - area1_cm2) / area1_cm2 * 100, 1)

    info = {
        "timepoint1": {
            "center": (t1_cx, t1_cy),
            "axes": (t1_rx, t1_ry),
            "area_cm2": area1_cm2,
            "label": "Baseline (Month 0)",
        },
        "timepoint2": {
            "center": (t2_cx, t2_cy),
            "axes": (t2_rx, t2_ry),
            "area_cm2": area2_cm2,
            "label": "Follow-up (Month 6)",
        },
        "growth_percent": growth_pct,
        "growth_mm": round((t2_rx - t1_rx) * 0.7, 1),
    }

    return scan1, scan2, mask1, mask2, info


# ─── Real data extraction from Task07 ────────────────────────────────────────

def extract_real_pair(data_dir="./data/Task07_Pancreas", patient_idx=0):
    """
    Extract two slices from the same patient volume at different depths
    to simulate longitudinal scans.
    """
    try:
        import nibabel as nib
    except ImportError:
        print("nibabel not installed — skipping real data extraction")
        return None, None

    img_dir = Path(data_dir) / "imagesTr"
    lbl_dir = Path(data_dir) / "labelsTr"

    if not img_dir.exists():
        print(f"Dataset not found at {data_dir}")
        return None, None

    vol_files = sorted([f for f in img_dir.glob("pancreas_0*.nii.gz")
                        if not f.name.startswith("._")])

    if not vol_files or patient_idx >= len(vol_files):
        print("No volumes found")
        return None, None

    vol_path = vol_files[patient_idx]
    lbl_path = lbl_dir / vol_path.name

    if not lbl_path.exists():
        print(f"Label not found for {vol_path.name}")
        return None, None

    print(f"Loading {vol_path.name}...")
    img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
    lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)

    # Find slices with tumor (label == 2 = tumor)
    tumor_slices = []
    for i in range(img_vol.shape[2]):
        lbl_sl = lbl_vol[:, :, i]
        tumor_pixels = (lbl_sl == 2).sum()
        if tumor_pixels > 30:
            tumor_slices.append((i, tumor_pixels))

    if len(tumor_slices) < 2:
        print(f"Not enough tumor slices in {vol_path.name}")
        return None, None

    # Pick two slices separated by ~20% of tumor extent
    tumor_slices.sort(key=lambda x: x[0])
    n = len(tumor_slices)
    idx1 = n // 4         # earlier in tumor
    idx2 = 3 * n // 4     # later in tumor (larger cross-section typically)

    slices_to_use = [tumor_slices[idx1][0], tumor_slices[idx2][0]]
    scans = []

    for sl_idx in slices_to_use:
        img_sl = img_vol[:, :, sl_idx]
        # HU windowing
        img_sl = np.clip(img_sl, -160, 240)
        img_sl = ((img_sl + 160) / 400.0 * 255).astype(np.uint8)
        img_sl = cv2.resize(img_sl, (512, 512))
        scans.append(img_sl)

    print(f"  Slice {slices_to_use[0]}: {tumor_slices[idx1][1]} tumor pixels")
    print(f"  Slice {slices_to_use[1]}: {tumor_slices[idx2][1]} tumor pixels")

    return scans[0], scans[1]


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  PancrAI — Longitudinal Test Scan Generator")
    print("=" * 55)

    output_dir = Path(".")

    # ── Generate synthetic pair ──
    print("\n[1] Generating synthetic longitudinal scan pair...")
    scan1, scan2, mask1, mask2, info = generate_scan_pair(size=512, seed=42)

    p1 = output_dir / "scan_patient_A_month0.png"
    p2 = output_dir / "scan_patient_A_month6.png"
    cv2.imwrite(str(p1), scan1)
    cv2.imwrite(str(p2), scan2)

    print(f"\n  Saved: {p1.name}")
    print(f"    Tumor center: {info['timepoint1']['center']}")
    print(f"    Tumor area  : {info['timepoint1']['area_cm2']} cm²")
    print(f"    Description : {info['timepoint1']['label']}")

    print(f"\n  Saved: {p2.name}")
    print(f"    Tumor center: {info['timepoint2']['center']}")
    print(f"    Tumor area  : {info['timepoint2']['area_cm2']} cm²")
    print(f"    Description : {info['timepoint2']['label']}")

    print(f"\n  Tumor growth : {info['growth_percent']:+.1f}%")
    print(f"  Growth (mm)  : +{info['growth_mm']} mm diameter")

    # ── Generate a second synthetic pair (different patient) ──
    print("\n[2] Generating second synthetic pair (stable tumor)...")
    scan3, scan4, _, _, info2 = generate_scan_pair(size=512, seed=99)

    # Make scan4 very similar to scan3 (stable disease)
    scan4_stable = scan3.copy()
    noise = np.random.normal(0, 3, scan3.shape).astype(np.int16)
    scan4_stable = np.clip(scan4_stable.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    p3 = output_dir / "scan_patient_B_month0.png"
    p4 = output_dir / "scan_patient_B_month3.png"
    cv2.imwrite(str(p3), scan3)
    cv2.imwrite(str(p4), scan4_stable)
    print(f"  Saved: {p3.name} and {p4.name} (stable disease)")

    # ── Try real data ──
    print("\n[3] Attempting real data extraction from Task07...")
    real1, real2 = extract_real_pair("./data/Task07_Pancreas", patient_idx=2)
    if real1 is not None and real2 is not None:
        r1 = output_dir / "scan_real_patient_early_slice.png"
        r2 = output_dir / "scan_real_patient_late_slice.png"
        cv2.imwrite(str(r1), real1)
        cv2.imwrite(str(r2), real2)
        print(f"  Saved: {r1.name}")
        print(f"  Saved: {r2.name}")
        print("  These are real CT slices from the same patient volume")
    else:
        print("  Skipped — using synthetic scans only")

    # ── Summary ──
    print("\n" + "=" * 55)
    print("  Files ready for longitudinal comparison testing:")
    print("=" * 55)

    files = [
        ("scan_patient_A_month0.png", "Scan 1 — Growing tumor baseline"),
        ("scan_patient_A_month6.png", "Scan 2 — Same tumor 6 months later (+35%)"),
        ("scan_patient_B_month0.png", "Scan 3 — Stable tumor baseline"),
        ("scan_patient_B_month3.png", "Scan 4 — Same tumor 3 months later (stable)"),
    ]

    if real1 is not None:
        files += [
            ("scan_real_patient_early_slice.png", "Scan 5 — Real CT early slice"),
            ("scan_real_patient_late_slice.png",  "Scan 6 — Real CT later slice"),
        ]

    for fname, desc in files:
        if Path(fname).exists():
            size_kb = Path(fname).stat().st_size // 1024
            print(f"  {fname} ({size_kb} KB)")
            print(f"    → {desc}")

    print("\nHow to use in PancrAI:")
    print("  1. Go to 'Longitudinal Comparison' page")
    print("  2. Upload scan_patient_A_month0.png as Scan 1")
    print("     Set date: 2024-01-01")
    print("  3. Upload scan_patient_A_month6.png as Scan 2")
    print("     Set date: 2024-07-01")
    print("  4. Click 'Compare Scans'")
    print("  5. You should see ~35% tumor growth with the difference map")
    print()
    print("For stable disease test:")
    print("  Upload scan_patient_B_month0.png and scan_patient_B_month3.png")
    print("  The tumor area should be nearly identical (stable disease)")


if __name__ == "__main__":
    main()
