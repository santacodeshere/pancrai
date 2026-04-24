"""
PancrAI — Demo Data Generator
Creates synthetic CT-like images and masks for testing the full pipeline
without a real medical dataset.

Usage:
    python demo_data_generator.py --output_dir ./data/demo --n_train 200 --n_val 50
"""

import os
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm


def generate_ct_background(size: int, rng: np.random.Generator) -> np.ndarray:
    """
    Simulate a CT slice background with realistic soft-tissue texture.
    Uses Perlin-like layered noise to mimic organ boundaries.
    """
    img = np.zeros((size, size), dtype=np.float32)

    # Base soft-tissue gray level: ~40-80 HU mapped to 0.3-0.5 range
    img += rng.uniform(0.30, 0.45)

    # Add large-scale organ structure via low-freq noise
    for scale in [8, 16, 32]:
        noise = rng.random((size // scale, size // scale)).astype(np.float32)
        noise = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        img += noise * 0.04

    # Simulate pancreatic region — slightly brighter oval in center-left
    cx, cy = int(size * 0.45), int(size * 0.55)
    rx, ry = int(size * 0.20), int(size * 0.10)
    yy, xx = np.ogrid[:size, :size]
    pancreas_region = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    img[pancreas_region] += rng.uniform(0.05, 0.12)

    # Add fine-grain noise (image noise)
    img += rng.normal(0, 0.02, (size, size)).astype(np.float32)

    return np.clip(img, 0, 1)


def generate_tumor(
    size: int,
    rng: np.random.Generator,
    tumor_type: str = "random",
) -> tuple:
    """
    Generate a realistic-looking tumor blob and its binary mask.

    tumor_type: 'benign', 'malignant', 'cystic', or 'random'

    Returns: (tumor_intensity_mask float32, binary_mask uint8)
    """
    if tumor_type == "random":
        tumor_type = rng.choice(["benign", "malignant", "cystic"])

    intensity_mask = np.zeros((size, size), dtype=np.float32)
    binary_mask = np.zeros((size, size), dtype=np.uint8)

    # Tumor center — biased toward center of image (pancreatic region)
    cx = int(rng.uniform(size * 0.30, size * 0.65))
    cy = int(rng.uniform(size * 0.35, size * 0.65))

    if tumor_type == "malignant":
        # Irregular, spiculated margin — larger, higher intensity
        rx = rng.integers(int(size * 0.08), int(size * 0.18))
        ry = rng.integers(int(size * 0.06), int(size * 0.15))
        intensity_boost = rng.uniform(0.25, 0.40)
        # Add irregular spiculations
        n_spicules = rng.integers(5, 12)
        for _ in range(n_spicules):
            angle = rng.uniform(0, 2 * np.pi)
            ext = rng.uniform(1.0, 1.8)
            spx = int(cx + rx * ext * np.cos(angle))
            spy = int(cy + ry * ext * np.sin(angle))
            cv2.line(binary_mask, (cx, cy), (spx, spy), 255, rng.integers(2, 5))

    elif tumor_type == "benign":
        # Well-defined, smooth, smaller
        rx = rng.integers(int(size * 0.04), int(size * 0.10))
        ry = rng.integers(int(size * 0.04), int(size * 0.10))
        intensity_boost = rng.uniform(0.15, 0.25)

    else:  # cystic
        # Round, low intensity center (fluid-filled)
        rx = rng.integers(int(size * 0.05), int(size * 0.12))
        ry = rng.integers(int(size * 0.05), int(size * 0.12))
        intensity_boost = rng.uniform(-0.10, 0.05)  # cysts are darker (fluid)

    # Draw main tumor ellipse
    yy, xx = np.ogrid[:size, :size]
    ellipse = ((xx - cx) / max(rx, 1)) ** 2 + ((yy - cy) / max(ry, 1)) ** 2 <= 1.0
    binary_mask[ellipse] = 255

    # Apply morphological closing to smooth irregular borders
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # Intensity
    intensity_mask[binary_mask > 0] = intensity_boost
    # Gradient falloff at edges
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    max_dist = dist_transform.max()
    if max_dist > 0:
        gradient = dist_transform / max_dist
        intensity_mask = intensity_mask * (0.6 + 0.4 * gradient)

    return intensity_mask, binary_mask, tumor_type


def generate_sample(
    size: int,
    rng: np.random.Generator,
    tumor_probability: float = 0.70,
) -> tuple:
    """
    Generate a single (image, mask) training sample.

    Returns: (image uint8, mask uint8, has_tumor bool, tumor_type str)
    """
    bg = generate_ct_background(size, rng)
    has_tumor = rng.random() < tumor_probability
    tumor_type = "none"

    if has_tumor:
        tumor_intensity, tumor_mask, tumor_type = generate_tumor(size, rng)
        # Composite: add tumor intensity to background
        image_float = np.clip(bg + tumor_intensity, 0, 1)
    else:
        image_float = bg
        tumor_mask = np.zeros((size, size), dtype=np.uint8)

    # Convert to uint8
    image_u8 = (image_float * 255).astype(np.uint8)
    return image_u8, tumor_mask, has_tumor, tumor_type


def create_demo_dataset(
    output_dir: str,
    n_train: int = 200,
    n_val: int = 50,
    img_size: int = 224,
    seed: int = 42,
):
    """
    Create a full demo dataset with paired images and masks.

    Directory structure:
        output_dir/
            images/    ← PNG images
            masks/     ← PNG binary masks
            split.json ← train/val file lists
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    img_dir = out_path / "images"
    mask_dir = out_path / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    total = n_train + n_val
    stats = {"no_tumor": 0, "benign": 0, "malignant": 0, "cystic": 0}
    train_files = []
    val_files = []

    print(f"\n[DemoGen] Generating {total} synthetic CT samples ({n_train} train, {n_val} val)...")
    print(f"[DemoGen] Output: {out_path.resolve()}")

    for i in tqdm(range(total), desc="Generating samples"):
        image, mask, has_tumor, tumor_type = generate_sample(img_size, rng)

        # Save
        fname = f"sample_{i:05d}.png"
        cv2.imwrite(str(img_dir / fname), image)
        cv2.imwrite(str(mask_dir / fname), mask)

        # Track stats
        stats[tumor_type if has_tumor else "no_tumor"] += 1

        if i < n_train:
            train_files.append(fname)
        else:
            val_files.append(fname)

    # Save split info
    import json
    split_info = {
        "train": train_files,
        "val": val_files,
        "stats": stats,
        "img_size": img_size,
    }
    with open(out_path / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Print summary
    print(f"\n[DemoGen] ✓ Dataset created at {out_path.resolve()}")
    print(f"[DemoGen]   Total samples : {total}")
    print(f"[DemoGen]   Train         : {n_train}")
    print(f"[DemoGen]   Val           : {n_val}")
    print(f"[DemoGen]   No Tumor      : {stats['no_tumor']}")
    print(f"[DemoGen]   Benign        : {stats['benign']}")
    print(f"[DemoGen]   Malignant     : {stats['malignant']}")
    print(f"[DemoGen]   Cystic        : {stats['cystic']}")
    print(f"\n[DemoGen] To train on this data:")
    print(f"   python -m ml.train --data_dir {output_dir}")


def generate_sample_scan(output_path: str = "./sample_ct.png", size: int = 256):
    """
    Generate a single demo scan image for quick UI testing.

    Args:
        output_path: Where to save the PNG.
        size: Image size in pixels.
    """
    rng = np.random.default_rng(99)
    image, mask, has_tumor, tumor_type = generate_sample(size, rng, tumor_probability=1.0)

    cv2.imwrite(output_path, image)
    mask_path = output_path.replace(".png", "_mask.png")
    cv2.imwrite(mask_path, mask)

    print(f"[DemoGen] Sample scan saved: {output_path}")
    print(f"[DemoGen] Sample mask saved: {mask_path}")
    print(f"[DemoGen] Tumor type: {tumor_type}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic CT-like training data for PancrAI"
    )
    parser.add_argument("--output_dir", type=str, default="./data/demo",
                        help="Output directory for the dataset")
    parser.add_argument("--n_train", type=int, default=200,
                        help="Number of training samples")
    parser.add_argument("--n_val", type=int, default=50,
                        help="Number of validation samples")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Image size (square)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample_only", action="store_true",
                        help="Generate just one sample scan for UI testing")
    args = parser.parse_args()

    if args.sample_only:
        generate_sample_scan("./sample_ct.png", args.img_size)
    else:
        create_demo_dataset(
            output_dir=args.output_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            img_size=args.img_size,
            seed=args.seed,
        )
