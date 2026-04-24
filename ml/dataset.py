"""
PancrAI — PyTorch Dataset
Supports Medical Segmentation Decathlon Task07 (NIfTI) and NIH Pancreas-CT.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Callable, List, Tuple


class PancreasDataset(Dataset):
    """
    PyTorch Dataset for pancreatic tumor segmentation.

    Supports:
    - Medical Segmentation Decathlon Task07 (NIfTI format)
    - NIH Pancreas-CT (NIfTI format)
    - Custom paired image/mask directories (PNG/JPEG)

    Directory structure expected:
    data_dir/
      imagesTr/       ← training volumes (.nii.gz)
      labelsTr/       ← training masks   (.nii.gz)
      imagesVal/      ← validation volumes
      labelsVal/      ← validation masks

    Or for 2D slices:
    data_dir/
      images/         ← PNG slices
      masks/          ← PNG masks (binary)

    Args:
        data_dir: Root data directory.
        split: 'train' or 'val'.
        img_size: Target image size (square).
        transform: Albumentations transform pipeline.
        max_slices_per_volume: Limit slices per NIfTI volume (for memory).
        slice_axis: Axis along which to extract 2D slices (0=sagittal, 1=coronal, 2=axial).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        max_slices_per_volume: int = 30,
        slice_axis: int = 2,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.max_slices = max_slices_per_volume
        self.slice_axis = slice_axis

        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self._load_dataset()

    def _load_dataset(self):
        """Auto-detect dataset format and load image-mask pairs."""
        # Try NIfTI format first (Decathlon / NIH)
        img_dir = self.data_dir / (f"imagesTr" if self.split == "train" else "imagesVal")
        lbl_dir = self.data_dir / (f"labelsTr" if self.split == "train" else "labelsVal")

        if img_dir.exists() and lbl_dir.exists():
            self._load_nifti_volumes(img_dir, lbl_dir)
            print(f"[Dataset] Loaded {len(self.samples)} slices from NIfTI volumes ({self.split})")
            return

        # Fallback: flat 2D PNG/JPEG slices
        img_dir2 = self.data_dir / "images"
        lbl_dir2 = self.data_dir / "masks"
        if img_dir2.exists() and lbl_dir2.exists():
            self._load_2d_slices(img_dir2, lbl_dir2)
            print(f"[Dataset] Loaded {len(self.samples)} 2D slices ({self.split})")
            return

        # No data found — generate synthetic demo samples
        print(f"[Dataset] Warning: No data found at {self.data_dir}. "
              "Generating synthetic demo data for testing.")
        self._generate_synthetic(n=200 if self.split == "train" else 50)

    def _load_nifti_volumes(self, img_dir: Path, lbl_dir: Path):
        """Extract 2D slices from NIfTI volumes."""
        try:
            import nibabel as nib
        except ImportError:
            print("[Dataset] nibabel not installed. Falling back to synthetic data.")
            self._generate_synthetic()
            return

        import cv2

        nii_files = sorted(img_dir.glob("*.nii.gz")) + sorted(img_dir.glob("*.nii"))

        for vol_path in nii_files:
            # Find corresponding label
            lbl_path = lbl_dir / vol_path.name
            if not lbl_path.exists():
                lbl_path = lbl_dir / vol_path.name.replace("_0000", "")
            if not lbl_path.exists():
                continue

            try:
                img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
                lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)

                # Binarize label (any non-zero = tumor)
                lbl_vol = (lbl_vol > 0).astype(np.float32)

                n_slices = img_vol.shape[self.slice_axis]
                # Select slices that contain tumor (positive) + some background
                tumor_slices = []
                background_slices = []

                for i in range(n_slices):
                    if self.slice_axis == 2:
                        img_sl = img_vol[:, :, i]
                        lbl_sl = lbl_vol[:, :, i]
                    elif self.slice_axis == 1:
                        img_sl = img_vol[:, i, :]
                        lbl_sl = lbl_vol[:, i, :]
                    else:
                        img_sl = img_vol[i, :, :]
                        lbl_sl = lbl_vol[i, :, :]

                    if lbl_sl.sum() > 10:
                        tumor_slices.append((img_sl, lbl_sl))
                    else:
                        background_slices.append((img_sl, lbl_sl))

                # Balance: all tumor slices + equal background
                selected = tumor_slices
                bg_count = min(len(tumor_slices), len(background_slices))
                if bg_count > 0:
                    selected += random.sample(background_slices, bg_count)

                # Limit per volume
                if len(selected) > self.max_slices:
                    selected = random.sample(selected, self.max_slices)

                for img_sl, lbl_sl in selected:
                    img_norm = self._normalize_slice(img_sl)
                    img_resized = cv2.resize(img_norm, (self.img_size, self.img_size))
                    lbl_resized = cv2.resize(lbl_sl, (self.img_size, self.img_size),
                                             interpolation=cv2.INTER_NEAREST)
                    self.samples.append((img_resized, lbl_resized))

            except Exception as e:
                print(f"[Dataset] Error loading {vol_path}: {e}")
                continue

    def _load_2d_slices(self, img_dir: Path, lbl_dir: Path):
        """Load paired 2D PNG/JPEG image and mask slices."""
        import cv2
        exts = {".png", ".jpg", ".jpeg"}
        img_files = sorted([f for f in img_dir.iterdir() if f.suffix.lower() in exts])

        # Split train/val
        random.seed(42)
        random.shuffle(img_files)
        split_idx = int(len(img_files) * 0.85)
        if self.split == "train":
            img_files = img_files[:split_idx]
        else:
            img_files = img_files[split_idx:]

        for img_path in img_files:
            lbl_path = lbl_dir / img_path.name
            if not lbl_path.exists():
                continue
            try:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                lbl = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (self.img_size, self.img_size))
                lbl = cv2.resize(lbl, (self.img_size, self.img_size),
                                 interpolation=cv2.INTER_NEAREST)
                img_norm = img.astype(np.float32) / 255.0
                lbl_bin = (lbl > 127).astype(np.float32)
                self.samples.append((img_norm, lbl_bin))
            except Exception:
                continue

    def _generate_synthetic(self, n: int = 200):
        """
        Generate synthetic training samples for pipeline testing.
        Creates circular 'tumor' blobs on random background.
        """
        rng = np.random.default_rng(42)
        h, w = self.img_size, self.img_size

        for _ in range(n):
            # Background: random grayscale noise texture
            img = rng.random((h, w)).astype(np.float32)
            img = (img * 0.4 + 0.2).clip(0, 1)  # soft tissue range

            mask = np.zeros((h, w), dtype=np.float32)
            has_tumor = rng.random() > 0.3   # 70% have tumor

            if has_tumor:
                # Random elliptical tumor blob
                cx = rng.integers(h // 4, 3 * h // 4)
                cy = rng.integers(w // 4, 3 * w // 4)
                rx = rng.integers(10, h // 5)
                ry = rng.integers(10, w // 5)

                yy, xx = np.ogrid[:h, :w]
                ellipse = ((yy - cx) / rx) ** 2 + ((xx - cy) / ry) ** 2 <= 1.0
                mask[ellipse] = 1.0
                # Enhance intensity in tumor region
                img[ellipse] = np.clip(img[ellipse] + 0.3 + rng.random(ellipse.sum()) * 0.2, 0, 1)

            self.samples.append((img, mask))

    @staticmethod
    def _normalize_slice(sl: np.ndarray) -> np.ndarray:
        """Normalize a single slice to [0, 1]."""
        sl_min, sl_max = sl.min(), sl.max()
        if sl_max > sl_min:
            return ((sl - sl_min) / (sl_max - sl_min)).astype(np.float32)
        return np.zeros_like(sl, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self.samples[idx]

        # Apply augmentation if provided
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Convert to 3-channel tensor with ImageNet normalization
        if isinstance(img, np.ndarray):
            img_3ch = np.stack([img, img, img], axis=0).astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img_3ch = (img_3ch - mean) / std
            img_tensor = torch.from_numpy(img_3ch)
        else:
            img_tensor = img  # Already a tensor from albumentations

        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(
                mask.astype(np.float32)).unsqueeze(0)  # (1, H, W)
        else:
            mask_tensor = mask.unsqueeze(0) if mask.dim() == 2 else mask

        return img_tensor, mask_tensor
