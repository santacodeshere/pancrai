"""
PancrAI — Medical Image Augmentation
Augmentations applied consistently to both image and mask.
Uses albumentations for efficient paired transforms.
"""

import numpy as np
from typing import Dict, Any

try:
    import albumentations as A
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False


def get_train_transforms(img_size: int = 224):
    """
    Training augmentation pipeline with medical-specific transforms.
    Applied to both image and mask consistently.
    """
    if not HAS_ALBUMENTATIONS:
        return _numpy_transforms(train=True, img_size=img_size)

    return A.Compose([
        # Geometric transforms (applied to both image and mask)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, interpolation=1, border_mode=0, p=0.5),
        A.ElasticTransform(
            alpha=120, sigma=120 * 0.05,
            interpolation=1, border_mode=0, p=0.3
        ),
        A.RandomResizedCrop(
            size=(img_size, img_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=1,
            p=0.4
        ),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
            interpolation=1, border_mode=0, p=0.4
        ),

        # Intensity transforms (image only — mask stays binary)
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),

        # Ensure output is resized to target
        A.Resize(img_size, img_size),
    ])


def get_val_transforms(img_size: int = 224):
    """
    Validation transforms: only resize, no augmentation.
    """
    if not HAS_ALBUMENTATIONS:
        return _numpy_transforms(train=False, img_size=img_size)

    return A.Compose([
        A.Resize(img_size, img_size),
    ])


# ─── Numpy Fallback Transforms ───────────────────────────────────────────────

class NumpyAugmentation:
    """
    Pure-numpy fallback augmentation when albumentations is not installed.
    """

    def __init__(self, train: bool = True, img_size: int = 224):
        self.train = train
        self.img_size = img_size

    def __call__(self, image: np.ndarray,
                 mask: np.ndarray) -> Dict[str, np.ndarray]:
        import cv2

        img = image.astype(np.float32)
        msk = mask.astype(np.float32)

        if self.train:
            if np.random.random() > 0.5:
                img = np.fliplr(img).copy()
                msk = np.fliplr(msk).copy()

            if np.random.random() > 0.7:
                img = np.flipud(img).copy()
                msk = np.flipud(msk).copy()

            if np.random.random() > 0.5:
                angle = np.random.uniform(-15, 15)
                h, w = img.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
                msk = cv2.warpAffine(msk, M, (w, h), flags=cv2.INTER_NEAREST)

            if np.random.random() > 0.5:
                alpha = np.random.uniform(0.8, 1.2)
                beta = np.random.uniform(-0.1, 0.1)
                img = np.clip(img * alpha + beta, 0.0, 1.0)

            if np.random.random() > 0.7:
                noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)

            if np.random.random() > 0.6:
                scale = np.random.uniform(0.9, 1.1)
                h, w = img.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                img = cv2.resize(img, (new_w, new_h))
                msk = cv2.resize(msk, (new_w, new_h),
                                 interpolation=cv2.INTER_NEAREST)
                img = _crop_or_pad(img, self.img_size)
                msk = _crop_or_pad(msk, self.img_size)

        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_NEAREST)

        return {"image": img, "mask": msk}


def _crop_or_pad(arr: np.ndarray, target: int) -> np.ndarray:
    """Crop center or zero-pad array to target square size."""
    h, w = arr.shape[:2]
    if h >= target and w >= target:
        y0 = (h - target) // 2
        x0 = (w - target) // 2
        return arr[y0:y0 + target, x0:x0 + target]
    else:
        pad_h = max(0, target - h)
        pad_w = max(0, target - w)
        return np.pad(arr, ((0, pad_h), (0, pad_w)), mode="constant")


def _numpy_transforms(train: bool, img_size: int):
    """Return numpy augmentation callable."""
    return NumpyAugmentation(train=train, img_size=img_size)