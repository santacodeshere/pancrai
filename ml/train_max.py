"""
PancrAI — Maximum Quality Training Configuration
Combines Medical Segmentation Decathlon Task07 + NIH Pancreas-CT datasets.
Applies state-of-the-art training techniques for best possible performance.

Usage:
    python -m ml.train_max \
        --decathlon_dir ./data/Task07_Pancreas \
        --nih_dir       ./data/NIH_nifti \
        --output_dir    ./weights \
        --epochs        150

With GPU:
    python -m ml.train_max \
        --decathlon_dir ./data/Task07_Pancreas \
        --nih_dir       ./data/NIH_nifti \
        --img_size      512 \
        --batch_size    4 \
        --epochs        150 \
        --mixed_precision
"""

import os
import time
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from pathlib import Path
from typing import Optional, List, Tuple, Callable
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

from app.models.transunet import TransUNet
from ml.losses import CombinedDiceBCELoss, FocalTverskyLoss
from ml.metrics import dice_score, iou_score, sensitivity, specificity, hausdorff_distance
from ml.augmentation import get_train_transforms, get_val_transforms


# ─── Combined Loss (best for pancreas segmentation) ──────────────────────────

class MaxQualityLoss(nn.Module):
    """
    Combined loss proven to work best for small pancreatic tumor segmentation:
      0.4 × Dice + 0.3 × BCE + 0.3 × FocalTversky

    FocalTversky penalizes false negatives (missed tumors) more heavily,
    which is critical in medical imaging where missing a tumor is dangerous.
    """
    def __init__(self):
        super().__init__()
        self.dice_bce = CombinedDiceBCELoss(dice_weight=0.4, bce_weight=0.3)
        self.focal_tversky = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        return self.dice_bce(logits, targets) + 0.3 * self.focal_tversky(logits, targets)


# ─── Enhanced Dataset with multi-window CT support ───────────────────────────

class EnhancedPancreasDataset(Dataset):
    """
    Enhanced dataset that extracts 2D slices from NIfTI volumes with:
    - Multi-window CT processing (soft tissue + pancreas-specific windows)
    - Positive/negative slice balancing (3:1 tumor:background ratio)
    - Multi-axis slicing (axial + coronal views for more diversity)
    - Per-volume statistics normalization

    Supports both Decathlon Task07 and NIH Pancreas-CT layouts.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        transform: Optional[Callable] = None,
        max_slices_per_volume: int = 50,
        positive_ratio: float = 0.75,
        multi_axis: bool = True,
    ):
        self.img_size = img_size
        self.transform = transform
        self.max_slices = max_slices_per_volume
        self.positive_ratio = positive_ratio
        self.multi_axis = multi_axis
        self.samples: List[Tuple[np.ndarray, np.ndarray]] = []
        self.is_positive: List[bool] = []   # track which samples have tumor
        self._load(Path(data_dir), split)

    def _load(self, data_dir: Path, split: str):
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required. Run: pip install nibabel"
            )

        # Detect layout: Decathlon vs NIH (flat)
        img_dir = data_dir / ("imagesTr" if split == "train" else "imagesTs")
        lbl_dir = data_dir / ("labelsTr" if split == "train" else "labelsTs")

        # Fallback: flat layout (NIH style after conversion)
        if not img_dir.exists():
            img_dir = data_dir / "images"
            lbl_dir = data_dir / "masks"

        if not img_dir.exists():
            raise FileNotFoundError(
                f"No image directory found in {data_dir}.\n"
                f"Expected: {data_dir}/imagesTr/ or {data_dir}/images/"
            )

        vol_files = sorted(img_dir.glob("*.nii.gz")) + sorted(img_dir.glob("*.nii"))
        if not vol_files:
            raise FileNotFoundError(f"No .nii.gz files found in {img_dir}")

        # 85/15 train/val split if no separate val dir
        if split == "val" and not (data_dir / "imagesTs").exists():
            random.seed(42)
            random.shuffle(vol_files)
            cut = int(len(vol_files) * 0.85)
            vol_files = vol_files[cut:]
        elif split == "train" and not (data_dir / "imagesTs").exists():
            random.seed(42)
            random.shuffle(vol_files)
            cut = int(len(vol_files) * 0.85)
            vol_files = vol_files[:cut]

        print(f"[Dataset] Loading {len(vol_files)} volumes ({split}) from {data_dir.name}...")

        for vol_path in tqdm(vol_files, desc=f"Loading {split}", leave=False):
            lbl_path = lbl_dir / vol_path.name
            if not lbl_path.exists():
                # Try without _0000 suffix (Decathlon naming)
                lbl_path = lbl_dir / vol_path.name.replace("_0000", "")
            if not lbl_path.exists():
                continue

            try:
                img_nib = nib.load(str(vol_path))
                lbl_nib = nib.load(str(lbl_path))
                img_vol = img_nib.get_fdata().astype(np.float32)
                lbl_vol = (lbl_nib.get_fdata() > 0).astype(np.float32)

                # Extract slices along axial axis (and coronal if multi_axis)
                axes = [2] if not self.multi_axis else [2, 1]
                for axis in axes:
                    self._extract_slices(img_vol, lbl_vol, axis)

            except Exception as e:
                print(f"  [!] Skipping {vol_path.name}: {e}")
                continue

        print(f"[Dataset] {split}: {len(self.samples)} slices "
              f"({sum(self.is_positive)} positive, "
              f"{len(self.samples)-sum(self.is_positive)} negative)")

    def _extract_slices(self, img_vol: np.ndarray,
                        lbl_vol: np.ndarray, axis: int):
        """Extract and balance 2D slices from a 3D volume."""
        n = img_vol.shape[axis]
        positive_slices = []
        negative_slices = []

        for i in range(n):
            if axis == 2:
                img_sl = img_vol[:, :, i]
                lbl_sl = lbl_vol[:, :, i]
            elif axis == 1:
                img_sl = img_vol[:, i, :]
                lbl_sl = lbl_vol[:, i, :]
            else:
                img_sl = img_vol[i, :, :]
                lbl_sl = lbl_vol[i, :, :]

            img_proc = self._process_ct_slice(img_sl)
            lbl_bin = (lbl_sl > 0).astype(np.float32)

            if lbl_bin.sum() > 20:
                positive_slices.append((img_proc, lbl_bin))
            elif lbl_bin.sum() == 0:
                negative_slices.append((img_proc, lbl_bin))

        # All positive slices + proportional negatives
        selected_pos = positive_slices
        n_neg = min(
            int(len(selected_pos) * (1 - self.positive_ratio) / self.positive_ratio),
            len(negative_slices)
        )
        selected_neg = random.sample(negative_slices, n_neg) if n_neg > 0 else []
        selected = selected_pos + selected_neg

        # Limit per volume
        if len(selected) > self.max_slices:
            selected = random.sample(selected, self.max_slices)

        for img_sl, lbl_sl in selected:
            img_r = cv2.resize(img_sl, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_LINEAR)
            lbl_r = cv2.resize(lbl_sl, (self.img_size, self.img_size),
                               interpolation=cv2.INTER_NEAREST)
            self.samples.append((img_r, lbl_r))
            self.is_positive.append(lbl_sl.sum() > 0)

    def _process_ct_slice(self, sl: np.ndarray) -> np.ndarray:
        """
        Apply dual-window CT processing.
        Combines soft-tissue window and pancreas-specific window
        into a normalized float32 image.
        """
        # Window 1: Soft tissue (WC=40, WW=400)
        w1 = np.clip(sl, -160, 240)
        w1 = (w1 + 160) / 400.0

        # Window 2: Pancreas-specific (WC=50, WW=200)
        w2 = np.clip(sl, -50, 150)
        w2 = (w2 + 50) / 200.0

        # Average both windows for richer contrast information
        combined = (w1 * 0.6 + w2 * 0.4).astype(np.float32)
        return np.clip(combined, 0.0, 1.0)

    def get_sampler_weights(self) -> List[float]:
        """
        Returns per-sample weights for WeightedRandomSampler.
        Positive (tumor) samples get higher weight to ensure
        they appear more frequently during training.
        """
        pos_weight = 3.0
        neg_weight = 1.0
        return [pos_weight if p else neg_weight for p in self.is_positive]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]

        if isinstance(img, np.ndarray):
            rgb = np.stack([img, img, img], axis=0).astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
            std  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
            rgb  = (rgb - mean) / std
            img_t = torch.from_numpy(rgb)
        else:
            img_t = img

        if isinstance(mask, np.ndarray):
            mask_t = torch.from_numpy(
                mask.astype(np.float32)).unsqueeze(0)
        else:
            mask_t = mask.unsqueeze(0) if mask.dim() == 2 else mask

        return img_t, mask_t


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion,
                    scaler, device, epoch, grad_clip=1.0):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    n = len(loader)
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)

    for imgs, masks in pbar:
        imgs  = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler:
            with autocast():
                logits = model(imgs)
                loss   = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            d = dice_score(torch.sigmoid(logits), masks)
        total_loss += loss.item()
        total_dice += d
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    sums = {"loss": 0, "dice": 0, "iou": 0,
            "sensitivity": 0, "specificity": 0, "hausdorff": 0}
    n = len(loader)

    for imgs, masks in tqdm(loader, desc="Val", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        preds  = torch.sigmoid(logits)
        sums["loss"]        += criterion(logits, masks).item()
        sums["dice"]        += dice_score(preds, masks)
        sums["iou"]         += iou_score(preds, masks)
        sums["sensitivity"] += sensitivity(preds, masks)
        sums["specificity"] += specificity(preds, masks)
        sums["hausdorff"]   += hausdorff_distance(preds, masks)

    return {k: v / n for k, v in sums.items()}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PancrAI Maximum Quality Training"
    )
    parser.add_argument("--decathlon_dir", type=str,
                        default="./data/Task07_Pancreas",
                        help="Path to Medical Segmentation Decathlon Task07")
    parser.add_argument("--nih_dir", type=str, default="",
                        help="Path to NIH Pancreas-CT (optional, adds more data)")
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=224,
                        help="224 for CPU/low-VRAM, 512 for high-VRAM GPU")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="8 for 8GB GPU, 4 for 6GB, 2 for CPU")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience")
    parser.add_argument("--mixed_precision", action="store_true",
                        default=False,
                        help="Use AMP (requires CUDA GPU, speeds up 2-3x)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_slices", type=int, default=50,
                        help="Max slices per volume (increase if RAM allows)")
    parser.add_argument("--multi_axis", action="store_true", default=True,
                        help="Extract axial + coronal slices (doubles data)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("  PancrAI — Maximum Quality Training")
    print(f"{'='*60}")
    print(f"  Device      : {device}")
    if device.type == "cuda":
        print(f"  GPU         : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM        : {mem:.1f} GB")
    print(f"  Image size  : {args.img_size}×{args.img_size}")
    print(f"  Batch size  : {args.batch_size}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Mixed prec  : {args.mixed_precision}")
    print(f"{'='*60}\n")

    # ── Build datasets ──
    train_datasets = []
    val_datasets   = []

    # Primary: Decathlon Task07
    if Path(args.decathlon_dir).exists():
        print(f"Loading Decathlon Task07 from {args.decathlon_dir}...")
        train_datasets.append(EnhancedPancreasDataset(
            args.decathlon_dir, "train", args.img_size,
            transform=get_train_transforms(args.img_size),
            max_slices_per_volume=args.max_slices,
            multi_axis=args.multi_axis,
        ))
        val_datasets.append(EnhancedPancreasDataset(
            args.decathlon_dir, "val", args.img_size,
            transform=get_val_transforms(args.img_size),
            max_slices_per_volume=args.max_slices,
            multi_axis=False,
        ))
    else:
        print(f"[!] Decathlon dir not found: {args.decathlon_dir}")
        print("    Falling back to synthetic data for testing...")
        from ml.dataset import PancreasDataset
        train_datasets.append(PancreasDataset(
            args.decathlon_dir, "train", args.img_size,
            transform=get_train_transforms(args.img_size),
        ))
        val_datasets.append(PancreasDataset(
            args.decathlon_dir, "val", args.img_size,
            transform=get_val_transforms(args.img_size),
        ))

    # Optional: NIH Pancreas-CT (additional data)
    if args.nih_dir and Path(args.nih_dir).exists():
        print(f"Loading NIH Pancreas-CT from {args.nih_dir}...")
        train_datasets.append(EnhancedPancreasDataset(
            args.nih_dir, "train", args.img_size,
            transform=get_train_transforms(args.img_size),
            max_slices_per_volume=args.max_slices,
            multi_axis=args.multi_axis,
        ))
        val_datasets.append(EnhancedPancreasDataset(
            args.nih_dir, "val", args.img_size,
            transform=get_val_transforms(args.img_size),
            max_slices_per_volume=args.max_slices,
            multi_axis=False,
        ))
        print("NIH dataset added — combined training for better generalization")

    # Combine datasets
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 \
               else train_datasets[0]
    val_ds   = ConcatDataset(val_datasets)   if len(val_datasets) > 1   \
               else val_datasets[0]

    print(f"\nTotal training slices : {len(train_ds)}")
    print(f"Total validation slices: {len(val_ds)}")

    # Weighted sampler — oversample tumor slices
    if hasattr(train_ds, "get_sampler_weights"):
        weights = train_ds.get_sampler_weights()
    else:
        # For ConcatDataset, merge weights from sub-datasets
        weights = []
        for ds in train_datasets:
            if hasattr(ds, "get_sampler_weights"):
                weights.extend(ds.get_sampler_weights())
            else:
                weights.extend([1.0] * len(ds))

    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ──
    model = TransUNet(img_size=args.img_size, pretrained=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {n_params:,}")

    # ── Optimizer: different LR for encoder vs decoder ──
    encoder_params = []
    decoder_params = []
    for name, param in model.named_parameters():
        if "enc" in name or "vit" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    optimizer = AdamW([
        {"params": encoder_params, "lr": args.lr * 0.1},   # slower for pretrained
        {"params": decoder_params, "lr": args.lr},          # faster for new decoder
    ], weight_decay=1e-4)

    # Cosine annealing with warm restarts — helps escape local minima
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2, eta_min=1e-7
    )

    criterion = MaxQualityLoss()
    scaler    = GradScaler() if (args.mixed_precision and
                                  device.type == "cuda") else None

    # ── Training loop ──
    best_dice     = 0.0
    patience_ctr  = 0
    history = {k: [] for k in [
        "train_loss", "train_dice", "val_loss", "val_dice",
        "val_iou", "val_sensitivity", "val_specificity", "val_hausdorff",
    ]}

    print(f"\nStarting training for up to {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_dice = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, epoch,
        )
        val_m = validate(model, val_loader, criterion, device)
        scheduler.step(epoch)

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[1]["lr"]

        print(
            f"Ep {epoch:03d}/{args.epochs} | "
            f"Loss {tr_loss:.4f} | Dice {tr_dice:.4f} | "
            f"ValDice {val_m['dice']:.4f} | "
            f"ValIoU {val_m['iou']:.4f} | "
            f"Sens {val_m['sensitivity']:.4f} | "
            f"HD {val_m['hausdorff']:.1f} | "
            f"LR {lr_now:.2e} | {elapsed:.0f}s"
        )

        # Record history
        history["train_loss"].append(tr_loss)
        history["train_dice"].append(tr_dice)
        history["val_loss"].append(val_m["loss"])
        history["val_dice"].append(val_m["dice"])
        history["val_iou"].append(val_m["iou"])
        history["val_sensitivity"].append(val_m["sensitivity"])
        history["val_specificity"].append(val_m["specificity"])
        history["val_hausdorff"].append(val_m["hausdorff"])

        # Save best
        if val_m["dice"] > best_dice:
            best_dice    = val_m["dice"]
            patience_ctr = 0
            ckpt = os.path.join(args.output_dir, "transunet_best.pth")
            torch.save({
                "epoch":      epoch,
                "model":      model.state_dict(),
                "optimizer":  optimizer.state_dict(),
                "best_dice":  best_dice,
                "val_metrics": val_m,
                "config":     vars(args),
            }, ckpt)
            print(f"  ✓ Best model saved — Dice={best_dice:.4f}")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {args.patience} epochs)")
                break

        # Save periodic checkpoint every 25 epochs
        if epoch % 25 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(args.output_dir,
                             f"transunet_epoch{epoch:03d}.pth")
            )

    # ── Save history and plots ──
    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    _plot_history(history, args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best Validation Dice : {best_dice:.4f}")
    print(f"  Weights saved to     : {args.output_dir}/transunet_best.pth")
    print(f"{'='*60}\n")


def _plot_history(history: dict, output_dir: str):
    """Save training curve plots."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PancrAI — Training History", fontsize=14, fontweight="bold")
    pairs = [
        ("train_loss",     "val_loss",        "Loss",        axes[0, 0]),
        ("train_dice",     "val_dice",        "Dice Score",  axes[0, 1]),
        (None,             "val_iou",         "IoU",         axes[0, 2]),
        (None,             "val_sensitivity", "Sensitivity", axes[1, 0]),
        (None,             "val_specificity", "Specificity", axes[1, 1]),
        (None,             "val_hausdorff",   "Hausdorff",   axes[1, 2]),
    ]
    for train_k, val_k, title, ax in pairs:
        eps = range(1, len(history.get(val_k, [])) + 1)
        if train_k and train_k in history:
            ax.plot(eps, history[train_k], label="Train",
                    color="#1565C0", linewidth=2)
        if val_k in history:
            ax.plot(eps, history[val_k], label="Val",
                    color="#E53935", linewidth=2)
        ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {output_dir}/training_curves.png")


if __name__ == "__main__":
    main()