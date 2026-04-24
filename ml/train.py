"""
PancrAI — Full Training Pipeline
Trains TransUNet on Medical Segmentation Decathlon Task07 / NIH Pancreas-CT.
"""

import os
import time
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from ml.augmentation import get_train_transforms, get_val_transforms
from ml.losses import CombinedDiceBCELoss
from ml.metrics import dice_score, iou_score, sensitivity, specificity, hausdorff_distance
from app.models.transunet import TransUNet


# ─── Configuration ───────────────────────────────────────────────────────────

def get_config():
    parser = argparse.ArgumentParser(description="Train TransUNet for pancreatic tumor segmentation")
    parser.add_argument("--data_dir", type=str, default="./data/Task07_Pancreas")
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (increased to 50)")
    parser.add_argument("--mixed_precision", action="store_true", default=True)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="0 = main process only (safer on Windows)")
    parser.add_argument("--val_split", type=float, default=0.15,
                        help="Fraction of volumes to use for validation")
    parser.add_argument("--max_slices", type=int, default=10,
                        help="Max slices per volume (10 = fast, 30 = thorough)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# ─── Dataset builder with real train/val split ────────────────────────────────

def build_datasets(cfg):
    """
    Split NIfTI volumes in imagesTr/ into train and val sets.
    This avoids the synthetic fallback caused by missing imagesVal/ folder.
    """
    import nibabel as nib
    import cv2

    img_dir = Path(cfg.data_dir) / "imagesTr"
    lbl_dir = Path(cfg.data_dir) / "labelsTr"

    # Get all valid volume pairs
    all_vols = sorted([
        f for f in img_dir.glob("*.nii.gz")
        if not f.name.startswith("._")
        and (lbl_dir / f.name).exists()
    ])

    print(f"[Dataset] Found {len(all_vols)} valid volume pairs")

    # Reproducible shuffle and split
    random.seed(cfg.seed)
    random.shuffle(all_vols)
    n_val = max(1, int(len(all_vols) * cfg.val_split))
    val_vols = all_vols[:n_val]
    train_vols = all_vols[n_val:]

    print(f"[Dataset] Train volumes: {len(train_vols)} | Val volumes: {len(val_vols)}")

    def extract_slices(vol_paths, max_slices, is_train):
        """Extract 2D slices from a list of NIfTI volumes."""
        samples = []
        for vol_path in vol_paths:
            lbl_path = lbl_dir / vol_path.name
            try:
                img_vol = nib.load(str(vol_path)).get_fdata().astype(np.float32)
                lbl_vol = nib.load(str(lbl_path)).get_fdata().astype(np.float32)
                lbl_vol = (lbl_vol > 0).astype(np.float32)

                n_slices = img_vol.shape[2]
                tumor_slices, bg_slices = [], []

                for i in range(n_slices):
                    img_sl = img_vol[:, :, i]
                    lbl_sl = lbl_vol[:, :, i]
                    # Normalize slice
                    sl_min, sl_max = img_sl.min(), img_sl.max()
                    if sl_max > sl_min:
                        img_sl = (img_sl - sl_min) / (sl_max - sl_min)
                    else:
                        img_sl = np.zeros_like(img_sl)
                    # Resize
                    img_sl = cv2.resize(img_sl.astype(np.float32),
                                        (cfg.img_size, cfg.img_size))
                    lbl_sl = cv2.resize(lbl_sl, (cfg.img_size, cfg.img_size),
                                        interpolation=cv2.INTER_NEAREST)
                    if lbl_sl.sum() > 10:
                        tumor_slices.append((img_sl, lbl_sl))
                    else:
                        bg_slices.append((img_sl, lbl_sl))

                # Balance tumor vs background
                selected = tumor_slices.copy()
                bg_count = min(len(tumor_slices), len(bg_slices))
                if bg_count > 0:
                    selected += random.sample(bg_slices, bg_count)

                # Limit slices per volume
                if len(selected) > max_slices:
                    selected = random.sample(selected, max_slices)

                samples.extend(selected)

            except Exception as e:
                print(f"[Dataset] Skipping {vol_path.name}: {e}")
                continue

        return samples

    print("[Dataset] Loading train slices...")
    train_samples = extract_slices(train_vols, cfg.max_slices, is_train=True)
    print(f"[Dataset] Train slices: {len(train_samples)}")

    print("[Dataset] Loading val slices...")
    val_samples = extract_slices(val_vols, cfg.max_slices, is_train=False)
    print(f"[Dataset] Val slices: {len(val_samples)}")

    return train_samples, val_samples


# ─── Simple Dataset wrapper ───────────────────────────────────────────────────

class SliceDataset(torch.utils.data.Dataset):
    """Wraps a list of (image, mask) numpy arrays with optional transforms."""

    def __init__(self, samples, transform=None, img_size=224):
        self.samples = samples
        self.transform = transform
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self.samples[idx]
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)

        if self.transform:
            try:
                aug = self.transform(image=img, mask=mask)
                img = aug["image"]
                mask = aug["mask"]
            except Exception:
                pass

        # Convert to 3-channel tensor
        if isinstance(img, np.ndarray):
            img_3ch = np.stack([img, img, img], axis=0).astype(np.float32)
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
            img_3ch = (img_3ch - mean) / std
            img_tensor = torch.from_numpy(img_3ch)
        else:
            img_tensor = img

        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        else:
            mask_tensor = mask.unsqueeze(0) if mask.dim() == 2 else mask

        return img_tensor, mask_tensor


# ─── Training Loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    n_batches = len(loader)

    pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [Train]", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast("cuda"):
                logits = model(images)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        with torch.no_grad():
            preds = torch.sigmoid(logits)
            d = dice_score(preds, masks)

        total_loss += loss.item()
        total_dice += d
        pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

    return total_loss / n_batches, total_dice / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    metrics_sum = {"dice": 0.0, "iou": 0.0, "sensitivity": 0.0,
                   "specificity": 0.0, "hausdorff": 0.0}
    n_batches = len(loader)

    for images, masks in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device)
        masks  = masks.to(device)
        logits = model(images)
        loss   = criterion(logits, masks)
        total_loss += loss.item()

        preds = torch.sigmoid(logits)
        metrics_sum["dice"]        += dice_score(preds, masks)
        metrics_sum["iou"]         += iou_score(preds, masks)
        metrics_sum["sensitivity"] += sensitivity(preds, masks)
        metrics_sum["specificity"] += specificity(preds, masks)
        metrics_sum["hausdorff"]   += hausdorff_distance(preds, masks)

    avg = {k: v / n_batches for k, v in metrics_sum.items()}
    avg["loss"] = total_loss / n_batches
    return avg


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_training_curves(history, output_dir):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("PancrAI Training Curves", fontsize=14, fontweight="bold")

    metrics = [
        ("loss",        "Loss",               axes[0, 0]),
        ("dice",        "Dice Score",          axes[0, 1]),
        ("iou",         "IoU Score",           axes[0, 2]),
        ("sensitivity", "Sensitivity",         axes[1, 0]),
        ("specificity", "Specificity",         axes[1, 1]),
        ("hausdorff",   "Hausdorff Distance",  axes[1, 2]),
    ]
    for key, label, ax in metrics:
        train_vals = history.get(f"train_{key}", [])
        val_vals   = history.get(f"val_{key}",   [])
        epochs     = range(1, max(len(train_vals), len(val_vals)) + 1)
        if train_vals:
            ax.plot(list(epochs)[:len(train_vals)], train_vals,
                    label="Train", color="#1565C0", linewidth=2)
        if val_vals:
            ax.plot(list(epochs)[:len(val_vals)], val_vals,
                    label="Val", color="#E53935", linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "training_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Train] Curves saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg = get_config()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Build real train/val split from imagesTr ──
    train_samples, val_samples = build_datasets(cfg)

    train_ds = SliceDataset(train_samples,
                            transform=get_train_transforms(cfg.img_size),
                            img_size=cfg.img_size)
    val_ds   = SliceDataset(val_samples,
                            transform=get_val_transforms(cfg.img_size),
                            img_size=cfg.img_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True, num_workers=cfg.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size,
                              shuffle=False, num_workers=cfg.num_workers,
                              pin_memory=True)

    print(f"[Train] Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")

    # ── Model ──
    model = TransUNet(img_size=cfg.img_size, pretrained=True)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] Trainable parameters: {n_params:,}")

    # ── Optimizer / Scheduler / Loss ──
    optimizer = AdamW(model.parameters(), lr=cfg.lr,
                      weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    criterion = CombinedDiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    scaler    = GradScaler("cuda") if cfg.mixed_precision and torch.cuda.is_available() else None

    # ── Training Loop ──
    best_dice       = 0.0
    patience_counter = 0
    history = {k: [] for k in [
        "train_loss", "val_loss", "val_dice", "val_iou",
        "val_sensitivity", "val_specificity", "val_hausdorff", "train_dice"
    ]}

    print(f"[Train] Starting training for {cfg.epochs} epochs "
          f"(patience={cfg.patience}, max_slices={cfg.max_slices})...")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device, epoch)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"Train Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Dice: {val_metrics['dice']:.4f} | "
            f"Val IoU: {val_metrics['iou']:.4f} | "
            f"Sensitivity: {val_metrics['sensitivity']:.4f} | "
            f"Hausdorff: {val_metrics['hausdorff']:.2f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Record history
        history["train_loss"].append(train_loss)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_metrics["loss"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_sensitivity"].append(val_metrics["sensitivity"])
        history["val_specificity"].append(val_metrics["specificity"])
        history["val_hausdorff"].append(val_metrics["hausdorff"])

        # Save best model
        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            patience_counter = 0
            ckpt_path = os.path.join(cfg.output_dir, "transunet_best.pth")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_dice": best_dice,
                "config": vars(cfg),
            }, ckpt_path)
            print(f"  ✓ Best model saved — Dice: {best_dice:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"[Train] Early stopping at epoch {epoch} "
                      f"(no improvement for {cfg.patience} epochs)")
                break

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(cfg.output_dir, f"transunet_epoch{epoch}.pth"))

        # Save history after every epoch (so you can check progress)
        with open(os.path.join(cfg.output_dir, "training_history.json"), "w") as f:
            json.dump(history, f, indent=2)

    plot_training_curves(history, cfg.output_dir)
    print(f"\n[Train] Done. Best Val Dice: {best_dice:.4f}")
    print(f"[Train] Weights → {os.path.join(cfg.output_dir, 'transunet_best.pth')}")


if __name__ == "__main__":
    main()