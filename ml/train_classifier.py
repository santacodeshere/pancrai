"""
PancrAI — EfficientNetB4 Classifier Training
Fine-tunes the EfficientNetB4 classifier on tumor class labels.

Dataset: expects a directory with class subdirectories:
    data/classifier/
        0_no_tumor/         ← cropped regions with no tumor
        1_benign/           ← benign tumor crops
        2_malignant/        ← malignant (PDAC) crops
        3_cystic/           ← cystic (IPMN) crops

Or auto-generates synthetic class samples for testing.

Usage:
    python -m ml.train_classifier --data_dir ./data/classifier --epochs 50
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
import cv2
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from app.models.classifier import PancreasTumorClassifier, CLASS_NAMES


# ─── Dataset ─────────────────────────────────────────────────────────────────

class TumorCropDataset(Dataset):
    """
    Dataset of pre-cropped tumor/background regions for classification.

    Directory structure:
        data_dir/
            0_no_tumor/*.png
            1_benign/*.png
            2_malignant/*.png
            3_cystic/*.png

    Falls back to synthetic generation if no real data found.
    """

    def __init__(self, data_dir: str, split: str = "train",
                 img_size: int = 224, augment: bool = True):
        self.img_size = img_size
        self.augment = augment
        self.samples = []   # list of (image_array, class_idx)
        self._load(data_dir, split)

    def _load(self, data_dir: str, split: str):
        class_dirs = {
            0: "0_no_tumor", 1: "1_benign",
            2: "2_malignant", 3: "3_cystic",
        }
        found_any = False
        for cls_idx, dirname in class_dirs.items():
            cls_dir = Path(data_dir) / dirname
            if not cls_dir.exists():
                continue
            files = list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpg"))
            if not files:
                continue
            found_any = True

            # Split 85/15
            import random
            random.seed(42)
            random.shuffle(files)
            cut = int(len(files) * 0.85)
            files = files[:cut] if split == "train" else files[cut:]

            for f in files:
                img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.samples.append((img, cls_idx))

        if not found_any:
            print(f"[ClassifierDataset] No data found — generating synthetic samples")
            self._generate_synthetic(200 if split == "train" else 50)
        else:
            print(f"[ClassifierDataset] Loaded {len(self.samples)} samples ({split})")

    def _generate_synthetic(self, n: int):
        """Generate synthetic crop images per class."""
        from demo_data_generator import generate_ct_background, generate_tumor
        rng = np.random.default_rng(99)
        classes = [0, 1, 2, 3]
        tumor_types = [None, "benign", "malignant", "cystic"]

        for cls_idx, ttype in zip(classes, tumor_types):
            for _ in range(n // 4):
                bg = generate_ct_background(self.img_size, rng)
                if ttype is None:
                    img = (bg * 255).astype(np.uint8)
                else:
                    ti, tm, _ = generate_tumor(self.img_size, rng, ttype)
                    img = np.clip((bg + ti) * 255, 0, 255).astype(np.uint8)
                self.samples.append((img, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        img = cv2.resize(img, (self.img_size, self.img_size))

        if self.augment:
            img = self._augment(img)

        # Normalize and convert to 3-channel tensor
        img_f = img.astype(np.float32) / 255.0
        rgb = np.stack([img_f] * 3, axis=0)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        rgb = (rgb - mean) / std

        return torch.from_numpy(rgb), torch.tensor(label, dtype=torch.long)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        if np.random.random() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.random() > 0.5:
            img = np.flipud(img).copy()
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 5, img.shape).astype(np.int16)
            img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        return img


# ─── Training ─────────────────────────────────────────────────────────────────

def train_classifier(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ClassifierTrain] Device: {device}")
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_ds = TumorCropDataset(cfg.data_dir, "train", cfg.img_size, augment=True)
    val_ds = TumorCropDataset(cfg.data_dir, "val", cfg.img_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                               shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size,
                             shuffle=False, num_workers=2)

    model = PancreasTumorClassifier(pretrained=cfg.pretrained).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if torch.cuda.is_available() else None

    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, cfg.epochs + 1):
        # ── Train ──
        model.train()
        t_loss = 0.0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                with autocast():
                    loss = criterion(model(X), y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(X), y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            t_loss += loss.item()
        scheduler.step()

        # ── Validate ──
        model.eval()
        v_loss, correct, total = 0.0, 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                v_loss += criterion(logits, y).item()
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(y.cpu().tolist())

        acc = correct / max(total, 1)
        avg_tloss = t_loss / len(train_loader)
        avg_vloss = v_loss / len(val_loader)

        print(f"Epoch {epoch:03d} | TrainLoss: {avg_tloss:.4f} | "
              f"ValLoss: {avg_vloss:.4f} | ValAcc: {acc*100:.1f}%")

        history["train_loss"].append(avg_tloss)
        history["val_loss"].append(avg_vloss)
        history["val_acc"].append(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "best_acc": best_acc},
                       os.path.join(cfg.output_dir, "efficientnet_best.pth"))
            print(f"  ✓ Best model saved (Acc={best_acc*100:.1f}%)")

    # Final classification report
    print("\n" + classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

    # Save history
    with open(os.path.join(cfg.output_dir, "classifier_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].legend(); axes[0].set_title("Loss")
    axes[1].plot(history["val_acc"], label="Val Accuracy", color="green")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.output_dir, "classifier_curves.png"))
    print(f"\n[ClassifierTrain] Best accuracy: {best_acc*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/classifier")
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pretrained", action="store_true", default=True)
    cfg = parser.parse_args()
    train_classifier(cfg)
