"""
PancrAI — Model Evaluation
Evaluate trained TransUNet on a test set and generate a full metrics report.
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml.dataset import PancreasDataset
from ml.augmentation import get_val_transforms
from ml.metrics import compute_all_metrics, dice_score
from app.models.transunet import build_transunet


def evaluate(model, loader, device, threshold=0.5):
    """Run evaluation and return per-sample and aggregate metrics."""
    model.eval()
    all_metrics = []
    visual_samples = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            preds = model.predict(images)
            metrics = compute_all_metrics(preds, masks, threshold)
            all_metrics.append(metrics)

            # Store a few samples for visualization
            if len(visual_samples) < 8:
                visual_samples.append({
                    "image": images[0, 0].cpu().numpy(),
                    "mask": masks[0, 0].cpu().numpy(),
                    "pred": preds[0, 0].cpu().numpy(),
                })

    # Aggregate
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        agg[key] = {
            "mean": round(float(np.mean(vals)), 4),
            "std": round(float(np.std(vals)), 4),
            "min": round(float(np.min(vals)), 4),
            "max": round(float(np.max(vals)), 4),
        }

    return agg, visual_samples


def plot_predictions(samples, output_dir, n=8):
    """Plot sample predictions with ground truth overlay."""
    n = min(n, len(samples))
    fig = plt.figure(figsize=(4 * n, 10))
    gs = gridspec.GridSpec(3, n, hspace=0.1, wspace=0.05)

    for i, s in enumerate(samples[:n]):
        img = s["image"]
        mask_gt = s["mask"]
        pred = s["pred"]
        pred_bin = (pred > 0.5).astype(float)

        d = dice_score(
            torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
            torch.from_numpy(mask_gt).unsqueeze(0).unsqueeze(0),
        )

        ax1 = fig.add_subplot(gs[0, i])
        ax1.imshow(img, cmap="gray")
        ax1.set_title("Input", fontsize=8)
        ax1.axis("off")

        ax2 = fig.add_subplot(gs[1, i])
        ax2.imshow(img, cmap="gray")
        ax2.imshow(mask_gt, alpha=0.4, cmap="Reds")
        ax2.set_title("GT Mask", fontsize=8)
        ax2.axis("off")

        ax3 = fig.add_subplot(gs[2, i])
        ax3.imshow(img, cmap="gray")
        ax3.imshow(pred_bin, alpha=0.4, cmap="Blues")
        ax3.set_title(f"Pred (D={d:.3f})", fontsize=8)
        ax3.axis("off")

    plt.suptitle("PancrAI — Segmentation Predictions", fontsize=12, fontweight="bold")
    out_path = os.path.join(output_dir, "evaluation_predictions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Prediction visualization saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/Task07_Pancreas")
    parser.add_argument("--weights", type=str, default="./weights/transunet_best.pth")
    parser.add_argument("--output_dir", type=str, default="./weights")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    test_ds = PancreasDataset(
        data_dir=args.data_dir,
        split="val",
        img_size=args.img_size,
        transform=get_val_transforms(args.img_size),
    )
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=2)

    # Load model
    model = build_transunet(img_size=args.img_size, weights_path=args.weights)
    model.to(device)

    # Evaluate
    print(f"Evaluating on {len(test_ds)} samples...")
    metrics, visuals = evaluate(model, loader, device, args.threshold)

    # Print report
    print("\n" + "=" * 50)
    print("PancrAI — Evaluation Results")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}  "
              f"[{v['min']:.4f}, {v['max']:.4f}]")
    print("=" * 50)

    # Save metrics
    out_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_path}")

    # Save visualization
    plot_predictions(visuals, args.output_dir)


if __name__ == "__main__":
    main()
