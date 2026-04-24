"""
PancrAI — Inference Benchmark
Measures latency, throughput, and memory usage of the full pipeline.

Usage:
    python -m ml.benchmark
    python -m ml.benchmark --n_runs 100 --batch_size 4 --img_size 224
"""

import argparse
import time
import numpy as np
import torch
import gc
from typing import Dict


def benchmark_segmentation(model, img_size: int = 224,
                            n_runs: int = 50, device: str = "cpu") -> Dict:
    """Benchmark TransUNet segmentation inference."""
    model.eval()
    model.to(device)

    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # Timed runs
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        "mean_latency_ms": round(np.mean(latencies), 2),
        "p50_latency_ms": round(np.percentile(latencies, 50), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "p99_latency_ms": round(np.percentile(latencies, 99), 2),
        "throughput_fps": round(1000 / np.mean(latencies), 1),
        "n_runs": n_runs,
        "device": device,
        "img_size": img_size,
    }


def benchmark_classifier(model, img_size: int = 224,
                          n_runs: int = 50, device: str = "cpu") -> Dict:
    """Benchmark EfficientNetB4 classifier inference."""
    model.eval()
    model.to(device)
    dummy = torch.randn(1, 3, img_size, img_size).to(device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(dummy)
        if device == "cuda":
            torch.cuda.synchronize()
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_latency_ms": round(np.mean(latencies), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "throughput_fps": round(1000 / np.mean(latencies), 1),
    }


def benchmark_preprocessing(img_size: int = 224, n_runs: int = 100) -> Dict:
    """Benchmark the preprocessing pipeline."""
    from app.services.preprocessing import run_full_pipeline, preprocess_to_tensor
    rng = np.random.default_rng(42)
    dummy_img = (rng.random((512, 512)) * 255).astype(np.uint8)

    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        steps = run_full_pipeline(dummy_img, (img_size, img_size))
        t = preprocess_to_tensor(dummy_img, (img_size, img_size))
        latencies.append((time.perf_counter() - start) * 1000)

    return {
        "mean_latency_ms": round(np.mean(latencies), 2),
        "p95_latency_ms": round(np.percentile(latencies, 95), 2),
        "n_steps": len(steps),
    }


def get_model_params(model) -> Dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": f"{total:,}",
        "trainable_params": f"{trainable:,}",
        "approx_size_mb": round(total * 4 / 1e6, 1),   # float32 = 4 bytes
    }


def main():
    parser = argparse.ArgumentParser(description="PancrAI Inference Benchmark")
    parser.add_argument("--n_runs", type=int, default=50)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["cpu", "cuda", "auto"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"\n{'='*60}")
    print("  PancrAI — Inference Benchmark")
    print(f"  Device: {device.upper()} | Img size: {args.img_size} | Runs: {args.n_runs}")
    print(f"{'='*60}\n")

    # Load models
    from app.models.transunet import TransUNet
    from app.models.classifier import PancreasTumorClassifier

    print("Loading models...")
    seg = TransUNet(img_size=args.img_size, pretrained=False)
    cls = PancreasTumorClassifier(pretrained=False)

    # Model info
    seg_params = get_model_params(seg)
    cls_params = get_model_params(cls)

    print(f"\n{'─'*40}")
    print("  Model Parameters")
    print(f"{'─'*40}")
    print(f"  TransUNet     : {seg_params['total_params']} params "
          f"(~{seg_params['approx_size_mb']} MB)")
    print(f"  EfficientNetB4: {cls_params['total_params']} params "
          f"(~{cls_params['approx_size_mb']} MB)")

    # Preprocessing
    print(f"\n{'─'*40}")
    print("  Preprocessing Pipeline")
    print(f"{'─'*40}")
    pp = benchmark_preprocessing(args.img_size, args.n_runs)
    print(f"  Mean latency  : {pp['mean_latency_ms']:.2f} ms")
    print(f"  P95 latency   : {pp['p95_latency_ms']:.2f} ms")
    print(f"  Pipeline steps: {pp['n_steps']}")

    # Segmentation
    print(f"\n{'─'*40}")
    print("  TransUNet Segmentation")
    print(f"{'─'*40}")
    sb = benchmark_segmentation(seg, args.img_size, args.n_runs, device)
    print(f"  Mean latency  : {sb['mean_latency_ms']:.2f} ms")
    print(f"  P50 latency   : {sb['p50_latency_ms']:.2f} ms")
    print(f"  P95 latency   : {sb['p95_latency_ms']:.2f} ms")
    print(f"  P99 latency   : {sb['p99_latency_ms']:.2f} ms")
    print(f"  Throughput    : {sb['throughput_fps']:.1f} FPS")

    # Classification
    print(f"\n{'─'*40}")
    print("  EfficientNetB4 Classification")
    print(f"{'─'*40}")
    cb = benchmark_classifier(cls, args.img_size, args.n_runs, device)
    print(f"  Mean latency  : {cb['mean_latency_ms']:.2f} ms")
    print(f"  P95 latency   : {cb['p95_latency_ms']:.2f} ms")
    print(f"  Throughput    : {cb['throughput_fps']:.1f} FPS")

    # Total pipeline estimate
    total_ms = pp["mean_latency_ms"] + sb["mean_latency_ms"] + cb["mean_latency_ms"]
    print(f"\n{'─'*40}")
    print("  Full Pipeline Estimate (no GradCAM/MC)")
    print(f"{'─'*40}")
    print(f"  Preprocessing + Seg + Cls: ~{total_ms:.1f} ms")
    print(f"  ({total_ms/1000:.2f}s per scan)")
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
