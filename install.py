"""
PancrAI — Install Helper
Detects your platform and Python version, then installs the correct
packages in the right order.

Usage:
    python install.py           # auto-detect and install everything
    python install.py --cpu     # force CPU-only PyTorch (smaller download)
    python install.py --cuda    # force CUDA PyTorch (requires NVIDIA GPU)
    python install.py --check   # only check what's installed
"""

import sys
import subprocess
import platform
import argparse


def run(cmd: list, check: bool = True) -> int:
    print(f"\n>>> {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if check and result.returncode != 0:
        print(f"[!] Command failed with code {result.returncode}")
    return result.returncode


def get_python() -> list:
    return [sys.executable, "-m", "pip"]


def check_installed(package: str) -> tuple:
    """Return (installed: bool, version: str)."""
    try:
        import importlib.metadata
        ver = importlib.metadata.version(package)
        return True, ver
    except Exception:
        return False, ""


def install_torch(mode: str = "auto"):
    """Install PyTorch for the correct platform."""
    pip = get_python()
    py = get_python()

    print("\n" + "="*50)
    print("  Installing PyTorch")
    print("="*50)

    # Check if already installed
    installed, ver = check_installed("torch")
    if installed:
        print(f"  torch {ver} already installed — skipping")
        return

    if mode == "cpu":
        print("  Mode: CPU-only")
        run(pip + ["install", "torch", "torchvision",
                   "--index-url", "https://download.pytorch.org/whl/cpu"])
        return

    if mode == "cuda":
        print("  Mode: CUDA (NVIDIA GPU)")
        run(pip + ["install", "torch", "torchvision",
                   "--index-url", "https://download.pytorch.org/whl/cu124"])
        return

    # Auto-detect
    system = platform.system()
    machine = platform.machine().lower()

    if system == "Darwin" and "arm" in machine:
        # Apple Silicon — MPS backend, use default index
        print("  Detected: Apple Silicon Mac (MPS)")
        run(pip + ["install", "torch", "torchvision"])

    elif system == "Windows":
        # Windows — try CUDA first, fall back to CPU
        print("  Detected: Windows")
        print("  Trying CUDA build (NVIDIA GPU)...")
        rc = run(pip + ["install", "torch", "torchvision",
                         "--index-url", "https://download.pytorch.org/whl/cu124"],
                  check=False)
        if rc != 0:
            print("  CUDA build failed — falling back to CPU-only")
            run(pip + ["install", "torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cpu"])

    else:
        # Linux — default index (includes CUDA builds)
        print("  Detected: Linux")
        run(pip + ["install", "torch", "torchvision"])


def install_core():
    """Install core non-torch requirements."""
    pip = get_python()

    core_packages = [
        # Framework
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.7.1",
        # Image processing
        "opencv-python>=4.9.0",
        "Pillow>=10.3.0",
        "scikit-image>=0.22.0",
        # ML
        "timm>=1.0.0",
        "einops>=0.8.0",
        # Scientific
        "numpy>=1.26.0",
        "scipy>=1.13.0",
        "matplotlib>=3.8.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        # Database
        "sqlalchemy>=2.0.30",
        "aiosqlite>=0.20.0",
        # Frontend
        "streamlit>=1.35.0",
        "plotly>=5.22.0",
        # Utilities
        "python-dotenv>=1.0.1",
        "httpx>=0.27.0",
        "aiofiles>=23.2.1",
        "requests>=2.32.0",
        "tqdm>=4.66.0",
        "reportlab>=4.2.0",
        # Testing
        "pytest>=8.2.0",
        "pytest-asyncio>=0.23.0",
    ]

    print("\n" + "="*50)
    print("  Installing core packages")
    print("="*50)
    run(pip + ["install"] + core_packages)


def install_ai_apis():
    """Install AI API clients (optional but recommended)."""
    pip = get_python()

    print("\n" + "="*50)
    print("  Installing AI API clients (Gemini + Groq)")
    print("="*50)
    print("  These are optional. Skip if you don't need AI reports/chat.")
    print("  Get free keys at: aistudio.google.com and console.groq.com")

    run(pip + ["install", "google-generativeai>=0.7.0", "groq>=0.9.0"], check=False)


def install_medical_imaging():
    """Install optional medical imaging libraries."""
    pip = get_python()

    print("\n" + "="*50)
    print("  Installing medical imaging libraries (optional)")
    print("="*50)
    print("  Required only for DICOM (.dcm) and NIfTI (.nii) file support.")
    print("  PNG/JPEG uploads work without these.")

    packages = [
        ("pydicom", "pydicom>=2.4.0"),
        ("nibabel", "nibabel>=5.2.0"),
    ]
    for name, spec in packages:
        installed, ver = check_installed(name)
        if installed:
            print(f"  {name} {ver} already installed")
        else:
            rc = run(pip + ["install", spec], check=False)
            if rc != 0:
                print(f"  [!] {name} install failed — DICOM/NIfTI support disabled")


def check_installation():
    """Print a table of what's installed."""
    packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("fastapi", "FastAPI"),
        ("uvicorn", "uvicorn"),
        ("streamlit", "Streamlit"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sqlalchemy", "SQLAlchemy"),
        ("plotly", "Plotly"),
        ("google.generativeai", "google-generativeai"),
        ("groq", "Groq"),
        ("pydicom", "pydicom"),
        ("nibabel", "nibabel"),
        ("reportlab", "ReportLab"),
    ]

    print("\n" + "="*50)
    print("  Installed Packages")
    print("="*50)
    for pkg, name in packages:
        ok, ver = check_installed(pkg)
        status = f"✓ {ver}" if ok else "✗ NOT INSTALLED"
        print(f"  {'✓' if ok else '✗'}  {name:<25} {ver if ok else 'not installed'}")

    # PyTorch GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            print(f"\n  GPU: {gpu} (CUDA {torch.version.cuda})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n  GPU: Apple MPS available")
        else:
            print("\n  GPU: CPU only")
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="PancrAI install helper")
    parser.add_argument("--cpu", action="store_true", help="Force CPU-only PyTorch")
    parser.add_argument("--cuda", action="store_true", help="Force CUDA PyTorch")
    parser.add_argument("--check", action="store_true", help="Check installation only")
    parser.add_argument("--skip-torch", action="store_true",
                        help="Skip PyTorch install (if already installed)")
    parser.add_argument("--skip-medical", action="store_true",
                        help="Skip pydicom/nibabel (not needed for PNG/JPEG)")
    parser.add_argument("--skip-apis", action="store_true",
                        help="Skip Gemini/Groq API clients")
    args = parser.parse_args()

    print("="*50)
    print("  PancrAI — Dependency Installer")
    print(f"  Python {sys.version}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("="*50)

    if args.check:
        check_installation()
        return

    # Upgrade pip first
    run(get_python() + ["install", "--upgrade", "pip"])

    # Install in order
    if not args.skip_torch:
        mode = "cpu" if args.cpu else "cuda" if args.cuda else "auto"
        install_torch(mode)

    install_core()

    if not args.skip_apis:
        install_ai_apis()

    if not args.skip_medical:
        install_medical_imaging()

    # Final check
    check_installation()

    print("\n" + "="*50)
    print("  Installation complete!")
    print("")
    print("  Next steps:")
    print("  1. cp .env.example .env")
    print("     (add your Gemini and Groq API keys)")
    print("")
    print("  2. python run.py")
    print("     (starts both FastAPI + Streamlit)")
    print("")
    print("  3. Open http://localhost:8501")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
