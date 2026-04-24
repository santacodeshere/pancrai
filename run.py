#!/usr/bin/env python3
"""
PancrAI — Startup Script
Launches both the FastAPI backend and Streamlit frontend.
Handles environment checks, database init, and process management.

Usage:
    python run.py                  # start everything
    python run.py --api-only       # start only the FastAPI backend
    python run.py --ui-only        # start only the Streamlit frontend
    python run.py --check          # only run environment checks
    python run.py --demo           # generate a sample scan and launch UI
"""

import os
import sys
import time
import signal
import argparse
import subprocess
import threading
from pathlib import Path


# ─── ANSI Colors ─────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BLUE   = "\033[94m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"


def cprint(color: str, msg: str):
    print(f"{color}{msg}{RESET}")


# ─── Environment Checks ───────────────────────────────────────────────────────

def check_env():
    """Run pre-flight checks and print a status report."""
    cprint(BOLD + BLUE, "\n╔══════════════════════════════════════════╗")
    cprint(BOLD + BLUE, "║        PancrAI — Pre-flight Checks        ║")
    cprint(BOLD + BLUE, "╚══════════════════════════════════════════╝\n")

    all_ok = True

    # Python version
    py_ver = sys.version_info
    ok = py_ver >= (3, 10)
    status = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
    print(f"  {status}  Python {py_ver.major}.{py_ver.minor}.{py_ver.micro}  "
          f"{'(ok)' if ok else '(need 3.10+)'}")
    all_ok = all_ok and ok

    # Core packages
    packages = [
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("streamlit", "Streamlit"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("sqlalchemy", "SQLAlchemy"),
        ("plotly", "Plotly"),
        ("PIL", "Pillow"),
        ("timm", "timm"),
        ("einops", "einops"),
    ]
    for pkg, name in packages:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  {GREEN}✓{RESET}  {name} {ver}")
        except ImportError:
            cprint(RED, f"  ✗  {name} — NOT INSTALLED")
            all_ok = False

    # Optional packages
    optional = [
        ("pydicom", "pydicom (DICOM support)"),
        ("nibabel", "nibabel (NIfTI support)"),
        ("albumentations", "albumentations (augmentation)"),
        ("google.generativeai", "google-generativeai (Gemini reports)"),
        ("groq", "groq (Chat assistant)"),
        ("reportlab", "reportlab (PDF export)"),
    ]
    print()
    for pkg, name in optional:
        try:
            __import__(pkg)
            print(f"  {GREEN}✓{RESET}  {name}")
        except ImportError:
            print(f"  {YELLOW}○{RESET}  {name} — optional, not installed")

    # Environment variables
    print()
    env_vars = [
        ("GEMINI_API_KEY", "Gemini API key (AI reports)"),
        ("GROQ_API_KEY", "Groq API key (Chat assistant)"),
    ]
    for var, desc in env_vars:
        val = os.getenv(var, "")
        if val and val != f"your_{var.lower()}_here":
            print(f"  {GREEN}✓{RESET}  {desc} — configured")
        else:
            print(f"  {YELLOW}○{RESET}  {desc} — not set (offline mode)")

    # Model weights
    print()
    seg_weights = os.getenv("MODEL_WEIGHTS_PATH", "./weights/transunet_best.pth")
    cls_weights = os.getenv("CLASSIFIER_WEIGHTS_PATH", "./weights/efficientnet_best.pth")
    for path, desc in [(seg_weights, "TransUNet weights"), (cls_weights, "Classifier weights")]:
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / 1e6
            print(f"  {GREEN}✓{RESET}  {desc} — {path} ({size_mb:.1f} MB)")
        else:
            print(f"  {YELLOW}○{RESET}  {desc} — not found (demo mode with random weights)")

    # CUDA
    print()
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  {GREEN}✓{RESET}  CUDA available — {gpu} ({mem:.1f} GB)")
        else:
            print(f"  {YELLOW}○{RESET}  CUDA not available — using CPU (inference will be slower)")
    except Exception:
        pass

    # Directories
    print()
    for d in ["./uploads", "./weights", "./data"]:
        Path(d).mkdir(exist_ok=True)
        print(f"  {GREEN}✓{RESET}  Directory ready: {d}")

    print()
    if all_ok:
        cprint(GREEN + BOLD, "  ✓ All required dependencies satisfied!\n")
    else:
        cprint(RED + BOLD, "  ✗ Some required packages are missing.")
        cprint(YELLOW, "  Run: pip install -r requirements.txt\n")

    return all_ok


# ─── Process Management ────────────────────────────────────────────────────────

processes = []


def stream_output(proc, prefix: str, color: str):
    """Stream subprocess output with a colored prefix."""
    for line in iter(proc.stdout.readline, b""):
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            print(f"{color}[{prefix}]{RESET} {text}")


def start_api(host: str = "0.0.0.0", port: int = 8000) -> subprocess.Popen:
    """Start the FastAPI backend via uvicorn."""
    cprint(CYAN, f"\n[PancrAI] Starting FastAPI backend on http://{host}:{port} ...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", host, "--port", str(port), "--reload"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=Path(__file__).parent,
    )
    t = threading.Thread(target=stream_output, args=(proc, "API", CYAN), daemon=True)
    t.start()
    processes.append(proc)
    return proc


def start_ui(port: int = 8501, api_url: str = "http://localhost:8000/api/v1") -> subprocess.Popen:
    """Start the Streamlit frontend."""
    cprint(GREEN, f"\n[PancrAI] Starting Streamlit UI on http://localhost:{port} ...")
    env = os.environ.copy()
    env["API_BASE_URL"] = api_url

    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run",
         "frontend/streamlit_app.py",
         "--server.port", str(port),
         "--server.address", "localhost",
         "--server.headless", "true",
         "--browser.gatherUsageStats", "false"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path(__file__).parent,
    )
    t = threading.Thread(target=stream_output, args=(proc, "UI", GREEN), daemon=True)
    t.start()
    processes.append(proc)
    return proc


def wait_for_api(host: str = "localhost", port: int = 8000,
                 timeout: int = 30) -> bool:
    """Poll the API health endpoint until it responds."""
    import urllib.request
    url = f"http://{host}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def shutdown(signum=None, frame=None):
    """Gracefully terminate all child processes."""
    cprint(YELLOW, "\n[PancrAI] Shutting down...")
    for p in processes:
        try:
            p.terminate()
        except Exception:
            pass
    for p in processes:
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    cprint(GREEN, "[PancrAI] All processes stopped. Goodbye!\n")
    sys.exit(0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="PancrAI — Startup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-only", action="store_true",
                        help="Start only the FastAPI backend")
    parser.add_argument("--ui-only", action="store_true",
                        help="Start only the Streamlit frontend")
    parser.add_argument("--check", action="store_true",
                        help="Run environment checks only")
    parser.add_argument("--demo", action="store_true",
                        help="Generate a demo scan and launch the UI")
    parser.add_argument("--api-host", default="0.0.0.0",
                        help="FastAPI host (default: 0.0.0.0)")
    parser.add_argument("--api-port", type=int, default=8000,
                        help="FastAPI port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501,
                        help="Streamlit port (default: 8501)")
    parser.add_argument("--skip-checks", action="store_true",
                        help="Skip pre-flight environment checks")
    args = parser.parse_args()

    # Load .env
    if Path(".env").exists():
        from dotenv import load_dotenv
        load_dotenv()

    # Banner
    cprint(BOLD + BLUE, r"""
  ____                      _    ___ 
 |  _ \ __ _ _ __   ___ _ __/ \  |_ _|
 | |_) / _` | '_ \ / __| '__/ _ \  | | 
 |  __/ (_| | | | | (__| | / ___ \ | | 
 |_|   \__,_|_| |_|\___|_|/_/   \_\___|
                                        
 Intelligent Pancreatic Tumor Detection
""")

    # Environment check
    if not args.skip_checks:
        ok = check_env()
        if args.check:
            sys.exit(0 if ok else 1)
    else:
        cprint(YELLOW, "[PancrAI] Skipping environment checks (--skip-checks)\n")

    # Demo mode: generate sample scan
    if args.demo:
        cprint(CYAN, "[PancrAI] Generating demo scan...")
        try:
            from demo_data_generator import generate_sample_scan
            generate_sample_scan("./sample_ct.png")
            cprint(GREEN, "[PancrAI] Demo scan created: ./sample_ct.png")
        except Exception as e:
            cprint(YELLOW, f"[PancrAI] Demo generation warning: {e}")

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Launch processes
    api_proc = None
    ui_proc = None

    if not args.ui_only:
        api_proc = start_api(args.api_host, args.api_port)
        # Give the API a moment to start
        time.sleep(2)
        if not wait_for_api(
            "localhost" if args.api_host == "0.0.0.0" else args.api_host,
            args.api_port,
            timeout=30,
        ):
            cprint(YELLOW, "[PancrAI] API did not respond in 30s — continuing anyway")
        else:
            cprint(GREEN, f"[PancrAI] ✓ API ready at http://localhost:{args.api_port}")
            cprint(GREEN, f"[PancrAI] ✓ API docs at http://localhost:{args.api_port}/docs")

    if not args.api_only:
        api_url = f"http://localhost:{args.api_port}/api/v1"
        ui_proc = start_ui(args.ui_port, api_url)
        time.sleep(3)
        cprint(GREEN, f"\n[PancrAI] ✓ UI ready at http://localhost:{args.ui_port}")

    # Final summary
    cprint(BOLD + GREEN, "\n╔══════════════════════════════════════════╗")
    cprint(BOLD + GREEN, "║           PancrAI is Running! 🚀          ║")
    cprint(BOLD + GREEN, "╚══════════════════════════════════════════╝")
    if not args.ui_only:
        cprint(CYAN,  f"  API       → http://localhost:{args.api_port}")
        cprint(CYAN,  f"  API Docs  → http://localhost:{args.api_port}/docs")
    if not args.api_only:
        cprint(GREEN, f"  UI        → http://localhost:{args.ui_port}")
    cprint(YELLOW, "\n  Press Ctrl+C to stop all services\n")

    # Keep main thread alive
    try:
        while True:
            # Check if child processes are still running
            if api_proc and api_proc.poll() is not None:
                cprint(RED, "[PancrAI] API process exited unexpectedly!")
                break
            if ui_proc and ui_proc.poll() is not None:
                cprint(RED, "[PancrAI] UI process exited unexpectedly!")
                break
            time.sleep(2)
    except KeyboardInterrupt:
        pass
    finally:
        shutdown()


if __name__ == "__main__":
    main()
