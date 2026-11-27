"""CLI entrypoint: Start the local Decision AI API server & Evaluation Studio Web UI."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
VENDOR_KEV_DIR = Path(__file__).resolve().parents[1] / "vendor" / "kev"
for _p in (SRC_DIR, VENDOR_KEV_DIR):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import uvicorn

from decision_eval.runners import pick_device
from decision_eval.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Host Kev-0.8B and Qwen-0.8B behind one local HTTP API + Web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"], help="Force same backend (mlx or torch) across both Kev and Qwen.")
    parser.add_argument("--kev-run", default="jaredpalmer/kev-0.8b", help="Set '' to disable Kev.")
    parser.add_argument("--qwen-base-model", default="Qwen/Qwen3.5-0.8B-Base", help="Set '' to disable.")
    parser.add_argument("--qwen-instruct-model", default="", help="Optional Qwen/Qwen3.5-0.8B for JSON baseline.")
    parser.add_argument("--open-browser", action="store_true", help="Open Web UI automatically in browser.")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.5")
    os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.0")
    from decision_eval.paths import configure_bundled_hf_cache

    bundled_cache = configure_bundled_hf_cache()
    if bundled_cache:
        print(f"Using bundled model cache: {bundled_cache}")
    device = pick_device(args.device)
    app = create_app(
        kev_run=args.kev_run or None,
        qwen_base_model=args.qwen_base_model or None,
        qwen_instruct_model=args.qwen_instruct_model or None,
        device=device,
        backend=args.backend,
    )
    if args.open_browser:
        url = f"http://{args.host}:{args.port}"
        threading.Timer(2.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
