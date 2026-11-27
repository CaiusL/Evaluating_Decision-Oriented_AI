"""Pre-download Kev-0.8B and Qwen3.5-0.8B weights for offline M4 Mac hosting.

Usage:
  # Default (Hugging Face Hub): downloads Kev-0.8B + Qwen3.5-0.8B-Base
  python scripts/download_models.py

  # Also download the Instruct variant for the structured-JSON baseline
  python scripts/download_models.py --include-instruct

  # Optional: use ModelScope instead of Hugging Face for Qwen weights
  python scripts/download_models.py --qwen-source modelscope
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 * 1024)


def _check_env() -> None:
    print(f"[env] Python {sys.version.split()[0]}")
    if sys.version_info >= (3, 14):
        print("[warn] Python 3.14 detected; Python 3.12 or 3.13 is recommended for jaredpalmer/kev.")
    for pkg in ("mlx", "torch", "transformers", "huggingface_hub", "fastapi", "uvicorn", "kev"):
        ok = importlib.util.find_spec(pkg) is not None
        status = "OK" if ok else "MISSING"
        print(f"[env] {pkg:16s}: {status}")
        if pkg == "mlx" and not ok:
            print("      -> Install 'mlx' on Apple Silicon so Kev-0.8B runs fast (~45ms vs ~2s on MPS).")


def _download_hf(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[download:hf] Fetching {repo_id} ...")
    local_dir = snapshot_download(repo_id=repo_id)
    p = Path(local_dir)
    print(f"[done] {repo_id} -> {p} ({_dir_size_mb(p):.1f} MB)")
    return p


def _download_modelscope(repo_id: str) -> Path:
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[download:modelscope] Fetching {repo_id} ...")
    local_dir = snapshot_download(repo_id)
    p = Path(local_dir)
    print(f"[done] {repo_id} -> {p} ({_dir_size_mb(p):.1f} MB)")
    return p


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download Kev-0.8B and Qwen3.5-0.8B models.")
    parser.add_argument("--kev-repo", default="jaredpalmer/kev-0.8b")
    parser.add_argument("--qwen-base-repo", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--qwen-instruct-repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--include-instruct", action="store_true", help="Also download Qwen3.5-0.8B Instruct.")
    parser.add_argument("--qwen-source", choices=["hf", "modelscope"], default="hf")
    parser.add_argument("--hf-endpoint", default="", help="Optional HF mirror URL (e.g. https://hf-mirror.com).")
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    _check_env()
    print()

    # 1. Kev-0.8B adapter + pointer head (always from Hugging Face)
    _download_hf(args.kev_repo)

    # 2. Qwen3.5-0.8B-Base (shared backbone for Kev-0.8B + log-likelihood baseline)
    if args.qwen_source == "modelscope":
        base_path = _download_modelscope(args.qwen_base_repo)
        print(f"\n[tip] Export before starting the API:\n  export MODELSCOPE_BASE=\"{base_path}\"")
    else:
        _download_hf(args.qwen_base_repo)

    # 3. Optional Qwen3.5-0.8B Instruct (for structured-JSON baseline)
    if args.include_instruct:
        if args.qwen_source == "modelscope":
            _download_modelscope(args.qwen_instruct_repo)
        else:
            _download_hf(args.qwen_instruct_repo)

    print("\nAll requested models are cached. Start the local API with:")
    if args.include_instruct:
        print(
            f"  python scripts/serve_decision_api.py --port 8008 "
            f"--qwen-instruct-model {args.qwen_instruct_repo}"
        )
    else:
        print("  python scripts/serve_decision_api.py --port 8008")


if __name__ == "__main__":
    main()
