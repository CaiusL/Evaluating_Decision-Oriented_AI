"""Pre-download Kev-0.8B and Qwen3.5-0.8B weights (defaults to ModelScope: https://www.modelscope.cn).

ModelScope repositories:
  - jaredpalmer/kev-0.8b       -> https://www.modelscope.cn/models/jaredpalmer/kev-0.8b
  - Qwen/Qwen3.5-0.8B-Base     -> https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B-Base
  - Qwen/Qwen3.5-0.8B          -> https://www.modelscope.cn/models/Qwen/Qwen3.5-0.8B

Usage:
  # Default (ModelScope): downloads Kev-0.8B + Qwen3.5-0.8B-Base into models/modelscope/
  python decision_eval/cli/download_models.py

  # Also download the Instruct variant for the structured-JSON baseline
  python decision_eval/cli/download_models.py --include-instruct

  # Optional: use Hugging Face Hub instead of ModelScope
  python decision_eval/cli/download_models.py --source hf
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from decision_eval.paths import MODELSCOPE_DIR, resolve_local_model_path


def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total / (1024 * 1024)


def _check_env() -> None:
    print(f"[env] Python {sys.version.split()[0]}")
    if sys.version_info >= (3, 14):
        print("[warn] Python 3.14 detected; Python 3.12 or 3.13 is recommended for jaredpalmer/kev.")
    for pkg in ("mlx", "mlx_lm", "torch", "transformers", "modelscope", "huggingface_hub", "fastapi", "uvicorn", "kev"):
        ok = importlib.util.find_spec(pkg) is not None
        status = "OK" if ok else "MISSING"
        print(f"[env] {pkg:16s}: {status}")


def _download_hf(repo_id: str) -> Path:
    from huggingface_hub import snapshot_download

    print(f"[download:hf] Fetching {repo_id} ...")
    local_dir = snapshot_download(repo_id=repo_id)
    p = Path(local_dir)
    print(f"[done] {repo_id} -> {p} ({_dir_size_mb(p):.1f} MB)")
    return p


def _download_modelscope(repo_id: str, fallback_hf: bool = True) -> Path:
    existing = resolve_local_model_path(repo_id)
    if existing != repo_id and Path(existing).is_dir():
        p = Path(existing)
        print(f"[cached:modelscope] {repo_id} -> {p} ({_dir_size_mb(p):.1f} MB)")
        return p

    target_dir = MODELSCOPE_DIR / repo_id
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        from modelscope.hub.snapshot_download import snapshot_download

        print(f"[download:modelscope] Fetching https://www.modelscope.cn/models/{repo_id} ...")
        local_dir = snapshot_download(repo_id, local_dir=str(target_dir))
        p = Path(local_dir)
        print(f"[done] {repo_id} -> {p} ({_dir_size_mb(p):.1f} MB)")
        return p
    except Exception as exc:
        if not fallback_hf:
            raise
        print(f"[warn] ModelScope download failed for {repo_id} ({exc}); falling back to Hugging Face...")
        return _download_hf(repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download Kev-0.8B and Qwen3.5-0.8B models.")
    parser.add_argument("--kev-repo", default="jaredpalmer/kev-0.8b")
    parser.add_argument("--qwen-base-repo", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--qwen-instruct-repo", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--include-instruct", action="store_true", help="Also download Qwen3.5-0.8B Instruct.")
    parser.add_argument("--source", choices=["modelscope", "hf"], default="modelscope", help="Model hub source (default: modelscope).")
    parser.add_argument("--qwen-source", choices=["modelscope", "hf"], default="", help="Deprecated alias for --source.")
    parser.add_argument("--hf-endpoint", default="", help="Optional HF mirror URL (e.g. https://hf-mirror.com).")
    args = parser.parse_args()

    source = args.qwen_source or args.source
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    _check_env()
    print()

    repos = [args.kev_repo, args.qwen_base_repo]
    if args.include_instruct:
        repos.append(args.qwen_instruct_repo)

    for repo_id in repos:
        if source == "modelscope":
            _download_modelscope(repo_id, fallback_hf=True)
        else:
            _download_hf(repo_id)

    print("\nAll requested models are cached. Start the local API with:")
    if args.include_instruct:
        print(
            f"  python decision_eval/cli/serve.py --port 8008 "
            f"--qwen-instruct-model {args.qwen_instruct_repo}"
        )
    else:
        print("  python decision_eval/cli/serve.py --port 8008")


if __name__ == "__main__":
    main()
