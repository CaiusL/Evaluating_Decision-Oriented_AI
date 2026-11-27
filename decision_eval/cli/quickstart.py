"""One-command setup & launcher for teammates on Apple Silicon Macs.

Usage (works with macOS system python3 — zero manual setup):
  python3 decision_eval/cli/quickstart.py

Automatically:
  1. Ensures `uv` is available (installs via Homebrew if `brew` is present).
  2. Creates `.venv-decision` with Python 3.13 if missing.
  3. Installs `decision_eval/requirements.txt` (including mlx, mlx-lm, kev) if missing.
  4. Uses bundled `models/hf_cache/` if present, otherwise downloads Kev-0.8B & Qwen3.5-0.8B.
  5. Starts the local API server and opens the Evaluation Studio Web UI in your browser.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

CLI_DIR = Path(__file__).resolve().parent
EVAL_ROOT = CLI_DIR.parent
REPO_ROOT = EVAL_ROOT.parent
VENV_DIR = REPO_ROOT / ".venv-decision"
VENV_PY = VENV_DIR / "bin" / "python"
REQ_FILE = EVAL_ROOT / "requirements.txt"
BUNDLED_CACHE = REPO_ROOT / "models" / "hf_cache"


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _find_uv() -> str | None:
    found = shutil.which("uv")
    if found:
        return found
    for candidate in [
        Path.home() / ".local" / "bin" / "uv",
        Path("/opt/homebrew/bin/uv"),
        Path("/usr/local/bin/uv"),
    ]:
        if candidate.is_file():
            return str(candidate)
    return None


def _ensure_uv() -> str:
    uv = _find_uv()
    if uv:
        return uv
    brew = shutil.which("brew") or ("/opt/homebrew/bin/brew" if Path("/opt/homebrew/bin/brew").is_file() else None)
    if brew:
        print("uv not found — installing uv via Homebrew...")
        _run([brew, "install", "uv"])
        uv = _find_uv()
        if uv:
            return uv
    raise SystemExit(
        "Could not find `uv` or `brew`.\n"
        "Please run these two commands once in Terminal, then re-run quickstart.py:\n"
        '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"\n'
        "  brew install uv"
    )


def _deps_ready() -> bool:
    if not VENV_PY.is_file():
        return False
    proc = subprocess.run(
        [str(VENV_PY), "-c", "import fastapi, uvicorn, torch, transformers, kev, mlx, mlx_lm, modelscope"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description="One-command setup + ModelScope download + launch Web UI.")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"])
    parser.add_argument("--source", default="modelscope", choices=["modelscope", "hf"], help="Model download source (default: modelscope).")
    parser.add_argument("--include-instruct", action="store_true", help="Also download/load separate Qwen3.5-0.8B Instruct.")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    uv = _ensure_uv()

    if not VENV_PY.is_file():
        print("Creating Python 3.13 virtual environment (.venv-decision)...")
        _run([uv, "venv", "--python", "3.13", str(VENV_DIR)])

    if not _deps_ready():
        print("Installing dependencies from decision_eval/requirements.txt...")
        _run([uv, "pip", "install", "--python", str(VENV_PY), "-r", str(REQ_FILE)])

    env = os.environ.copy()
    has_bundled_hf = BUNDLED_CACHE.is_dir() and any(BUNDLED_CACHE.iterdir())
    if has_bundled_hf:
        print(f"Found bundled model cache at {BUNDLED_CACHE} — skipping download.")
        env["HF_HUB_CACHE"] = str(BUNDLED_CACHE)
        env["HUGGINGFACE_HUB_CACHE"] = str(BUNDLED_CACHE)
    else:
        dl_cmd = [str(VENV_PY), str(CLI_DIR / "download_models.py"), "--source", args.source]
        if args.include_instruct:
            dl_cmd.append("--include-instruct")
        _run(dl_cmd, env=env)

    serve_cmd = [
        str(VENV_PY),
        str(CLI_DIR / "serve.py"),
        "--port",
        str(args.port),
        "--backend",
        args.backend,
    ]
    if args.include_instruct:
        serve_cmd.extend(["--qwen-instruct-model", "Qwen/Qwen3.5-0.8B"])
    if not args.no_browser:
        serve_cmd.append("--open-browser")

    print("\nStarting Decision AI Evaluation Studio...")
    _run(serve_cmd, env=env)


if __name__ == "__main__":
    main()
