"""Centralized path resolution for the decision_eval package."""
from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
EVAL_ROOT = SRC_DIR.parent
REPO_ROOT = EVAL_ROOT.parent

UI_HTML = EVAL_ROOT / "ui" / "index.html"
CONFIG_DIR = EVAL_ROOT / "config"
DATA_DIR = REPO_ROOT / "data"
BENCHMARKS_DIR = DATA_DIR / "benchmarks_v2"
CAR_BUILD_DIR = DATA_DIR / "car_build_bench"
MODELS_DIR = REPO_ROOT / "models"
HF_CACHE_DIR = MODELS_DIR / "hf_cache"
MODELSCOPE_DIR = MODELS_DIR / "modelscope"
M2_QWEN_LORA_S42_DIR = MODELS_DIR / "m2_qwen3.5_0.8b_lora_s42"
CAR_BUILD_NN_NPZ = MODELS_DIR / "car_build_nn.npz"
DRIVER_NN_NPZ = (
    REPO_ROOT / "submissions" / "racing" / "jev_rl" / "weights" / "jev_rl.mapdata.npz"
    if (REPO_ROOT / "submissions" / "racing" / "jev_rl" / "weights" / "jev_rl.mapdata.npz").is_file()
    else MODELS_DIR / "jev_rl_v7.mapdata.npz"
)
REPORTS_DIR = REPO_ROOT / "reports" / "full_benchmark_v2"
CHARTS_DIR = REPORTS_DIR / "charts"


def configure_bundled_hf_cache() -> Path | None:
    """If models/hf_cache/ was bundled in the zip, point Hugging Face to it automatically."""
    import os

    if HF_CACHE_DIR.is_dir() and any(HF_CACHE_DIR.iterdir()):
        os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HF_CACHE_DIR))
        return HF_CACHE_DIR
    return None


def _looks_like_model_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return any(
        (path / name).is_file()
        for name in ("config.json", "adapter_config.json", "run.json", "tokenizer.json")
    ) or any(path.glob("*.safetensors"))


def resolve_local_model_path(repo_id: str) -> str:
    """Return a local ModelScope directory for `repo_id` if present, else `repo_id` unchanged."""
    import os

    if not repo_id:
        return repo_id
    direct = Path(repo_id).expanduser()
    if direct.is_dir():
        return str(direct)

    candidates: list[Path] = [MODELSCOPE_DIR / repo_id]
    if repo_id.startswith("Qwen/Qwen3.5-0.8B-Base") and os.environ.get("MODELSCOPE_BASE"):
        candidates.insert(0, Path(os.environ["MODELSCOPE_BASE"]).expanduser())

    ms_home = Path.home() / ".cache" / "modelscope"
    candidates.extend([
        ms_home / "hub" / "models" / repo_id,
        ms_home / "hub" / repo_id,
        ms_home / repo_id,
    ])
    for cand in candidates:
        if _looks_like_model_dir(cand):
            return str(cand)
    return repo_id
