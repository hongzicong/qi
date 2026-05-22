from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBERO_REPO = PROJECT_ROOT / "tests" / "third_party" / "LIBERO"
LIBERO_PACKAGE_ROOT = LIBERO_REPO / "libero" / "libero"
WAN22_MODEL_ID = Path("Wan-AI") / "Wan2.2-TI2V-5B"
WAN22_REQUIRED_FILES = (
    "Wan2.2_VAE.pth",
    "models_t5_umt5-xxl-enc-bf16.pth",
)


def _has_wan22_cache(base_dir: Path) -> bool:
    model_dir = base_dir / WAN22_MODEL_ID
    return all((model_dir / filename).is_file() for filename in WAN22_REQUIRED_FILES)


def _bootstrap_model_cache() -> None:
    if os.environ.get("DIFFSYNTH_MODEL_BASE_PATH"):
        return

    shared_cache = PROJECT_ROOT.parent / "checkpoints"
    if _has_wan22_cache(shared_cache):
        os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(shared_cache)


def bootstrap_libero() -> Path:
    if not LIBERO_PACKAGE_ROOT.is_dir():
        raise FileNotFoundError(
            "Vendored LIBERO package is missing. Expected: "
            f"{LIBERO_PACKAGE_ROOT}"
        )

    _bootstrap_model_cache()

    libero_repo_str = str(LIBERO_REPO)
    if libero_repo_str not in sys.path:
        sys.path.insert(0, libero_repo_str)

    project_hash = hashlib.sha1(str(PROJECT_ROOT).encode("utf-8")).hexdigest()[:12]
    config_dir = Path(f"/tmp/qi_libero_config_{project_hash}")
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)

    config = {
        "benchmark_root": str(LIBERO_PACKAGE_ROOT),
        "bddl_files": str(LIBERO_PACKAGE_ROOT / "bddl_files"),
        "init_states": str(LIBERO_PACKAGE_ROOT / "init_files"),
        "datasets": str(LIBERO_REPO / "libero" / "datasets"),
        "assets": str(LIBERO_PACKAGE_ROOT / "assets"),
    }
    config_file = config_dir / "config.yaml"
    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return LIBERO_REPO
