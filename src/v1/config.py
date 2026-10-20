from __future__ import annotations

import os
from pathlib import Path


def _load_default_env() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_default_env()

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("AIDETECT_DATA_DIR", "/content/data"))
WORK_DIR = Path(os.environ.get("AIDETECT_WORK_DIR", str(ROOT / "artifacts")))
REPO = "nebula/GenImage-arrow"
TRAIN_GEN = "sd14"
SEED = 42

GENS = {
    "sd14": "stable_diffusion_v_1_4",
    "sd15": "stable_diffusion_v_1_5",
    "wukong": "wukong",
    "adm": "ADM",
    "glide": "glide",
    "vqdm": "VQDM",
    "midjourney": "Midjourney",
    "biggan": "BigGAN",
}

ZIPS = dict.fromkeys(GENS)
N_TRAIN = 16000
N_VAL = 2000
N_TEST = 3000
IMG_SIZE = 256
CLIP_IMG_SIZE = 224

__all__ = [
    "ROOT",
    "DATA_DIR",
    "WORK_DIR",
    "REPO",
    "TRAIN_GEN",
    "SEED",
    "GENS",
    "ZIPS",
    "N_TRAIN",
    "N_VAL",
    "N_TEST",
    "IMG_SIZE",
    "CLIP_IMG_SIZE",
]
