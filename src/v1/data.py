from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from datasets import Dataset
except ImportError:  # dataset support is optional until the user chooses to run the download pipeline
    Dataset = None

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:  # pragma: no cover
    HfApi = None
    hf_hub_download = None

from PIL import Image

from .config import DATA_DIR, GENS, REPO, SEED


def shards(split: str, folder: str) -> List[str]:
    if HfApi is None:
        raise ImportError("The optional Hugging Face client is not installed. Install it only when you want to run the dataset download pipeline.")
    api = HfApi()
    all_files = api.list_repo_files(REPO, repo_type="dataset")
    prefix = f"data/{split}/{folder}/"
    return sorted(f for f in all_files if f.startswith(prefix) and f.endswith(".arrow"))


def sniff_ext(blob: bytes) -> str:
    if blob[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[:4] == b"RIFF":
        return ".webp"
    return ".png"


def count(directory: str | Path) -> int:
    return len(os.listdir(directory)) if os.path.isdir(directory) else 0


def convert(gen: str, hf_split: str, quotas: Dict[Tuple[str, str], int], p_valid: float = 0.11):
    """Download shards and write original image bytes to local folders."""
    tmp_dir = Path("/tmp/aidetect_hf")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    files = shards(hf_split, GENS[gen])
    rng = random.Random(SEED)
    rng.shuffle(files)

    for (split_name, kind) in quotas:
        (DATA_DIR / gen / split_name / kind).mkdir(parents=True, exist_ok=True)

    cnt = {(split_name, kind): count(DATA_DIR / gen / split_name / kind) for (split_name, kind) in quotas}
    done_file = DATA_DIR / gen / f".done_{hf_split}.json"
    done = set(json.loads(done_file.read_text())) if done_file.exists() else set()
    local_splits = sorted({split_name for split_name, _ in quotas})

    if Dataset is None or hf_hub_download is None:
        raise ImportError("Dataset download support is unavailable because the optional Hugging Face packages are not installed. Install them only when you want to run the dataset download pipeline.")

    for file_name in files:
        if all(cnt[key] >= n for key, n in quotas.items()):
            break
        if file_name in done:
            continue

        t0 = time.time()
        path = hf_hub_download(REPO, file_name, repo_type="dataset", local_dir=str(tmp_dir))
        ds = Dataset.from_file(path)
        names = ds.features["label"].names
        shard_id = os.path.basename(file_name).split("-")[1]

        for i, row in enumerate(ds):
            kind = "ai" if names[row["label"]] == "fake" else "nature"
            if len(local_splits) > 1:
                order = ["valid", "train"] if rng.random() < p_valid else ["train", "valid"]
            else:
                order = local_splits

            target_split = next(
                (split_name for split_name in order if (split_name, kind) in quotas and cnt[(split_name, kind)] < quotas[(split_name, kind)]),
                None,
            )
            if target_split is None:
                continue

            blob = row["image"]
            if isinstance(blob, dict):
                blob = blob["bytes"]
            base = os.path.splitext(os.path.basename(str(row["image_path"])))[0]
            out_name = f"{shard_id}_{i}_{base}{sniff_ext(blob)}"
            out_path = DATA_DIR / gen / target_split / kind / out_name
            out_path.write_bytes(blob)
            cnt[(target_split, kind)] += 1

        del ds
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(os.path.expanduser("~/.cache/huggingface/xet"), ignore_errors=True)
        done.add(file_name)
        done_file.write_text(json.dumps(sorted(done)))
        print(f"{gen} {hf_split} {shard_id}: {round(time.time() - t0)}s | " + " ".join(f"{k[0]}/{k[1]}={v}" for k, v in cnt.items()))

    return cnt


def find_dir(gen: str, split: str, kind: str):
    for root, _, _ in os.walk(str(DATA_DIR / gen)):
        if os.path.basename(root) == kind and os.path.basename(os.path.dirname(root)) == split:
            return root
    raise FileNotFoundError(f"Could not find {gen}/{split}/{kind}")


def list_imgs(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".webp")
    items = []
    for root, _, files in os.walk(folder):
        for fn in files:
            if fn.lower().endswith(exts):
                items.append(os.path.join(root, fn))
    return sorted(items)


def make_items(gen: str, split: str, n_per_class: int, seed: int = SEED):
    fake = list_imgs(find_dir(gen, split, "ai"))
    real = list_imgs(find_dir(gen, split, "nature"))
    rng = random.Random(seed)
    rng.shuffle(fake)
    rng.shuffle(real)
    return [(p, 1) for p in fake[:n_per_class]] + [(p, 0) for p in real[:n_per_class]]


def build_dataset_lists():
    from .config import N_TRAIN, N_VAL, N_TEST, TRAIN_GEN, GENS

    lists = {
        "train": make_items(TRAIN_GEN, "train", N_TRAIN),
        "val": make_items(TRAIN_GEN, "valid", N_VAL),
    }
    for gen in GENS:
        lists[f"test_{gen}"] = make_items(gen, "val", N_TEST)
    return lists


def summarize_dataset_counts():
    rows = []
    for gen in GENS:
        for split in ["train", "valid", "val"]:
            for kind in ["ai", "nature"]:
                folder = DATA_DIR / gen / split / kind
                if folder.is_dir():
                    rows.append((gen, split, kind, count(folder)))
    return rows


__all__ = [
    "shards",
    "sniff_ext",
    "count",
    "convert",
    "find_dir",
    "list_imgs",
    "make_items",
    "build_dataset_lists",
    "summarize_dataset_counts",
]
