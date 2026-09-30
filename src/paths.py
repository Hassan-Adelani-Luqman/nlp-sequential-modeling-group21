"""Environment-aware paths so the same code runs on Kaggle, Colab and locally.

Resolution order for the raw data directory (the folder holding Train.csv):
  1. the ``SWN_DATA_DIR`` environment variable, if set;
  2. Kaggle: any ``/kaggle/input/*`` folder that contains Train.csv;
  3. Colab: ``/content/data/raw`` or ``/content/drive/MyDrive/group21-swahili-audio``;
  4. local: ``<repo>/data/raw``.

Outputs (results, checkpoints, caches) go to ``/kaggle/working`` on Kaggle and to
the repo root everywhere else. Split files always live in the repo
(``data/splits``) because they are committed and shared by the whole group.
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
CONFIGS_DIR = REPO_ROOT / "configs"

# Private Kaggle dataset shared with the group, as "<owner>/<slug>".
KAGGLE_DATASET = os.environ.get("SWN_KAGGLE_DATASET", "luqmanhassanadelani/group21-swahili-audio")
TRAIN_FILE = "Train.csv"


def detect_env() -> str:
    """Return ``"kaggle"``, ``"colab"`` or ``"local"``."""
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle/input").is_dir():
        return "kaggle"
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return "colab"
    except ModuleNotFoundError:  # the parent "google" package is not installed
        pass
    return "local"


def _contains_train(folder: Path) -> bool:
    return folder.is_dir() and any(p.name.lower() == TRAIN_FILE.lower() for p in folder.iterdir())


def _candidate_data_dirs(env: str) -> list[Path]:
    candidates: list[Path] = []
    if os.environ.get("SWN_DATA_DIR"):
        candidates.append(Path(os.environ["SWN_DATA_DIR"]))
    if env == "kaggle":
        kaggle_input = Path("/kaggle/input")
        candidates.append(kaggle_input / KAGGLE_DATASET.split("/")[-1])
        if kaggle_input.is_dir():
            candidates.extend(sorted(p for p in kaggle_input.iterdir() if p.is_dir()))
            # Kaggle may mount datasets deeper (e.g. /kaggle/input/<owner>/<slug>/...): find Train.csv at any depth
            candidates.extend(sorted({p.parent for p in kaggle_input.rglob(TRAIN_FILE)}))
    elif env == "colab":
        candidates += [Path("/content/data/raw"), Path("/content/drive/MyDrive/group21-swahili-audio")]
    candidates.append(REPO_ROOT / "data" / "raw")
    return candidates


def get_data_dir() -> Path:
    """Locate the folder containing the Zindi data (CSVs + Swahili_words/), or raise with setup instructions."""
    env = detect_env()
    for folder in _candidate_data_dirs(env):
        if _contains_train(folder):
            return folder
    raise FileNotFoundError(
        f"Could not find {TRAIN_FILE} (environment: {env}). "
        "Kaggle: attach the group dataset via 'Add Input'. "
        "Colab: run src.paths.download_from_kaggle() or mount Drive. "
        "Local: put the Zindi data (CSVs + Swahili_words/) in data/raw/. "
        "Or set SWN_DATA_DIR to the folder that holds Train.csv. See data/README.md."
    )


def get_output_dir() -> Path:
    """Writable root for results/checkpoints (``/kaggle/working`` on Kaggle).

    Override with ``SWN_OUTPUT_DIR``, e.g. to write to Google Drive on Colab or to a scratch folder for smoke runs.
    """
    if os.environ.get("SWN_OUTPUT_DIR"):
        return Path(os.environ["SWN_OUTPUT_DIR"])
    return Path("/kaggle/working") if detect_env() == "kaggle" else REPO_ROOT


def results_dir() -> Path:
    return get_output_dir() / "results"


def checkpoints_dir() -> Path:
    return get_output_dir() / "checkpoints"


def cache_dir() -> Path:
    return results_dir() / "cache"


def download_from_kaggle(dest: Path | str = "/content/data/raw", dataset: str = KAGGLE_DATASET) -> Path:
    """Download the group's private Kaggle dataset (for Colab or local use).

    Requires the ``kaggle`` package and an API token in ``~/.kaggle/kaggle.json``
    (or the KAGGLE_USERNAME / KAGGLE_KEY environment variables). ``dataset`` must be
    the full ``"<owner>/<slug>"`` reference; set SWN_KAGGLE_DATASET to change the default.
    """
    if "/" not in dataset:
        raise ValueError('Pass the full Kaggle reference, e.g. "owner/group21-swahili-audio".')
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(dest), "--unzip"],
        check=True,
    )
    return dest
