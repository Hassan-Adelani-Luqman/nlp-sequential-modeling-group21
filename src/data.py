"""Loading and splitting the Zindi Swahili Audio Classification data.

Every model reads its data through this module, so all five approaches are
trained and evaluated on exactly the same clips.

Data layout (as downloaded from Zindi)::

    <data_dir>/Train.csv           Word_id, Swahili_word, English_translation   (4,200 rows)
    <data_dir>/Test.csv            Word_id                                       (1,800 rows)
    <data_dir>/SampleSubmission.csv
    <data_dir>/Swahili_words/*.wav 16 kHz mono 16-bit PCM                        (6,000 files)

Typical use::

    from src.data import load_split, load_splits, load_audio, split_hash
    train = load_split("train")                 # id, path, label, gloss, label_id
    wav = load_audio(train.path[0])             # float32 waveform at 16 kHz
    print(split_hash())                         # must match data/splits/manifest.json

The frozen split is created once (by M1) with ``python -m src.data --make-splits``
and committed. Feature extraction lives in ``src/features.py``.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.paths import SPLITS_DIR, get_data_dir

SPLIT_NAMES = ("train", "val", "test")
TARGET_SR = 16_000  # native rate of the data and the rate wav2vec2 / XLS-R were pretrained on

ID_COLUMN = "Word_id"
LABEL_COLUMN = "Swahili_word"
GLOSS_COLUMN = "English_translation"
AUDIO_DIRNAME = "Swahili_words"


# --------------------------------------------------------------------------- locating files
def _find_csv(data_dir: Path, name: str) -> Path:
    for path in Path(data_dir).iterdir():
        if path.name.lower() == name.lower():
            return path
    raise FileNotFoundError(f"{name} not found in {data_dir}")


def audio_dir(data_dir: Path | str | None = None) -> Path:
    """Folder holding the .wav files (handles the nested folder some unzip tools create)."""
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    for candidate in (data_dir / AUDIO_DIRNAME / AUDIO_DIRNAME, data_dir / AUDIO_DIRNAME, data_dir):
        if candidate.is_dir() and next(candidate.glob("*.wav"), None) is not None:
            return candidate
    first = next(data_dir.rglob("*.wav"), None)
    if first is None:
        raise FileNotFoundError(f"No .wav files under {data_dir}; unzip Swahili_words.zip there (see data/README.md)")
    return first.parent


# --------------------------------------------------------------------------- metadata & audio
def load_metadata(split: str = "train", data_dir: Path | str | None = None) -> pd.DataFrame:
    """Load Train.csv or Test.csv as ``id, path[, label, gloss]`` and check every file exists."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    src = pd.read_csv(_find_csv(data_dir, "Train.csv" if split == "train" else "Test.csv"))
    if ID_COLUMN not in src.columns:
        raise ValueError(f"Expected a '{ID_COLUMN}' column, found {list(src.columns)}")

    folder = audio_dir(data_dir)
    df = pd.DataFrame({"id": src[ID_COLUMN].astype(str)})
    df["path"] = [str(folder / i) for i in df["id"]]
    if split == "train":
        df["label"] = src[LABEL_COLUMN].astype(str).str.strip()
        df["gloss"] = src[GLOSS_COLUMN].astype(str).str.strip() if GLOSS_COLUMN in src else ""

    if df["id"].duplicated().any():
        raise ValueError(f"Duplicate ids in {split} metadata")
    missing = [p for p in df["path"] if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {split} audio files are missing, e.g. {missing[0]}")
    return df


def load_audio(path: Path | str, sr: int = TARGET_SR) -> np.ndarray:
    """Read a clip as a mono float32 waveform in [-1, 1], resampled to ``sr`` if needed."""
    import soundfile as sf

    wav, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=sr)
    return wav.astype(np.float32, copy=False)


def file_md5(path: Path | str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_duplicate_audio(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose audio bytes are identical to an earlier row (``df`` needs a ``path`` column).

    Returns the later copies with an added ``md5`` and ``duplicate_of`` column.
    """
    md5 = df["path"].map(file_md5)
    first_id = df.groupby(md5)["id"].transform("first")
    dups = df[md5.duplicated(keep="first")].copy()
    dups["md5"] = md5[dups.index]
    dups["duplicate_of"] = first_id[dups.index]
    return dups


# --------------------------------------------------------------------------- splits
def make_splits(
    seed: int = 42,
    val_size: float = 0.15,
    test_size: float = 0.15,
    data_dir: Path | str | None = None,
    splits_dir: Path | str = SPLITS_DIR,
    overwrite: bool = False,
) -> dict:
    """Create the frozen stratified train/val/test split and write it to ``splits_dir``.

    Byte-identical duplicate recordings are dropped first (keeping the first copy)
    so a clip can never appear in both train and test. Writes
    ``{train,val,test}_ids.csv``, ``label_map.json`` and ``manifest.json``.
    Refuses to overwrite an existing split unless ``overwrite=True``, because every
    group member's results depend on it.
    """
    splits_dir = Path(splits_dir)
    if (splits_dir / "manifest.json").exists() and not overwrite:
        raise FileExistsError(f"A frozen split already exists in {splits_dir}; pass overwrite=True to replace it.")

    df = load_metadata("train", data_dir)
    dups = find_duplicate_audio(df)
    label_of = df.set_index("id")["label"]
    n_conflicts = int((dups["label"].values != label_of[dups["duplicate_of"]].values).sum())
    df = df[~df["id"].isin(dups["id"])]

    train_df, holdout = train_test_split(
        df, test_size=val_size + test_size, stratify=df["label"], random_state=seed
    )
    val_df, test_df = train_test_split(
        holdout, test_size=test_size / (val_size + test_size), stratify=holdout["label"], random_state=seed
    )
    parts = {"train": train_df, "val": val_df, "test": test_df}

    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        part[["id", "label"]].to_csv(splits_dir / f"{name}_ids.csv", index=False)

    labels = sorted(df["label"].unique())
    label2id = {label: i for i, label in enumerate(labels)}
    (splits_dir / "label_map.json").write_text(json.dumps(label2id, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "fractions": {"train": round(1 - val_size - test_size, 4), "val": val_size, "test": test_size},
        "sizes": {name: len(part) for name, part in parts.items()},
        "class_counts": {name: part["label"].value_counts().sort_index().to_dict() for name, part in parts.items()},
        "glosses": df.drop_duplicates("label").set_index("label")["gloss"].sort_index().to_dict(),
        "source": {
            "n_source_rows": int(len(label_of)),
            "n_duplicates_dropped": int(len(dups)),
            "n_label_conflicts": n_conflicts,
            "dropped_duplicate_ids": dups["id"].tolist(),
        },
        "split_hash": split_hash(splits_dir),
    }
    (splits_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest


def split_hash(splits_dir: Path | str = SPLITS_DIR) -> str:
    """MD5 over the sorted ids of each split. Identical hash = identical split."""
    md5 = hashlib.md5()
    for name in SPLIT_NAMES:
        ids = pd.read_csv(Path(splits_dir) / f"{name}_ids.csv", dtype={"id": str})["id"]
        md5.update(f"{name}:{','.join(sorted(ids))}\n".encode("utf-8"))
    return md5.hexdigest()


def verify_splits(splits_dir: Path | str = SPLITS_DIR) -> bool:
    """True if the split files on disk match the hash recorded in manifest.json."""
    manifest = json.loads((Path(splits_dir) / "manifest.json").read_text(encoding="utf-8"))
    return split_hash(splits_dir) == manifest["split_hash"]


def load_label_map(splits_dir: Path | str = SPLITS_DIR) -> dict[str, int]:
    return json.loads((Path(splits_dir) / "label_map.json").read_text(encoding="utf-8"))


def label_names(splits_dir: Path | str = SPLITS_DIR) -> list[str]:
    """Class names ordered by label id (use for plot axes and probability columns)."""
    label2id = load_label_map(splits_dir)
    return sorted(label2id, key=label2id.get)


def _select(meta: pd.DataFrame, name: str, splits_dir: Path, label2id: dict[str, int]) -> pd.DataFrame:
    if name not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    ids = pd.read_csv(splits_dir / f"{name}_ids.csv", dtype={"id": str})[["id"]]
    df = ids.merge(meta, on="id", how="left", validate="one_to_one")
    if df["path"].isna().any():
        missing = int(df["path"].isna().sum())
        raise ValueError(f"{missing} ids in {name}_ids.csv are missing from Train.csv - is this the same data?")
    df["label_id"] = df["label"].map(label2id).astype(int)
    return df


def load_split(name: str, data_dir: Path | str | None = None, splits_dir: Path | str = SPLITS_DIR) -> pd.DataFrame:
    """Return one split as ``id, path, label, gloss, label_id``, in the frozen order."""
    splits_dir = Path(splits_dir)
    return _select(load_metadata("train", data_dir), name, splits_dir, load_label_map(splits_dir))


def load_splits(data_dir: Path | str | None = None, splits_dir: Path | str = SPLITS_DIR) -> dict[str, pd.DataFrame]:
    """Return ``{"train": df, "val": df, "test": df}``."""
    splits_dir = Path(splits_dir)
    meta = load_metadata("train", data_dir)
    label2id = load_label_map(splits_dir)
    return {name: _select(meta, name, splits_dir, label2id) for name in SPLIT_NAMES}


def load_competition_test(data_dir: Path | str | None = None) -> pd.DataFrame:
    """The unlabelled Zindi Test.csv as ``id, path`` (only for an optional leaderboard submission)."""
    return load_metadata("test", data_dir)


if __name__ == "__main__":
    # python -m src.data --make-splits   (M1, once)   |   python -m src.data   (anyone: verify)
    import argparse

    parser = argparse.ArgumentParser(description="Create or verify the frozen train/val/test split.")
    parser.add_argument("--make-splits", action="store_true", help="create data/splits (refuses to overwrite)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing split (coordinate with the group!)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.make_splits:
        info = make_splits(seed=args.seed, overwrite=args.overwrite)
        info["source"]["dropped_duplicate_ids"] = info["source"]["dropped_duplicate_ids"][:10]
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        manifest = json.loads((SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
        ok = verify_splits()
        sizes = {name: len(df) for name, df in load_splits().items()}
        print(f"split hash : {split_hash()}  ({'OK - matches manifest' if ok else 'MISMATCH with manifest!'})")
        print(f"sizes      : {sizes}  (manifest: {manifest['sizes']})")
        print(f"classes    : {label_names()}")
