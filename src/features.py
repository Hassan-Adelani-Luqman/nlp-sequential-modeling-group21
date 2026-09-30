"""Audio preprocessing and acoustic features shared by all models.

Pipeline per clip::

    load_audio (16 kHz mono) -> crop (energy window | trim | none) -> peak-normalise -> features

Why an energy window by default: clips are 2-62 s long (median 4.3 s) but a
spoken digit lasts under a second, and a single ``top_db`` trim leaves long
noisy tails in many clips. ``energy_crop`` keeps the fixed-length window with
the most signal energy, which contains the word in the typical clip and gives
every model the same input length. The window length is an R2 ablation
(1.0 / 1.5 / 2.0 s).

Feature types (frame rate 100 Hz: 25 ms window, 10 ms hop):
  * ``logmel``     - (T, 64) log-mel spectrogram           -> A3, A4
  * ``mfcc``       - (T, 120) 40 MFCC + delta + delta-delta -> A2 (MFCC-13 variant), A3/A4 ablation
  * ``mfcc_stats`` - (480,) mean/std/min/max of ``mfcc``    -> A1 (no temporal order)
  * ``wave``       - raw cropped waveform                   -> A5

Features for a whole split are computed once and cached with ``cached_features``.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from src.data import TARGET_SR, load_audio
from src.paths import cache_dir

# Part of every cache key. Bump it whenever preprocessing or feature code changes,
# so old cached features (local or on Kaggle) are never silently reused.
FEATURE_VERSION = 2  # v2: energy_crop centres the word in the window

N_FFT = 400        # 25 ms at 16 kHz
HOP_LENGTH = 160   # 10 ms -> 100 frames per second
N_MELS = 64
N_MFCC = 40


# --------------------------------------------------------------------------- waveform preprocessing
def trim_silence(wav: np.ndarray, top_db: float = 30, pad_ms: int = 50, sr: int = TARGET_SR) -> np.ndarray:
    """Strip leading/trailing audio quieter than ``top_db`` below the peak, keeping ``pad_ms`` margins."""
    import librosa

    _, (start, end) = librosa.effects.trim(wav, top_db=top_db, frame_length=N_FFT, hop_length=HOP_LENGTH)
    pad = int(sr * pad_ms / 1000)
    return wav[max(0, start - pad): min(len(wav), end + pad)]


def fix_length(wav: np.ndarray, seconds: float, sr: int = TARGET_SR) -> np.ndarray:
    """Centre-crop or symmetrically zero-pad to exactly ``seconds``."""
    target = int(round(seconds * sr))
    if len(wav) >= target:
        start = (len(wav) - target) // 2
        return wav[start: start + target]
    left = (target - len(wav)) // 2
    return np.pad(wav, (left, target - len(wav) - left))


def energy_crop(wav: np.ndarray, seconds: float = 1.5, sr: int = TARGET_SR, tolerance: float = 0.01) -> np.ndarray:
    """Keep the ``seconds``-long window with the highest energy, with the word centred in it.

    When the whole word fits in the window, many start positions hold almost the
    same energy and a plain argmax lands anywhere on that plateau (decided by
    background noise), so the word would sit at a random offset. We therefore
    take the middle of the contiguous run of starts within ``tolerance`` of the
    maximum energy around the argmax. Clips shorter than the window are zero-padded.
    """
    target = int(round(seconds * sr))
    if len(wav) <= target:
        return fix_length(wav, seconds, sr)
    cumulative = np.concatenate([[0.0], np.cumsum(wav.astype(np.float64) ** 2)])
    window_energy = cumulative[target:] - cumulative[:-target]
    best = int(np.argmax(window_energy))
    below = np.flatnonzero(window_energy < (1 - tolerance) * window_energy[best])
    before, after = below[below < best], below[below > best]
    lo = before[-1] + 1 if before.size else 0
    hi = after[0] - 1 if after.size else len(window_energy) - 1
    start = (lo + hi) // 2
    return wav[start: start + target]


def peak_normalise(wav: np.ndarray, peak: float = 0.95) -> np.ndarray:
    top = np.abs(wav).max()
    return wav if top < 1e-6 else (wav * (peak / top)).astype(np.float32)


def preprocess(wav: np.ndarray, crop: str = "energy", seconds: float | None = 1.5,
               top_db: float = 30, normalise: bool = True) -> np.ndarray:
    """Apply the cropping strategy used throughout the project.

    ``crop``: ``"energy"`` (fixed window around the loudest region), ``"trim"``
    (silence trim, then ``fix_length`` if ``seconds`` is given, else variable
    length - used by the HMM), or ``"none"`` (``fix_length`` only, if ``seconds``).
    """
    if crop == "energy":
        if seconds is None:
            raise ValueError("energy crop needs a fixed length in seconds")
        wav = energy_crop(wav, seconds)
    elif crop == "trim":
        wav = trim_silence(wav, top_db)
        if seconds is not None:
            wav = fix_length(wav, seconds)
    elif crop == "none":
        if seconds is not None:
            wav = fix_length(wav, seconds)
    else:
        raise ValueError("crop must be 'energy', 'trim' or 'none'")
    return peak_normalise(wav) if normalise else wav


# --------------------------------------------------------------------------- features
def log_mel(wav: np.ndarray, sr: int = TARGET_SR, n_mels: int = N_MELS) -> np.ndarray:
    """Log-mel spectrogram, shape (frames, n_mels), in dB."""
    import librosa

    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=n_mels)
    return librosa.power_to_db(mel, ref=1.0, top_db=80.0).T.astype(np.float32)


def mfcc(wav: np.ndarray, sr: int = TARGET_SR, n_mfcc: int = N_MFCC, deltas: bool = True) -> np.ndarray:
    """MFCCs (+ delta, delta-delta), shape (frames, n_mfcc * 3 if deltas else n_mfcc)."""
    import librosa

    mel_db = log_mel(wav, sr).T
    coeffs = librosa.feature.mfcc(S=mel_db, n_mfcc=n_mfcc)
    if deltas:
        n_frames = coeffs.shape[1]
        # librosa needs an odd delta window no longer than the clip; 9 frames (90 ms) normally.
        width = 9 if n_frames >= 9 else max(3, n_frames if n_frames % 2 else n_frames - 1)
        coeffs = np.vstack([coeffs, librosa.feature.delta(coeffs, width=width),
                            librosa.feature.delta(coeffs, order=2, width=width)])
    return coeffs.T.astype(np.float32)


def mfcc_stats(wav: np.ndarray, sr: int = TARGET_SR, n_mfcc: int = N_MFCC) -> np.ndarray:
    """Order-agnostic clip vector: mean, std, min, max of every MFCC/delta channel."""
    frames = mfcc(wav, sr, n_mfcc)
    return np.concatenate([frames.mean(0), frames.std(0), frames.min(0), frames.max(0)]).astype(np.float32)


FEATURES = {
    "logmel": log_mel,
    "mfcc": mfcc,
    "mfcc_stats": mfcc_stats,
    "wave": lambda wav, **_: wav.astype(np.float32),
}


def extract(path, feature: str = "logmel", crop: str = "energy", seconds: float | None = 1.5,
            top_db: float = 30, **feature_kwargs) -> np.ndarray:
    """Load one clip and return its features."""
    wav = preprocess(load_audio(path), crop=crop, seconds=seconds, top_db=top_db)
    return FEATURES[feature](wav, **feature_kwargs)


def is_variable_length(feature: str, seconds: float | None) -> bool:
    """Frame features without a fixed clip length are returned as a list of (T_i, C) arrays."""
    return seconds is None and feature not in ("mfcc_stats",)


def extract_many(paths, feature: str = "logmel", n_jobs: int = -1, seconds: float | None = 1.5, **kwargs):
    """Features for many clips in parallel.

    Returns a stacked array, or a list of arrays when ``is_variable_length(feature, seconds)``.
    """
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=n_jobs)(delayed(extract)(p, feature, seconds=seconds, **kwargs) for p in paths)
    return out if is_variable_length(feature, seconds) else np.stack(out)


# Default feature sets, one per model family. Use them as ``cached_features(df, **PRESETS["logmel"])``
# so every member computes (and shares) identical features.
PRESETS = {
    "logmel": {"feature": "logmel", "crop": "energy", "seconds": 1.5},                        # A3, A4
    "mfcc_stats": {"feature": "mfcc_stats", "crop": "energy", "seconds": 1.5},                # A1
    "mfcc13_trim": {"feature": "mfcc", "crop": "trim", "seconds": None, "n_mfcc": 13},       # A2 (variable length)
}


def cache_filename(df, feature: str = "logmel", crop: str = "energy", seconds: float | None = 1.5,
                   top_db: float = 30, **feature_kwargs) -> str:
    """Deterministic cache file name for these clips and settings (same on every machine)."""
    params = {"feature": feature, "crop": crop, "seconds": seconds, "top_db": top_db,
              "version": FEATURE_VERSION, **feature_kwargs}
    key = hashlib.md5((json.dumps(params, sort_keys=True) + ",".join(df["id"])).encode()).hexdigest()[:10]
    length = f"{seconds}s" if seconds is not None else "varlen"
    return f"{feature}_{crop}_{length}_v{FEATURE_VERSION}_{key}.npz"


def _cache_search_dirs() -> list[Path]:
    """The writable cache first, then read-only copies: ``SWN_FEATURE_CACHE`` and any Kaggle input."""
    dirs = [cache_dir()]
    if os.environ.get("SWN_FEATURE_CACHE"):
        dirs.append(Path(os.environ["SWN_FEATURE_CACHE"]))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        dirs.extend(sorted(p for p in kaggle_input.iterdir() if p.is_dir()))
        # the features dataset may be mounted deeper: locate it by its manifest
        dirs.extend(sorted({p.parent for p in kaggle_input.rglob("features_manifest.json")}))
    return dirs


def find_cached(name: str) -> Path | None:
    for folder in _cache_search_dirs():
        for candidate in (folder / name, folder / "cache" / name):
            if candidate.is_file():
                return candidate
    return None


def cached_features(df, feature: str = "logmel", crop: str = "energy", seconds: float | None = 1.5,
                    top_db: float = 30, n_jobs: int = -1, refresh: bool = False, **feature_kwargs):
    """Features for every row of a split DataFrame (needs ``id`` and ``path``), cached on disk.

    The cache key covers the clip ids and all preprocessing parameters, so
    changing any setting creates a new cache file instead of silently reusing
    stale features. Cached files are also found in read-only locations, such as
    the group's features dataset attached as a Kaggle input, so GPU sessions skip
    extraction. Returns an array, or a list when ``is_variable_length(feature, seconds)``.
    """
    name = cache_filename(df, feature, crop, seconds, top_db, **feature_kwargs)
    found = None if refresh else find_cached(name)
    if found is not None:
        stored = np.load(found, allow_pickle=False)
        if "lengths" in stored:
            return np.split(stored["X"], np.cumsum(stored["lengths"])[:-1])
        return stored["X"]
    path = cache_dir() / name

    X = extract_many(df["path"], feature, n_jobs=n_jobs, crop=crop, seconds=seconds, top_db=top_db, **feature_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(X, list):  # variable-length sequences: store concatenated frames + lengths (hmmlearn format)
        np.savez_compressed(path, X=np.concatenate(X), lengths=np.array([len(x) for x in X]))
    else:
        np.savez_compressed(path, X=X)
    return X


def utterance_cmvn(seqs: list[np.ndarray], eps: float = 1e-6) -> list[np.ndarray]:
    """Per-utterance cepstral mean and variance normalisation.

    Each clip is normalised with its own mean and std. This removes stationary
    channel effects (different phones and microphones across ~300 speakers)
    that a global Standardizer cannot, which is standard practice for GMM-HMM ASR.
    """
    return [((s - s.mean(0)) / (s.std(0) + eps)).astype(np.float32) for s in seqs]


def prefix_stats(path, fraction: float, seconds: float = 1.5, top_db: float = 30,
                 n_mfcc: int = N_MFCC) -> np.ndarray:
    """``mfcc_stats`` of only the first ``fraction`` of the spoken word (for the "how early" analysis).

    The word is located by the centred energy window, then its leading and
    trailing silence inside that window is trimmed, and the first ``fraction``
    of what remains is kept (at least 100 ms, so delta features still fit).
    """
    word = trim_silence(energy_crop(load_audio(path), seconds), top_db=top_db, pad_ms=0)
    keep = max(int(len(word) * fraction), TARGET_SR // 10)
    return mfcc_stats(peak_normalise(word[:keep]), n_mfcc=n_mfcc)


class Standardizer:
    """Per-channel mean/variance normalisation, fitted on the training split only.

    Works on (N, T, C) frame features, (N, C) vectors, or a list of (T_i, C) arrays.
    """

    def __init__(self, eps: float = 1e-6):
        self.eps = eps
        self.mean_ = None
        self.std_ = None

    @staticmethod
    def _frames(X) -> np.ndarray:
        if isinstance(X, list):
            return np.concatenate(X)
        return X.reshape(-1, X.shape[-1])

    def fit(self, X) -> "Standardizer":
        frames = self._frames(X)
        self.mean_ = frames.mean(0)
        self.std_ = frames.std(0) + self.eps
        return self

    def transform(self, X):
        if isinstance(X, list):
            return [((x - self.mean_) / self.std_).astype(np.float32) for x in X]
        return ((X - self.mean_) / self.std_).astype(np.float32)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def build_cache(presets: dict | None = None, n_jobs: int = -1, refresh: bool = False) -> dict:
    """Compute every preset for the train/val/test splits and write ``cache/features_manifest.json``.

    Run once (``python -m src.features``); upload ``results/cache/`` as the group's
    private features dataset so nobody has to re-extract on Kaggle.
    """
    from src.data import load_splits, split_hash

    presets = presets or PRESETS
    splits = load_splits()
    manifest = {"split_hash": split_hash(), "feature_version": FEATURE_VERSION, "presets": {}}
    for preset, params in presets.items():
        manifest["presets"][preset] = {"params": params, "splits": {}}
        for split, df in splits.items():
            start = time.time()
            X = cached_features(df, n_jobs=n_jobs, refresh=refresh, **params)
            shape = [len(X), "variable", int(X[0].shape[1])] if isinstance(X, list) else list(X.shape)
            manifest["presets"][preset]["splits"][split] = {"file": cache_filename(df, **params), "shape": shape}
            print(f"{preset:12s} {split:5s} shape={shape}  ({time.time() - start:.0f}s)", flush=True)
    out = cache_dir() / "features_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Precompute the shared feature presets for all splits.")
    parser.add_argument("--refresh", action="store_true", help="recompute even if cached")
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    build_cache(n_jobs=args.n_jobs, refresh=args.refresh)
    print(f"cache: {cache_dir()}")
