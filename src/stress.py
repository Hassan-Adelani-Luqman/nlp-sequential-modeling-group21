"""Temporal-order stress test: do the models depend on the ORDER of the sounds?

The word is first located with the centred energy window on the original clip.
Only then is the window perturbed, so every variant contains exactly the same
sounds and only their order changes:

* ``original`` - unchanged;
* ``reversed`` - the waveform played backwards;
* ``shuffled`` - the window cut into ~100 ms chunks, played in a random order.

Features are then computed by each model's own pipeline. Reversal leaves static
MFCC statistics exactly unchanged, but flips the sign of delta features, which
describe ~90 ms of local change. So only a static-statistics model is fully
order-free. Models that use order (A2-A5) should collapse, especially on words
that differ mainly in order (*tisa* / *sita*). The cuts between shuffled chunks
add small discontinuities, which mostly show up in delta features.
"""
from __future__ import annotations

import numpy as np

from src.data import TARGET_SR, load_audio
from src.features import FEATURES, energy_crop, peak_normalise

PERTURBATIONS = ("original", "reversed", "shuffled")


def perturb(wav: np.ndarray, kind: str, seed: int = 0, chunk_ms: int = 100, sr: int = TARGET_SR) -> np.ndarray:
    if kind == "original":
        return wav
    if kind == "reversed":
        return wav[::-1].copy()
    if kind == "shuffled":
        n_chunks = max(2, int(round(len(wav) / (sr * chunk_ms / 1000))))
        chunks = np.array_split(wav, n_chunks)
        order = np.random.default_rng(seed).permutation(n_chunks)
        if n_chunks > 1 and np.all(order == np.arange(n_chunks)):   # never leave the order unchanged
            order = np.roll(order, 1)
        return np.concatenate([chunks[i] for i in order])
    raise ValueError(f"kind must be one of {PERTURBATIONS}")


def perturbed_feature(path, kind: str, feature: str, seconds: float, seed: int = 0, **feature_kwargs) -> np.ndarray:
    """Features of one clip after locating the word on the original audio and perturbing that window."""
    window = energy_crop(load_audio(path), seconds)
    return FEATURES[feature](peak_normalise(perturb(window, kind, seed=seed)), **feature_kwargs)


def perturbed_features(paths, kind: str, feature: str, seconds: float, n_jobs: int = -1, **feature_kwargs):
    """Stacked features for many clips; clip ``i`` is shuffled with seed ``i`` (reproducible)."""
    from joblib import Parallel, delayed

    out = Parallel(n_jobs=n_jobs)(
        delayed(perturbed_feature)(p, kind, feature, seconds, seed=i, **feature_kwargs) for i, p in enumerate(paths))
    return np.stack(out)
