"""Smoke tests for the shared foundations (run with ``pytest -q``).

Most tests use a small synthetic dataset (12 "words" = 12 tone frequencies,
buried in silence and noise) so they run anywhere in a few seconds. The last
test checks the real committed split and is skipped when the data is absent.
"""
import json
import shutil

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from src import data, evaluate, features, utils
from src.paths import SPLITS_DIR

WORDS = ["hapana", "kumi", "mbili", "moja", "nane", "ndio", "nne", "saba", "sita", "tano", "tatu", "tisa"]
SR = data.TARGET_SR


def tone_clip(freq: float, total_s: float, burst_start_s: float, burst_s: float = 0.6, seed: int = 0) -> np.ndarray:
    """Quiet noise with a loud sine burst - a stand-in for a word surrounded by silence."""
    rng = np.random.default_rng(seed)
    wav = 0.005 * rng.standard_normal(int(total_s * SR))
    t = np.arange(int(burst_s * SR)) / SR
    start = int(burst_start_s * SR)
    wav[start: start + len(t)] += 0.5 * np.sin(2 * np.pi * freq * t)
    return wav.astype(np.float32)


@pytest.fixture()
def fake_data(tmp_path):
    """Zindi-shaped data: 20 clips per word, plus one byte-identical duplicate with a wrong label."""
    audio = tmp_path / "raw" / "Swahili_words"
    audio.mkdir(parents=True)
    rows = []
    for c, word in enumerate(WORDS):
        for i in range(20):
            wid = f"id_{word}_{i}.wav"
            clip = tone_clip(freq=200 + 60 * c, total_s=2.0 + i % 3, burst_start_s=0.3 + 0.1 * (i % 10), seed=i)
            sf.write(audio / wid, clip, SR, subtype="PCM_16")
            rows.append({"Word_id": wid, "Swahili_word": word, "English_translation": f"gloss_{word}"})
    shutil.copy(audio / rows[0]["Word_id"], audio / "id_dup.wav")
    rows.append({"Word_id": "id_dup.wav", "Swahili_word": "tisa", "English_translation": "nine"})
    raw = tmp_path / "raw"
    pd.DataFrame(rows).to_csv(raw / "Train.csv", index=False)
    pd.DataFrame({"Word_id": [r["Word_id"] for r in rows[:5]]}).to_csv(raw / "Test.csv", index=False)
    return raw, tmp_path / "splits"


# --------------------------------------------------------------------------- data
def test_metadata_and_audio(fake_data):
    raw, _ = fake_data
    meta = data.load_metadata("train", raw)
    assert list(meta.columns) == ["id", "path", "label", "gloss"] and len(meta) == 241
    wav = data.load_audio(meta.path[0])
    assert wav.dtype == np.float32 and wav.ndim == 1 and len(wav) == 2 * SR
    assert len(data.load_competition_test(raw)) == 5

    dups = data.find_duplicate_audio(meta)
    assert list(dups["id"]) == ["id_dup.wav"] and dups["duplicate_of"].iloc[0] == meta.id[0]


def test_splits_are_stratified_disjoint_and_frozen(fake_data):
    raw, splits_dir = fake_data
    manifest = data.make_splits(seed=42, data_dir=raw, splits_dir=splits_dir)
    assert manifest["sizes"] == {"train": 168, "val": 36, "test": 36}  # 70/15/15 of 240 after dedup
    assert manifest["source"]["n_duplicates_dropped"] == 1
    assert manifest["source"]["n_label_conflicts"] == 1
    for counts, n in zip(manifest["class_counts"].values(), (14, 3, 3)):
        assert set(counts.values()) == {n}  # exact stratification

    splits = data.load_splits(data_dir=raw, splits_dir=splits_dir)
    ids = [set(df["id"]) for df in splits.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert "id_dup.wav" not in set().union(*ids)
    assert data.verify_splits(splits_dir) and data.label_names(splits_dir) == WORDS

    with pytest.raises(FileExistsError):
        data.make_splits(data_dir=raw, splits_dir=splits_dir)
    again = data.make_splits(seed=42, data_dir=raw, splits_dir=splits_dir, overwrite=True)
    assert again["split_hash"] == manifest["split_hash"]  # same seed -> same split


# --------------------------------------------------------------------------- features
def test_energy_crop_finds_the_word():
    wav = tone_clip(440, total_s=5.0, burst_start_s=3.0)
    crop = features.energy_crop(wav, seconds=1.0)
    assert len(crop) == SR
    assert (crop ** 2).sum() > 0.95 * (wav ** 2).sum()  # the burst is inside the window
    energy = crop.astype(np.float64) ** 2
    centroid_s = (np.arange(len(crop)) * energy).sum() / energy.sum() / SR
    assert centroid_s == pytest.approx(0.5, abs=0.05)  # ... and centred in it, not at a noise-chosen offset
    assert len(features.energy_crop(wav[: SR // 2], seconds=1.0)) == SR  # short clips are padded


def test_trim_and_fix_length():
    wav = tone_clip(440, total_s=3.0, burst_start_s=1.0, burst_s=0.5)
    trimmed = features.trim_silence(wav, top_db=30)
    assert 0.5 * SR <= len(trimmed) <= 0.7 * SR
    assert len(features.fix_length(trimmed, 1.5)) == int(1.5 * SR)
    assert len(features.fix_length(wav, 1.5)) == int(1.5 * SR)


def test_feature_shapes():
    wav = features.preprocess(tone_clip(440, 4.0, 2.0), crop="energy", seconds=1.5)
    assert np.abs(wav).max() == pytest.approx(0.95, abs=1e-3)
    assert features.log_mel(wav).shape == (151, 64)  # 1.5 s at a 10 ms hop
    assert features.mfcc(wav).shape == (151, 120)
    assert features.mfcc(wav, n_mfcc=13).shape == (151, 39)
    assert features.mfcc_stats(wav).shape == (480,)


def test_cached_features_roundtrip(fake_data, tmp_path, monkeypatch):
    raw, splits_dir = fake_data
    monkeypatch.setattr(features, "cache_dir", lambda: tmp_path / "cache")
    data.make_splits(data_dir=raw, splits_dir=splits_dir)
    val = data.load_split("val", raw, splits_dir)

    X = features.cached_features(val, "logmel", n_jobs=1)
    assert X.shape == (36, 151, 64)
    assert np.array_equal(X, features.cached_features(val, "logmel", n_jobs=1))  # served from cache
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 1

    seqs = features.cached_features(val, "mfcc", crop="trim", seconds=None, n_jobs=1, n_mfcc=13)
    assert isinstance(seqs, list) and len(seqs) == 36 and seqs[0].shape[1] == 39
    again = features.cached_features(val, "mfcc", crop="trim", seconds=None, n_jobs=1, n_mfcc=13)
    assert all(np.array_equal(a, b) for a, b in zip(seqs, again))

    # A read-only copy (e.g. the group's features dataset attached on Kaggle) is used without re-extracting.
    shared = tmp_path / "shared_features"
    shared.mkdir()
    name = features.cache_filename(val, "logmel")
    shutil.move(tmp_path / "cache" / name, shared / name)
    monkeypatch.setenv("SWN_FEATURE_CACHE", str(shared))
    monkeypatch.setattr(features, "extract_many", lambda *a, **k: pytest.fail("should load from the shared cache"))
    assert np.array_equal(X, features.cached_features(val, "logmel"))
    assert "_v" in name  # feature version is part of the key

    Z = features.Standardizer().fit_transform(X)
    assert np.allclose(Z.reshape(-1, 64).mean(0), 0, atol=1e-3)
    Zs = features.Standardizer().fit_transform(seqs)  # variable-length list input
    assert len(Zs) == 36 and np.allclose(np.concatenate(Zs).mean(0), 0, atol=1e-3)


# --------------------------------------------------------------------------- evaluation & logging
def test_compute_metrics():
    y_true = np.array(list(range(12)) * 3)
    perfect = np.eye(12)[y_true] * 0.89 + 0.01
    m = evaluate.compute_metrics(y_true, perfect, WORDS)
    assert m["macro_f1"] == 1.0 and m["accuracy"] == 1.0 and m["roc_auc_ovr"] == 1.0
    assert m["log_loss"] < 0.15 and set(m["per_class"]) == set(WORDS)
    uniform = np.full((36, 12), 1 / 12)
    assert evaluate.compute_metrics(y_true, uniform)["log_loss"] == pytest.approx(np.log(12))


def test_significance_helpers():
    y = np.array([0, 1] * 50)
    point, low, high = evaluate.bootstrap_ci(y, y, n_resamples=50)
    assert point == low == high == 1.0
    res = evaluate.mcnemar_test(y, y, 1 - y)
    assert res["only_a_correct"] == 100 and res["p_value"] < 1e-6


def test_prediction_roundtrip(tmp_path):
    prob = np.random.default_rng(0).dirichlet(np.ones(12), size=8)
    path = evaluate.save_predictions("A1-R0-01", 42, "val", [f"d{i}" for i in range(8)], [0] * 8, prob, WORDS, tmp_path)
    df, loaded = evaluate.load_predictions(path)
    assert path.name == "A1-R0-01_seed42_val.csv"
    np.testing.assert_allclose(loaded, prob)
    assert (df["y_pred"] == prob.argmax(axis=1)).all()


def test_experiment_logging(tmp_path):
    runs = tmp_path / "runs"
    utils.log_experiment({"exp_id": "A3-R1-01", "seed": 42, "member": "M2", "val_macro_f1": 0.812345}, runs)
    utils.log_experiment({"exp_id": "A1-R0-01", "seed": 42, "member": "M1"}, runs)
    with pytest.raises(FileExistsError):
        utils.log_experiment({"exp_id": "A3-R1-01", "seed": 42}, runs)
    with pytest.raises(ValueError):
        utils.log_experiment({"exp_id": "bilstm-1", "seed": 42}, runs)
    with pytest.raises(KeyError):
        utils.log_experiment({"exp_id": "A3-R1-02", "seed": 42, "f1": 0.8}, runs)

    table = utils.build_experiment_table(runs)
    assert list(table["exp_id"]) == ["A1-R0-01", "A3-R1-01"]
    assert table.loc[1, "val_macro_f1"] == 0.8123 and table.loc[1, "approach"] == "A3"
    assert (tmp_path / "experiments.csv").exists()


# --------------------------------------------------------------------------- real data
@pytest.mark.skipif(not (SPLITS_DIR / "manifest.json").exists(), reason="frozen split not created yet")
def test_committed_split_is_intact():
    assert data.verify_splits(), "data/splits files changed - they no longer match manifest.json"
    manifest = json.loads((SPLITS_DIR / "manifest.json").read_text(encoding="utf-8"))
    try:
        splits = data.load_splits()
    except FileNotFoundError:
        pytest.skip("audio data not available in this environment")
    assert {k: len(v) for k, v in splits.items()} == manifest["sizes"]
