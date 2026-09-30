"""Tests for the neural-model infrastructure: augmentation, the A3 BiLSTM, the training loop, the runner.

Run with ``pytest -q``. They use small synthetic data and finish in well under a minute on CPU.
"""
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.augment import LogMelAugment  # noqa: E402
from src.models.bilstm import BiLSTMAttention, count_parameters  # noqa: E402
from src.train_torch import TrainConfig, predict_logits, train_model  # noqa: E402


def fake_logmel(n_frames=151, n_mels=64, seed=0):
    """Silence (-80 dB) with a loud 'word' in the middle, like a centred energy-window crop."""
    rng = np.random.default_rng(seed)
    x = np.full((n_frames, n_mels), -80.0) + rng.normal(0, 1, (n_frames, n_mels))
    x[50:100] = -20.0 + rng.normal(0, 3, (50, n_mels))
    return x.astype(np.float32)


def mirror_task(n_per_class=40, n_frames=30, seed=0):
    """Two classes of identical frames in opposite order ("tisa" vs "sita")."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for label, (a, b) in enumerate([(-1.0, 1.0), (1.0, -1.0)]):
        for _ in range(n_per_class):
            seq = np.vstack([np.full((n_frames // 2, 4), a), np.full((n_frames // 2, 4), b)])
            X.append(seq + 0.3 * rng.standard_normal(seq.shape))
            y.append(label)
    X, y = np.array(X, dtype=np.float32), np.array(y)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ------------------------------------------------------------------ augmentation
def test_augment_preserves_shape_and_range():
    x = fake_logmel()
    for aug in (LogMelAugment(seed=0), LogMelAugment.specaugment_only(seed=0)):
        out = aug(x)
        assert out.shape == x.shape and out.dtype == np.float32
        assert out.max() <= x.max() + 1.0  # noise injection can only add power, and only a little at 10-30 dB SNR
    a, b = LogMelAugment(seed=3)(x), LogMelAugment(seed=3)(x)
    assert np.array_equal(a, b)  # seeded -> reproducible


def test_shift_pads_with_silence_instead_of_wrapping():
    x = fake_logmel()
    aug = LogMelAugment(time_masks=0, freq_masks=0, stretch=None, noise_snr_db=None, shift_frames=10, seed=1)
    out = aug(x)
    assert np.isclose(np.abs(out - x).max(), 0) or out[:10].mean() < -70 or out[-10:].mean() < -70


def test_noise_raises_the_silence_floor():
    x = fake_logmel()
    aug = LogMelAugment(time_masks=0, freq_masks=0, stretch=None, shift_frames=0, noise_snr_db=(10, 10), p_noise=1.0, seed=0)
    out = aug(x)
    assert out[:40].mean() > x[:40].mean() + 20  # silence now sits ~SNR dB below the word, not 60 dB
    assert abs(out[60:90].mean() - x[60:90].mean()) < 2  # the word itself barely changes


def test_specaugment_masks_something():
    x = fake_logmel()
    aug = LogMelAugment(time_masks=2, time_width=20, freq_masks=2, freq_width=8, shift_frames=0, stretch=None,
                        noise_snr_db=None, seed=5)
    assert (aug(x) != x).any()


# ------------------------------------------------------------------ model
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("pooling", ["attention", "last", "mean"])
def test_bilstm_shapes(bidirectional, pooling):
    model = BiLSTMAttention(n_features=64, bidirectional=bidirectional, pooling=pooling)
    logits = model(torch.randn(3, 151, 64))
    assert logits.shape == (3, 12)
    if pooling == "attention":
        _, w = model(torch.randn(3, 151, 64), return_attention=True)
        assert w.shape == (3, 151) and torch.allclose(w.sum(1), torch.ones(3))
    assert count_parameters(model) > 0


def test_conv_frontend_variant():
    assert BiLSTMAttention(n_features=64, conv_frontend=True)(torch.randn(2, 151, 64)).shape == (2, 12)


# ------------------------------------------------------------------ training loop
def test_training_learns_temporal_order_and_is_reproducible(tmp_path):
    X, y = mirror_task()
    X_tr, y_tr, X_va, y_va = X[:60], y[:60], X[60:], y[60:]
    cfg = TrainConfig(epochs=15, batch_size=16, lr=5e-3, patience=15, seed=0)
    make = lambda: BiLSTMAttention(n_features=4, n_classes=2, hidden=16, layers=1)
    r1 = train_model(make, X_tr, y_tr, X_va, y_va, cfg, verbose=False, checkpoint=tmp_path / "m.pt")
    r2 = train_model(make, X_tr, y_tr, X_va, y_va, cfg, verbose=False)
    assert (r1.val_logits.argmax(1) == y_va).mean() >= 0.95  # the order is learnt
    np.testing.assert_allclose(r1.val_logits, r2.val_logits, atol=1e-5)  # same seed -> same model
    assert {"epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1"} <= set(r1.history.columns)
    assert 1 <= r1.best_epoch <= len(r1.history)
    assert (tmp_path / "m.pt").exists()
    np.testing.assert_allclose(predict_logits(r1.model, X_va, r1.scaler), r1.val_logits, atol=1e-5)

    from src.train_torch import load_checkpoint  # a reloaded checkpoint reproduces the trained model exactly

    model, scaler, ckpt = load_checkpoint(tmp_path / "m.pt", make)
    np.testing.assert_allclose(predict_logits(model, X_va, scaler), r1.val_logits, atol=1e-5)
    assert ckpt["best_epoch"] == r1.best_epoch


# ------------------------------------------------------------------ experiment runner
def test_runner_logs_then_reuses(tmp_path, monkeypatch):
    from src.experiment import ExperimentRunner

    monkeypatch.setenv("SWN_OUTPUT_DIR", str(tmp_path))
    names = ["a", "b", "c"]
    val = pd.DataFrame({"id": [f"c{i}" for i in range(30)], "label_id": np.repeat([0, 1, 2], 10)})
    probs = np.eye(3)[val.label_id] * 0.8 + 0.2 / 3
    calls = []

    def fit_predict():
        calls.append(1)
        return probs, {"note": "x", "history": pd.DataFrame({"epoch": [1, 2], "val_loss": [0.9, 0.5]})}

    runner = ExperimentRunner(val, names, member="M2", notebook="nb.ipynb")
    first = runner.run("A3-R1-01", "first", "why", fit_predict, {"hidden": 128})
    second = runner.run("A3-R1-01", "first", "why", fit_predict, {"hidden": 128})
    assert len(calls) == 1  # the second call reused the logged run
    assert first["metrics"]["accuracy"] == second["metrics"]["accuracy"] == 1.0
    assert list(second["extra"]["history"].columns) == ["epoch", "val_loss"]
    assert (tmp_path / "results" / "runs" / "A3-R1-01_seed42.json").exists()
    assert list(runner.summary("A3").index) == ["A3-R1-01"]
