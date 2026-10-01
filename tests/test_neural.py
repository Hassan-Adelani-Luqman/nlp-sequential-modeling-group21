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


def test_global_fill_reproduces_logged_behaviour_and_channel_fill_respects_scales():
    """The default must not change (logged runs depend on it); fill='channel' must keep MFCC channels in range."""
    rng = np.random.default_rng(0)
    mfcc_like = np.hstack([rng.normal(-600, 20, (151, 1)), rng.normal(0, 1, (151, 119))]).astype(np.float32)
    kw = dict(time_masks=2, freq_masks=2, shift_frames=10, stretch=(0.9, 1.1), p_stretch=1.0, noise_snr_db=None)
    legacy = LogMelAugment(seed=4, **kw)(mfcc_like)
    assert legacy[:, 1:].min() < -100   # global fill writes c0's scale into the delta channels (the bug)
    fixed = LogMelAugment(fill="channel", seed=4, **kw)(mfcc_like)
    assert fixed[:, 1:].min() > mfcc_like[:, 1:].min() - 1e-3   # every channel stays within its own range
    assert fixed[:, 0].min() >= mfcc_like[:, 0].min() - 1e-3
    np.testing.assert_array_equal(legacy, LogMelAugment(seed=4, **kw)(mfcc_like))   # default is unchanged and seeded


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


@pytest.mark.parametrize("blocks", [3, 6])
def test_tcresnet_shapes_cam_and_receptive_field(blocks):
    from src.models.tcresnet import TCResNet

    model = TCResNet(n_features=64, blocks=blocks).eval()
    x = torch.randn(2, 151, 64)
    logits, cam = model(x, return_cam=True)
    assert logits.shape == (2, 12) and cam.shape[:2] == (2, 12)
    assert torch.allclose(cam.mean(-1), logits, atol=1e-5)  # CAM averaged over time recovers the logits exactly
    assert model.receptive_field_frames() > 100             # context spans most of a 1-1.5 s window
    assert 20_000 < count_parameters(model) < 1_000_000


def test_tcresnet_learns_temporal_order():
    """Mirror-image sequences: a temporal convolution with enough context must separate them."""
    from src.models.tcresnet import TCResNet

    X, y = mirror_task(n_frames=40)
    cfg = TrainConfig(epochs=20, batch_size=16, lr=5e-3, patience=20, seed=0)
    r = train_model(lambda: TCResNet(n_features=4, n_classes=2, width=1.0, dropout=0.0), X[:60], y[:60], X[60:], y[60:],
                    cfg, verbose=False)
    assert (r.val_logits.argmax(1) == y[60:]).mean() >= 0.95


# ------------------------------------------------------------------ A5: wav2vec 2.0 / XLS-R on raw waveforms
def fake_wave(seed=0):
    """1.5 s window with a 1 s 'word' (noise burst) in the middle, like a centred energy-window crop."""
    wav = np.zeros(24000, np.float32)
    wav[4000:20000] = np.random.default_rng(seed).normal(0, 0.3, 16000)
    return wav


def test_wave_augment_keeps_length_is_seeded_and_hits_the_snr():
    from src.augment import WaveAugment

    wav = fake_wave()
    out = WaveAugment(seed=1)(wav)
    assert out.shape == wav.shape and out.dtype == np.float32
    assert np.array_equal(WaveAugment(seed=3)(wav), WaveAugment(seed=3)(wav))   # seeded -> reproducible
    slow = WaveAugment(speed=(0.9, 0.9), p_speed=1.0, shift_ms=0, noise_snr_db=None)(wav)
    assert (slow != 0).sum() > (wav != 0).sum()                                 # slower speech: the word lasts longer
    shifted = WaveAugment(speed=None, noise_snr_db=None, shift_ms=100, seed=2)(wav)
    assert np.abs(shifted).sum() > 0 and not np.array_equal(shifted, wav)     # shifted with silence, not wrapped
    noisy = WaveAugment(speed=None, shift_ms=0, noise_snr_db=(20, 20), p_noise=1.0, seed=0)(wav)
    # The reference is the louder half of the frames, which here all belong to the word.
    snr_vs_word = 10 * np.log10((wav[4000:20000] ** 2).mean() / ((noisy - wav) ** 2).mean())
    assert 19 < snr_vs_word < 21


@pytest.fixture
def tiny_wav2vec2():
    pytest.importorskip("transformers")
    from src.models.wav2vec2 import tiny_config

    return tiny_config()


def test_wav2vec2_classifier_freezing_groups_and_normalisation(tiny_wav2vec2):
    from src.models.wav2vec2 import Wav2Vec2Classifier

    model = Wav2Vec2Classifier(config=tiny_wav2vec2).eval()
    x = torch.randn(3, 16000)
    with torch.no_grad():
        assert model(x).shape == (3, 12)
        assert torch.allclose(model(x), model(4 * x + 1), atol=1e-4)            # each clip is normalised in the model
    assert not any(p.requires_grad for p in model.net.wav2vec2.feature_extractor.parameters())
    assert [g["lr"] for g in model.optimizer_groups(3e-5)] == [3e-5, 1e-3]     # encoder, then the new head
    probe = Wav2Vec2Classifier(config=tiny_wav2vec2, freeze="encoder")
    assert [g["lr"] for g in probe.optimizer_groups(3e-5)] == [1e-3]           # frozen encoder: only the head trains
    weights = Wav2Vec2Classifier(config=tiny_wav2vec2, weighted_layer_sum=True).layer_weights()
    assert len(weights) == 3 and np.isclose(sum(weights), 1.0)                 # input embedding + 2 layers
    assert model.layer_weights() is None


def test_layer_weights_start_equal_after_loading_a_pretrained_checkpoint(tiny_wav2vec2, tmp_path):
    """A checkpoint has no layer weights, and from_pretrained left them as arbitrary memory (first A5-R2-06 run)."""
    from transformers import Wav2Vec2Model
    from src.models.wav2vec2 import Wav2Vec2Classifier

    Wav2Vec2Model(tiny_wav2vec2).save_pretrained(tmp_path)          # an encoder-only checkpoint, like XLS-R's
    model = Wav2Vec2Classifier(backbone=str(tmp_path), weighted_layer_sum=True)
    np.testing.assert_array_equal(model.net.layer_weights.detach().numpy(), np.full(3, 1 / 3, np.float32))


def test_train_model_on_raw_waveforms(tiny_wav2vec2, tmp_path):
    from src.models.wav2vec2 import Wav2Vec2Classifier
    from src.train_torch import load_checkpoint

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], 12)
    t = np.arange(8000) / 16000
    X = np.stack([rng.uniform(0.2, 1.0) * np.sin(2 * np.pi * (300 if c == 0 else 1200) * t) for c in y])
    X = (X + 0.05 * rng.standard_normal(X.shape)).astype(np.float32)          # two "words": a low and a high tone
    cfg = TrainConfig(epochs=2, batch_size=8, lr=1e-3, patience=2, seed=0, standardize=False, eval_batch_size=5)
    make = lambda: Wav2Vec2Classifier(n_classes=2, config=tiny_wav2vec2, head_lr=1e-2)
    r = train_model(make, X, y, X, y, cfg, verbose=False, checkpoint=tmp_path / "a5.pt")
    assert r.scaler.mean_ == 0 and r.scaler.std_ == 1                          # waveforms are not standardised
    assert r.val_logits.shape == (24, 2) and len(r.history) == 2
    model, scaler, _ = load_checkpoint(tmp_path / "a5.pt", make)
    np.testing.assert_allclose(predict_logits(model, X, scaler), r.val_logits, atol=1e-4)


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
    with pytest.raises(ValueError, match="different parameters"):   # never reuse a run for another config
        runner.run("A3-R1-01", "first", "why", fit_predict, {"hidden": 64})
