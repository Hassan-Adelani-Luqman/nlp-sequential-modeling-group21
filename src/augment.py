"""On-the-fly augmentation of log-mel spectrograms, and of raw waveforms for A5 (training split only).

Augmentation works on the cached log-mel features in dB, so GPU sessions never
decode audio. It uses no external data (the competition rules allow only the
provided audio). Every transform is synthetic:

* SpecAugment time and frequency masking (Park et al., 2019);
* time shift of up to +-``shift_frames`` (10 ms each), padding with silence rather than wrapping;
* tempo perturbation: frames resampled by a factor in ``stretch`` (0.9-1.1x speed);
* noise injection at a random SNR (dB) in the mel *power* domain. This
  approximates additive white noise, which with area-normalised mel filters
  adds roughly equal power to every band.

Usage::

    aug = LogMelAugment(seed=42)          # all transforms
    aug = LogMelAugment.specaugment_only()
    x_aug = aug(x)                        # x: (frames, mels) log-mel in dB
"""
from __future__ import annotations

import numpy as np

TOP_DB = 80.0  # dynamic range used when the log-mel features were computed (src.features.log_mel)


class LogMelAugment:
    """``fill`` sets what padding and masks are filled with.

    * ``"global"`` (default): the clip's overall minimum (padding) and mean (masks). Correct for log-mel,
      where every band shares one dB scale. Kept as the default so logged runs (e.g. A3) reproduce exactly.
    * ``"channel"``: each channel's own minimum and mean. **Required for MFCC-type inputs.** Their channels
      have very different scales (c0 is often near -600, while the deltas sit near 0), so a global value
      written into every channel creates huge outliers. In A4 this collapsed training to chance (run A4-R2-03).
    """

    def __init__(self, time_masks: int = 2, time_width: int = 20, freq_masks: int = 2, freq_width: int = 8,
                 shift_frames: int = 10, stretch: tuple[float, float] | None = (0.9, 1.1), p_stretch: float = 0.5,
                 noise_snr_db: tuple[float, float] | None = (10.0, 30.0), p_noise: float = 0.5,
                 fill: str = "global", seed: int | None = None):
        if fill not in ("global", "channel"):
            raise ValueError("fill must be 'global' or 'channel'")
        self.time_masks, self.time_width = time_masks, time_width
        self.freq_masks, self.freq_width = freq_masks, freq_width
        self.shift_frames = shift_frames
        self.stretch, self.p_stretch = stretch, p_stretch
        self.noise_snr_db, self.p_noise = noise_snr_db, p_noise
        self.fill = fill
        self.rng = np.random.default_rng(seed)

    @classmethod
    def specaugment_only(cls, seed: int | None = None, fill: str = "global") -> "LogMelAugment":
        return cls(shift_frames=0, stretch=None, noise_snr_db=None, fill=fill, seed=seed)

    def describe(self) -> dict:
        """Settings, for the experiment log."""
        return {"time_masks": self.time_masks, "time_width": self.time_width, "freq_masks": self.freq_masks,
                "freq_width": self.freq_width, "shift_frames": self.shift_frames, "stretch": self.stretch,
                "noise_snr_db": self.noise_snr_db, "fill": self.fill}

    def _floor(self, x: np.ndarray):
        return x.min(axis=0) if self.fill == "channel" else x.min()

    def _centre(self, x: np.ndarray):
        return x.mean(axis=0) if self.fill == "channel" else x.mean()

    # ------------------------------------------------------------------ transforms
    def _stretch(self, x: np.ndarray) -> np.ndarray:
        """Resample frames by a speed factor, then centre-crop/pad back to the original length."""
        rate = self.rng.uniform(*self.stretch)
        n = x.shape[0]
        new_n = max(2, int(round(n / rate)))
        src = np.linspace(0, n - 1, new_n)
        stretched = np.stack([np.interp(src, np.arange(n), x[:, m]) for m in range(x.shape[1])], axis=1)
        if new_n >= n:
            start = (new_n - n) // 2
            return stretched[start: start + n]
        pad = n - new_n
        out = np.empty_like(x)
        out[:] = self._floor(x)                          # silence-like padding (per channel if fill="channel")
        out[pad // 2: pad // 2 + new_n] = stretched
        return out

    def _shift(self, x: np.ndarray) -> np.ndarray:
        k = int(self.rng.integers(-self.shift_frames, self.shift_frames + 1))
        if k == 0:
            return x
        out = np.empty_like(x)
        out[:] = self._floor(x)                          # silence, not wrap-around
        if k > 0:
            out[k:] = x[:-k]
        else:
            out[:k] = x[-k:]
        return out

    def _noise(self, x: np.ndarray) -> np.ndarray:
        """Add noise at a target SNR relative to the speech frames' mean power."""
        power = 10.0 ** (x / 10.0)
        frame_power = power.mean(axis=1)
        speech = frame_power >= np.median(frame_power)  # louder half of the frames ~ the word
        signal = power[speech].mean()
        snr = self.rng.uniform(*self.noise_snr_db)
        noise = signal / (10.0 ** (snr / 10.0)) * self.rng.exponential(1.0, size=x.shape)  # chi-square-like per bin
        out = 10.0 * np.log10(power + noise)
        return np.maximum(out, out.max() - TOP_DB)

    def _mask(self, x: np.ndarray) -> np.ndarray:
        x = x.copy()
        fill = self._centre(x)                           # scalar, or one value per channel
        n_frames, n_mels = x.shape
        for _ in range(self.time_masks):
            w = int(self.rng.integers(0, self.time_width + 1))
            if w:
                t0 = int(self.rng.integers(0, max(1, n_frames - w)))
                x[t0: t0 + w, :] = fill
        for _ in range(self.freq_masks):
            w = int(self.rng.integers(0, self.freq_width + 1))
            if w:
                f0 = int(self.rng.integers(0, max(1, n_mels - w)))
                x[:, f0: f0 + w] = fill if np.ndim(fill) == 0 else fill[f0: f0 + w]
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.stretch and self.rng.random() < self.p_stretch:
            x = self._stretch(x)
        if self.shift_frames:
            x = self._shift(x)
        if self.noise_snr_db and self.rng.random() < self.p_noise:
            x = self._noise(x)
        return self._mask(x).astype(np.float32)


class WaveAugment:
    """On-the-fly augmentation of raw waveforms (A5), the counterpart of ``LogMelAugment``'s shift, tempo and noise.

    * time shift of up to +-``shift_ms``, padding with silence (zeros) rather than wrapping;
    * speed perturbation: the clip is resampled (linear interpolation) by a factor in ``speed``. Unlike the
      log-mel tempo change, this changes the pitch as well as the tempo;
    * white noise at a random SNR (dB) relative to the louder half of the clip's 10 ms frames.

    There is no masking here: wav2vec 2.0 masks spans of its own latent frames while it is fine-tuned.
    """

    def __init__(self, shift_ms: int = 100, speed: tuple[float, float] | None = (0.9, 1.1), p_speed: float = 0.5,
                 noise_snr_db: tuple[float, float] | None = (10.0, 30.0), p_noise: float = 0.5, sr: int = 16000,
                 seed: int | None = None):
        self.shift_ms, self.speed, self.p_speed = shift_ms, speed, p_speed
        self.noise_snr_db, self.p_noise, self.sr = noise_snr_db, p_noise, sr
        self.rng = np.random.default_rng(seed)

    def describe(self) -> dict:
        """Settings, for the experiment log."""
        return {"shift_ms": self.shift_ms, "speed": self.speed, "noise_snr_db": self.noise_snr_db}

    def _speed(self, x: np.ndarray) -> np.ndarray:
        """Resample by a speed factor, then centre-crop or pad with silence back to the original length."""
        rate = self.rng.uniform(*self.speed)
        n = len(x)
        new_n = max(2, int(round(n / rate)))
        fast = np.interp(np.linspace(0, n - 1, new_n), np.arange(n), x)
        if new_n >= n:
            start = (new_n - n) // 2
            return fast[start: start + n]
        out = np.zeros_like(x)
        pad = (n - new_n) // 2
        out[pad: pad + new_n] = fast
        return out

    def _shift(self, x: np.ndarray) -> np.ndarray:
        k = int(self.rng.integers(-self.shift_ms, self.shift_ms + 1)) * self.sr // 1000   # ms -> samples
        if k == 0:
            return x
        out = np.zeros_like(x)
        if k > 0:
            out[k:] = x[:-k]
        else:
            out[:k] = x[-k:]
        return out

    def _noise(self, x: np.ndarray) -> np.ndarray:
        hop = self.sr // 100
        frames = x[: len(x) // hop * hop].reshape(-1, hop)
        power = (frames ** 2).mean(1)
        signal = power[power >= np.median(power)].mean()           # louder half of the frames ~ the word
        snr = self.rng.uniform(*self.noise_snr_db)
        return x + self.rng.normal(0.0, np.sqrt(signal / 10.0 ** (snr / 10.0)), size=x.shape)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        if self.speed and self.rng.random() < self.p_speed:
            x = self._speed(x)
        if self.shift_ms:
            x = self._shift(x)
        if self.noise_snr_db and self.rng.random() < self.p_noise:
            x = self._noise(x)
        return x.astype(np.float32)
