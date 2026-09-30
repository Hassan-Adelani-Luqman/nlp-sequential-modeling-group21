"""Shared PyTorch training loop for the frame-sequence models (A3 BiLSTM, A4 TC-ResNet).

* Inputs: cached features (N, frames, channels) in their natural units (dB for log-mel).
  Augmentation (``src.augment``) is applied to training clips on the fly, then every
  clip is standardised with statistics fitted on the training split only.
* Optimisation: AdamW, one-cycle (or cosine) learning-rate schedule, gradient
  clipping, and cross-entropy with label smoothing. Mixed precision is used on CUDA.
* Model selection: early stopping on validation **log loss** (plain cross-entropy
  without label smoothing, i.e. the official metric). The best epoch's weights are restored.
* Reproducibility: pass a *function* that builds the model; the loop seeds every
  generator before calling it, so weight initialisation and batch order repeat exactly.

Typical use::

    result = train_model(lambda: BiLSTMAttention(n_features=64), X_tr, y_tr, X_va, y_va, TrainConfig())
    result.val_logits, result.history, result.best_epoch
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.features import Standardizer
from src.utils import get_device, set_seed


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3               # peak learning rate of the one-cycle schedule
    weight_decay: float = 1e-2
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    patience: int = 5              # epochs without a validation log-loss improvement before stopping
    min_delta: float = 1e-4
    scheduler: str = "onecycle"    # "onecycle" | "cosine"
    amp: bool = True               # mixed precision, only used on CUDA
    num_workers: int = 0           # 0 is safest in notebooks on Windows; 2 on Kaggle/Colab
    seed: int = 42


@dataclass
class TrainResult:
    model: nn.Module
    history: pd.DataFrame          # epoch, train_loss, val_loss, val_acc, val_macro_f1, lr, seconds
    best_epoch: int
    val_logits: np.ndarray         # logits of the restored best model on the validation split
    scaler: Standardizer


class FeatureDataset(Dataset):
    """In-memory feature clips; augments (optional) and then standardises each clip."""

    def __init__(self, X: np.ndarray, y: np.ndarray, scaler: Standardizer, augment: Callable | None = None):
        self.X, self.y, self.scaler, self.augment = X, np.asarray(y, dtype=np.int64), scaler, augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        if self.augment is not None:
            x = self.augment(x)
        return torch.from_numpy(self.scaler.transform(np.asarray(x, dtype=np.float32))), int(self.y[i])


@torch.no_grad()
def predict_logits(model: nn.Module, X: np.ndarray, scaler: Standardizer, batch_size: int = 256,
                   device: torch.device | None = None) -> np.ndarray:
    """Logits for raw (unstandardised) features, in evaluation mode."""
    device = device or next(model.parameters()).device
    model.eval()
    use_amp = device.type == "cuda"
    out = []
    for start in range(0, len(X), batch_size):
        xb = torch.from_numpy(scaler.transform(X[start: start + batch_size])).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out.append(model(xb).float().cpu())
    return torch.cat(out).numpy()


def load_checkpoint(path: Path | str, model_fn: Callable[[], nn.Module],
                    device: torch.device | str = "cpu") -> tuple[nn.Module, Standardizer, dict]:
    """Rebuild a trained model and its training-split scaler from a checkpoint written by ``train_model``.

    Returns ``(model in eval mode, scaler, raw checkpoint dict)``. ``weights_only=False`` is needed
    because our checkpoints also store the scaler's numpy arrays; only load checkpoints you created.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_fn()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    scaler = Standardizer()
    scaler.mean_, scaler.std_ = ckpt["scaler_mean"], ckpt["scaler_std"]
    return model, scaler, ckpt


def _make_scheduler(optimizer, cfg: TrainConfig, total_steps: int):
    if cfg.scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.1)
    if cfg.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    raise ValueError("scheduler must be 'onecycle' or 'cosine'")


def train_model(model_fn: Callable[[], nn.Module], X_tr: np.ndarray, y_tr, X_va: np.ndarray, y_va,
                cfg: TrainConfig | None = None, augment: Callable | None = None,
                device: torch.device | None = None, checkpoint: Path | str | None = None,
                verbose: bool = True) -> TrainResult:
    """Train with early stopping on validation log loss; returns the best epoch's model and logits."""
    cfg = cfg or TrainConfig()
    device = device or get_device()
    set_seed(cfg.seed)
    model = model_fn().to(device)

    scaler = Standardizer().fit(X_tr)
    y_tr, y_va = np.array(y_tr, dtype=np.int64), np.array(y_va, dtype=np.int64)  # writable copies (pandas views are read-only)
    loader = DataLoader(FeatureDataset(X_tr, y_tr, scaler, augment), batch_size=cfg.batch_size, shuffle=True,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(cfg.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = _make_scheduler(optimizer, cfg, cfg.epochs * len(loader))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    use_amp = cfg.amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    y_va_t = torch.from_numpy(y_va)

    history, best_loss, best_epoch, best_state, best_logits, stale = [], np.inf, 0, None, None, 0
    for epoch in range(1, cfg.epochs + 1):
        start = time.time()
        model.train()
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = criterion(model(xb), yb)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            scheduler.step()
            total += loss.item() * len(yb)
            seen += len(yb)

        logits = predict_logits(model, X_va, scaler, device=device)
        val_loss = nn.functional.cross_entropy(torch.from_numpy(logits), y_va_t).item()  # no smoothing: = log loss
        pred = logits.argmax(1)
        history.append({"epoch": epoch, "train_loss": total / seen, "val_loss": val_loss,
                        "val_acc": float((pred == y_va).mean()),
                        "val_macro_f1": float(f1_score(y_va, pred, average="macro")),
                        "lr": optimizer.param_groups[0]["lr"], "seconds": time.time() - start})
        improved = val_loss < best_loss - cfg.min_delta
        if improved:
            best_loss, best_epoch, best_logits, stale = val_loss, epoch, logits, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if verbose:
            h = history[-1]
            print(f"  epoch {epoch:2d}  train {h['train_loss']:.3f}  val loss {val_loss:.3f}  acc {h['val_acc']:.1%}"
                  f"  {h['seconds']:.0f}s{'  *' if improved else ''}", flush=True)
        if stale >= cfg.patience:
            break

    model.load_state_dict(best_state)
    if checkpoint is not None:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_state, "config": asdict(cfg), "best_epoch": best_epoch,
                    "scaler_mean": scaler.mean_, "scaler_std": scaler.std_}, checkpoint)
    return TrainResult(model, pd.DataFrame(history), best_epoch, best_logits, scaler)
