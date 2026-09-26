"""Shared evaluation: metrics, prediction files, significance tests and plots.

All five approaches are scored with ``compute_metrics`` so numbers are
comparable. Each run also saves its per-document probabilities with
``save_predictions``; the final comparison and error analysis (notebook 06)
are rebuilt from those files, never from numbers typed by hand.

Metric choices (see report, Section 4):
  * log loss (co-primary) - the official Zindi metric, rewards calibrated probabilities;
  * macro-F1 (co-primary) and accuracy - the 12 words are balanced, so accuracy is
    not inflated; macro-F1 exposes any word the model systematically fails;
  * one-vs-rest ROC-AUC and expected calibration error (ECE).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.paths import results_dir


# --------------------------------------------------------------------------- metrics
def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """ECE over equal-width confidence bins of the top-class probability."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (confidence > lo) & (confidence <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def compute_metrics(y_true, y_prob, label_names: list[str] | None = None) -> dict:
    """All project metrics from integer labels and an (n_docs, n_classes) probability matrix.

    Returns a flat dict of headline metrics plus ``per_class`` (precision,
    recall, F1 and support for each class).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_classes = y_prob.shape[1]
    labels = list(range(n_classes))
    names = label_names or [str(i) for i in labels]
    y_pred = y_prob.argmax(axis=1)

    # Renormalise defensively (e.g. calibrated SVM outputs may not sum exactly to 1).
    y_prob = np.clip(y_prob, 1e-12, 1.0)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro", labels=labels)
    except ValueError:  # a class absent from y_true (only happens on tiny smoke-test data)
        auc = float("nan")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob, labels=labels)),
        "roc_auc_ovr": float(auc),
        "ece": expected_calibration_error(y_true, y_prob),
        "per_class": {
            name: {"precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)}
            for name, p, r, f, s in zip(names, precision, recall, f1, support)
        },
    }


# --------------------------------------------------------------------------- prediction files
def save_predictions(
    exp_id: str,
    seed: int,
    split: str,
    ids,
    y_true,
    y_prob,
    label_names: list[str],
    out_dir: Path | str | None = None,
) -> Path:
    """Write ``results/predictions/<exp_id>_seed<seed>_<split>.csv``.

    Columns: ``id, y_true, y_pred, prob_<class>...`` (labels are integer ids).
    """
    y_prob = np.asarray(y_prob, dtype=float)
    df = pd.DataFrame({"id": list(ids), "y_true": np.asarray(y_true, dtype=int), "y_pred": y_prob.argmax(axis=1)})
    for i, name in enumerate(label_names):
        df[f"prob_{name}"] = y_prob[:, i]
    out_dir = Path(out_dir) if out_dir else results_dir() / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_id}_seed{seed}_{split}.csv"
    df.to_csv(path, index=False)
    return path


def load_predictions(path: Path | str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return the prediction frame and its probability matrix."""
    df = pd.read_csv(path, dtype={"id": str})
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    return df, df[prob_cols].to_numpy()


# --------------------------------------------------------------------------- uncertainty & significance
def bootstrap_ci(y_true, y_pred, metric=None, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 0):
    """Percentile bootstrap CI for a metric (default macro-F1). Returns (point, low, high)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metric = metric or (lambda t, p: f1_score(t, p, average="macro", zero_division=0))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = [metric(y_true[idx], y_pred[idx]) for idx in (rng.integers(0, n, n) for _ in range(n_resamples))]
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, y_pred)), float(low), float(high)


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """Exact McNemar test on the discordant pairs of two classifiers (Dietterich, 1998)."""
    y_true = np.asarray(y_true)
    a_right = np.asarray(pred_a) == y_true
    b_right = np.asarray(pred_b) == y_true
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    n = only_a + only_b
    p_value = 1.0 if n == 0 else binomtest(only_a, n, 0.5).pvalue
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": float(p_value)}


# --------------------------------------------------------------------------- plots
def plot_confusion_matrix(y_true, y_pred, label_names: list[str], title: str = "", normalize: bool = True, ax=None):
    """Row-normalised confusion matrix (recall per true class on the diagonal)."""
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    values = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(label_names)), label_names, rotation=45, ha="right")
    ax.set_yticks(range(len(label_names)), label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    threshold = values.max() / 2
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            text = f"{values[i, j]:.2f}" if normalize else str(values[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > threshold else "black")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    return ax


def plot_learning_curves(history: pd.DataFrame, title: str = "", axes=None):
    """Plot train/val loss and val macro-F1 per epoch.

    ``history`` needs columns ``epoch, train_loss, val_loss, val_macro_f1``
    (the per-epoch file written by the training loops).
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", label="train")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", label="val")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title=f"{title} loss".strip())
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_f1"], marker="o", color="tab:green")
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", title=f"{title} macro-F1".strip())
    for ax in axes:
        ax.grid(alpha=0.3)
    return axes
