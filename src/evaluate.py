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


# Word pairs that sound alike (EDA 6). tisa/sita are anagrams: same sounds, different order.
KEY_PAIRS = [("tisa", "sita"), ("nne", "nane"), ("tatu", "tano"), ("saba", "sita")]


def pair_confusion(y_true, y_pred, label_names: list[str], a: str, b: str) -> float:
    """Share of clips of word ``a`` or ``b`` that were predicted as the other word of the pair (NaN if absent)."""
    if a not in label_names or b not in label_names:
        return float("nan")
    ia, ib = label_names.index(a), label_names.index(b)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    in_pair = np.isin(y_true, [ia, ib])
    swapped = ((y_true == ia) & (y_pred == ib)) | ((y_true == ib) & (y_pred == ia))
    return float(swapped[in_pair].mean()) if in_pair.any() else float("nan")


# --------------------------------------------------------------------------- calibration
def softmax(scores, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(scores, y_true, bounds: tuple[float, float] = (1e-3, 1e4)) -> float:
    """Temperature T minimising the log loss of ``softmax(scores / T)`` (Guo et al., 2017).

    ``scores`` are per-class logits: network logits, log-probabilities, or HMM log-likelihoods.
    One scalar changes confidence but never the predicted class, so accuracy and F1 are unchanged.
    """
    from scipy.optimize import minimize_scalar

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    rows = np.arange(len(y))

    def nll(log_t):
        return -np.log(np.clip(softmax(scores, np.exp(log_t))[rows, y], 1e-12, None)).mean()

    return float(np.exp(minimize_scalar(nll, bounds=np.log(bounds), method="bounded").x))


def cross_fitted_temperature(scores, y_true, n_splits: int = 5, seed: int = 0):
    """Out-of-fold calibrated probabilities on the split used for fitting the temperature.

    Each fold is calibrated with a temperature fitted on the other folds, so the
    resulting log loss is not optimistically biased. Returns ``(probs, T)``, where
    ``T`` is fitted on all rows (use it on the test split later).
    """
    from sklearn.model_selection import StratifiedKFold

    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y_true, dtype=int)
    probs = np.zeros_like(scores)
    for fit_idx, eval_idx in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(scores, y):
        probs[eval_idx] = softmax(scores[eval_idx], fit_temperature(scores[fit_idx], y[fit_idx]))
    return probs, fit_temperature(scores, y)


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


def paired_bootstrap(y_true, prob_a, prob_b, metric=None, n_resamples: int = 1000, alpha: float = 0.05,
                     seed: int = 0):
    """Percentile bootstrap CI for ``metric(a) - metric(b)`` on the same resampled clips (default: log loss).

    Returns (difference, low, high). Resampling clips jointly keeps the pairing, so the interval reflects
    how much the gap between two models depends on which clips happen to be in the test split.
    """
    y_true = np.asarray(y_true)
    prob_a, prob_b = np.asarray(prob_a), np.asarray(prob_b)
    labels = list(range(prob_a.shape[1]))
    metric = metric or (lambda t, p: log_loss(t, np.clip(p, 1e-12, 1), labels=labels))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        diffs.append(metric(y_true[idx], prob_a[idx]) - metric(y_true[idx], prob_b[idx]))
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(metric(y_true, prob_a) - metric(y_true, prob_b)), float(low), float(high)


def holm_correction(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (step-down), returned in the input order (Holm, 1979)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, min(1.0, (m - rank) * p[i]))   # monotone: never below an earlier adjusted value
        adjusted[i] = running
    return adjusted


def reliability_curve(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Top-class confidence against accuracy in equal-width bins (non-empty bins only), for reliability diagrams."""
    y_true, y_prob = np.asarray(y_true), np.asarray(y_prob)
    confidence = y_prob.max(axis=1)
    correct = (y_prob.argmax(axis=1) == y_true).astype(float)
    bins = np.clip(np.digitize(confidence, np.linspace(0, 1, n_bins + 1)[1:-1], right=True), 0, n_bins - 1)
    rows = [{"bin": b, "confidence": confidence[bins == b].mean(), "accuracy": correct[bins == b].mean(),
             "count": int((bins == b).sum())} for b in range(n_bins) if (bins == b).any()]
    return pd.DataFrame(rows)


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
def plot_confusion_matrix(y_true, y_pred, label_names: list[str], title: str = "", normalize: bool = True,
                          ax=None, annotate_min: float = 0.05, colorbar: bool = True):
    """Row-normalised confusion matrix (recall per true class on the diagonal).

    Only cells at or above ``annotate_min`` are labelled (the diagonal and the
    real confusions), so a 12x12 matrix stays readable. The colour carries the rest.
    """
    import matplotlib.pyplot as plt

    from src.plotting import INK, SURFACE, blue_cmap

    cm = confusion_matrix(y_true, y_pred, labels=range(len(label_names)))
    values = cm / cm.sum(axis=1, keepdims=True).clip(min=1) if normalize else cm
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(values, cmap=blue_cmap(), vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(label_names)), label_names, rotation=45, ha="right")
    ax.set_yticks(range(len(label_names)), label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.grid(False)
    threshold = values.max() * 0.55
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            if values[i, j] >= annotate_min:
                text = f"{values[i, j]:.2f}".lstrip("0") if normalize else str(values[i, j])
                ax.text(j, i, text, ha="center", va="center", fontsize=6.5,
                        color=SURFACE if values[i, j] > threshold else INK)
    if colorbar:
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02).outline.set_visible(False)
    return ax


def plot_learning_curves(history: pd.DataFrame, title: str = "", axes=None):
    """Plot train/val loss and val macro-F1 per epoch.

    ``history`` needs columns ``epoch, train_loss, val_loss, val_macro_f1``
    (the per-epoch file written by the training loops).
    """
    import matplotlib.pyplot as plt

    from src.plotting import SERIES

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    # The training objective includes label smoothing and is computed on augmented clips, so it sits above the
    # validation log loss by construction; the labels say so, so the curves are not misread as a bug.
    axes[0].plot(history["epoch"], history["train_loss"], marker="o", color=SERIES[0],
                 label="train objective (label-smoothed, augmented)")
    axes[0].plot(history["epoch"], history["val_loss"], marker="o", color=SERIES[1], label="validation log loss")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title=f"{title} loss".strip())
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_f1"], marker="o", color=SERIES[0])
    axes[1].set(xlabel="Epoch", ylabel="Validation macro-F1", title=f"{title} macro-F1".strip())
    return axes
