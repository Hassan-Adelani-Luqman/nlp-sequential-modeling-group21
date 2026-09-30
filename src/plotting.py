"""Shared figure style so every member's plots look like one report.

Colours come from a validated reference palette (light mode, report surface):
  * categorical slots are used in a fixed order and follow the entity, never its
    rank - A1 is always slot 1, A2 slot 2, and so on;
  * magnitudes (confusion matrices, grid heatmaps) use one blue ramp, light -> dark;
  * emphasis means highlighting one series and greying the rest.
Slots 1-3 pass the colour-vision-deficiency checks. Slot 3 (aqua) is below
3:1 contrast, so charts using it carry direct labels.

Usage::

    from src.plotting import apply_style, SERIES, MODEL_COLORS, BLUE_RAMP
    apply_style()
"""
from __future__ import annotations

from pathlib import Path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
DEEMPHASIS = "#c3c2b7"  # grey for non-highlighted series

# Categorical slots, fixed order (blue, orange, aqua, yellow, magenta, green, violet, red).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MODEL_COLORS = {f"A{i + 1}": SERIES[i] for i in range(5)}  # each approach keeps its colour in every figure

# Sequential single-hue ramp (steps 100 -> 700), light surface first so zero recedes.
BLUE_RAMP = [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def blue_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("project_blue", BLUE_RAMP)


def apply_style() -> None:
    """Recessive axes and grid, thin marks, and the system sans font for all later matplotlib figures."""
    import matplotlib as mpl

    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"],
        "axes.prop_cycle": mpl.cycler(color=SERIES),
    })


def progression_chart(df, best_id: str, color: str, title: str, name: str | None = None):
    """Validation log loss and accuracy per run (one bar each); the selected run is highlighted, the rest greyed.

    ``df`` is indexed by exp_id with ``val_logloss`` and ``val_acc`` columns (``ExperimentRunner.summary``).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(df) + 1.2), sharey=True)
    ids = list(df.index)[::-1]
    colors = [color if i == best_id else DEEMPHASIS for i in ids]
    for ax, col, label, fmt in [(axes[0], "val_logloss", "Validation log loss (lower is better)", "{:.3f}"),
                                (axes[1], "val_acc", "Validation accuracy", "{:.1%}")]:
        vals = df.loc[ids, col]
        ax.barh(ids, vals, color=colors, height=0.6)
        for y, v in enumerate(vals):
            ax.text(v, y, "  " + fmt.format(v), va="center", fontsize=8, color=INK_SECONDARY)
        ax.set_xlabel(label)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, vals.max() * 1.18)
    axes[0].set_title(title, loc="left")
    fig.tight_layout()
    if name:
        save_figure(fig, name)
    return fig


def save_figure(fig, name: str, out_dir: Path | str | None = None) -> Path:
    """Save to ``results/figures/<name>.png`` at 300 dpi (print quality for the report)."""
    from src.paths import results_dir

    out_dir = Path(out_dir) if out_dir else results_dir() / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    return path
