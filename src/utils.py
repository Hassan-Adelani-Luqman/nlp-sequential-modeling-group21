"""Seeding, config loading and experiment tracking.

Experiment tracking: every training run calls ``log_experiment(...)`` once. Each
run is stored as its own JSON file in ``results/runs/`` (one file per run means
four people can log runs in parallel without git merge conflicts), and
``build_experiment_table()`` collects them into ``results/experiments.csv``,
which is the table used in the report.

Experiment ids follow ``A{approach}-R{round}-{nn}``, e.g. ``A3-R2-04``:
approach 1-5 (A1 MFCC-stats SVM, A2 GMM-HMM, A3 BiLSTM, A4 TC-ResNet, A5 XLS-R),
round 0-4 (R0 baselines, R1 defaults, R2 tuning, R3 final seeds, R4 data-size).
"""
from __future__ import annotations

import json
import os
import random
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.paths import results_dir

EXP_ID_RE = re.compile(r"^A[1-5]-R[0-4]-\d{2}$")

EXPERIMENT_COLUMNS = [
    "exp_id", "date", "member", "approach", "round", "config_file", "seed",
    "change_vs_previous", "rationale",
    "val_macro_f1", "val_logloss", "val_acc",
    "test_macro_f1", "test_logloss", "test_acc",
    "train_time_min", "gpu", "notes",
]


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gpu_name() -> str:
    try:
        import torch

        return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def load_config(path: Path | str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_experiment(row: dict, runs_dir: Path | str | None = None, overwrite: bool = False) -> Path:
    """Record one run as ``results/runs/<exp_id>_seed<seed>.json``.

    ``row`` must contain ``exp_id`` and ``seed`` and may only use keys from
    ``EXPERIMENT_COLUMNS``; missing keys are left blank. Float metrics are rounded
    to 4 decimals. ``approach`` and ``round`` are filled from the id if absent.
    """
    unknown = set(row) - set(EXPERIMENT_COLUMNS)
    if unknown:
        raise KeyError(f"Unknown experiment fields {sorted(unknown)}; allowed: {EXPERIMENT_COLUMNS}")
    exp_id = row.get("exp_id", "")
    if not EXP_ID_RE.match(exp_id):
        raise ValueError(f"exp_id '{exp_id}' must look like 'A3-R2-04'")
    if "seed" not in row:
        raise KeyError("row must include 'seed'")

    record = {col: row.get(col, "") for col in EXPERIMENT_COLUMNS}
    record["approach"] = record["approach"] or exp_id.split("-")[0]
    record["round"] = record["round"] or exp_id.split("-")[1]
    record["date"] = record["date"] or datetime.now().strftime("%Y-%m-%d %H:%M")
    record = {k: round(v, 4) if isinstance(v, float) else v for k, v in record.items()}

    path = run_path(exp_id, record["seed"], runs_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path.name} already logged; use a new exp_id or overwrite=True")
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def record_test_metrics(exp_id: str, seed: int, metrics: dict, runs_dir: Path | str | None = None) -> Path:
    """Fill the ``test_*`` columns of an already logged run (Phase 7: each final model is scored on test once)."""
    path = run_path(exp_id, seed, runs_dir)
    if not path.exists():
        raise FileNotFoundError(f"{path.name} is not logged")
    record = json.loads(path.read_text(encoding="utf-8"))
    record.update({"test_macro_f1": round(float(metrics["macro_f1"]), 4),
                   "test_logloss": round(float(metrics["log_loss"]), 4),
                   "test_acc": round(float(metrics["accuracy"]), 4)})
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def run_path(exp_id: str, seed: int, runs_dir: Path | str | None = None) -> Path:
    runs_dir = Path(runs_dir) if runs_dir else results_dir() / "runs"
    return runs_dir / f"{exp_id}_seed{seed}.json"


def load_run(exp_id: str, seed: int, runs_dir: Path | str | None = None) -> dict | None:
    """The logged record of a run, or None if it has not been run yet.

    Notebooks use this to reuse finished runs instead of retraining: this keeps
    them fast to re-execute on Colab, while ``RERUN = True`` still reproduces everything.
    """
    path = run_path(exp_id, seed, runs_dir)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def build_experiment_table(runs_dir: Path | str | None = None, out_csv: Path | str | None = None) -> pd.DataFrame:
    """Collect all run JSONs into one table sorted by id and seed, and save it as CSV."""
    runs_dir = Path(runs_dir) if runs_dir else results_dir() / "runs"
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(runs_dir.glob("*.json"))]
    table = pd.DataFrame(records, columns=EXPERIMENT_COLUMNS)
    if not table.empty:
        table = table.sort_values(["exp_id", "seed"]).reset_index(drop=True)
    out_csv = Path(out_csv) if out_csv else runs_dir.parent / "experiments.csv"
    table.to_csv(out_csv, index=False)
    return table
