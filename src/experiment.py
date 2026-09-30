"""One way to run, log and reuse experiments, for every model notebook.

``ExperimentRunner.run`` trains and evaluates one configuration on the
validation split. If that configuration is already logged it reuses the result,
so notebooks re-execute in seconds on Colab; ``rerun=True`` retrains everything.
Each trained run writes:

* ``results/runs/<exp_id>_seed<seed>.json``      - the experiment-log record (``src.utils.log_experiment``)
* ``results/predictions/<exp_id>_seed<seed>_val.csv`` - per-clip probabilities
* ``results/metrics/<exp_id>_seed<seed>_val.json``    - all metrics, word-pair confusions, params, extra info
* ``results/metrics/<exp_id>_seed<seed>_history.csv`` - per-epoch learning curve (neural models)

Usage::

    runner = ExperimentRunner(val, names, member="M2", notebook="notebooks/03_bilstm.ipynb")
    r = runner.run("A3-R1-01", "first A3 run", "why", fit_predict, params)
    # fit_predict() -> (val_probs, extra); put a per-epoch DataFrame in extra["history"] to save it
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from src.evaluate import KEY_PAIRS, compute_metrics, load_predictions, pair_confusion, save_predictions
from src.paths import results_dir
from src.utils import gpu_name, load_run, log_experiment


class ExperimentRunner:
    def __init__(self, split_df: pd.DataFrame, label_names: list[str], member: str, notebook: str,
                 seed: int = 42, rerun: bool = False, split: str = "val"):
        self.df, self.names, self.member, self.notebook = split_df, label_names, member, notebook
        self.y = split_df["label_id"].to_numpy()
        self.seed, self.rerun, self.split = seed, rerun, split
        self.rows: list[dict] = []

    def _paths(self, exp_id: str):
        res = results_dir()
        stem = f"{exp_id}_seed{self.seed}"
        return (res / "predictions" / f"{stem}_{self.split}.csv", res / "metrics" / f"{stem}_{self.split}.json",
                res / "metrics" / f"{stem}_history.csv")

    def run(self, exp_id: str, change: str, rationale: str, fit_predict, params: dict) -> dict:
        pred_file, metrics_file, history_file = self._paths(exp_id)
        record = load_run(exp_id, self.seed)
        if record and pred_file.exists() and metrics_file.exists() and not self.rerun:
            _, probs = load_predictions(pred_file)
            extra = json.loads(metrics_file.read_text(encoding="utf-8"))["extra"]
            if history_file.exists():
                extra["history"] = pd.read_csv(history_file)
            minutes, status = float(record["train_time_min"]), "reused"
        else:
            start = time.time()
            probs, extra = fit_predict()
            minutes, status = (time.time() - start) / 60, "trained"

        m = compute_metrics(self.y, probs, self.names)
        pairs = {f"{a}/{b}": pair_confusion(self.y, np.asarray(probs).argmax(1), self.names, a, b) for a, b in KEY_PAIRS}
        if status == "trained":
            history = extra.pop("history", None)
            save_predictions(exp_id, self.seed, self.split, self.df["id"], self.y, probs, self.names)
            metrics_file.parent.mkdir(parents=True, exist_ok=True)
            metrics_file.write_text(json.dumps({"params": params, "extra": extra, "metrics": m, "pair_confusion": pairs},
                                               indent=2, default=str), encoding="utf-8")
            if history is not None:
                history.to_csv(history_file, index=False)
                extra["history"] = history
            log_experiment({"exp_id": exp_id, "seed": self.seed, "member": self.member, "config_file": self.notebook,
                            "change_vs_previous": change, "rationale": rationale,
                            "val_macro_f1": m["macro_f1"], "val_logloss": m["log_loss"], "val_acc": m["accuracy"],
                            "train_time_min": minutes, "gpu": gpu_name(),
                            "notes": json.dumps({**params, **{k: v for k, v in extra.items() if k != "history"}},
                                                default=str)},
                           overwrite=self.rerun)
        self.rows.append({"exp_id": exp_id, "change": change, "val_logloss": m["log_loss"], "val_acc": m["accuracy"],
                          "val_macro_f1": m["macro_f1"], "ece": m["ece"], "tisa/sita": pairs["tisa/sita"],
                          "nne/nane": pairs["nne/nane"], "minutes": minutes})
        print(f"{exp_id} [{status}] log loss {m['log_loss']:.3f} · acc {m['accuracy']:.1%} · macro-F1 {m['macro_f1']:.3f}"
              f" · ECE {m['ece']:.3f} · tisa/sita {pairs['tisa/sita']:.1%} · {minutes:.1f} min", flush=True)
        return {"metrics": m, "probs": np.asarray(probs), "extra": extra, "params": params, "exp_id": exp_id}

    def summary(self, prefix: str = "") -> pd.DataFrame:
        rows = [r for r in self.rows if r["exp_id"].startswith(prefix)]
        return pd.DataFrame(rows).drop_duplicates("exp_id", keep="last").set_index("exp_id")

    @staticmethod
    def styled(df: pd.DataFrame):
        return df.style.format({"val_logloss": "{:.3f}", "val_acc": "{:.1%}", "val_macro_f1": "{:.3f}", "ece": "{:.3f}",
                                "tisa/sita": "{:.1%}", "nne/nane": "{:.1%}", "minutes": "{:.1f}"})
