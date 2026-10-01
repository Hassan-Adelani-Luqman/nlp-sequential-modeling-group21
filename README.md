# Sequential Models for Swahili Spoken-Word Recognition — Group 21

> *How effectively can sequential modelling approaches recognise spoken Swahili words,
> and what evidence supports the strengths and limitations of each approach?*

Formative Assignment 2 (NLP & Language Technologies). We compare five approaches on the
[Zindi Swahili Audio Classification](https://zindi.world/competitions/swahili-audio-classification/data)
dataset:
- 4,200 labelled clips, recorded by crowd workers in Kenya.
- 12 balanced classes: the digits *moja … kumi* (one to ten), plus *ndio* (yes) and *hapana* (no).
- The official metric is **log loss**.

A spoken word is a sequence of acoustic frames, and its identity depends on the *order* of
its sounds. *tisa* (nine) and *sita* (six) contain the same sounds in a different order.

| ID | Approach | Uses temporal order? | Owner |
|---|---|---|---|
| A1 | MFCC summary statistics + logistic regression | no (bag of frames; its Δ features keep ~90 ms of local change) | M1 |
| A2 | GMM-HMM per word | yes (Markov states) | M1 |
| A3 | BiLSTM + attention on log-mel frames | yes (recurrent) | M2 |
| A4 | TC-ResNet (1D convolutions over time) | yes (convolutional) | M3 |
| A5 | Fine-tuned XLS-R-300M on raw audio (vs wav2vec2-base) | yes (self-attention, pretrained) | M4 |
## Quick start

### Kaggle (main compute)
1. New notebook → **Settings**: Accelerator **GPU T4 x2** (or P100), Internet **On**.
2. **Add Input** → the private dataset `group21-swahili-audio` (see [data/README.md](data/README.md)).
3. First cell:
   ```python
   !git clone https://github.com/Hassan-Adelani-Luqman/nlp-sequential-modeling-group21.git
   %cd nlp-sequential-modeling-group21
   !pip install -q -r requirements.txt
   ```
4. Run [notebooks/00_setup_check.ipynb](notebooks/00_setup_check.ipynb) once to confirm your split hash matches.

### Colab
Same as Kaggle, but get the data first. Either mount Drive with the data in `MyDrive/group21-swahili-audio/`,
or use `src.paths.download_from_kaggle()` (see [data/README.md](data/README.md)).

### Local
```bash
python -m venv .venv && .venv/Scripts/activate      # Windows (use bin/activate on Linux/macOS)
pip install torch torchaudio                        # see pytorch.org for the right build
pip install -r requirements.txt
# put Train.csv, Test.csv, SampleSubmission.csv and Swahili_words/ in data/raw/
python -m src.data                                  # verify the frozen split
pytest -q                                           # smoke tests
```

### Building the report
The report reads its figures and the main results table straight from `results/`, so it always shows the logged
numbers. Either upload `report/` and `results/` to Overleaf (pdfLaTeX, main file `report/main.tex`), or build
locally with [Tectonic](https://tectonic-typesetting.github.io/):
```bash
cd report && tectonic main.tex      # writes report/main.pdf
```

## Using the shared code
```python
from src.data import load_split, label_names
from src.features import PRESETS, cached_features, Standardizer
from src.evaluate import compute_metrics, save_predictions
from src.utils import set_seed, log_experiment

set_seed(42)
train, val = load_split("train"), load_split("val")          # id, path, label, gloss, label_id
X_tr = cached_features(train, **PRESETS["logmel"])             # (2940, 151, 64), centred 1.5 s energy window
X_va = cached_features(val, **PRESETS["logmel"])
scaler = Standardizer().fit(X_tr)                              # fit on train only
X_tr, X_va = scaler.transform(X_tr), scaler.transform(X_va)
# ... train a model, get val probabilities `prob` (630 x 12) ...
metrics = compute_metrics(val.label_id, prob, label_names())
save_predictions("A3-R1-01", 42, "val", val.id, val.label_id, prob, label_names())
log_experiment({"exp_id": "A3-R1-01", "seed": 42, "member": "M2",
                "val_logloss": metrics["log_loss"], "val_macro_f1": metrics["macro_f1"],
                "val_acc": metrics["accuracy"], "change_vs_previous": "first run",
                "rationale": "default config from Plan.md"})
```

**Shared presets.** Use these so everyone trains on identical features:

| Preset | Output | Used by |
|---|---|---|
| `PRESETS["logmel"]` | (N, 151, 64) | A3, A4 |
| `PRESETS["mfcc_stats"]` | (N, 480) | A1 |
| `PRESETS["mfcc13_trim"]` | list of (T_i, 39) | A2 |

Build them all once with `python -m src.features`. On Kaggle, attach the group's
**features dataset** and `cached_features` loads from it instead of re-extracting
(see [data/README.md](data/README.md)).

Feature options (`src/features.py`):

| Feature | Output | Used by |
|---|---|---|
| `"logmel"` | (T, 64) | A3, A4 |
| `"mfcc"` | (T, 120); `n_mfcc=13` gives 39 | A2, ablations |
| `"mfcc_stats"` | (480,) | A1 |
| `"wave"` | raw waveform | A5 |

Crop options:

| `crop` | Behaviour |
|---|---|
| `"energy"` (default) | Fixed window over the loudest part of the clip, with the word centred |
| `"trim"` + `seconds=None` | Silence-trimmed, variable length; returns a list, for the HMM |
| `"none"` | Only padding or cropping to `seconds` |

## Repository layout
```
configs/        base.yaml (shared settings) + one YAML per experiment
data/           README with download steps; splits/ = frozen split (committed); raw/ = git-ignored audio
notebooks/      00_setup_check, 01_eda, 02_baselines, 03_bilstm, 04_tcresnet, 05_xlsr, 06_results_error_analysis, 07_order_stress_test
src/            paths.py · data.py · features.py · utils.py · evaluate.py · (augment.py, train_torch.py, train_hf.py, models/)
results/        runs/ (one JSON per run) · experiments.csv · predictions/ · metrics/ · figures/   (cache/ is git-ignored)
tests/          smoke tests on synthetic audio (pytest)
report/         report PDF, lit notes, AI-use log
```

## Conventions (everyone)
- **Data:** always use `src.data.load_split(...)` and `src.features.cached_features(...)`. Never re-split. Fit normalisation on train only.
- **Seeds:** call `src.utils.set_seed(seed)` at the top of every run.
- **Experiment ids:** `A{approach}-R{round}-{nn}`, e.g. `A3-R2-04`.
  - Rounds: R0 baselines · R1 defaults · R2 tuning · R3 final 3 seeds · R4 data-size curve.
- **Every run** does two things:
  - calls `log_experiment`, with `change_vs_previous` and `rationale` filled in;
  - saves val/test probabilities with `save_predictions`.
- **Model selection** uses validation log loss only. The test split is used in R3.
- **Git:** branch `feat/<member>-<task>` → PR → one reviewer → merge.
  - Never commit audio, features, `kaggle.json` or model weights; the competition rules forbid redistributing the data.

## Team
| Member | Role |
|---|---|
| M1 · Hassan Adelani Luqman | Data pipeline, features, evaluation, baselines A1/A2, README |
| M2 · Haguma Bianca | Augmentation, PyTorch training loop, BiLSTM A3 |
| M3 · Emmanuel Mukasa | EDA, TC-ResNet A4, error analysis |
| M4 · Emmanuel Nsabagasani | HF training, XLS-R A5, results |

Links: [report](report/main.pdf) · [demo video](https://youtu.be/Gv1f1HYyZ18) · [contribution tracker](https://docs.google.com/spreadsheets/d/1lV5C-49tpxNkssPY9lWKOiY321QZCv3a-YPMJONuf9E/edit?usp=sharing).
