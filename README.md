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

The full project plan is in [Plan.md](Plan.md).

## Status
- [x] Phase 1: `src/paths.py`, `src/data.py`, `src/features.py`, `src/utils.py`, `src/evaluate.py`, tests
- [x] Frozen split committed in `data/splits/` (hash `b74d294f…`)
- [x] Phase 4 + R2 for the classical baselines ([notebooks/02_baselines.ipynb](notebooks/02_baselines.ipynb), 21 logged runs)
  - A1 (MFCC stats + logistic regression): **77.9%** val accuracy, log loss 0.725
  - A2 (12-state × 8-Gaussian GMM-HMM): **95.2%** val accuracy, log loss 0.180
  - *tisa/sita* confusion: 6.7% (A1) vs 0% (A2)
- [x] A3 rounds R1 + R2 ([notebooks/03_bilstm.ipynb](notebooks/03_bilstm.ipynb), 13 logged runs on a Kaggle T4)
  - Conv1d + BiLSTM + attention on MFCC-40: **97.0%** val accuracy, log loss 0.134
  - 0% *tisa/sita*, *nne/nane* and *tatu/tano* confusion
  - Seed check (3 seeds): val log loss 0.137 ± 0.004, accuracy 96.9% ± 0.2; checkpoints saved for the Phase 7 test run
  - Interpretability: attention concentrates on the word (median 100% of the weight); a streaming, forward-only A3 reaches 80% accuracy 500 ms into the word
- [x] Phase 3 EDA ([notebooks/01_eda.ipynb](notebooks/01_eda.ipynb)): durations, loudness and SNR, spectra, sound-alike pairs, t-SNE, pitch. Every finding ends in a modelling decision.
- [x] A4 rounds R1 + R2 ([notebooks/04_tcresnet.ipynb](notebooks/04_tcresnet.ipynb), 12 logged runs, mostly on a Kaggle T4)
  - TC-ResNet8-1.5 on MFCC-40 from a 2.0 s window: **97.1%** val accuracy, log loss 0.119, with 150K parameters (A3: 855K)
  - Seed check (3 seeds): val log loss 0.135 ± 0.023, accuracy 96.8% ± 0.3. A4 ties with A3 on average, but is less stable across seeds.
  - A failed run (augmentation fill bug) and the runs it made stale are kept in [results/failed_runs/](results/failed_runs/) and [results/superseded_runs/](results/superseded_runs/)
- [x] A5 rounds R1 + R2 ([notebooks/05_xlsr.ipynb](notebooks/05_xlsr.ipynb), 9 logged runs on a Kaggle T4)
  - XLS-R-300M fine-tuned on the raw waveform, with waveform augmentation: **97.9%** val accuracy, log loss 0.101, the best of the five approaches
  - Seed check (3 seeds): val log loss 0.1015 ± 0.0009, accuracy 98.0% ± 0.1, the most stable model. Checkpoints saved for the Phase 7 test run (1.2 GB each, not committed)
  - English-only wav2vec2-base comes close (0.120). A frozen encoder with only the head trained fails (27%). Every A5 error is also an A3 error.
- [x] Temporal-order stress test ([notebooks/07_order_stress_test.ipynb](notebooks/07_order_stress_test.ipynb)): on reversed audio, A2–A5 fall from 95–98% to 21–39% accuracy, while an order-free control is unchanged. Shuffled 100 ms chunks hardly affect A5 (89%).
- [x] Report drafts: [evaluation metrics](report/sections/evaluation_metrics.tex) (M1), [methodology](report/sections/methodology.tex) (M2–M4), [related work](report/sections/related_work.tex) (M2–M4), [dataset & EDA](report/sections/dataset_eda.tex) (M3), [limitations & responsible AI](report/sections/limitations_responsible_ai.tex) (M3), [literature notes](report/lit_notes.md) (M1–M4), [verified bibliography](report/references.bib)
- [x] Phase 7 final evaluation on the test split ([notebooks/06_results_error_analysis.ipynb](notebooks/06_results_error_analysis.ipynb), scored once)
  - Test log loss / accuracy: A5 **0.093 / 98.1%**, A4 0.125 / 96.3%, A3 0.170 / 96.0%, A2 0.186 / 96.5%, A1 0.700 / 79.5% (mean of 3 seeds for A3–A5)
  - A5 is significantly more accurate than every other model (McNemar, Holm-corrected p ≤ 0.039). Its log-loss lead over A4 is not significant. A3 and A4 tie on accuracy, but A4 has the lower log loss.
  - Data-size curve (R4): with 24 clips per word, A5 already reaches 96.5%. Without pretraining, the HMM is the most data-efficient model.
  - Efficiency: A5 takes 871 ms per clip on one CPU thread and 1.26 GB, against 9 ms and 0.6 MB for A4
  - Results table for the report: [results/tables/test_results.tex](results/tables/test_results.tex)
- [x] Phase 8 error analysis ([notebooks/06_results_error_analysis.ipynb](notebooks/06_results_error_analysis.ipynb), Part 4; [report section](report/sections/error_analysis.tex))
  - Spelling similarity does not predict confusions. The anagram pair *sita/tisa* drops from 8.6% (A1) to 1.0–1.9% for the order-aware models.
  - The same 10 test clips defeat A2–A5. Compared with all clips, they are far more often very quiet (38% vs 4%) or contain several utterances (52% vs 25%), and 3 are label-error candidates.
  - Noise is the main remaining risk: 8–13% errors below 30 dB SNR, against at most 1.5% above
  - Label check without a listener: confident learning plus an external Swahili recogniser (MMS). One likely label error, three words said in English, two crop failures, and the rest mostly other speech or none. Removing the clear cases changes test accuracy by 0.3 points. Optional human check: [results/metrics/listening_sheet.csv](results/metrics/listening_sheet.csv)
- [ ] Report Results & Discussion (M4); Phases 9–12
- [ ] Report, demo video, contribution tracker (links added on submission)

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
| M1 | Data pipeline, features, evaluation, baselines A1/A2, README |
| M2 | Augmentation, PyTorch training loop, BiLSTM A3 |
| M3 | EDA, TC-ResNet A4, error analysis |
| M4 | HF training, XLS-R A5, results |

Links (added on submission): report · demo video · contribution tracker.
