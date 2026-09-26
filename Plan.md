# Formative 2 — Phased Project Plan
**Group 21 · 4 members · 7 days · Kaggle GPUs · PyTorch + Hugging Face**
**Challenge: Swahili Audio Classification (Zindi) — spoken-word recognition**

---

## Context
The brief asks us to run a small empirical research project on one question: *"How effectively can sequential modelling approaches address a real-world NLP or language-technology problem, and what evidence supports the strengths and limitations of the selected approaches?"*

We first chose Swahili News, but that competition's data is not accessible to us. The **Swahili Audio Classification** data is, so we switched to it. The brief explicitly allows speech: *"speech may be represented as sequences of audio signals or acoustic features."*

**Deliverables**
- A concise PDF report: Introduction, Related Work, Dataset & EDA, Methodology, Results & Discussion, Error Analysis & Limitations, Conclusion & Future Work, References, AI-use disclosure.
- A clean GitHub repo that runs end-to-end on Colab.
- A 7–10 min demo video.
- A contribution tracker. Every member must train at least one model.

**What the rubric rewards (50 pts)**

| Criterion | Pts |
|---|---|
| Experimentation & Rigor | 12 |
| Data Understanding & EDA | 8 |
| Evaluation Metrics | 5 |
| Results & Visualizations | 5 |
| Discussion & Analysis | 5 |
| Technical Implementation | 5 |
| Demo | 5 |
| Academic Writing & Citations | 3 |
| Originality & Responsible AI | 2 |

The quality of the investigation matters more than the score.

### The dataset (verified from the files)

| Item | Value |
|---|---|
| Task | Classify an audio clip as one of **12 spoken Swahili words** |
| Classes | *moja, mbili, tatu, nne, tano, sita, saba, nane, tisa, kumi* (one to ten) + *ndio* (yes), *hapana* (no) |
| Train | 4,200 clips, **perfectly balanced** (350 per word). Columns: `Word_id` (`.wav` filename), `Swahili_word`, `English_translation` |
| Test | 1,800 unlabelled clips. `SampleSubmission.csv` needs 12 probability columns |
| Audio | `Swahili_words.zip` (506 MB) → 6,000 `.wav` files (~900 MB). Recorded by ~300 crowd workers in Kenya (Zindi, 2021) |
| Official metric | **Log loss** |
| Rules | Openly available pretrained models are allowed. **Only competition data may be used** (no external noise or speech corpora). Data must **not be redistributed** |
| Not provided | Speaker IDs, demographics, recording conditions. **Sample rate and duration will be measured in the EDA** |

### Why this suits a sequential-modelling study
- A spoken word is a **time series of acoustic frames** (~100 frames/s). The class is set by the *order* of its sounds.
- The label set contains a natural experiment. **_tisa_ (nine) and _sita_ (six) contain the same sounds in a different order.** A bag-of-frames model should confuse them, and a sequence model should not. Other near-minimal pairs are *nne* / *nane* (four / eight) and *tatu* / *tano* (three / five, same onset "ta-").
- Real-world relevance: Swahili has 100M+ speakers. Spoken digits and yes/no are the core of voice interfaces for **mobile money, IVR hotlines, agriculture and health helplines, and low-literacy users**, and Swahili is under-served by commercial ASR.

### Five approaches
The brief asks for five, with at least three sequential neural models; the "Final Submission" text says "three". We do five: 3 sequential neural + 2 classical. Confirm with the lecturer.

---

## Phase overview

| Phase | Name | Day(s) | Lead | Rubric criteria served |
|---|---|---|---|---|
| 0 | Kick-off and team setup | 1 (AM) | All | — |
| 1 | Data acquisition and shared audio foundations | 1 | M1 (+M2) | Technical Implementation |
| 2 | Literature review | 1–3 (parallel) | All | Related Work, Metrics, Writing |
| 3 | Exploratory audio analysis | 1–2 | M3 (+M1) | Data Understanding |
| 4 | Classical baselines (A1, A2) | 2 | M1 | Experimentation |
| 5 | Neural sequence models, first pass (A3, A4, A5) | 3 | M2, M3, M4 | Experimentation |
| 6 | Iterative tuning and ablations | 4 | M2, M3, M4 (+M1) | Experimentation (12 pts) |
| 7 | Final evaluation and statistical comparison | 5 | M4 (+all) | Metrics, Results |
| 8 | Error analysis and responsible AI | 5 | M3 (+all) | Discussion, Responsible AI |
| 9 | Report writing | 3–6 (rolling) | All | Writing, Discussion |
| 10 | Repo polish and reproducibility | 6–7 | M1 (+all) | Technical Implementation |
| 11 | Demo video | 6 | All | Demo |
| 12 | Final QA and submission | 7 | All | — |

### The five approaches (details in Phases 4–5)

| ID | Approach | Family | Uses temporal order? | Owner |
|---|---|---|---|---|
| **A1** | MFCC summary statistics + SVM (RBF) | Classical, discriminative | **No** (bag of frames) | M1 |
| **A2** | GMM-HMM per word (left-to-right) | Classical, generative sequence model | Yes (Markov states) | M1 |
| **A3** | BiLSTM + attention on log-mel frames | Neural, recurrent | Yes | M2 |
| **A4** | Temporal CNN (TC-ResNet, 1D convolutions over time) | Neural, convolutional | Yes (local, then global) | M3 |
| **A5** | Fine-tuned **XLS-R-300M** on raw waveform (vs. wav2vec2-base) | Neural, self-attention, self-supervised pretrained | Yes (global attention) | M4 |

A1 vs A2 compares no order and explicit order among classical models. A3, A4 and A5 compare recurrent, convolutional and attention-based sequence modelling.

---

## Phase 0 — Kick-off and team setup (Day 1, morning, ~1 h)
**Goal:** everyone agrees on the plan, roles and tools before any code is written.

**Tasks**
1. **Hold a 30-min kick-off call.**
   - Walk through this plan and confirm the switch to the audio challenge.
   - Agree daily stand-ups: 15 min each morning and evening.
2. **Assign roles** (each member trains at least one model):

   | Member | Trains | Other ownership | Report sections |
   |---|---|---|---|
   | **M1** | A1 MFCC+SVM, A2 GMM-HMM | Repo, `data.py`, `features.py`, `paths.py`, `evaluate.py`, Kaggle dataset, README | Dataset & EDA (with M3), Evaluation Metrics |
   | **M2** | A3 BiLSTM + Attention | `train_torch.py`, `augment.py` (SpecAugment etc.) | Related Work (RNN/CNN KWS), Methodology |
   | **M3** | A4 TC-ResNet | EDA notebook, error-analysis notebook | Error Analysis & Limitations, Responsible AI |
   | **M4** | A5 XLS-R / wav2vec2 | `train_hf.py`, results notebook, final figures, optional demo app | Introduction, Related Work (self-supervised speech, African speech), Results & Discussion |

3. **Every member joins the Zindi competition.** The rules forbid sharing the data with non-participants, so this keeps the shared Kaggle dataset compliant.
4. **Create the shared spaces:**
   - A GitHub repo with all members as collaborators.
   - An Overleaf project using the IEEE conference template.
   - A copy of the contribution tracker.
   - A group chat.
5. **Agree the Git workflow:**
   - `main` holds only working code.
   - Each member works on a branch: `feat/<member>-<task>`.
   - Changes go in by PR, with one reviewer.
   - Never commit audio, features or model weights.
6. **Each member verifies a Kaggle account** (phone-verified, which enables GPU and internet).

**Outputs:** repo, Overleaf, tracker, chat, role table.
**Done when:** all 4 members can push, edit Overleaf, open a Kaggle GPU notebook, and have joined the Zindi competition.

---

## Phase 1 — Data acquisition and shared audio foundations (Day 1)
**Goal:** one frozen split, one cached feature pipeline, and one evaluation module, so all five models see identical data.

> Status: `src/paths.py`, `src/utils.py`, `src/evaluate.py` and the split logic were already built for the text plan and are **reused as they are**. `src/data.py` and the tests need adapting for audio, and `src/features.py` is new.

**Tasks**
1. **Get the data** (M1).
   - Download `Train.csv`, `Test.csv`, `SampleSubmission.csv` and **`Swahili_words.zip`** from Zindi. The starter notebook is optional reading.
   - Unzip into `data/raw/Swahili_words/`.
   - Upload the CSVs and the zip as a **private** Kaggle Dataset `group21-swahili-audio`, shared only with the 3 other members.
2. **Adapt `src/data.py`** (M1):
   - `load_metadata(split)`: map `Word_id` → `id`, `Swahili_word` → `label`, and keep `English_translation`. Check that every id has a `.wav` file.
   - `audio_dir()`: locate the folder of `.wav` files, which may be nested after unzipping.
   - `load_audio(id, sr=16000)`: read with `soundfile`, convert to mono, resample to **16 kHz** (the rate wav2vec2/XLS-R were pretrained on), and return a float32 waveform.
   - **Integrity checks:** list unreadable or empty files, and find exact duplicate recordings by MD5 of the audio bytes. Drop duplicates before splitting and record the counts in the manifest.
   - **Frozen split** (reuse `make_splits`): stratified **70/15/15**, seed 42 → 2,940 / 630 / 630 clips (245 / 52–53 / 52–53 per class). We use 70/15/15 instead of 80/10/10 so the test set has ~52 clips per word and per-class results are reliable. The split files, `label_map.json` and `manifest.json` (with the split hash) are committed.
3. **Write `src/features.py`** (M1, used by everyone):
   - `trim_silence(wav, top_db=30)`: `librosa.effects.trim` with 50 ms padding kept.
   - `fix_length(wav, seconds)`: centre-pad or crop to a fixed duration, set from the EDA (≈95th percentile of trimmed durations).
   - `log_mel(wav)`: 64 mel bands, 25 ms window, 10 ms hop → shape (T frames × 64).
   - `mfcc(wav)`: 40 MFCCs, optionally + Δ + ΔΔ.
   - `mfcc_stats(wav)`: mean, std, min and max per coefficient (the A1 input).
   - `extract_all(name, fn)`: compute features for all 6,000 clips **once** and cache them to `cache/features_<name>.npz`, so no model re-decodes audio. Use multiprocessing (~2–3 min on a Kaggle CPU).
   - Per-channel **mean/variance normalisation** fitted on the train split only.
4. **Reuse the existing modules unchanged:**
   - `src/utils.py`: `set_seed` and `log_experiment` (one JSON per run) → `results/experiments.csv`.
   - `src/evaluate.py`: `compute_metrics` (macro-F1, accuracy, log loss, ROC-AUC, ECE, per-class), `save_predictions`, `bootstrap_ci`, `mcnemar_test`, confusion-matrix and learning-curve plots.
   - `src/paths.py`: Kaggle, Colab and local detection.
5. **Update `requirements.txt`:** add `librosa`, `soundfile` and `hmmlearn`. `torchaudio` is already on Kaggle and Colab.
6. **Update `tests/test_smoke.py`:** replace the text fixtures with synthetic audio. Generate 12 classes of short sine or chirp `.wav` files and check loading, resampling, trimming, feature shapes, the split and the metrics.

**Outputs:** Kaggle dataset, committed split files, working `data.py` and `features.py`, cached features, passing tests.
**Done when:** every member runs `00_setup_check.ipynb` on Kaggle, gets the same split hash, and loads a cached log-mel tensor.

---

## Phase 2 — Literature review (Days 1–3, in parallel)
**Goal:** a focused review that justifies the task framing, the five approaches, the features, the metrics and the augmentation.

**Tasks**
1. **Each member reads 5–7 sources** and writes a 3–5 sentence note per paper covering task, method, data, result, and *how it informs our decision*. Notes go in `report/lit_notes.md`.

   | Member | Reading |
   |---|---|
   | M1 | Davis & Mermelstein (1980) MFCC · Rabiner (1989) HMM tutorial · Warden (2018) Speech Commands · Sokolova & Lapalme (2009) metrics · Guo et al. (2017) calibration and temperature scaling · Dietterich (1998) statistical tests |
   | M2 | Hochreiter & Schmidhuber (1997) LSTM · Graves et al. (2013) speech recognition with deep RNNs · de Andrade et al. (2018) *neural attention model for speech command recognition* (BiLSTM + attention, the direct precedent for A3) · Bahdanau et al. (2015) attention · Park et al. (2019) SpecAugment |
   | M3 | Sainath & Parada (2015) CNNs for small-footprint KWS · Choi et al. (2019) TC-ResNet (A4) · Bai et al. (2018) TCN · Menon et al. (2018) ASR-free keyword spotting for humanitarian radio monitoring (African languages) · Doumbouya et al. (2021) radio archives and voice assistants for low-literacy users (West Africa) |
   | M4 | Vaswani et al. (2017) · Baevski et al. (2020) wav2vec 2.0 · Conneau et al. (2021) XLSR · Babu et al. (2022) XLS-R · Pratap et al. (2023) MMS · Berg et al. (2021) Keyword Transformer · Gong et al. (2021) AST · Ardila et al. (2020) Common Voice (contains Swahili) · Joshi et al. (2020) linguistic diversity |

2. **Build the shared `references.bib`** (≥25 entries), including the Zindi dataset page and software: PyTorch, torchaudio, librosa, hmmlearn, scikit-learn, HF Transformers.
3. **Write a model-justification table** (M4 compiles). One row per approach with these columns: *data property addressed*, *literature evidence*, *expected strength*, *expected weakness*.

**Outputs:** `lit_notes.md`, `references.bib`, justification table.
**Done when:** every approach, feature choice and metric has at least 2 supporting citations.

---

## Phase 3 — Exploratory audio analysis (Days 1–2)
**Goal:** understand the recordings and turn each finding into an explicit modelling decision (8 pts).

**Tasks** (in `notebooks/01_eda.ipynb`; every analysis ends with a **"→ Decision:"** cell)

| # | Analysis | How | Decision it informs |
|---|---|---|---|
| 1 | Class balance | Counts per class, train/val/test (350 each confirmed) | No class weighting needed; accuracy ≈ balanced accuracy; macro-F1 still reported |
| 2 | File properties | Sample rate, channels, bit depth, codec; unreadable/empty files | Resample to 16 kHz mono; drop corrupt files |
| 3 | Duration | Histogram of raw and **silence-trimmed** durations, per class (box plots) | Fixed input length (≈95th percentile of trimmed durations); crop/pad strategy |
| 4 | Silence and loudness | Leading and trailing silence %, RMS level (dBFS), peak level, clipping rate, rough SNR (speech-frame vs non-speech-frame energy) | Silence trimming on or off (tested in R2); RMS normalisation; noise augmentation strength |
| 5 | What the words look like | Waveform + log-mel spectrogram for 2 examples per word; **mean spectrogram per class** | Frame and hop choice; shows the temporal structure the models must learn |
| 6 | Phonetic similarity | 12×12 matrix of edit distance between the word spellings (Swahili spelling is nearly phonemic); highlight *tisa/sita*, *nne/nane*, *tatu/tano*, *saba/sita* | Predicts the confusions; motivates sequence models; feeds error analysis |
| 7 | Feature-space separability | t-SNE/UMAP of MFCC-mean vectors, coloured by class | How far an order-agnostic representation can go (sets expectations for A1) |
| 8 | "How early is a word recognisable?" | Train A1 on the **first 25 / 50 / 75 / 100 %** of each trimmed utterance; plot accuracy per class | How much temporal context is needed, and which words share onsets (the key sequential analysis) |
| 9 | Duplicates and speaker variability | Byte-hash duplicates; acoustic near-duplicates (cosine similarity of MFCC-mean vectors > 0.99); pitch (F0) distribution as a rough proxy for speaker variety | Leakage check; motivates augmentation; documents the "no speaker IDs" limitation |

- Save figures to `results/figures/eda_*.png` at 300 dpi with a fixed colour per word.
- End the notebook with an **"EDA → design decisions"** table, which is reused in the report and the video.

**Outputs:** `01_eda.ipynb`, ~10 figures, the decisions table. The final values of `target_sr`, `fix_length`, `top_db` and `n_mels` go into `configs/base.yaml`.
**Done when:** every analysis has a stated decision and the feature settings are frozen.

---

## Phase 4 — Classical baselines A1 and A2 (Day 2)
**Goal:** strong classical references, one without temporal order and one with it (round **R0**).

**A1 — MFCC statistics + SVM** (`src/models/baselines.py`, `notebooks/02_baselines.ipynb`)
- Input: 40 MFCC + Δ + ΔΔ → mean, std, min and max over time (480-d vector). **All temporal order is discarded.**
- Model: `StandardScaler` → `SVC(kernel="rbf", probability=True)`. Grid C ∈ {1, 10, 100}, γ ∈ {scale, 1e-3, 1e-2}, using **5-fold stratified CV on train**, then checked on val.
- Experiments:
  - `A1-R0-01`: MFCC-13 stats
  - `A1-R0-02`: MFCC-40 + Δ/ΔΔ stats
  - `A1-R0-03`: Logistic regression instead of SVM
  - `A1-R0-04`: with vs without silence trimming
- Also produces the "how early" curve for EDA analysis 8.

**A2 — GMM-HMM per word** (`hmmlearn`)
- One left-to-right `GMMHMM` per class on MFCC-13 + Δ + ΔΔ frame sequences (39-d), with diagonal covariances.
- Prediction = the class whose HMM gives the highest log-likelihood. Probabilities come from a softmax over the per-class log-likelihoods, with a **temperature tuned on val** so log loss is meaningful.
- Grid: states ∈ {3, 5, 8} × mixtures ∈ {1, 2, 4}.
- **Purpose:** the classic isolated-word recogniser (Rabiner, 1989). It models order explicitly but is shallow and generative. It sets the bar that the neural sequence models must beat.

**Outputs:** logged runs, prediction files, confusion matrices, `configs/a1_best.yaml`, `configs/a2_best.yaml`.
**Done when:** both baselines are logged with an "observation → next step" note. For example: "A1 confuses tisa↔sita at X% → evidence that order matters".

---

## Phase 5 — Neural sequence models, first pass (Day 3)
**Goal:** get A3, A4 and A5 training correctly with sensible defaults (round **R1**).

**Shared loop `src/train_torch.py`** (M2, used by A3 and A4)
- Dataset: cached log-mel (T × 64) + label. Augmentation is applied **on the fly in train only**.
- Loss: cross-entropy with label smoothing 0.1. Optimiser: AdamW. OneCycle or cosine LR schedule. Gradient clip 1.0. AMP fp16.
- Early stopping on **val log loss** (patience 5). Best checkpoint saved to `/kaggle/working`.
- Writes per-epoch `train_loss, val_loss, val_macro_f1, val_acc` to `results/metrics/<exp_id>_history.csv`.

**`src/augment.py`** (M2). Only synthetic transforms, because the rules forbid external noise data:
- SpecAugment time and frequency masks (Park et al., 2019).
- Random time shift (±100 ms).
- Gaussian noise at a random SNR of 10–30 dB.
- Speed or pitch perturbation (0.9–1.1×).

**A3 — BiLSTM + attention** (M2, `src/models/bilstm.py`, `notebooks/03_bilstm.ipynb`)
- Architecture: log-mel (T × 64) → optional 2-layer Conv1d front-end → **2-layer BiLSTM** (hidden 128 per direction, dropout 0.3) → additive attention pooling over time → Linear(256 → 12).
- Defaults: batch 64, lr 1e-3, ≤40 epochs.
- The model returns **attention weights per frame**, for the Phase 8 plots (which part of the word does it attend to?).

**A4 — TC-ResNet** (M3, `src/models/tcresnet.py`, `notebooks/04_tcresnet.ipynb`)
- Treats the 64 mel bands as channels and convolves **over time only** (Choi et al., 2019).
- Architecture: Conv1d(64 → 16k, kernel 3) → 3–6 residual blocks (kernel 9, stride 2) → global average pool → Linear(→ 12). About 60–300k parameters.
- Defaults: batch 64, lr 1e-3, ≤40 epochs.
- Also report the **receptive field in milliseconds**, and plot class activation over time.

**A5 — Self-supervised pretrained transformer** (M4, `src/train_hf.py`, `notebooks/05_xlsr.ipynb`)
- Model: `facebook/wav2vec2-xls-r-300m` (pretrained on 128 languages, **including Swahili**) via `AutoModelForAudioClassification`, on the **raw 16 kHz waveform** padded or cropped to the fixed length.
- Settings: freeze the CNN feature encoder, lr 3e-5, warmup 10%, 10 epochs, batch 16 (gradient accumulation if needed), `fp16=True`, best model selected by val log loss.
- Fallback if T4 memory or time is tight: `facebook/wav2vec2-base`, which is also the R2 comparison.
- Expected: about 1–3 min per epoch on a T4 for 2,940 short clips.

**Kaggle speed settings**
- Accelerator: GPU **T4 x2** (use 1 GPU) or **P100**. Internet ON.
- Extract features once in a CPU session and save them as a Kaggle dataset version (`group21-swahili-audio-features`), so GPU sessions never decode audio.
- `num_workers=2`, `pin_memory=True`.
- For long runs use **Save Version → Save & Run All**.
- Download `results/` from the notebook output and commit it through a PR.

**Experiments:** `A3-R1-01`, `A4-R1-01`, `A5-R1-01`.
**Outputs:** learning curves and first confusion matrices.
**Done when:** all three train cleanly, clearly beat A1 on val (or the reason is understood), and are logged with an observation note. For example: "A3 overfits after epoch 12 → R2: add SpecAugment".

---

## Phase 6 — Iterative tuning and ablations (Day 4)
**Goal:** systematic, well-tracked experiments that each follow from a previous observation (12 pts).

**Rules**
- Change **one factor at a time** relative to that model's current best config.
- Log every run, including failures, with the `change_vs_previous` and `rationale` columns filled in.
- Select on **validation** only. The test split stays untouched until Phase 7.

| Factor | Variants | Models | Research question |
|---|---|---|---|
| Input representation | MFCC-40 vs log-mel-64 (vs raw waveform for A5) | A3, A4 | Does a learned or less compressed representation help? |
| Silence trimming | off / on | A1, A3, A4 | Does removing non-speech frames help sequence models focus? (EDA 4) |
| Input length | 1.0 / 1.5 / 2.0 s (from EDA 3) | A3, A4, A5 | How much temporal context is needed? |
| Augmentation | none → SpecAugment → + shift/noise/speed | A3, A4, A5 | Does augmentation make up for only ~245 clips per word and unknown speakers? |
| Direction | unidirectional LSTM vs BiLSTM | A3 | Does future context help? (A unidirectional model is needed for streaming) |
| Pooling | last hidden state vs attention | A3 | Does attention find the discriminative segment? |
| Receptive field | 3 vs 6 residual blocks / kernel 9 vs 15 | A4 | How far across time must a convolution see? |
| Pretraining language | **XLS-R-300M (multilingual incl. Swahili) vs wav2vec2-base (English only)** | A5 | Does multilingual pretraining transfer better to Swahili? |
| Fine-tuning depth | frozen encoder + head vs full fine-tune | A5 | Cost vs accuracy in low-resource fine-tuning |
| Calibration | none vs **temperature scaling** on val (Guo et al., 2017) | all | Can post-hoc calibration cut log loss (the official metric) without changing accuracy? |
| **Temporal-order stress test** (evaluation only) | original vs **time-reversed** vs **frame-shuffled** val inputs | all | How much does each model rely on temporal order? A1 should be unaffected; true sequence models should collapse |

- Budget: about 5–7 runs per neural model, 25–35 logged runs in total.
- M1 (baselines done) runs A3/A4 sweeps on a second Kaggle account to spread the GPU quota.
- End of day: each owner writes a **5-line progression summary** and freezes `configs/a{3,4,5}_best.yaml`.
- *Optional stretch:* a Keyword Transformer (Berg et al., 2021) trained **from scratch** on log-mel. It separates the effect of the attention architecture from the effect of pretraining.

**Outputs:** filled `experiments.csv`, ablation tables and plots, frozen best configs.
**Done when:** each approach has a best config chosen on validation, supported by at least 3 logged ablations.

---

## Phase 7 — Final evaluation and statistical comparison (Day 5)
**Goal:** a fair, statistically grounded comparison on the untouched test split (rounds **R3** and **R4**).

**Tasks**
1. **Final runs (R3):**
   - Train each best config with **3 seeds** (42, 13, 7). A1 and A2 are deterministic, so they need one run each.
   - Evaluate on the test split and save per-seed predictions.
2. **Main results table** (M4, `notebooks/06_results_error_analysis.ipynb`):
   - Columns: Approach, #Params, Train time, Inference ms/clip, **Log loss (mean ± std)**, **Macro-F1 (mean ± std)**, Accuracy, ROC-AUC, ECE.
   - Add **bootstrap 95% CIs** (1,000 resamples).
3. **Significance testing:** **McNemar's test** between the best model and each of the others, with Holm correction.
4. **Figures:**
   - 12×12 normalised confusion matrices (grid of 5).
   - One-vs-rest ROC curves for the top 2 models.
   - Learning curves.
   - **Reliability diagrams** before and after temperature scaling.
   - Per-class F1 grouped bars.
   - **Temporal-order stress-test** bar chart (original / reversed / shuffled × 5 models).
5. **Data-size curve (R4):** train A1, A3 and A5 on 10 / 25 / 50 / 100% of the training clips (≈25 → 245 per word). Plot test accuracy and log loss against clips per word. This shows how much labelled speech a low-resource language needs, with and without pretraining.
6. **Optional:** a late Zindi submission (best model trained on train+val) as an external log-loss check.
7. **Efficiency table:** parameters, model size (MB), training time and CPU inference latency. This matters for on-device or IVR deployment.

**Outputs:** final tables (CSV + LaTeX) and all figures in `results/figures/`.
**Done when:** every number in the report is regenerated from `results/predictions/` by notebook 06.

---

## Phase 8 — Error analysis and responsible AI (Day 5)
**Goal:** evidence-based understanding of *why* and *where* the models fail.

**Tasks** (M3 leads, each owner contributes their model's section)
1. **Confusion vs phonetic similarity:** correlate the pairwise confusion rate with the edit distance between the words (EDA 6). Report the anagram pair *tisa/sita* and the pairs *nne/nane* and *tatu/tano* for each model. This is the core evidence on how well each model uses order.
2. **Model-agreement analysis:** *all correct* / *all wrong* / *only model X wrong*.
   - **Listen to** ~15 clips that every model gets wrong and tag them: mislabelled, truncated, silent or very noisy, clipped, strong accent, wrong word spoken.
   - Errors only one model makes point to that architecture's weakness.
3. **Stratified performance:** accuracy by duration bucket, by estimated SNR bucket, and by loudness bucket. This links errors back to EDA 3–4.
4. **Interpretability:**
   - A3 attention weights overlaid on spectrograms: does it attend to the vowel that separates *nne* and *nane*?
   - A4 class-activation over time.
   - A5 confident errors, from the reliability diagram.
5. **Early-recognition curve:** accuracy of A3 (unidirectional variant) when it sees only the first *x* ms. This shows at what point each word becomes distinguishable.
6. **Limitations:**
   - **No speaker IDs**, so the same speaker may appear in both train and test, making scores optimistic. *Stretch mitigation:* cluster ECAPA speaker embeddings (SpeechBrain, openly available) into pseudo-speakers and re-evaluate with a grouped split.
   - A closed set of 12 words with no "unknown" or "silence" class, so real users saying other words get confident wrong answers.
   - Unknown recording devices and demographics.
   - A single dataset and a single region (Kenya).
   - The external-data ban limits augmentation.
   - Compute limits on large models.
7. **Responsible AI:**
   - Accent and dialect coverage (Kenyan vs Tanzanian vs Congolese Swahili) and unknown gender and age balance, which risk unequal error rates.
   - Consent and data protection: **the no-redistribution rule is respected** and no audio is committed.
   - Voice data is biometric.
   - Misuse (surveillance) vs benefit (inclusive voice interfaces).
   - Energy use: GPU hours are logged.
8. **AI-use disclosure:** keep the running `report/ai_use_log.md` (already started).

**Outputs:** error-analysis section of notebook 06, 5–7 figures and tables, updated `ai_use_log.md`.
**Done when:** each failure category is backed by concrete clips and numbers.

---

## Phase 9 — Report writing (Days 3–6, rolling)
**Goal:** a concise, coherent, IEEE-cited report of about 8–12 pages plus references.

| Section | Owner | Content | Draft by |
|---|---|---|---|
| Abstract | M4 | Problem, approaches, key numbers, main finding (≤200 words) | Day 6 |
| 1. Introduction | M4 | Swahili voice interfaces, low-resource speech, research question, contributions | Day 4 |
| 2. Related Work | M2 + M4 | Keyword spotting (HMM → CNN/RNN → transformers/SSL); African and low-resource speech; how the literature shaped our choices | Day 4 |
| 3. Dataset & EDA | M1 + M3 | Data description; 3–4 key figures (durations, spectrograms, phonetic-similarity matrix, "how early" curve); decisions table | Day 3 |
| 4. Methodology | M2 (+M1 metrics) | Preprocessing, features, the five approaches + justification table, training, tuning protocol, validation strategy, **metrics and their justification** | Day 4 |
| 5. Results & Discussion | M4 | Main table, ablations, order stress test, data-size curve, calibration, efficiency; interpretation linked to architecture, EDA and literature | Day 5 |
| 6. Error Analysis & Limitations | M3 | Phase 8 content | Day 5 |
| 7. Conclusion & Future Work | All | Answer the research question; future work: an unknown/silence class, speaker-grouped evaluation, streaming KWS, more Swahili words, on-device models | Day 6 |
| Responsible AI + AI disclosure | M3 | Phase 8 items 7–8 | Day 6 |
| References | M1 | IEEE, ≥25 entries | Day 6 |
| Links box | M1 | GitHub repo, demo video, contribution tracker | Day 7 |

**Writing rules**
- Every claim is backed by a number, figure or citation.
- Write in our own words, with no generic filler.
- Figures are captioned and referenced in the text.
- Use consistent model names (A1–A5).

**Done when:** each section has been reviewed by 2 members other than its author.

---

## Phase 10 — Repo polish and reproducibility (Days 6–7)
**Goal:** a clean, modular repo that runs end-to-end on Colab with minimal setup (5 pts).

**Tasks**
1. **README:**
   - Summary and research question.
   - Data instructions (join Zindi → download → Kaggle dataset / Drive).
   - Setup for Colab, Kaggle and local.
   - "Open in Colab" badges and the notebook run order.
   - A results table and key figure.
   - The repo layout.
   - Team and contributions, plus links to the report, video and tracker.
2. **Setup cell in every notebook:** detect the environment → clone → `pip install -r requirements.txt` → locate the data → set seeds.
3. **Colab data path:** mount Drive (with `Swahili_words.zip` uploaded by the user) or download from Kaggle with the user's API token. The data must never be public.
4. **Clean-up:** docstrings, no hard-coded paths, YAML configs, notebook outputs kept, and **pinned versions** in `requirements.txt`.
5. **Fresh-runtime test:** run notebooks 01 → 06 top to bottom on a **fresh Colab T4** and on Kaggle.
6. Tag `v1.0-submission`.

**Done when:** a member who didn't write a notebook can run it from the README alone.

---

## Phase 11 — Demo video (Day 6)
**Goal:** a professional 7–10 minute video in which every member presents.

| Segment | Time | Presenter |
|---|---|---|
| Problem, real-world relevance, research question | 1 min | M4 |
| Data and EDA (spectrograms, durations, the tisa/sita anagram, decisions table) | 1.5 min | M1 / M3 |
| The five approaches and why each was chosen | 2 min | Each owner, ~30 s |
| Experiment progression (R0 → R4) and key ablations | 1.5 min | M2 |
| Results, significance, order stress test | 1.5 min | M4 |
| Error analysis, limitations, responsible AI | 1.5 min | M3 |
| Conclusion and future work | 0.5 min | M1 |

**Tasks**
1. Build a ~12-slide deck from the report figures.
2. *Optional, high impact:* a short **live demo**. A Gradio app in a Colab or Kaggle notebook where a member speaks a Swahili digit and the best model predicts it. The recording is used only for the demo, never for training.
3. Record with Zoom or OBS (screen + camera). Show the repo and one notebook briefly.
4. Upload as an unlisted YouTube video or a Drive link.
5. Rehearse likely questions: why log loss, why MFCC vs log-mel, what the order stress test shows, why XLS-R over wav2vec2-base, and what speaker leakage means.

**Done when:** the video is 7–10 min, every member speaks, and the link works in an incognito window.

---

## Phase 12 — Final QA and submission (Day 7)

**Checklist**
- [ ] The report has all sections, the AI disclosure, the responsible-AI discussion and the 3 links (repo, video, tracker).
- [ ] Figures and tables are numbered, captioned and referenced; the numbers match `results/`.
- [ ] IEEE references are complete, including the dataset, pretrained models (XLS-R, wav2vec2) and libraries.
- [ ] `experiments.csv` has ≥25 runs with the rationale columns filled in.
- [ ] The repo runs on fresh Colab; **no audio, features or weights are committed**; the release is tagged.
- [ ] The contribution tracker is complete for every member.
- [ ] The video and tracker links are viewable by the grader.
- [ ] An internal Q&A confirms each member can explain every part.
- [ ] Submitted before the deadline.

---

## Evaluation metrics (reference for Phases 4–8)
- **Log loss (co-primary).** The official Zindi metric. It rewards calibrated probabilities, which matter when a voice interface must decide whether to act on a prediction or ask the user to repeat.
- **Macro-F1 (co-primary) and accuracy.** Classes are exactly balanced, so accuracy is not inflated by a majority class. Macro-F1 shows whether any word is systematically failed (Sokolova & Lapalme, 2009).
- **Per-class P/R/F1, 12×12 confusion matrix, OvR ROC-AUC.**
- **ECE and reliability diagrams**, with temperature scaling (Guo et al., 2017).
- **Uncertainty:** mean ± std over 3 seeds, bootstrap 95% CIs, and McNemar with Holm correction (Dietterich, 1998).
- **Limitations to discuss:**
  - Log loss punishes a single confident error heavily.
  - A 630-clip test set gives CIs of about ±2–3 accuracy points.
  - Possible speaker overlap makes every score optimistic.

## Day-by-day summary

| Day | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 1 | P0; P1 data, split, `features.py`, Kaggle dataset | P2 reading; `augment.py` | P3 EDA start (file properties, durations); P2 reading | P2 reading; load XLS-R on Kaggle and time one batch |
| 2 | P4 A1 + A2 (R0); "how early" curve | `train_torch.py`; A3 skeleton | P3 EDA complete; A4 skeleton | `train_hf.py`; A5 skeleton; justification table |
| 3 | Metrics section; EDA write-up | A3 R1 | A4 R1; EDA section draft | A5 R1; Introduction draft |
| 4 | A3/A4 sweeps (2nd account); temperature scaling for all; references | A3 R2; Methodology draft | A4 R2; order stress-test script | A5 R2 (XLS-R vs wav2vec2, freezing, LR); Related Work |
| 5 | Data-size curve (A1); help with P7 | A3 R3 seeds; attention plots; early-recognition curve | A4 R3 seeds; P8 error analysis (listening study) | A5 R3 seeds; P7 results notebook and tables |
| 6 | P10 README and cleanup | Report review; slides | Limitations, Responsible AI; slides | Results & Discussion; abstract; slides; Gradio demo |
| 6–7 | **All: P11 video, P9 final edits** | | | |
| 7 | **All: P10 fresh Colab test, P12 QA and submission** | | | |

## Verification
- `pytest -q` passes on synthetic audio: loading, resampling, trimming, feature shapes, split disjointness and stratification, metrics.
- Every training script completes a **64-clip CPU smoke run** in under 2 minutes.
- The split hash is identical for all members (`python -m src.data`).
- Re-running a final config with the same seed reproduces test accuracy within ±0.5 points.
- Notebooks 01 → 06 run top to bottom on a fresh Colab runtime and regenerate every report figure and table from `results/`.
