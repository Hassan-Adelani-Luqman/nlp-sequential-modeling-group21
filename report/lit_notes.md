# Literature notes (Phase 2)

One entry per paper, 3–5 sentences, using the template below. The last field matters most: the rubric
rewards literature that *informed our decisions*, not a list of summaries. Full references live in
[references.bib](references.bib) (IEEE style in the report). Use the same citation key here.

```
### <citation key>: <short title> (<first author>, <year>)
- Task / data:
- Method:
- Key result:
- How it informs our project:
```

Reading assignments (Plan.md, Phase 2):

| Member | Papers |
|---|---|
| M1 | Davis & Mermelstein (1980) · Rabiner (1989) · Warden (2018) · Sokolova & Lapalme (2009) · Guo et al. (2017) · Dietterich (1998), plus supporting references for the metrics section |
| M2 | Hochreiter & Schmidhuber (1997) · Graves et al. (2013) · de Andrade et al. (2018) · Bahdanau et al. (2015) · Park et al. (2019) |
| M3 | Sainath & Parada (2015) · Choi et al. (2019) · Bai et al. (2018) · Menon et al. (2018) · Doumbouya et al. (2021) |
| M4 | Vaswani et al. (2017) · Baevski et al. (2020) · Conneau et al. (2021) · Babu et al. (2022) · Pratap et al. (2023) · Berg et al. (2021) · Gong et al. (2021) · Ardila et al. (2020) · Joshi et al. (2020) |

---

## M1: features, classical models, evaluation

### davis1980comparison: parametric representations for word recognition (Davis, 1980)
- **Task / data:** speaker-dependent recognition of monosyllabic words spoken in continuous sentences (two speakers).
- **Method:** compared mel-frequency cepstrum, linear-frequency cepstrum, linear-prediction cepstrum, linear-prediction spectrum and reflection coefficients as the frame representation.
- **Key result:** ten mel-frequency cepstral coefficients computed every 6.4 ms performed best (96.5% and 95.0% for the two speakers).
- **How it informs our project:**
  - This is the origin of MFCCs as the default acoustic representation. A1 and A2 use MFCCs over 25 ms windows with a 10 ms hop.
  - Our A1 result echoes theirs: a compact set (MFCC-13 + Δ + ΔΔ) beat MFCC-40 (log loss 0.783 vs 0.880).
  - Their setting had two speakers; ours has about 300 crowd-workers on different phones. That is why A2 also needed per-utterance normalisation (CMVN), which raised its accuracy from 53% to 76%.

### rabiner1989tutorial: HMMs for speech recognition (Rabiner, 1989)
- **Task / data:** a tutorial on hidden Markov model theory and its application to speech recognition, including isolated-word recognition.
- **Method:** the three HMM problems (evaluation by the forward algorithm, decoding by Viterbi, training by Baum–Welch re-estimation), plus practical issues: left-to-right (Bakis) topologies for speech, scaling, initial parameter estimates, and insufficient training data.
- **Key result:** a practical recipe for word recognisers: one HMM per word, with the word whose model gives the highest likelihood chosen.
- **How it informs our project:**
  - A2 follows this recipe directly: one left-to-right GMM-HMM per word, trained with Baum–Welch, classified by maximum likelihood.
  - Rabiner's emphasis on good initial emission estimates motivated our flat-start initialisation from a uniform segmentation.
  - His discussion of insufficient training data anticipates the failure we hit: starved mixture components in 8-state models produced NaN parameters, which we fixed with small MAP priors.

### warden2018speech: Speech Commands (Warden, 2018)
- **Task / data:** an audio dataset of short spoken words for training and evaluating keyword-spotting systems.
- **Method:** describes the data collection, and a methodology for reproducible, comparable accuracy metrics with baseline results.
- **Key result:** a standard benchmark and evaluation protocol for limited-vocabulary speech recognition.
- **How it informs our project:**
  - Our task is the same kind of problem (short isolated words, a closed vocabulary), so we follow the practice of reporting accuracy on a fixed held-out split.
  - Two contrasts shape our limitations section:
    - Its protocol keeps each speaker's recordings in a single partition, so test accuracy reflects unseen speakers. We cannot, because our data has no speaker IDs.
    - Its standard benchmark adds "silence" and "unknown word" classes, which our closed 12-word set lacks.

### sokolova2009systematic: performance measures for classification (Sokolova, 2009)
- **Task / data:** a systematic analysis of 24 performance measures across binary, multi-class, multi-labelled and hierarchical classification.
- **Method:** studies which changes to the confusion matrix leave each measure unchanged ("measure invariance").
- **Key result:** different measures are sensitive to different aspects of a classifier's errors, so the choice of measure should match the task.
- **How it informs our project:**
  - Justifies reporting several complementary measures rather than one.
  - Accuracy suits our balanced 12 classes. Macro-averaged F1 weights every word equally and exposes words a model systematically fails, such as A1's 66% recall on *mbili*.
  - Per-class scores and the confusion matrix show *which* words fail, which a single number hides.

### guo2017calibration: calibration of neural networks (Guo, 2017)
- **Task / data:** confidence calibration of modern image and text classifiers.
- **Method:** measures miscalibration with expected calibration error (ECE) and reliability diagrams, and compares post-hoc calibration methods.
- **Key result:** modern neural networks are poorly calibrated. Temperature scaling, a single-parameter variant of Platt scaling, is surprisingly effective.
- **How it informs our project:**
  - The official metric is log loss, so calibration directly affects our score.
  - We report ECE (15 bins, as in this paper) and reliability diagrams, and apply temperature scaling to every model.
  - We use the same one-parameter idea on the HMM's per-frame log-likelihoods. There the fitted temperature (0.38 for the final HMM) *sharpens* scores that were under-confident.
  - Logistic regression (A1) needed no correction (T = 0.99), as expected for a model trained by minimising log loss.
  - We cross-fit the temperature on validation, so the reported log loss is not optimistic.

### dietterich1998approximate: statistical tests for comparing classifiers (Dietterich, 1998)
- **Task / data:** comparing supervised learning algorithms using statistical tests.
- **Method:** measures the Type I error of five approximate tests.
- **Key result:** McNemar's test and a new 5×2 cross-validated t-test have acceptable Type I error. McNemar's test is recommended when an algorithm can be run only once, and the 5×2 cv test when it can be run ten times.
- **How it informs our project:** the final comparison (R3) uses the exact McNemar test on the shared test clips. Retraining the pretrained transformer (A5) ten times for a 5×2 cv test is beyond our compute budget. Holm's correction handles the several comparisons against the best model.

**Supporting references for the metrics section**

These were checked against their abstracts. Each gets one line, with the key in `references.bib`.

- **gneiting2007strictly:** proper scoring rules. Log loss is strictly proper, so it rewards honest probabilities. This is why the official metric is appropriate for a system that must act on its confidence.
- **platt1999probabilistic, wu2004probability:** Platt's sigmoid maps SVM scores to probabilities; Wu et al. combine pairwise probabilities into multi-class estimates. libsvm's `probability=True` uses both. This explains our measured result: those probabilities gave log loss 0.880, against 1.230 for one-vs-rest sigmoid calibration, at the same accuracy.
- **naeini2015obtaining:** introduces Bayesian Binning into Quantiles and is a standard source for the binned ECE measure we report.
- **cawley2010overfitting:** over-fitting the model-selection criterion biases performance estimates. This is our caveat that R0/R2 validation scores (the best of many configurations) are optimistic, and why the test split is used only once.
- **mcnemar1947note, holm1979simple, efron1993introduction:** the McNemar test, Holm's multiple-comparison procedure, and bootstrap confidence intervals used in R3.
- **opitz2019macro:** two different "macro F1" formulas exist and can rank classifiers differently. We use the mean of per-class F1 scores and say so in the report.

---

## M2: recurrent models, attention, augmentation
*To be written by M2.*

## M3: convolutional keyword spotting, African-language speech
*To be written by M3.*

## M4: transformers and self-supervised speech models
*To be written by M4.*
