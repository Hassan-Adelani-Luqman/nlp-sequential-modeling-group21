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

### hochreiter1997long: Long short-term memory (Hochreiter, 1997)
- **Task / data:** learning long-range dependencies in sequences with recurrent networks.
- **Method:** LSTM units, which keep error flow constant through gated "constant error carousels", so gradients neither vanish nor explode over long time lags.
- **Key result:** LSTM can learn to bridge time lags of more than 1000 discrete time steps, where earlier recurrent methods failed.
- **How it informs our project:**
  - A3 uses LSTM cells because a word spans about 151 frames (1.5 s at 100 frames/s), and the evidence that separates *tisa* from *sita*, or *tano* from *tatu*, is spread across the word.
  - A plain recurrent network would struggle to carry information across that span. EDA 8 showed that the second syllable matters.

### graves2013speech: Deep recurrent networks for speech recognition (Graves, 2013)
- **Task / data:** phoneme recognition on TIMIT.
- **Method:** *deep* (stacked) and *bidirectional* LSTM networks, combining several levels of representation with long-range context in both directions.
- **Key result:** deep LSTM networks reached a 17.7% test error on the TIMIT phoneme benchmark.
- **How it informs our project:**
  - This is the precedent for A3's two stacked bidirectional LSTM layers on acoustic frames.
  - The bidirectional choice is tested directly in R2 (A3-R2-05, forward-only). That also asks whether a streaming-capable model would lose accuracy.

### bahdanau2015neural: Attention for sequence models (Bahdanau, 2015)
- **Task / data:** English–French neural machine translation.
- **Method:** replaces the fixed-length encoding of the sequence with a learned soft search: an additive (tanh) score for each position, and a weighted sum over positions.
- **Key result:** performance comparable to the state-of-the-art phrase-based system, by removing the fixed-length bottleneck.
- **How it informs our project:**
  - A3 pools the LSTM outputs over time with this additive attention, instead of using only the final hidden state. R2 compares attention with last-state (A3-R2-03) and mean (A3-R2-04) pooling.
  - The attention weights also let the error analysis show which frames each decision relied on.

### deandrade2018neural: Attention model for speech commands (de Andrade, 2018)
- **Task / data:** keyword recognition on Google Speech Commands (V1 and V2), the closest benchmark to our task.
- **Method:** a convolutional recurrent network with attention. The attention weights show which parts of the audio the network used.
- **Key result:** "94.1% on Google Speech Commands dataset V1 and 94.5% on V2 (for the 20-commands recognition task)", with only 202K trainable parameters.
- **How it informs our project:**
  - This is the direct precedent for A3: recurrence plus attention on spectrogram features for spoken-word classification.
  - We deliberately leave out its convolutional layers by default, so that A3 is a purely recurrent model and contrasts cleanly with the convolutional A4. A3-R2-12 adds them back to test that choice.
  - Their ~94% on a far larger dataset is a useful reference for what is achievable. We have only 245 training clips per word.

### park2019specaugment: SpecAugment (Park, 2019)
- **Task / data:** end-to-end speech recognition (LibriSpeech 960 h, Switchboard 300 h).
- **Method:** augmentation applied directly to filter-bank features: time warping, masking blocks of frequency channels, and masking blocks of time steps.
- **Key result:** state-of-the-art results on both tasks, e.g. "6.8% WER on test-other without the use of a language model" on LibriSpeech, against 7.5% for the previous best hybrid system.
- **How it informs our project:**
  - With only 245 clips per word, A3 is at risk of over-fitting. A3-R2-01 adds SpecAugment's time and frequency masking.
  - A3-R2-02 extends it with synthetic time shift, tempo change and noise. Only synthetic transforms are allowed, because the competition bans external data.
  - We use a global tempo change rather than SpecAugment's time warping, and apply masking in the log-mel domain as the paper does.

## M3: convolutional keyword spotting, African-language speech

### sainath2015convolutional: CNNs for small-footprint keyword spotting (Sainath, 2015)
- **Task / data:** keyword spotting on devices with tight compute limits.
- **Method:** convolutional architectures designed for two budgets: a limited number of multiplications, and a limited number of parameters.
- **Key result:** the CNNs gave "between a 27-44% relative improvement in false reject rate compared to a DNN, while fitting into the constraints of each application."
- **How it informs our project:** this established convolution as the standard architecture for keyword spotting. It motivates a convolutional approach (A4) alongside the recurrent one (A3). Our binding constraint is different, though: very little data (245 clips per word) rather than little compute.

### choi2019temporal: TC-ResNet, temporal convolution for keyword spotting (Choi, 2019)
- **Task / data:** real-time keyword spotting on mobile phones, evaluated on Google Speech Commands.
- **Method:** temporal convolutions over the frames, with the frequency axis treated as channels, in a compact residual network. This replaces the deep 2-D convolutions of earlier work.
- **Key result:** "more than 385× speedup on Google Pixel 1", while surpassing the accuracy of the previous state-of-the-art model.
- **How it informs our project:**
  - A4 *is* this architecture. Our default TC-ResNet8 at width 1.5 has 146K parameters.
  - R2 tests the paper's own design axes: depth (TC-ResNet8 vs TC-ResNet14), kernel width and channel width. It also compares their MFCC input with log-mel.

### bai2018empirical: Convolutional vs recurrent networks for sequences (Bai, 2018)
- **Task / data:** a systematic comparison of generic convolutional and recurrent architectures across standard sequence-modelling benchmarks.
- **Method:** a simple temporal convolutional network (TCN), with dilated causal convolutions and residual connections, against LSTMs and GRUs under comparable settings.
- **Key result:** "a simple convolutional architecture outperforms canonical recurrent networks such as LSTMs across a diverse range of tasks and datasets, while demonstrating longer effective memory."
- **How it informs our project:**
  - This is the direct question behind our central comparison, recurrence (A3) against convolution (A4). Bai et al. would predict A4 matches or beats A3.
  - Our experiment tests whether that holds for very short sequences (one spoken word, about 1 s) and scarce training data.

### menon2018fast: ASR-free keyword spotting for humanitarian monitoring (Menon, 2018)
- **Task / data:** keyword spotting to support UN relief programmes in parts of Africa where languages are extremely under-resourced, by monitoring radio broadcasts.
- **Method:** dynamic time warping (DTW) on a small set of recorded isolated keywords provides the supervision for training a CNN keyword spotter. The set is "1920 recorded keywords (40 keyword types, 34 minutes of speech)".
- **Key result:** the DTW-supervised CNN "substantially outperforms a CNN classifier trained only on the keywords, improving the area under the ROC curve from 0.54 to 0.64."
- **Caution:** the target context is Uganda (Luganda, Acholi and other regional languages), but the experiments themselves use **South African English** radio news, where transcriptions exist to measure performance. We cite it for the motivation and the low-resource regime, not as evidence about African-language audio.
- **How it informs our project:**
  - It shows the real humanitarian demand for small-vocabulary keyword spotting in African languages.
  - Their data budget (34 minutes of keywords) is similar in scale to ours: 4,200 clips, each word under 1 s.
  - It also shows how scarce labelled African-language keyword audio is. That makes the Swahili data we have valuable.

### doumbouya2021using: Radio archives for low-resource speech recognition (Doumbouya, 2021)
- **Task / data:** speech recognition for West African languages, towards a voice assistant for users who cannot read.
  - The "West African Radio Corpus" has 142 hours of audio in more than 10 languages, from Guinean radio stations.
  - A labelled virtual-assistant corpus has 10K clips in four languages: French, Maninka, Susu and Pular.
- **Method:** self-supervised speech representation learning (a wav2vec model, "West African wav2vec") on noisy, unlabelled radio archives.
- **Key result:**
  - The pretrained encoder performs similarly to the baseline on multilingual speech recognition, and significantly outperforms it on West African language identification.
  - The paper shares "the first-ever speech recognition models for Maninka, Pular and Susu".
- **How it informs our project:**
  - It grounds our real-world motivation: voice interfaces for low-literacy users in African languages, built around small vocabularies such as digits and commands.
  - It supports self-supervised pretraining on unlabelled audio as the route for low-resource languages, which is the idea behind A5 (XLS-R).

## M4: transformers and self-supervised speech models
*To be written by M4.*
