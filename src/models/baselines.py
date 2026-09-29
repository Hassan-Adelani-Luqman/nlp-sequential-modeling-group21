"""Classical baselines: A1 (MFCC statistics + SVM) and A2 (GMM-HMM per word).

A1 throws temporal order away: each clip becomes one vector of frame statistics,
so an anagram pair such as *tisa* / *sita* looks almost identical to it.
A2 models order explicitly with a left-to-right hidden Markov model per word,
the classic isolated-word recogniser (Rabiner, 1989). Comparing the two shows
how much order information matters before any neural model is involved.
"""
from __future__ import annotations

import logging
import os
import warnings

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.evaluate import softmax

# --------------------------------------------------------------------------- A1
A1_GRIDS = {
    "svm": {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 1e-3, 1e-2]},
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
}

# scikit-learn 1.9 deprecates SVC(probability=True) in favour of CalibratedClassifierCV.
# We keep it deliberately: on our validation split libsvm's pairwise-coupled Platt
# probabilities give log loss 0.880 (ECE 0.10), versus 1.230 for the recommended
# sigmoid replacement and 1.153 for temperature calibration, at the same accuracy
# (see notebooks/02_baselines.ipynb). requirements.txt pins scikit-learn < 1.11.
_PROBABILITY_WARNING = "The `probability` parameter was deprecated"
warnings.filterwarnings("ignore", message=_PROBABILITY_WARNING, category=FutureWarning)
# joblib worker processes (GridSearchCV n_jobs) do not inherit warning filters, but do inherit the environment.
if _PROBABILITY_WARNING not in os.environ.get("PYTHONWARNINGS", ""):
    os.environ["PYTHONWARNINGS"] = ",".join(
        filter(None, [os.environ.get("PYTHONWARNINGS"), f"ignore:{_PROBABILITY_WARNING}:FutureWarning"]))


def make_a1(kind: str = "svm", seed: int = 42) -> Pipeline:
    """Standardise the clip vector, then an RBF SVM (libsvm Platt probabilities) or logistic regression."""
    if kind == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=seed)
    elif kind == "logreg":
        clf = LogisticRegression(max_iter=5000)
    else:
        raise ValueError("kind must be 'svm' or 'logreg'")
    return Pipeline([("scale", StandardScaler()), ("clf", clf)])


def tune_a1(X, y, kind: str = "svm", seed: int = 42, n_splits: int = 5, n_jobs: int = -1) -> GridSearchCV:
    """Grid search with stratified k-fold CV on the training split, selected by log loss.

    The best configuration is refitted on the whole training split
    (``search.best_estimator_``). Validation data is never touched here.
    """
    search = GridSearchCV(
        make_a1(kind, seed), A1_GRIDS[kind], scoring="neg_log_loss",
        cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed), n_jobs=n_jobs, refit=True,
    )
    return search.fit(X, y)


# --------------------------------------------------------------------------- A2
def left_to_right_transmat(n_states: int, stay: float = 0.5) -> np.ndarray:
    """Each state either repeats or moves to the next one; the last state absorbs."""
    trans = np.zeros((n_states, n_states))
    for i in range(n_states - 1):
        trans[i, i], trans[i, i + 1] = stay, 1 - stay
    trans[-1, -1] = 1.0
    return trans


def _score_all(model, seqs) -> np.ndarray:
    """Per-frame average log-likelihood of every sequence under one word model."""
    return np.array([model.score(s) / len(s) for s in seqs])


class GMMHMMClassifier:
    """One left-to-right GMM-HMM per word; a clip is assigned to the best-scoring word model.

    * Topology: left-to-right without skips. Zero transitions stay zero under
      Baum-Welch, so the model can only move forward through the word.
    * Emissions: ``n_mix`` diagonal Gaussians per state.
    * Initialisation: flat start. Every training sequence is cut into
      ``n_states`` equal segments and state *i* is initialised from a GMM fitted
      to all *i*-th segments. This ties states to time positions from the start,
      rather than to arbitrary spectral clusters.
    * Scores: per-frame average log-likelihoods, so confidence does not depend
      on clip length. ``predict_proba`` applies a softmax with ``temperature_``
      (fit it on validation data with ``src.evaluate.fit_temperature``).
    * MAP priors (``map_priors=True``): with maximum-likelihood EM, a mixture
      component that receives no frames gets weight 0 and a 0/0 variance, and
      NaNs then spread through the whole model. This happened for 8-state models
      on small data. Small conjugate priors prevent it: a Dirichlet pseudo-count
      of 1 per mixture weight and per allowed transition, one pseudo-frame of
      variance 0.1 per dimension (on standardised features), and a tiny pull of
      each mean toward 0. They are negligible next to the thousands of frames
      each state normally sees.
    """

    def __init__(self, n_states: int = 5, n_mix: int = 2, n_iter: int = 20, min_covar: float = 1e-3,
                 map_priors: bool = True, seed: int = 42, n_jobs: int = -1):
        self.n_states = n_states
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.min_covar = min_covar
        self.map_priors = map_priors
        self.seed = seed
        self.n_jobs = n_jobs
        self.temperature_ = 1.0

    def _priors(self) -> dict:
        if not self.map_priors:
            return {}
        # hmmlearn's diagonal M-step: var = (stats + 2*covars_weight) / (n + 1 + 2*(covars_prior + 1)),
        # so covars_prior=-1 adds one pseudo-frame and covars_weight=0.05 makes its variance 0.1.
        # Transitions get one pseudo-count on the *allowed* moves only (stay / next), so a state that no
        # sequence reaches keeps a valid row while forbidden transitions stay exactly zero.
        allowed = left_to_right_transmat(self.n_states) > 0
        return {"weights_prior": 2.0, "means_prior": 0.0, "means_weight": 1e-2,
                "covars_prior": -1.0, "covars_weight": 0.05, "transmat_prior": 1.0 + allowed}

    def _fit_one(self, seqs: list[np.ndarray]):
        from hmmlearn.hmm import GMMHMM
        from sklearn.mixture import GaussianMixture

        logging.getLogger("hmmlearn").setLevel(logging.ERROR)  # silence per-iteration convergence chatter
        n, k, d = self.n_states, self.n_mix, seqs[0].shape[1]

        means, covars, weights = np.zeros((n, k, d)), np.zeros((n, k, d)), np.zeros((n, k))
        segments = [np.concatenate([np.array_split(s, n)[i] for s in seqs]) for i in range(n)]
        for i, frames in enumerate(segments):
            gm = GaussianMixture(k, covariance_type="diag", reg_covar=self.min_covar, random_state=self.seed)
            gm.fit(frames)
            means[i], covars[i], weights[i] = gm.means_, gm.covariances_, gm.weights_

        model = GMMHMM(n_components=n, n_mix=k, covariance_type="diag", n_iter=self.n_iter, tol=1e-2,
                       min_covar=self.min_covar, init_params="", params="tmcw", random_state=self.seed,
                       **self._priors())
        model.n_features = d
        model.startprob_ = np.eye(n)[0]
        model.transmat_ = left_to_right_transmat(n)
        model.means_, model.covars_, model.weights_ = means, covars, weights
        model.fit(np.concatenate(seqs), [len(s) for s in seqs])
        # hmmlearn only validates parameters when scoring, so fail here with a clear message instead.
        if not all(np.isfinite(a).all() for a in (model.transmat_, model.means_, model.covars_, model.weights_)):
            raise FloatingPointError(f"GMM-HMM ({n} states x {k} mix) degenerated to NaN parameters")
        return model

    def fit(self, seqs: list[np.ndarray], y) -> "GMMHMMClassifier":
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_one)([s for s, label in zip(seqs, y) if label == c]) for c in self.classes_
        )
        return self

    def scores(self, seqs: list[np.ndarray]) -> np.ndarray:
        """(n_clips, n_classes) per-frame average log-likelihoods - the "logits" of the HMM."""
        cols = Parallel(n_jobs=self.n_jobs)(delayed(_score_all)(m, seqs) for m in self.models_)
        return np.column_stack(cols)

    def predict_proba(self, seqs: list[np.ndarray], scores: np.ndarray | None = None) -> np.ndarray:
        return softmax(self.scores(seqs) if scores is None else scores, self.temperature_)

    def predict(self, seqs: list[np.ndarray]) -> np.ndarray:
        return self.classes_[self.scores(seqs).argmax(axis=1)]
