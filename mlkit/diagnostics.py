"""Model-selection and bias/variance diagnostics (Course 2, Week 3).

This is the most practically useful week of the specialization: it is the machinery
that tells you *which* of the many possible next steps is worth your time, instead of
guessing. The core routine is `bias_variance_report`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "train_cv_test_split",
    "BiasVarianceReport",
    "bias_variance_report",
    "learning_curve",
    "select_by_cv",
    "threshold_sweep",
]


def train_cv_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.6,
    cv_frac: float = 0.2,
    seed: int = 0,
) -> tuple[np.ndarray, ...]:
    """Split into train / cross-validation / test, default 60/20/20.

    Three splits, not two, because the moment you use a set to *choose* something
    (degree, lambda, architecture, threshold) that set's error becomes an optimistic
    estimate of generalisation -- you picked the option that happened to look best on
    it. The cv set absorbs all the choosing; the test set stays untouched so that the
    final number is still an honest estimate.

    Returns (X_train, y_train, X_cv, y_cv, X_test, y_test).
    """
    if not 0 < train_frac < 1 or not 0 <= cv_frac < 1 or train_frac + cv_frac >= 1:
        raise ValueError("train_frac and cv_frac must be positive and sum to less than 1")
    X = np.asarray(X)
    y = np.asarray(y)
    m = X.shape[0]
    idx = np.random.default_rng(seed).permutation(m)
    n_train = int(round(train_frac * m))
    n_cv = int(round(cv_frac * m))
    tr, cv, te = idx[:n_train], idx[n_train : n_train + n_cv], idx[n_train + n_cv :]
    return X[tr], y[tr], X[cv], y[cv], X[te], y[te]


@dataclass
class BiasVarianceReport:
    j_train: float
    j_cv: float
    baseline: float
    avoidable_bias: float
    variance_gap: float
    diagnosis: str
    advice: list[str]

    def __str__(self) -> str:
        lines = [
            f"baseline (human/irreducible) : {self.baseline:.4f}",
            f"J_train                      : {self.j_train:.4f}",
            f"J_cv                         : {self.j_cv:.4f}",
            f"avoidable bias (train - base): {self.avoidable_bias:.4f}",
            f"variance gap   (cv - train)  : {self.variance_gap:.4f}",
            f"diagnosis                    : {self.diagnosis}",
        ]
        lines += [f"  - {a}" for a in self.advice]
        return "\n".join(lines)


def bias_variance_report(
    j_train: float,
    j_cv: float,
    baseline: float = 0.0,
    tolerance: float = 0.02,
) -> BiasVarianceReport:
    """Turn (J_train, J_cv, baseline) into a diagnosis and a to-do list.

    The baseline matters more than it first appears. J_train = 15% sounds terrible,
    but if humans also get 14% on this data then there is almost no avoidable bias and
    the model is doing fine -- the data is simply hard. Comparing J_train to 0 instead
    of to the baseline sends you chasing bias that does not exist.

    tolerance is the gap (in the same units as the errors) below which a difference is
    treated as noise rather than signal.
    """
    avoidable_bias = j_train - baseline
    variance_gap = j_cv - j_train
    high_bias = avoidable_bias > tolerance
    high_variance = variance_gap > tolerance

    if high_bias and high_variance:
        diagnosis = "high bias AND high variance"
        advice = [
            "rare in practice; check for a bug or a train/cv distribution mismatch first",
            "get more features / bigger model (bias) AND more data / more regularisation (variance)",
        ]
    elif high_bias:
        diagnosis = "high bias (underfitting)"
        advice = [
            "add features, or polynomial features",
            "use a bigger network / deeper tree",
            "DECREASE lambda",
            "more training data will NOT help -- the learning curve has already flattened",
        ]
    elif high_variance:
        diagnosis = "high variance (overfitting)"
        advice = [
            "get more training data (this is the reliable one)",
            "use a smaller feature set",
            "INCREASE lambda",
            "a bigger model will make it worse",
        ]
    else:
        diagnosis = "looks fine (low bias, low variance)"
        advice = ["evaluate on the test set once, then stop touching it"]

    return BiasVarianceReport(
        j_train=float(j_train),
        j_cv=float(j_cv),
        baseline=float(baseline),
        avoidable_bias=float(avoidable_bias),
        variance_gap=float(variance_gap),
        diagnosis=diagnosis,
        advice=advice,
    )


def learning_curve(
    fit_predict: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    error_fn: Callable[[np.ndarray, np.ndarray], float],
    sizes: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """J_train and J_cv as a function of training-set size.

    `fit_predict(X_sub, y_sub, X_eval)` must train on the subset and return
    (pred_on_X_sub, pred_on_X_eval). Returns (sizes, j_train, j_cv).

    Reading it (the point of the whole exercise):
      * both curves flatten and meet at a HIGH error -> high bias.
        More data will not help; the curves have already converged.
      * a wide persistent gap, J_train low -> high variance.
        More data will help: the gap closes as m grows.
    """
    m = X_train.shape[0]
    if sizes is None:
        sizes = np.unique(np.linspace(max(5, m // 10), m, 8).astype(int))
    rng = np.random.default_rng(seed)
    order = rng.permutation(m)

    j_tr, j_cv = [], []
    for size in sizes:
        idx = order[:size]
        pred_sub, pred_cv = fit_predict(X_train[idx], y_train[idx], X_cv)
        j_tr.append(error_fn(y_train[idx], pred_sub))
        j_cv.append(error_fn(y_cv, pred_cv))
    return np.asarray(sizes), np.asarray(j_tr), np.asarray(j_cv)


def select_by_cv(
    candidates: Sequence,
    fit_predict: Callable[[object, np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cv: np.ndarray,
    y_cv: np.ndarray,
    error_fn: Callable[[np.ndarray, np.ndarray], float],
) -> tuple[object, np.ndarray, np.ndarray]:
    """Pick the candidate (degree, lambda, architecture, ...) with the lowest J_cv.

    Returns (best_candidate, j_train_per_candidate, j_cv_per_candidate).
    """
    j_tr, j_cv = [], []
    for c in candidates:
        pred_tr, pred_cv = fit_predict(c, X_train, y_train, X_cv)
        j_tr.append(error_fn(y_train, pred_tr))
        j_cv.append(error_fn(y_cv, pred_cv))
    j_tr = np.asarray(j_tr)
    j_cv = np.asarray(j_cv)
    return candidates[int(np.argmin(j_cv))], j_tr, j_cv


def threshold_sweep(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precision / recall / F1 across decision thresholds (skewed-data section).

    Returns (thresholds, precision, recall, f1). Choose the threshold on the CV set:
    higher threshold buys precision at the cost of recall. F1 exists so that this
    trade-off can still be reduced to a single number to sort by.
    """
    from mlkit.metrics import f1_score, precision, recall

    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    thresholds = np.asarray(thresholds, dtype=float)

    p, r, f = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        p.append(precision(y_true, y_pred))
        r.append(recall(y_true, y_pred))
        f.append(f1_score(y_true, y_pred))
    return thresholds, np.asarray(p), np.asarray(r), np.asarray(f)
