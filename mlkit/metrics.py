"""Evaluation metrics (Course 2, Week 3 — including the skewed-dataset section).

The central lesson of the skewed-data video: on a dataset that is 99.5% negative,
"always predict 0" scores 99.5% accuracy while being useless. Precision and recall
are the pair that exposes this, and F1 collapses them back into one number so that
model selection still has a single criterion to sort by.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "mse",
    "rmse",
    "r2_score",
    "accuracy",
    "confusion_matrix",
    "precision",
    "recall",
    "f1_score",
    "classification_error",
]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error. Note: the course's J for linear regression is mse/2."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean((y_pred - y_true) ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def classification_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction misclassified — this is J_train / J_cv / J_test for classification."""
    return 1.0 - accuracy(y_true, y_pred)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> np.ndarray:
    """Rows = actual class, columns = predicted class."""
    y_true = np.asarray(y_true).ravel().astype(int)
    y_pred = np.asarray(y_pred).ravel().astype(int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FP): of everything flagged positive, how much really is.

    Undefined when nothing was flagged positive; returns 0.0 by convention, which is
    also what scikit-learn does with zero_division=0.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    predicted_positive = int(np.sum(y_pred == 1))
    return float(tp / predicted_positive) if predicted_positive else 0.0


def recall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TP / (TP + FN): of everything that really is positive, how much we caught."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    actual_positive = int(np.sum(y_true == 1))
    return float(tp / actual_positive) if actual_positive else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Harmonic mean 2PR/(P+R).

    Harmonic rather than arithmetic on purpose: the arithmetic mean of P=1.0, R=0.0
    is a respectable 0.5, while the harmonic mean is 0. F1 refuses to reward a model
    that abandons one of the two.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p + r == 0:
        return 0.0
    return float(2 * p * r / (p + r))
