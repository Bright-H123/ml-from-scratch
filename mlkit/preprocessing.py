"""Feature scaling and feature engineering (Course 1, Week 2).

Why scaling matters: gradient descent takes steps of the same size alpha in every
coordinate. If one feature ranges over [0, 2000] and another over [0, 5], the cost
surface is a long thin valley and the update either oscillates across the narrow
direction or crawls along the wide one. Rescaling makes the contours round, so a
single alpha works for all coordinates.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "ZScoreScaler",
    "zscore_normalize",
    "mean_normalize",
    "minmax_normalize",
    "polynomial_features",
]


class ZScoreScaler:
    """x' = (x - mu) / sigma, fitted on the training set only.

    Fitting on train and reusing mu/sigma on cv/test is the whole point: the model
    was trained in the rescaled coordinate system, so new data must be mapped into
    the *same* system. Re-fitting on the test set leaks information and shifts the
    coordinates out from under the learned weights.
    """

    def __init__(self) -> None:
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "ZScoreScaler":
        X = np.asarray(X, dtype=float)
        self.mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        # A constant column has sigma == 0; leave it untouched instead of dividing by 0.
        self.sigma = np.where(sigma == 0, 1.0, sigma)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("ZScoreScaler.fit must be called before transform")
        return (np.asarray(X, dtype=float) - self.mu) / self.sigma

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        if self.mu is None or self.sigma is None:
            raise RuntimeError("ZScoreScaler.fit must be called before inverse_transform")
        return np.asarray(X_scaled, dtype=float) * self.sigma + self.mu


def zscore_normalize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X_norm, mu, sigma). Matches the course's `zscore_normalize_features`."""
    scaler = ZScoreScaler().fit(X)
    return scaler.transform(X), scaler.mu, scaler.sigma


def mean_normalize(X: np.ndarray) -> np.ndarray:
    """x' = (x - mu) / (max - min). Centres at 0, roughly [-0.5, 0.5]."""
    X = np.asarray(X, dtype=float)
    span = X.max(axis=0) - X.min(axis=0)
    span = np.where(span == 0, 1.0, span)
    return (X - X.mean(axis=0)) / span


def minmax_normalize(X: np.ndarray) -> np.ndarray:
    """x' = (x - min) / (max - min). Maps to [0, 1]."""
    X = np.asarray(X, dtype=float)
    lo = X.min(axis=0)
    span = X.max(axis=0) - lo
    span = np.where(span == 0, 1.0, span)
    return (X - lo) / span


def polynomial_features(X: np.ndarray, degree: int, include_bias: bool = False) -> np.ndarray:
    """Raise each column to powers 1..degree (no cross terms).

    This is the course's "polynomial regression is just linear regression on
    engineered features" trick: the model stays linear *in the parameters*, which is
    why the same gradient descent code still applies.

    Note the practical consequence: x^4 of a feature that reaches 1000 is 1e12, so
    polynomial features make scaling mandatory rather than optional.
    """
    if degree < 1:
        raise ValueError("degree must be >= 1")
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    cols = [X**p for p in range(1, degree + 1)]
    out = np.hstack(cols)
    if include_bias:
        out = np.hstack([np.ones((out.shape[0], 1)), out])
    return out
