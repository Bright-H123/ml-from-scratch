"""Anomaly detection with Gaussian models (Course 3, Week 1).

Fit p(x) on data assumed to be (mostly) normal, then flag anything with p(x) < eps.
The unusual part relative to the rest of the course: the model is fitted on unlabelled
data, but epsilon is chosen using a small LABELLED cross-validation set.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "estimate_gaussian",
    "gaussian_density",
    "multivariate_gaussian_density",
    "select_threshold",
    "AnomalyDetector",
]


def estimate_gaussian(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Maximum-likelihood mu and sigma^2, per feature. Returns (mu, var).

    Note this uses the 1/m normalisation, not 1/(m-1). It is the maximum-likelihood
    estimate, and it is what the course uses; with the m values typical here the
    difference is irrelevant.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    var = X.var(axis=0)          # numpy's default ddof=0 is the ML estimate
    return mu, var


def gaussian_density(X: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """p(x) = prod_j N(x_j; mu_j, var_j), the independence assumption.

    Multiplying per-feature densities assumes the features are independent. The course
    notes this assumption is usually false and the algorithm works anyway -- but it is
    exactly why feature engineering matters here: if two features are only anomalous in
    COMBINATION (high CPU load with low network traffic), no product of marginals will
    notice. The fixes are to engineer a ratio feature (cpu/traffic) or to use the
    multivariate version below.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = np.asarray(mu, dtype=float)
    var = np.asarray(var, dtype=float)
    var = np.where(var == 0, 1e-12, var)     # a constant feature would divide by zero
    coeff = 1.0 / np.sqrt(2 * np.pi * var)
    exponent = np.exp(-((X - mu) ** 2) / (2 * var))
    return np.prod(coeff * exponent, axis=1)


def multivariate_gaussian_density(X: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Full covariance version — catches correlated anomalies the product version misses.

    Cost: it needs m > n (otherwise Sigma is singular) and O(n^3) to invert. The course
    prefers the independent version plus hand-made combination features.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    mu = np.asarray(mu, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    diff = X - mu
    det = np.linalg.det(Sigma)
    if det <= 0:
        raise np.linalg.LinAlgError("covariance matrix is singular; need more samples than features")
    inv = np.linalg.inv(Sigma)
    quad = np.einsum("ij,jk,ik->i", diff, inv, diff)
    return np.exp(-0.5 * quad) / np.sqrt((2 * np.pi) ** n * det)


def select_threshold(y_cv: np.ndarray, p_cv: np.ndarray, n_steps: int = 1000) -> tuple[float, float]:
    """Pick epsilon by maximising F1 on the labelled cv set. Returns (best_eps, best_f1).

    Accuracy is useless here: with 2% anomalies, "never flag anything" scores 98%.
    F1 is what forces the threshold to actually find the anomalies.
    """
    from mlkit.metrics import f1_score

    y_cv = np.asarray(y_cv).ravel()
    p_cv = np.asarray(p_cv, dtype=float).ravel()
    best_eps, best_f1 = 0.0, 0.0
    step = (p_cv.max() - p_cv.min()) / n_steps
    if step <= 0:
        return float(p_cv.min()), 0.0
    for eps in np.arange(p_cv.min(), p_cv.max(), step):
        f1 = f1_score(y_cv, (p_cv < eps).astype(int))
        if f1 > best_f1:
            best_f1, best_eps = f1, float(eps)
    return best_eps, float(best_f1)


class AnomalyDetector:
    """Fit on unlabelled (assumed-normal) data; tune epsilon on a labelled cv set."""

    def __init__(self, multivariate: bool = False) -> None:
        self.multivariate = multivariate
        self.mu: np.ndarray | None = None
        self.var: np.ndarray | None = None
        self.Sigma: np.ndarray | None = None
        self.epsilon: float = 0.0
        self.cv_f1_: float = 0.0

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        X = np.asarray(X, dtype=float)
        self.mu, self.var = estimate_gaussian(X)
        if self.multivariate:
            self.Sigma = np.cov(X, rowvar=False, bias=True)
        return self

    def density(self, X: np.ndarray) -> np.ndarray:
        if self.mu is None:
            raise RuntimeError("AnomalyDetector.fit must be called first")
        if self.multivariate:
            return multivariate_gaussian_density(X, self.mu, self.Sigma)
        return gaussian_density(X, self.mu, self.var)

    def tune(self, X_cv: np.ndarray, y_cv: np.ndarray) -> "AnomalyDetector":
        self.epsilon, self.cv_f1_ = select_threshold(y_cv, self.density(X_cv))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """1 = anomaly."""
        return (self.density(X) < self.epsilon).astype(int)
