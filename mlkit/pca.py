"""Principal Component Analysis (Course 3, Week 2).

Find a small set of orthogonal axes that keep as much of the data's variance as
possible, then use the coordinates along those axes instead of the original features.

The course is careful about what PCA is FOR: visualisation, primarily -- squashing 10
or 50 features down to 2 or 3 so they can be plotted. It also mentions two historical
uses that are no longer recommended: compressing data to save storage/transmission
(cheap now) and speeding up model training (modern optimisers handle wide inputs).
PCA is NOT the tool for overfitting; regularisation is.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PCA", "pca_fit", "project", "reconstruct"]


def pca_fit(X: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (components, mu, sigma, explained_variance_ratio).

    Steps, in the order the course gives them:
      1. mean-normalise every feature (mandatory -- PCA is about variance around the
         mean, and an uncentred feature's mean would dominate the first component)
      2. optionally scale to comparable ranges (done here) so that a feature measured
         in metres does not outrank one measured in kilometres purely by unit choice
      3. take the top-k right singular vectors of X as the new axes

    SVD is used rather than eigendecomposition of X^T X: it is the numerically stable
    route, and it avoids forming the covariance matrix at all.
    """
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    Xn = (X - mu) / sigma

    # full_matrices=False keeps Vt at (min(m,n), n), which is all we need.
    _, S, Vt = np.linalg.svd(Xn, full_matrices=False)
    total_var = float(np.sum(S**2))
    ratio = (S**2 / total_var) if total_var > 0 else np.zeros_like(S)
    return Vt[:n_components], mu, sigma, ratio[:n_components]


def project(X: np.ndarray, components: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """z = U_reduce^T x — coordinates in the new, lower-dimensional basis."""
    Xn = (np.asarray(X, dtype=float) - mu) / sigma
    return Xn @ np.asarray(components, dtype=float).T


def reconstruct(Z: np.ndarray, components: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """x_approx = U_reduce z, mapped back to the original units.

    This is lossy by construction: everything perpendicular to the kept components was
    thrown away. How lossy is exactly 1 - sum(explained_variance_ratio).
    """
    Xn = np.asarray(Z, dtype=float) @ np.asarray(components, dtype=float)
    return Xn * sigma + mu


class PCA:
    def __init__(self, n_components: int = 2) -> None:
        self.n_components = n_components
        self.components_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.sigma_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "PCA":
        self.components_, self.mu_, self.sigma_, self.explained_variance_ratio_ = pca_fit(
            X, self.n_components
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA.fit must be called before transform")
        return project(X, self.components_, self.mu_, self.sigma_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("PCA.fit must be called before inverse_transform")
        return reconstruct(Z, self.components_, self.mu_, self.sigma_)

    def reconstruction_error(self, X: np.ndarray) -> float:
        """Mean squared distance between X and its reconstruction."""
        return float(np.mean(np.sum((np.asarray(X, dtype=float) - self.inverse_transform(self.transform(X))) ** 2, axis=1)))
