"""Small synthetic datasets mirroring the ones used in the course labs.

Everything is seeded so demos and tests are reproducible.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "house_prices",
    "linear_data",
    "blobs",
    "two_class_2d",
    "moons",
    "polynomial_data",
    "anomaly_data",
    "ratings_matrix",
    "mushroom_data",
]


def house_prices() -> tuple[np.ndarray, np.ndarray]:
    """The Week 2 house dataset: size, bedrooms, floors, age -> price (1000s).

    Feature ranges differ by two orders of magnitude, which is exactly why the
    scaling lecture uses it.
    """
    X = np.array(
        [
            [952, 2, 1, 65],
            [1244, 3, 1, 64],
            [1947, 3, 2, 17],
            [1725, 3, 2, 42],
            [1959, 3, 2, 15],
            [1314, 2, 1, 14],
            [864, 2, 1, 66],
            [1836, 3, 1, 17],
            [1026, 3, 1, 43],
            [3194, 4, 2, 87],
            [788, 2, 1, 80],
            [1200, 2, 2, 17],
            [1557, 2, 1, 18],
            [1430, 3, 1, 20],
            [1220, 2, 1, 15],
        ],
        dtype=float,
    )
    y = np.array(
        [271.5, 300.0, 509.8, 394.0, 540.0, 415.0, 230.0, 560.0, 294.0, 718.2,
         200.0, 302.0, 468.0, 374.2, 388.0],
        dtype=float,
    )
    return X, y


def linear_data(m: int = 100, n: int = 3, noise: float = 1.0, seed: int = 0):
    """y = X w_true + b_true + gaussian noise. Returns (X, y, w_true, b_true)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(m, n))
    w_true = rng.uniform(-3, 3, size=n)
    b_true = float(rng.uniform(-2, 2))
    y = X @ w_true + b_true + rng.normal(scale=noise, size=m)
    return X, y, w_true, b_true


def polynomial_data(m: int = 60, noise: float = 3.0, seed: int = 0):
    """A quadratic truth, so degree-1 underfits and degree-15 overfits.

    This is the dataset shape used for the bias/variance and learning-curve figures.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, size=m)
    y = 0.8 * x**2 - 2.0 * x + 3.0 + rng.normal(scale=noise, size=m)
    return x.reshape(-1, 1), y


def blobs(m: int = 300, k: int = 3, spread: float = 0.8, seed: int = 0):
    """k well-separated 2-D gaussian clusters. Returns (X, true_labels, centres)."""
    rng = np.random.default_rng(seed)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
    centres = np.stack([4 * np.cos(angles), 4 * np.sin(angles)], axis=1)
    labels = rng.integers(0, k, size=m)
    X = centres[labels] + rng.normal(scale=spread, size=(m, 2))
    return X, labels, centres


def two_class_2d(m: int = 200, seed: int = 0, separation: float = 2.5):
    """Two linearly separable-ish gaussian classes in 2-D."""
    rng = np.random.default_rng(seed)
    m0 = m // 2
    X0 = rng.normal(loc=[-separation / 2, -separation / 2], scale=1.0, size=(m0, 2))
    X1 = rng.normal(loc=[separation / 2, separation / 2], scale=1.0, size=(m - m0, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(m0, dtype=int), np.ones(m - m0, dtype=int)])
    perm = rng.permutation(m)
    return X[perm], y[perm]


def moons(m: int = 200, noise: float = 0.2, seed: int = 0):
    """Two interleaving half-circles: not linearly separable.

    Used to show that logistic regression needs polynomial features here while a
    small neural network or a decision tree does not.
    """
    rng = np.random.default_rng(seed)
    m0 = m // 2
    t0 = np.linspace(0, np.pi, m0)
    t1 = np.linspace(0, np.pi, m - m0)
    X0 = np.stack([np.cos(t0), np.sin(t0)], axis=1)
    X1 = np.stack([1 - np.cos(t1), 0.5 - np.sin(t1)], axis=1)
    X = np.vstack([X0, X1]) + rng.normal(scale=noise, size=(m, 2))
    y = np.concatenate([np.zeros(m0, dtype=int), np.ones(m - m0, dtype=int)])
    perm = rng.permutation(m)
    return X[perm], y[perm]


def anomaly_data(m_normal: int = 300, m_anomaly: int = 15, seed: int = 0):
    """Server metrics (throughput, latency) with a handful of planted anomalies.

    Returns (X, y) where y == 1 marks an anomaly. Deliberately skewed, matching the
    course's point that anomaly detection is for the "very few positives" regime.
    """
    rng = np.random.default_rng(seed)
    X_normal = rng.multivariate_normal([14.0, 15.0], [[2.0, 0.5], [0.5, 2.0]], size=m_normal)
    X_anom = rng.uniform(low=[2.0, 2.0], high=[26.0, 28.0], size=(m_anomaly, 2))
    X = np.vstack([X_normal, X_anom])
    y = np.concatenate([np.zeros(m_normal, dtype=int), np.ones(m_anomaly, dtype=int)])
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def ratings_matrix(n_users: int = 20, n_items: int = 30, n_latent: int = 3,
                   density: float = 0.4, seed: int = 0):
    """Generate a low-rank ratings matrix with missing entries.

    Returns (Y, R, Y_full) where R[i, j] == 1 means user j rated item i, Y holds the
    observed ratings (0 where unrated), and Y_full is the noiseless ground truth used
    to check that collaborative filtering recovers the held-out entries.
    """
    rng = np.random.default_rng(seed)
    item_feats = rng.uniform(0, 1, size=(n_items, n_latent))
    user_prefs = rng.uniform(0, 1, size=(n_users, n_latent))
    Y_full = 1.0 + 4.0 * (item_feats @ user_prefs.T) / n_latent  # roughly 1..5 stars
    R = (rng.random((n_items, n_users)) < density).astype(int)
    Y = Y_full * R
    return Y, R, Y_full


def mushroom_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The Course 2 Week 4 mushroom toy set, one-hot already applied.

    Columns: brown_cap, tapering_stalk, solitary. Label: 1 = edible.
    """
    X = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
        ],
        dtype=float,
    )
    y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1, 0], dtype=int)
    return X, y, ["brown_cap", "tapering_stalk", "solitary"]
