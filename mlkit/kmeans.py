"""K-means clustering (Course 3, Week 1).

Two steps, repeated:
    1. assign   each point to the nearest centroid
    2. move     each centroid to the mean of the points assigned to it

Both steps decrease the same cost, which is why the algorithm always converges.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "find_closest_centroids",
    "compute_centroids",
    "distortion",
    "kmeans_init_centroids",
    "run_kmeans",
    "KMeans",
    "elbow_curve",
]


def find_closest_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Step 1. Returns idx (m,) with the index of the nearest centroid for each point.

    Squared distance is used rather than the distance itself: argmin is the same
    either way (sqrt is monotone) and skipping the sqrt is both faster and avoids a
    pointless loss of precision.
    """
    X = np.asarray(X, dtype=float)
    centroids = np.asarray(centroids, dtype=float)
    # (m, 1, n) - (1, K, n) -> (m, K, n), summed over n gives (m, K)
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1)


def compute_centroids(X: np.ndarray, idx: np.ndarray, K: int) -> np.ndarray:
    """Step 2. Each centroid becomes the mean of its assigned points.

    An empty cluster has no mean. The course mentions two fixes: drop the cluster
    (K becomes K-1) or reinitialise it randomly. Reinitialising is used here so that
    the caller always gets back exactly K centroids.
    """
    X = np.asarray(X, dtype=float)
    idx = np.asarray(idx)
    m, n = X.shape
    centroids = np.zeros((K, n))
    for k in range(K):
        members = X[idx == k]
        if members.size == 0:
            centroids[k] = X[np.random.default_rng().integers(0, m)]
        else:
            centroids[k] = members.mean(axis=0)
    return centroids


def distortion(X: np.ndarray, idx: np.ndarray, centroids: np.ndarray) -> float:
    """J = (1/m) * sum_i ||x_i - mu_{c_i}||^2 — the cost K-means minimises.

    Useful property to check when debugging: J can never increase, not on the assign
    step (each point moves to a closer centroid) and not on the move step (the mean
    is the minimiser of squared distance). If your J ever goes up, there is a bug.
    """
    X = np.asarray(X, dtype=float)
    diff = X - np.asarray(centroids, dtype=float)[np.asarray(idx)]
    return float(np.mean(np.sum(diff**2, axis=1)))


def kmeans_init_centroids(X: np.ndarray, K: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Pick K distinct training examples as the initial centroids.

    The course is explicit that this beats picking random points in space: starting on
    actual data guarantees no centroid begins in an empty region.
    """
    rng = np.random.default_rng() if rng is None else rng
    X = np.asarray(X, dtype=float)
    if K > X.shape[0]:
        raise ValueError("K cannot exceed the number of examples")
    return X[rng.choice(X.shape[0], size=K, replace=False)].copy()


def run_kmeans(
    X: np.ndarray,
    K: int,
    max_iters: int = 100,
    n_init: int = 10,
    tol: float = 1e-10,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    """K-means with multiple random restarts. Returns (centroids, idx, J, history).

    n_init matters more than it looks. K-means only ever finds a LOCAL minimum, and a
    bad initialisation produces visibly wrong clusterings (e.g. two centroids sharing
    one blob while another blob gets split). The fix in the course is blunt and
    effective: run it 50-1000 times and keep whichever run ended with the lowest J.
    """
    X = np.asarray(X, dtype=float)
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf, [])

    for _ in range(n_init):
        centroids = kmeans_init_centroids(X, K, rng)
        history: list[float] = []
        idx = np.zeros(X.shape[0], dtype=int)
        for _ in range(max_iters):
            idx = find_closest_centroids(X, centroids)
            new_centroids = compute_centroids(X, idx, K)
            history.append(distortion(X, idx, new_centroids))
            shift = np.max(np.abs(new_centroids - centroids))
            centroids = new_centroids
            if shift < tol:
                break                      # assignments have stopped changing
        j = distortion(X, idx, centroids)
        if j < best[2]:
            best = (centroids, idx, j, history)

    return best


class KMeans:
    def __init__(self, K: int, max_iters: int = 100, n_init: int = 10, seed: int = 0) -> None:
        self.K = K
        self.max_iters = max_iters
        self.n_init = n_init
        self.seed = seed
        self.centroids: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float = np.inf
        self.history_: list[float] = []

    def fit(self, X: np.ndarray) -> "KMeans":
        self.centroids, self.labels_, self.inertia_, self.history_ = run_kmeans(
            X, self.K, self.max_iters, self.n_init, seed=self.seed
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise RuntimeError("KMeans.fit must be called before predict")
        return find_closest_centroids(X, self.centroids)


def elbow_curve(X: np.ndarray, k_values=range(1, 9), n_init: int = 10, seed: int = 0):
    """J as a function of K, for the elbow method. Returns (k_values, distortions).

    Honest caveat from the lecture: J decreases monotonically with K by construction
    (more centroids can only fit tighter), so you cannot pick K by minimising J -- that
    would always choose K = m. The elbow is a visual heuristic and on real data the
    curve is often a smooth arc with no clear elbow at all.

    The recommendation that actually works: choose K by what the clusters are FOR.
    T-shirt sizes -> is 3 (S/M/L) or 5 (XS/S/M/L/XL) better for the business? That is a
    cost/benefit question, not a distortion question.
    """
    ks = list(k_values)
    return ks, [run_kmeans(X, k, n_init=n_init, seed=seed)[2] for k in ks]
