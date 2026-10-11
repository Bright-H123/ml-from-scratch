"""Recommender systems (Course 3, Week 2).

Two families:

  collaborative filtering -- learn item features X *and* user weights W from the
      ratings alone. No content description is needed; the algorithm infers what the
      latent features are from who agrees with whom.

  content-based filtering -- both users and items come with descriptive features; two
      networks map them into a shared space and the dot product is the score.

The thing that makes collaborative filtering different from every other supervised
model in the course: X is not given, it is a PARAMETER. Gradient descent updates W, b
and X simultaneously.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "cofi_cost",
    "cofi_gradient",
    "normalize_ratings",
    "CollaborativeFiltering",
    "similar_items",
    "content_based_score",
]


def cofi_cost(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    lambda_: float = 0.0,
) -> float:
    """J = 1/2 * sum_{(i,j): r=1} (w_j . x_i + b_j - y_ij)^2 + reg(W) + reg(X).

    Shapes follow the course's lab:
        X (n_items, n_features)   item feature vectors  -- learned
        W (n_users, n_features)   user weight vectors   -- learned
        b (1, n_users)            user bias             -- learned
        Y (n_items, n_users)      ratings
        R (n_items, n_users)      1 where a rating exists

    Multiplying the error by R is the whole trick for missing data: unrated entries
    contribute exactly zero to the cost and therefore zero gradient, so the model is
    never asked to explain a rating that does not exist.
    """
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    err = (X @ W.T + b - np.asarray(Y, dtype=float)) * np.asarray(R, dtype=float)
    j = 0.5 * float(np.sum(err**2))
    j += (lambda_ / 2) * (float(np.sum(W**2)) + float(np.sum(X**2)))
    return j


def cofi_gradient(
    X: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
    Y: np.ndarray,
    R: np.ndarray,
    lambda_: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dX, dW, db)."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W, dtype=float)
    b = np.asarray(b, dtype=float).reshape(1, -1)
    err = (X @ W.T + b - np.asarray(Y, dtype=float)) * np.asarray(R, dtype=float)
    dX = err @ W + lambda_ * X
    dW = err.T @ X + lambda_ * W
    db = err.sum(axis=0, keepdims=True)      # bias is not regularised
    return dX, dW, db


def normalize_ratings(Y: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Subtract each ITEM's mean rating (row-wise), over rated entries only.

    Returns (Y_norm, Y_mean). Predictions must add Y_mean back.

    Why it matters, from the lecture: a brand-new user has rated nothing, so every
    term of their cost is zero except the regulariser, which drives w to 0 -- the model
    predicts 0 stars for everything. After mean normalisation, w = 0 predicts the item's
    AVERAGE rating instead, which is a far more sensible default. It also speeds up
    convergence for everyone else.
    """
    Y = np.asarray(Y, dtype=float)
    R = np.asarray(R, dtype=float)
    counts = R.sum(axis=1)
    counts = np.where(counts == 0, 1.0, counts)      # unrated item -> mean 0
    Y_mean = (Y * R).sum(axis=1) / counts
    Y_norm = (Y - Y_mean[:, None]) * R               # keep unrated entries at 0
    return Y_norm, Y_mean


class CollaborativeFiltering:
    """Learns X, W, b jointly with Adam."""

    def __init__(self, n_features: int = 10, lambda_: float = 1.0, seed: int = 0) -> None:
        self.n_features = n_features
        self.lambda_ = lambda_
        self.seed = seed
        self.X: np.ndarray | None = None
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.Y_mean: np.ndarray | None = None
        self.cost_history: list[float] = []

    def fit(
        self,
        Y: np.ndarray,
        R: np.ndarray,
        iterations: int = 400,
        alpha: float = 0.1,
        normalize: bool = True,
    ) -> "CollaborativeFiltering":
        Y = np.asarray(Y, dtype=float)
        R = np.asarray(R, dtype=float)
        n_items, n_users = Y.shape
        rng = np.random.default_rng(self.seed)

        if normalize:
            Y_work, self.Y_mean = normalize_ratings(Y, R)
        else:
            Y_work, self.Y_mean = Y, np.zeros(n_items)

        # Random init breaks symmetry exactly as in a neural network: with X and W both
        # zero, every gradient is zero and nothing ever moves.
        self.X = rng.normal(scale=0.1, size=(n_items, self.n_features))
        self.W = rng.normal(scale=0.1, size=(n_users, self.n_features))
        self.b = np.zeros((1, n_users))

        m_state = [np.zeros_like(p) for p in (self.X, self.W, self.b)]
        v_state = [np.zeros_like(p) for p in (self.X, self.W, self.b)]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        self.cost_history = []

        for t in range(1, iterations + 1):
            grads = cofi_gradient(self.X, self.W, self.b, Y_work, R, self.lambda_)
            params = [self.X, self.W, self.b]
            for i, (p, g) in enumerate(zip(params, grads)):
                m_state[i] = beta1 * m_state[i] + (1 - beta1) * g
                v_state[i] = beta2 * v_state[i] + (1 - beta2) * g**2
                m_hat = m_state[i] / (1 - beta1**t)
                v_hat = v_state[i] / (1 - beta2**t)
                p -= alpha * m_hat / (np.sqrt(v_hat) + eps)
            self.cost_history.append(cofi_cost(self.X, self.W, self.b, Y_work, R, self.lambda_))
        return self

    def predict(self) -> np.ndarray:
        """Full (n_items, n_users) prediction matrix, with item means added back."""
        if self.X is None:
            raise RuntimeError("fit must be called before predict")
        return self.X @ self.W.T + self.b + self.Y_mean[:, None]

    def recommend(self, user: int, top_n: int = 5, exclude_rated: np.ndarray | None = None) -> np.ndarray:
        """Indices of the top_n predicted items for one user."""
        scores = self.predict()[:, user]
        if exclude_rated is not None:
            scores = scores.copy()
            scores[np.asarray(exclude_rated)[:, user] == 1] = -np.inf
        return np.argsort(scores)[::-1][:top_n]


def similar_items(X: np.ndarray, item: int, top_n: int = 5) -> np.ndarray:
    """Items whose learned feature vectors are closest to `item`'s.

    "Find related movies" needs no extra model: squared distance in the learned feature
    space is the whole method. The item itself is excluded.
    """
    X = np.asarray(X, dtype=float)
    d2 = np.sum((X - X[item]) ** 2, axis=1)
    d2[item] = np.inf
    return np.argsort(d2)[:top_n]


def content_based_score(v_user: np.ndarray, v_item: np.ndarray) -> np.ndarray:
    """Dot product of the two towers' output vectors.

    Content-based filtering runs user features through one network and item features
    through another; the two output vectors v_u and v_m must have the SAME dimension
    (they are dotted together) even though the input feature sets usually differ in
    size and meaning.

    In production this is split into two stages:
        retrieval -- cheaply pull a few hundred plausible candidates (similar to recently
                     watched, top in genre, most popular in country). Item vectors can be
                     precomputed offline, so this is just a nearest-neighbour lookup.
        ranking   -- run the full model on that shortlist only.
    Scoring the entire catalogue for every request is what this avoids.
    """
    v_user = np.atleast_2d(np.asarray(v_user, dtype=float))
    v_item = np.atleast_2d(np.asarray(v_item, dtype=float))
    if v_user.shape[1] != v_item.shape[1]:
        raise ValueError("user and item vectors must have the same output dimension")
    return np.sum(v_user * v_item, axis=1)
