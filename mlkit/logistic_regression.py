"""Logistic regression (Course 1, Week 3).

    model  f(x) = sigmoid(w . x + b),  interpreted as P(y = 1 | x)
    cost   J = -1/m * sum [ y log f + (1 - y) log(1 - f) ]
    grad   dJ/dw_j = 1/m * sum (f - y) x_j        <- identical *form* to linear regression

The identical gradient is not a coincidence and not the same algorithm: f() differs,
so the numbers differ. It falls out of pairing the sigmoid with cross-entropy, where
the sigmoid derivative in the chain rule cancels the denominator from the log.

Why not squared error: with f = sigmoid(...), J becomes non-convex in w, so gradient
descent can settle into a local minimum. Cross-entropy restores convexity.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "predict_proba",
    "predict",
    "compute_cost",
    "compute_gradient",
    "gradient_descent",
    "decision_boundary_2d",
    "LogisticRegression",
]

# Below this magnitude of z, exp() would overflow/underflow float64 before sigmoid
# saturates anyway, so clipping changes no answer but removes the warning.
_Z_CLIP = 500.0


def sigmoid(z: np.ndarray | float) -> np.ndarray:
    """1 / (1 + e^-z), numerically safe for large |z|."""
    z = np.clip(np.asarray(z, dtype=float), -_Z_CLIP, _Z_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    return sigmoid(np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b)


def predict(X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5) -> np.ndarray:
    """Threshold the probability. 0.5 is only the default, not a law.

    Raising the threshold buys precision at the cost of recall (predict positive only
    when confident); lowering it does the reverse. Pick it on the cv set against
    whichever cost matters -- see metrics.f1_score.
    """
    return (predict_proba(X, w, b) >= threshold).astype(int)


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
    """Binary cross-entropy, plus optional L2 on w (never on b)."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m = X.shape[0]
    f = predict_proba(X, w, b)
    # log(0) is -inf when a confident prediction is exactly right or exactly wrong;
    # clipping keeps the cost finite without moving the optimum.
    eps = 1e-15
    f = np.clip(f, eps, 1 - eps)
    cost = float(-np.mean(y * np.log(f) + (1 - y) * np.log(1 - f)))
    w_arr = np.asarray(w, dtype=float)
    reg = float(lambda_ * np.dot(w_arr, w_arr) / (2 * m))
    return cost + reg


def compute_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0
) -> tuple[np.ndarray, float]:
    """Return (dj_dw, dj_db) — same algebraic form as linear regression."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m = X.shape[0]
    err = predict_proba(X, w, b) - y
    dj_dw = (X.T @ err) / m + (lambda_ / m) * np.asarray(w, dtype=float)
    dj_db = float(err.sum() / m)
    return dj_dw, dj_db


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray | None = None,
    b_init: float = 0.0,
    alpha: float = 0.1,
    num_iters: int = 2000,
    lambda_: float = 0.0,
    record_every: int = 1,
) -> tuple[np.ndarray, float, list[float]]:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[1]
    w = np.zeros(n) if w_init is None else np.array(w_init, dtype=float)
    b = float(b_init)

    history: list[float] = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % record_every == 0:
            history.append(compute_cost(X, y, w, b, lambda_))
    return w, b, history


def decision_boundary_2d(w: np.ndarray, b: float, x0: np.ndarray) -> np.ndarray:
    """For 2 features, the boundary w0*x0 + w1*x1 + b = 0 solved for x1.

    Useful for plotting: the boundary is where z = 0, i.e. sigmoid(z) = 0.5.
    """
    w = np.asarray(w, dtype=float)
    if w.shape[0] != 2:
        raise ValueError("decision_boundary_2d requires exactly 2 features")
    if w[1] == 0:
        raise ValueError("w[1] == 0: boundary is vertical, cannot solve for x1")
    return -(w[0] * np.asarray(x0, dtype=float) + b) / w[1]


class LogisticRegression:
    def __init__(
        self, alpha: float = 0.1, num_iters: int = 2000, lambda_: float = 0.0, threshold: float = 0.5
    ) -> None:
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.threshold = threshold
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.cost_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        self.w, self.b, self.cost_history = gradient_descent(
            X, y, alpha=self.alpha, num_iters=self.num_iters, lambda_=self.lambda_
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("LogisticRegression.fit must be called before predict_proba")
        return predict_proba(X, self.w, self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.w is None:
            raise RuntimeError("LogisticRegression.fit must be called before cost")
        return compute_cost(X, y, self.w, self.b, self.lambda_)
