"""Linear regression (Course 1, Weeks 1-2).

The whole of Course 1 Week 1 in three lines:

    model     f(x) = w . x + b
    cost      J(w, b) = 1/(2m) * sum_i (f(x_i) - y_i)^2
    learning  w := w - alpha * dJ/dw,   b := b - alpha * dJ/db   (simultaneously)

The 1/2 in the cost has no statistical meaning; it cancels the 2 that falls out of
differentiating the square, so the gradient is clean.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "predict",
    "compute_cost",
    "compute_gradient",
    "gradient_descent",
    "normal_equation",
    "LinearRegression",
]


def predict(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """f(X) = X @ w + b, shape (m,)."""
    return np.asarray(X, dtype=float) @ np.asarray(w, dtype=float) + b


def compute_cost(X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0) -> float:
    """J(w, b) = 1/(2m) * sum (f - y)^2 + lambda/(2m) * sum w_j^2.

    b is deliberately left out of the regularisation term: b shifts the whole
    prediction up or down and shrinking it does nothing to control model complexity.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m = X.shape[0]
    err = predict(X, w, b) - y
    cost = float(np.dot(err, err) / (2 * m))
    reg = float(lambda_ * np.dot(np.asarray(w, dtype=float), np.asarray(w, dtype=float)) / (2 * m))
    return cost + reg


def compute_gradient(
    X: np.ndarray, y: np.ndarray, w: np.ndarray, b: float, lambda_: float = 0.0
) -> tuple[np.ndarray, float]:
    """Return (dj_dw, dj_db).

    dj_dw_j = 1/m * sum_i (f(x_i) - y_i) * x_ij  + lambda/m * w_j
    dj_db   = 1/m * sum_i (f(x_i) - y_i)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m = X.shape[0]
    err = predict(X, w, b) - y            # (m,)
    dj_dw = (X.T @ err) / m               # (n,)
    dj_db = float(err.sum() / m)
    dj_dw = dj_dw + (lambda_ / m) * np.asarray(w, dtype=float)
    return dj_dw, dj_db


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    w_init: np.ndarray | None = None,
    b_init: float = 0.0,
    alpha: float = 0.01,
    num_iters: int = 1000,
    lambda_: float = 0.0,
    record_every: int = 1,
) -> tuple[np.ndarray, float, list[float]]:
    """Batch gradient descent. Returns (w, b, cost_history).

    Reading the cost history is the diagnostic from the "choosing the learning rate"
    video: if J ever goes up, alpha is too large; if it falls but barely, alpha is too
    small. A correct run decreases on *every* iteration.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    n = X.shape[1]
    w = np.zeros(n) if w_init is None else np.array(w_init, dtype=float)
    b = float(b_init)

    history: list[float] = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)
        # Both updates use the *old* w and b -- that is what "simultaneous update" means.
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % record_every == 0:
            history.append(compute_cost(X, y, w, b, lambda_))
    return w, b, history


def normal_equation(X: np.ndarray, y: np.ndarray, lambda_: float = 0.0) -> tuple[np.ndarray, float]:
    """Closed-form solution, no iterations and no learning rate.

    Solves the augmented system [1, X] to get b and w at once. Only exists for linear
    regression, and inverting an (n+1)x(n+1) matrix costs O(n^3), so gradient descent
    wins once n gets into the thousands.

    lstsq is used instead of an explicit inverse so that a singular X^T X (perfectly
    collinear features) yields the minimum-norm solution rather than an exception.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    m, n = X.shape
    Xb = np.hstack([np.ones((m, 1)), X])          # (m, n+1), column 0 is the intercept
    if lambda_ == 0.0:
        theta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
    else:
        reg = lambda_ * np.eye(n + 1)
        reg[0, 0] = 0.0                            # do not regularise the intercept
        theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return theta[1:], float(theta[0])


class LinearRegression:
    """Thin sklearn-shaped wrapper so demos and tests read the same way."""

    def __init__(self, alpha: float = 0.01, num_iters: int = 1000, lambda_: float = 0.0) -> None:
        self.alpha = alpha
        self.num_iters = num_iters
        self.lambda_ = lambda_
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.cost_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray, method: str = "gd") -> "LinearRegression":
        if method == "gd":
            self.w, self.b, self.cost_history = gradient_descent(
                X, y, alpha=self.alpha, num_iters=self.num_iters, lambda_=self.lambda_
            )
        elif method == "normal":
            self.w, self.b = normal_equation(X, y, self.lambda_)
            self.cost_history = []
        else:
            raise ValueError("method must be 'gd' or 'normal'")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.w is None:
            raise RuntimeError("LinearRegression.fit must be called before predict")
        return predict(X, self.w, self.b)

    def cost(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.w is None:
            raise RuntimeError("LinearRegression.fit must be called before cost")
        return compute_cost(X, y, self.w, self.b, self.lambda_)
