"""Neural networks from scratch (Course 2, Weeks 1-2).

Covers the whole of C2W1 (forward propagation, layers, vectorisation) and C2W2
(activation choice, softmax, Adam, back propagation).

Layout convention used throughout: activations are stored as (m, units), i.e. one
ROW per example. This matches the course's `Dense(a_in, W, b)` labs and Keras, and
lets `A @ W + b` broadcast the bias over examples for free.

    A_prev  (m, n_in)
    W       (n_in, n_out)
    b       (n_out,)
    Z = A_prev @ W + b      -> (m, n_out)
    A = g(Z)

Keras equivalent of the models built here is given in notes/c2w1 and notes/c2w2.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigmoid",
    "relu",
    "linear",
    "softmax",
    "dense_forward",
    "Dense",
    "NeuralNetwork",
]

_Z_CLIP = 500.0


# --------------------------------------------------------------------------------------
# Activations. Each returns (value, derivative_fn) style is avoided; instead each
# activation knows how to turn dA into dZ, because softmax+CE does not factorise.
# --------------------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(np.asarray(z, dtype=float), -_Z_CLIP, _Z_CLIP)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    """max(0, z) — the default for hidden layers.

    Why not sigmoid in hidden layers: sigmoid is flat on BOTH ends, so gradients
    vanish in two directions and learning crawls. ReLU is flat only on the left, and
    it is cheaper to compute. Why not linear: stacking linear layers collapses to a
    single linear layer (W2(W1 x) = (W2 W1) x), so the network gains nothing at all
    from depth — it would just be an expensive linear regression.
    """
    return np.maximum(0.0, np.asarray(z, dtype=float))


def linear(z: np.ndarray) -> np.ndarray:
    """Identity, a.k.a. "no activation". Used on regression output layers."""
    return np.asarray(z, dtype=float)


def softmax(z: np.ndarray) -> np.ndarray:
    """Row-wise softmax: exp(z_k) / sum_j exp(z_j).

    The max is subtracted first. This changes nothing mathematically (the constant
    cancels between numerator and denominator) but keeps exp() from overflowing when
    logits are large — the same concern behind Keras's from_logits=True.
    """
    z = np.asarray(z, dtype=float)
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


_ACTIVATIONS = {"relu": relu, "sigmoid": sigmoid, "linear": linear, "softmax": softmax}


def _activation_backward(name: str, dA: np.ndarray, Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """dJ/dZ from dJ/dA for elementwise activations."""
    if name == "relu":
        return dA * (Z > 0)
    if name == "sigmoid":
        return dA * A * (1.0 - A)
    if name == "linear":
        return dA
    raise ValueError(f"no elementwise backward for activation {name!r}")


# --------------------------------------------------------------------------------------
# The course's explicit single-layer forward pass (C2W1 "Neural network implementation
# in Python"), kept because it is the clearest statement of what a layer does.
# --------------------------------------------------------------------------------------


def dense_forward(a_in: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str = "sigmoid") -> np.ndarray:
    """One dense layer, vectorised. a_in (m, n_in) -> (m, n_out)."""
    g = _ACTIVATIONS[activation]
    return g(np.asarray(a_in, dtype=float) @ np.asarray(W, dtype=float) + np.asarray(b, dtype=float))


class Dense:
    """A fully connected layer holding W, b and the cached values backprop needs."""

    def __init__(self, n_in: int, n_out: int, activation: str = "relu", rng: np.random.Generator | None = None):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}")
        rng = np.random.default_rng() if rng is None else rng
        # He initialisation for ReLU, Xavier otherwise. Symmetry must be broken: if all
        # weights start equal, every unit in a layer computes the same thing forever.
        scale = np.sqrt(2.0 / n_in) if activation == "relu" else np.sqrt(1.0 / n_in)
        self.W = rng.normal(scale=scale, size=(n_in, n_out))
        self.b = np.zeros(n_out)
        self.activation = activation
        self.Z: np.ndarray | None = None
        self.A: np.ndarray | None = None
        self.A_prev: np.ndarray | None = None
        self.dW: np.ndarray | None = None
        self.db: np.ndarray | None = None

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        self.A_prev = np.asarray(A_prev, dtype=float)
        self.Z = self.A_prev @ self.W + self.b
        self.A = _ACTIVATIONS[self.activation](self.Z)
        return self.A

    def backward(self, dZ: np.ndarray) -> np.ndarray:
        """Given dJ/dZ for this layer, store dW/db and return dJ/dA_prev."""
        m = self.A_prev.shape[0]
        self.dW = self.A_prev.T @ dZ / m
        self.db = dZ.sum(axis=0) / m
        return dZ @ self.W.T


class NeuralNetwork:
    """Sequential dense network trained with Adam.

    loss is inferred from the output activation, mirroring the course's pairing:
        sigmoid  -> binary crossentropy
        softmax  -> sparse categorical crossentropy
        linear   -> mean squared error
    """

    def __init__(self, layer_sizes: list[int], activations: list[str], lambda_: float = 0.0,
                 seed: int | None = 0) -> None:
        if len(activations) != len(layer_sizes) - 1:
            raise ValueError("need one activation per layer, i.e. len(layer_sizes) - 1")
        rng = np.random.default_rng(seed)
        self.layers = [
            Dense(layer_sizes[i], layer_sizes[i + 1], activations[i], rng)
            for i in range(len(activations))
        ]
        self.lambda_ = lambda_
        self.output_activation = activations[-1]
        self.cost_history: list[float] = []
        # Adam state, one pair of moment estimates per parameter tensor.
        self._m: list[dict[str, np.ndarray]] = []
        self._v: list[dict[str, np.ndarray]] = []
        self._t = 0

    # -- forward / cost ------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        A = np.asarray(X, dtype=float)
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def _l2_penalty(self, m: int) -> float:
        if self.lambda_ == 0.0:
            return 0.0
        total = sum(float(np.sum(layer.W**2)) for layer in self.layers)
        return self.lambda_ * total / (2 * m)

    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        A = self.forward(X)
        y = np.asarray(y)
        m = A.shape[0]
        eps = 1e-15
        if self.output_activation == "sigmoid":
            p = np.clip(A.ravel(), eps, 1 - eps)
            yf = y.astype(float).ravel()
            cost = float(-np.mean(yf * np.log(p) + (1 - yf) * np.log(1 - p)))
        elif self.output_activation == "softmax":
            p = np.clip(A[np.arange(m), y.astype(int).ravel()], eps, 1.0)
            cost = float(-np.mean(np.log(p)))
        elif self.output_activation == "linear":
            cost = float(np.mean((A.ravel() - y.astype(float).ravel()) ** 2) / 2)
        else:
            raise ValueError(f"no loss defined for output activation {self.output_activation!r}")
        return cost + self._l2_penalty(m)

    # -- backward ------------------------------------------------------------------

    def _output_dZ(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """dJ/dZ at the output layer.

        For all three pairings below this collapses to (prediction - target), which is
        exactly the cancellation described in notes/c1w3: the activation derivative
        eats the loss derivative's denominator. It is also why `from_logits=True` is
        numerically safer in Keras -- the intermediate probabilities never need to be
        formed at full precision.
        """
        m = A.shape[0]
        y = np.asarray(y)
        if self.output_activation == "sigmoid":
            return A - y.astype(float).reshape(m, 1)
        if self.output_activation == "softmax":
            dZ = A.copy()
            dZ[np.arange(m), y.astype(int).ravel()] -= 1.0
            return dZ
        if self.output_activation == "linear":
            return A - y.astype(float).reshape(m, -1)
        raise ValueError(f"no output gradient for {self.output_activation!r}")

    def backward(self, X: np.ndarray, y: np.ndarray) -> None:
        A = self.forward(X)
        m = A.shape[0]
        dZ = self._output_dZ(A, y)
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            dA_prev = layer.backward(dZ)
            if self.lambda_:
                layer.dW += (self.lambda_ / m) * layer.W
            if i > 0:
                prev = self.layers[i - 1]
                dZ = _activation_backward(prev.activation, dA_prev, prev.Z, prev.A)

    # -- optimisation --------------------------------------------------------------

    def _init_adam(self) -> None:
        self._m = [{"W": np.zeros_like(l.W), "b": np.zeros_like(l.b)} for l in self.layers]
        self._v = [{"W": np.zeros_like(l.W), "b": np.zeros_like(l.b)} for l in self.layers]
        self._t = 0

    def _adam_step(self, alpha: float, beta1: float, beta2: float, eps: float) -> None:
        """Adam: per-parameter adaptive learning rates.

        Intuition from the course: if a parameter keeps moving the same direction,
        speed it up; if it keeps oscillating, slow it down. The first moment m is the
        running average of the gradient (direction), the second moment v the running
        average of its square (magnitude); dividing one by sqrt of the other gives a
        step that is large for consistent-but-small gradients and small for erratic
        ones. Bias correction undoes the fact that m and v start at zero.
        """
        self._t += 1
        for i, layer in enumerate(self.layers):
            for name, grad, param in (("W", layer.dW, layer.W), ("b", layer.db, layer.b)):
                self._m[i][name] = beta1 * self._m[i][name] + (1 - beta1) * grad
                self._v[i][name] = beta2 * self._v[i][name] + (1 - beta2) * grad**2
                m_hat = self._m[i][name] / (1 - beta1**self._t)
                v_hat = self._v[i][name] / (1 - beta2**self._t)
                param -= alpha * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 200,
        alpha: float = 0.01,
        batch_size: int | None = None,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        seed: int = 0,
        verbose: bool = False,
    ) -> "NeuralNetwork":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        m = X.shape[0]
        self._init_adam()
        rng = np.random.default_rng(seed)
        bs = m if batch_size is None else min(batch_size, m)

        for epoch in range(epochs):
            order = rng.permutation(m)
            for start in range(0, m, bs):
                idx = order[start : start + bs]
                self.backward(X[idx], y[idx])
                self._adam_step(alpha, beta1, beta2, eps)
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            if verbose and (epoch % max(1, epochs // 10) == 0):
                print(f"epoch {epoch:5d}  cost {cost:.6f}")
        return self

    # -- prediction ----------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        A = self.forward(X)
        if self.output_activation == "sigmoid":
            return (A.ravel() >= threshold).astype(int)
        if self.output_activation == "softmax":
            return np.argmax(A, axis=1)
        return A.ravel() if A.shape[1] == 1 else A

    # -- gradient checking ---------------------------------------------------------

    def gradient_check(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> float:
        """Max abs difference between backprop gradients and finite differences.

        Run this whenever backprop is edited. A correct implementation lands around
        1e-9; anything above ~1e-5 means there is a bug.
        """
        self.backward(X, y)
        analytic = [(l.dW.copy(), l.db.copy()) for l in self.layers]
        worst = 0.0
        for i, layer in enumerate(self.layers):
            for name, param, grad in (("W", layer.W, analytic[i][0]), ("b", layer.b, analytic[i][1])):
                it = np.nditer(param, flags=["multi_index"])
                while not it.finished:
                    ix = it.multi_index
                    original = param[ix]
                    param[ix] = original + eps
                    j_plus = self.compute_cost(X, y)
                    param[ix] = original - eps
                    j_minus = self.compute_cost(X, y)
                    param[ix] = original
                    numeric = (j_plus - j_minus) / (2 * eps)
                    worst = max(worst, abs(numeric - grad[ix]))
                    it.iternext()
        return worst
