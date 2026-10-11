"""Decision trees (Course 2, Week 4).

Unlike everything else in the course, trees are not trained by gradient descent --
the split structure is discrete, so there is no gradient to follow. Instead the tree
is built greedily: at each node, try every (feature, threshold) pair, keep the one
that reduces impurity the most, recurse. Greedy means locally optimal at each node,
with no guarantee of a globally optimal tree; that is accepted because finding the
optimal tree is NP-hard and the greedy one works well in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "entropy",
    "gini",
    "information_gain",
    "best_split",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
]


def entropy(y: np.ndarray) -> float:
    """H(p) = -sum_k p_k log2 p_k.

    For binary labels this is the course's H(p1) curve: 0 when the node is pure
    (all one class, no uncertainty), 1.0 at p=0.5 (maximum uncertainty, a coin flip).
    0 * log(0) is defined as 0, which is the limit.
    """
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    # + 0.0 turns the -0.0 that a pure node produces back into a plain 0.0.
    return float(-np.sum(p * np.log2(p))) + 0.0


def gini(y: np.ndarray) -> float:
    """1 - sum p_k^2. Nearly interchangeable with entropy; cheaper (no log)."""
    y = np.asarray(y).ravel()
    if y.size == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - np.sum(p**2))


def _variance(y: np.ndarray) -> float:
    """Impurity measure for regression trees: replaces entropy."""
    y = np.asarray(y, dtype=float).ravel()
    if y.size == 0:
        return 0.0
    return float(np.var(y))


def information_gain(y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray,
                     criterion: str = "entropy") -> float:
    """IG = H(parent) - (w_left * H(left) + w_right * H(right)).

    The children are weighted by their sizes, not averaged plainly. Without the
    weights, splitting off a single pure example would look as valuable as splitting
    the node cleanly in half -- large impure nodes are the expensive ones.
    """
    impurity = {"entropy": entropy, "gini": gini, "variance": _variance}[criterion]
    n = len(y_left) + len(y_right)
    if n == 0:
        return 0.0
    w_left = len(y_left) / n
    w_right = len(y_right) / n
    return impurity(y_parent) - (w_left * impurity(y_left) + w_right * impurity(y_right))


def best_split(
    X: np.ndarray,
    y: np.ndarray,
    criterion: str = "entropy",
    feature_subset: np.ndarray | None = None,
) -> tuple[int | None, float, float]:
    """Exhaustively search (feature, threshold) for the largest information gain.

    Continuous features are handled the way the course describes: sort the unique
    values and try midpoints between consecutive ones. A binary/one-hot feature is
    just the special case with a single midpoint at 0.5.

    feature_subset restricts the search -- this is the hook random forests use.

    Returns (feature_index, threshold, gain); feature_index is None if no split helps.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    n_features = X.shape[1]
    features = np.arange(n_features) if feature_subset is None else np.asarray(feature_subset)

    best_gain = 0.0
    best_feat: int | None = None
    best_thr = 0.0

    for f in features:
        values = np.unique(X[:, f])
        if values.size < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0
        for thr in thresholds:
            mask = X[:, f] <= thr
            if mask.all() or not mask.any():
                continue
            gain = information_gain(y, y[mask], y[~mask], criterion)
            if gain > best_gain:
                best_gain, best_feat, best_thr = gain, int(f), float(thr)

    return best_feat, best_thr, best_gain


@dataclass
class Node:
    feature: int | None = None
    threshold: float = 0.0
    left: "Node | None" = None
    right: "Node | None" = None
    value: float | int | None = None   # leaf prediction
    proba: np.ndarray | None = None    # leaf class distribution (classifier only)
    n_samples: int = 0
    depth: int = 0
    gain: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class _BaseTree:
    def __init__(
        self,
        max_depth: int = 4,
        min_samples_split: int = 2,
        min_impurity_decrease: float = 0.0,
        criterion: str = "entropy",
        max_features: int | str | None = None,
        seed: int = 0,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.criterion = criterion
        self.max_features = max_features
        self.seed = seed
        self.root: Node | None = None
        self.n_classes_: int = 0
        self._rng = np.random.default_rng(seed)

    # -- stopping rules ------------------------------------------------------------
    #
    # The course lists four, all of which are really "stop before the tree memorises
    # the training set": node is pure, max depth reached, gain below a threshold, too
    # few samples to split. Each is a knob controlling the bias/variance trade-off --
    # deeper tree = lower bias, higher variance.

    def _n_features_to_try(self, n_features: int) -> np.ndarray | None:
        if self.max_features is None:
            return None
        if self.max_features == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(self.max_features, int):
            k = max(1, min(self.max_features, n_features))
        else:
            raise ValueError(f"bad max_features {self.max_features!r}")
        return self._rng.choice(n_features, size=k, replace=False)

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(n_samples=len(y), depth=depth)
        self._set_leaf_value(node, y)

        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            return node

        subset = self._n_features_to_try(X.shape[1])
        feat, thr, gain = best_split(X, y, self.criterion, subset)
        if feat is None or gain <= self.min_impurity_decrease:
            return node

        mask = X[:, feat] <= thr
        node.feature, node.threshold, node.gain = feat, thr, gain
        node.left = self._build(X[mask], y[mask], depth + 1)
        node.right = self._build(X[~mask], y[~mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self._prepare(y)
        self.root = self._build(X, y, 0)
        return self

    def _descend(self, x: np.ndarray) -> Node:
        node = self.root
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.root is None:
            raise RuntimeError("fit must be called before predict")
        X = np.asarray(X, dtype=float)
        return np.array([self._descend(x).value for x in X])

    def print_tree(self, feature_names: list[str] | None = None) -> str:
        """Render the tree as indented text — trees are supposed to be readable."""
        lines: list[str] = []

        def walk(node: Node, indent: int, branch: str) -> None:
            pad = "    " * indent + branch
            if node.is_leaf:
                lines.append(f"{pad}predict {node.value}  (n={node.n_samples})")
                return
            name = feature_names[node.feature] if feature_names else f"x[{node.feature}]"
            lines.append(
                f"{pad}{name} <= {node.threshold:.4g}?  (n={node.n_samples}, gain={node.gain:.4f})"
            )
            walk(node.left, indent + 1, "yes -> ")
            walk(node.right, indent + 1, "no  -> ")

        walk(self.root, 0, "")
        return "\n".join(lines)


class DecisionTreeClassifier(_BaseTree):
    def __init__(self, max_depth: int = 4, min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0, criterion: str = "entropy",
                 max_features: int | str | None = None, seed: int = 0) -> None:
        super().__init__(max_depth, min_samples_split, min_impurity_decrease,
                         criterion, max_features, seed)

    def _prepare(self, y: np.ndarray) -> None:
        self.n_classes_ = int(np.max(y)) + 1

    def _set_leaf_value(self, node: Node, y: np.ndarray) -> None:
        counts = np.bincount(np.asarray(y).astype(int), minlength=self.n_classes_)
        node.proba = counts / counts.sum()
        node.value = int(np.argmax(counts))   # majority vote

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.vstack([self._descend(x).proba for x in X])


class DecisionTreeRegressor(_BaseTree):
    """Same algorithm, variance instead of entropy, leaf predicts the mean.

    That is the entire difference. The resulting model is piecewise constant, which
    is why a single regression tree produces visible steps rather than a smooth curve.
    """

    def __init__(self, max_depth: int = 4, min_samples_split: int = 2,
                 min_impurity_decrease: float = 0.0,
                 max_features: int | str | None = None, seed: int = 0) -> None:
        super().__init__(max_depth, min_samples_split, min_impurity_decrease,
                         "variance", max_features, seed)

    def _prepare(self, y: np.ndarray) -> None:
        self.n_classes_ = 0

    def _set_leaf_value(self, node: Node, y: np.ndarray) -> None:
        node.value = float(np.mean(np.asarray(y, dtype=float)))
