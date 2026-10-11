"""Tree ensembles (Course 2, Week 4): bagging, random forest, gradient boosting.

Motivation from the lecture: a single decision tree is *highly sensitive to the data*
-- change one training example and the root split can change, producing a completely
different tree. Averaging many trees, each trained on a slightly different view of the
data, cancels that sensitivity out. Trees are individually high-variance, and
averaging is a variance-reduction operation.
"""

from __future__ import annotations

import numpy as np

from mlkit.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor

__all__ = [
    "bootstrap_sample",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingRegressor",
]


def bootstrap_sample(m: int, rng: np.random.Generator) -> np.ndarray:
    """Draw m indices WITH replacement — sampling with replacement is the point.

    Each bootstrap set is the same size as the original but contains roughly 63% of
    the distinct examples (1 - 1/e), some of them repeated. The ~37% left out are the
    "out-of-bag" samples, which give a free validation estimate.
    """
    return rng.integers(0, m, size=m)


class _BaseForest:
    """Bagged trees. With max_features set, this becomes a random forest.

    Plain bagging has a weakness the course calls out: even on different bootstrap
    samples, the trees keep choosing the same strong feature at the root, so they end
    up highly correlated and averaging them helps less than it should. Random forest
    fixes it by letting each split consider only a random subset of sqrt(n) features,
    forcing the trees to disagree.
    """

    def __init__(self, n_estimators: int = 50, max_depth: int = 6,
                 max_features: int | str | None = "sqrt", min_samples_split: int = 2,
                 seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees: list = []
        self.oob_indices_: list[np.ndarray] = []

    def _make_tree(self, seed: int):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        m = X.shape[0]
        rng = np.random.default_rng(self.seed)
        self.trees = []
        self.oob_indices_ = []
        for i in range(self.n_estimators):
            idx = bootstrap_sample(m, rng)
            oob = np.setdiff1d(np.arange(m), np.unique(idx))
            tree = self._make_tree(int(rng.integers(0, 2**31 - 1)))
            tree.fit(X[idx], y[idx])
            self.trees.append(tree)
            self.oob_indices_.append(oob)
        return self

    def _all_predictions(self, X: np.ndarray) -> np.ndarray:
        return np.stack([t.predict(X) for t in self.trees])   # (n_trees, m)


class RandomForestClassifier(_BaseForest):
    def __init__(self, n_estimators: int = 50, max_depth: int = 6,
                 max_features: int | str | None = "sqrt", min_samples_split: int = 2,
                 criterion: str = "entropy", seed: int = 0) -> None:
        super().__init__(n_estimators, max_depth, max_features, min_samples_split, seed)
        self.criterion = criterion
        self.n_classes_ = 0

    def _make_tree(self, seed: int):
        return DecisionTreeClassifier(
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            criterion=self.criterion, max_features=self.max_features, seed=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes_ = int(np.max(np.asarray(y))) + 1
        return super().fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Average the per-tree class distributions (soft voting)."""
        return np.mean([t.predict_proba(X) for t in self.trees], axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Majority vote across trees."""
        preds = self._all_predictions(X).astype(int)          # (n_trees, m)
        counts = np.apply_along_axis(
            lambda col: np.bincount(col, minlength=self.n_classes_), 0, preds
        )                                                      # (n_classes, m)
        return np.argmax(counts, axis=0)

    def oob_score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy using, for each example, only the trees that never saw it.

        This is a validation estimate that costs no held-out data.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        m = X.shape[0]
        votes = np.zeros((m, self.n_classes_))
        seen = np.zeros(m, dtype=bool)
        for tree, oob in zip(self.trees, self.oob_indices_):
            if oob.size == 0:
                continue
            p = tree.predict(X[oob]).astype(int)
            votes[oob, p] += 1
            seen[oob] = True
        if not seen.any():
            return float("nan")
        pred = np.argmax(votes[seen], axis=1)
        return float(np.mean(pred == y[seen]))


class RandomForestRegressor(_BaseForest):
    def _make_tree(self, seed: int):
        return DecisionTreeRegressor(
            max_depth=self.max_depth, min_samples_split=self.min_samples_split,
            max_features=self.max_features, seed=seed,
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._all_predictions(X).mean(axis=0)


class GradientBoostingRegressor:
    """Additive boosting, the idea behind XGBoost, in its simplest form.

    Difference from a random forest, which is the thing worth remembering:
      * forest    -- trees are built INDEPENDENTLY on bootstrap samples, then averaged
      * boosting  -- trees are built SEQUENTIALLY, each one fitting the residual error
                     that the trees so far still get wrong

    So boosting deliberately concentrates on the hard examples (the course calls this
    "deliberate practice"), while bagging treats all examples alike. That also makes
    boosting easier to overfit: too many rounds and it starts fitting noise. The
    learning rate shrinks each tree's contribution to slow that down.

    Real XGBoost adds second-order gradients, regularisation on leaf weights, column
    subsampling and a lot of engineering; this class keeps only the core loop.
    """

    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1,
                 max_depth: int = 3, min_samples_split: int = 2, seed: int = 0) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.seed = seed
        self.trees: list[DecisionTreeRegressor] = []
        self.f0: float = 0.0
        self.train_loss_: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostingRegressor":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        # With squared-error loss the optimal constant starting prediction is the mean.
        self.f0 = float(np.mean(y))
        pred = np.full_like(y, self.f0)
        self.trees = []
        self.train_loss_ = []
        for i in range(self.n_estimators):
            residual = y - pred            # = -dL/dpred for squared error, up to a factor
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                seed=self.seed + i,
            ).fit(X, residual)
            pred = pred + self.learning_rate * tree.predict(X)
            self.trees.append(tree)
            self.train_loss_.append(float(np.mean((y - pred) ** 2)))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        pred = np.full(X.shape[0], self.f0)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
