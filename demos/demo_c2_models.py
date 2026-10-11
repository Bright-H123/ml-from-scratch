"""Course 2 demo: decision boundaries of logistic regression, a neural network,
a single decision tree, and a random forest, on the same non-linear data.

Run:  python demos/demo_c2_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlkit import datasets, decision_tree as dt, ensembles as ens, logistic_regression as lg
from mlkit import metrics, neural_net as nn, preprocessing as pp

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def boundary_grid(X, n=250):
    pad = 0.5
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n),
    )
    return xx, yy, np.column_stack([xx.ravel(), yy.ravel()])


def main() -> None:
    X_raw, y = datasets.moons(m=400, noise=0.2, seed=3)
    scaler = pp.ZScoreScaler().fit(X_raw)
    X = scaler.transform(X_raw)
    xx, yy, grid = boundary_grid(X)

    models = []

    clf = lg.LogisticRegression(alpha=0.5, num_iters=5000).fit(X, y)
    models.append(("Logistic regression", clf.predict(grid), metrics.accuracy(y, clf.predict(X))))

    net = nn.NeuralNetwork([2, 16, 8, 1], ["relu", "relu", "sigmoid"], seed=1)
    net.fit(X, y, epochs=400, alpha=0.02, batch_size=32)
    models.append(("Neural network 16-8", net.predict(grid), metrics.accuracy(y, net.predict(X))))

    tree = dt.DecisionTreeClassifier(max_depth=5).fit(X, y)
    models.append((f"Decision tree (depth 5)", tree.predict(grid), metrics.accuracy(y, tree.predict(X))))

    rf = ens.RandomForestClassifier(n_estimators=60, max_depth=6, seed=0).fit(X, y)
    models.append(("Random forest (60 trees)", rf.predict(grid), metrics.accuracy(y, rf.predict(X))))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))
    for ax, (name, zz, acc) in zip(axes, models):
        ax.contourf(xx, yy, np.asarray(zz).reshape(xx.shape), levels=[-0.5, 0.5, 1.5],
                    colors=["#cfe3f7", "#f9d8d6"], alpha=0.9)
        ax.scatter(X[y == 0, 0], X[y == 0, 1], s=12, c="tab:blue", label="class 0")
        ax.scatter(X[y == 1, 0], X[y == 1, 1], s=12, c="tab:red", label="class 1")
        ax.set_title(f"{name}\ntrain accuracy {acc:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Same data, four decision boundaries — the tree's is axis-aligned by construction",
                 fontsize=11)
    fig.tight_layout()
    path = OUT / "c2_decision_boundaries.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}\n")

    for name, _, acc in models:
        print(f"{name:28s} train accuracy {acc:.4f}")
    print(f"\nrandom forest out-of-bag accuracy (the honest one): {rf.oob_score(X, y):.4f}")

    X_m, y_m, names = datasets.mushroom_data()
    print("\nThe course's mushroom tree, learned from scratch:")
    print(dt.DecisionTreeClassifier(max_depth=2).fit(X_m, y_m).print_tree(names))


if __name__ == "__main__":
    main()
