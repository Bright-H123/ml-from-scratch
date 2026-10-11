# ml_specialization

Study repo for Andrew Ng's **Machine Learning Specialization** (DeepLearning.AI / Stanford, 2022 edition,
3 courses / 10 weeks). Every algorithm from the course is implemented from scratch in NumPy and
validated against scikit-learn or an analytic ground truth.

- Notes are written in Chinese (`notes/`), code and comments in English.
- Start with [`KNOWLEDGE_MAP.md`](KNOWLEDGE_MAP.md) — the whole course on one page.

## Layout

```
KNOWLEDGE_MAP.md   Single-page map of all 3 courses
notes/             Per-week notes: intuition -> formula -> derivation -> pitfalls
mlkit/             From-scratch NumPy implementations (the reusable library)
demos/             Runnable scripts that produce figures
tests/             pytest suite, each module checked against sklearn / analytic results
```

## Environment

Verified on Python 3.14.5, numpy 2.4.5, scikit-learn 1.8.0, scipy 1.17.1, matplotlib 3.10.9.

TensorFlow is **not** required. The course uses Keras for the neural-network weeks; here the same
models are built with NumPy (`mlkit/neural_net.py`) and the equivalent Keras code is given in the
notes for reference.

```bash
pip install -r requirements.txt
```

## Run the tests

```bash
python -m pytest tests -q
```

112 tests, all passing. Every from-scratch implementation is checked against
scikit-learn where an equivalent exists, and every hand-derived gradient is checked
against central finite differences (backprop lands at ~1e-10).

## Run the demos

```bash
python demos/demo_c1_regression.py
python demos/demo_c2_diagnostics.py
python demos/demo_c2_models.py
python demos/demo_c3_unsupervised.py
```

Figures are written to `demos/figures/`.

## Module index

| Module | Course reference | Contents |
|---|---|---|
| `mlkit/datasets.py` | — | Small synthetic datasets used by demos and tests |
| `mlkit/metrics.py` | C2 W3 | MSE, accuracy, precision/recall/F1, confusion matrix |
| `mlkit/preprocessing.py` | C1 W2 | Z-score / mean / min-max normalisation, polynomial features |
| `mlkit/linear_regression.py` | C1 W1–W2 | Cost, gradient, batch gradient descent, normal equation, L2 |
| `mlkit/logistic_regression.py` | C1 W3 | Sigmoid, log loss, gradient descent, L2, decision boundary |
| `mlkit/neural_net.py` | C2 W1–W2 | Dense layers, ReLU/sigmoid/linear/softmax, backprop, Adam |
| `mlkit/diagnostics.py` | C2 W3 | Train/cv/test split, learning curves, bias-variance report |
| `mlkit/decision_tree.py` | C2 W4 | Entropy, information gain, classification + regression trees |
| `mlkit/ensembles.py` | C2 W4 | Bagging, random forest, gradient boosting |
| `mlkit/kmeans.py` | C3 W1 | K-means with multiple random restarts, distortion, elbow |
| `mlkit/anomaly.py` | C3 W1 | Gaussian and multivariate Gaussian density, epsilon by F1 |
| `mlkit/recommender.py` | C3 W2 | Collaborative filtering, mean normalisation, content-based |
| `mlkit/pca.py` | C3 W2 | SVD-based PCA, projection, reconstruction, variance explained |
| `mlkit/rl.py` | C3 W3 | MDP, Bellman value iteration, Q-learning, epsilon-greedy |
