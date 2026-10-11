"""mlkit — from-scratch NumPy implementations of every algorithm in Andrew Ng's
Machine Learning Specialization (2022, DeepLearning.AI / Stanford).

Submodules follow the course order:

    Course 1  preprocessing, linear_regression, logistic_regression
    Course 2  neural_net, diagnostics, decision_tree, ensembles, metrics
    Course 3  kmeans, anomaly, recommender, pca, rl

Import submodules explicitly, e.g. `from mlkit import linear_regression as lr`.
"""

__version__ = "0.1.0"

__all__ = [
    "anomaly",
    "datasets",
    "decision_tree",
    "diagnostics",
    "ensembles",
    "kmeans",
    "linear_regression",
    "logistic_regression",
    "metrics",
    "neural_net",
    "pca",
    "preprocessing",
    "recommender",
    "rl",
]
