"""Course 1 — linear regression, logistic regression, preprocessing, metrics.

Every non-trivial claim is checked against either scikit-learn or an analytic result,
and every hand-derived gradient is checked against finite differences.
"""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression as SkLinear
from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.metrics import f1_score as sk_f1
from sklearn.metrics import confusion_matrix as sk_cm
from sklearn.metrics import precision_score as sk_precision
from sklearn.metrics import recall_score as sk_recall

from mlkit import datasets, linear_regression as lr, logistic_regression as lg, metrics
from mlkit import preprocessing as pp


def numeric_gradient(cost_fn, w, b, eps=1e-6):
    """Central finite differences for (w vector, b scalar)."""
    dw = np.zeros_like(w, dtype=float)
    for j in range(len(w)):
        step = np.zeros_like(w, dtype=float)
        step[j] = eps
        dw[j] = (cost_fn(w + step, b) - cost_fn(w - step, b)) / (2 * eps)
    db = (cost_fn(w, b + eps) - cost_fn(w, b - eps)) / (2 * eps)
    return dw, db


# ---------------------------------------------------------------- preprocessing


def test_zscore_produces_zero_mean_unit_std():
    X, _, _, _ = datasets.linear_data(m=120, n=4, seed=3)
    Xs = pp.ZScoreScaler().fit_transform(X)
    assert np.allclose(Xs.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Xs.std(axis=0), 1.0, atol=1e-12)


def test_scaler_reuses_train_statistics_on_new_data():
    """The cv/test set must be mapped with the TRAIN mu/sigma, not its own."""
    X_train, _, _, _ = datasets.linear_data(m=100, n=3, seed=1)
    X_test = X_train + 5.0                      # deliberately shifted
    scaler = pp.ZScoreScaler().fit(X_train)
    transformed = scaler.transform(X_test)
    # If the scaler had re-fitted, this would be ~0; it must instead show the shift.
    assert np.all(transformed.mean(axis=0) > 1.0)


def test_scaler_handles_constant_column():
    X = np.column_stack([np.ones(20), np.arange(20.0)])
    Xs = pp.ZScoreScaler().fit_transform(X)
    assert np.all(np.isfinite(Xs))


def test_scaler_inverse_roundtrip():
    X, _, _, _ = datasets.linear_data(m=50, n=3, seed=9)
    s = pp.ZScoreScaler().fit(X)
    assert np.allclose(s.inverse_transform(s.transform(X)), X)


def test_scaler_raises_before_fit():
    with pytest.raises(RuntimeError):
        pp.ZScoreScaler().transform(np.ones((3, 2)))


def test_polynomial_features_shape_and_values():
    x = np.array([[2.0], [3.0]])
    P = pp.polynomial_features(x, degree=3)
    assert P.shape == (2, 3)
    assert np.allclose(P, [[2, 4, 8], [3, 9, 27]])


def test_polynomial_features_accepts_1d_input():
    P = pp.polynomial_features(np.array([2.0, 3.0]), degree=2)
    assert P.shape == (2, 2)


# ------------------------------------------------------------- linear regression


def test_normal_equation_matches_sklearn():
    X, y, _, _ = datasets.linear_data(m=200, n=4, noise=0.5, seed=1)
    w, b = lr.normal_equation(X, y)
    sk = SkLinear().fit(X, y)
    assert np.allclose(w, sk.coef_, atol=1e-10)
    assert np.isclose(b, sk.intercept_, atol=1e-10)


def test_normal_equation_recovers_true_parameters_without_noise():
    X, _, w_true, b_true = datasets.linear_data(m=200, n=3, noise=0.0, seed=4)
    y = X @ w_true + b_true
    w, b = lr.normal_equation(X, y)
    assert np.allclose(w, w_true, atol=1e-8)
    assert np.isclose(b, b_true, atol=1e-8)


def test_normal_equation_survives_collinear_features():
    """lstsq must return a minimum-norm solution instead of raising."""
    X, y, _, _ = datasets.linear_data(m=80, n=2, seed=2)
    X = np.column_stack([X, X[:, 0]])           # exact duplicate column
    w, b = lr.normal_equation(X, y)
    assert np.all(np.isfinite(w)) and np.isfinite(b)


def test_linear_gradient_matches_finite_differences():
    X, y, _, _ = datasets.linear_data(m=60, n=4, seed=5)
    w = np.array([0.3, -0.2, 0.5, 0.1])
    b = 0.4
    for lam in (0.0, 1.5):
        dw, db = lr.compute_gradient(X, y, w, b, lam)
        ndw, ndb = numeric_gradient(lambda ww, bb: lr.compute_cost(X, y, ww, bb, lam), w, b)
        assert np.allclose(dw, ndw, atol=1e-6)
        assert np.isclose(db, ndb, atol=1e-6)


def test_linear_cost_does_not_regularise_b():
    X, y, _, _ = datasets.linear_data(m=30, n=2, seed=6)
    w = np.zeros(2)
    assert lr.compute_cost(X, y, w, 3.0, 100.0) == pytest.approx(
        lr.compute_cost(X, y, w, 3.0, 0.0)
    )


def test_gradient_descent_cost_decreases_monotonically():
    X, y, _, _ = datasets.linear_data(m=150, n=4, seed=7)
    Xs = pp.ZScoreScaler().fit_transform(X)
    _, _, hist = lr.gradient_descent(Xs, y, alpha=0.3, num_iters=500)
    assert np.all(np.diff(hist) <= 1e-12)


def test_gradient_descent_converges_to_normal_equation():
    X, y, _, _ = datasets.linear_data(m=200, n=3, noise=0.3, seed=8)
    Xs = pp.ZScoreScaler().fit_transform(X)
    w_gd, b_gd, _ = lr.gradient_descent(Xs, y, alpha=0.3, num_iters=5000)
    w_ne, b_ne = lr.normal_equation(Xs, y)
    assert np.allclose(w_gd, w_ne, atol=1e-4)
    assert np.isclose(b_gd, b_ne, atol=1e-4)


def test_regularisation_shrinks_weights():
    X, y, _, _ = datasets.linear_data(m=60, n=5, noise=2.0, seed=11)
    w0, _ = lr.normal_equation(X, y, lambda_=0.0)
    w1, _ = lr.normal_equation(X, y, lambda_=50.0)
    assert np.sum(w1**2) < np.sum(w0**2)


def test_linear_regression_class_raises_before_fit():
    with pytest.raises(RuntimeError):
        lr.LinearRegression().predict(np.ones((2, 2)))


def test_linear_regression_rejects_unknown_method():
    X, y, _, _ = datasets.linear_data(m=20, n=2, seed=0)
    with pytest.raises(ValueError):
        lr.LinearRegression().fit(X, y, method="magic")


# ----------------------------------------------------------- logistic regression


def test_sigmoid_known_values_and_overflow_safety():
    assert lg.sigmoid(0.0) == pytest.approx(0.5)
    assert lg.sigmoid(np.array([1e6]))[0] == pytest.approx(1.0)
    assert lg.sigmoid(np.array([-1e6]))[0] == pytest.approx(0.0)
    assert np.all(np.isfinite(lg.sigmoid(np.array([-1e9, 0.0, 1e9]))))


def test_logistic_gradient_matches_finite_differences():
    X, y = datasets.two_class_2d(m=120, seed=12)
    w = np.array([0.4, -0.7])
    b = 0.2
    for lam in (0.0, 2.0):
        dw, db = lg.compute_gradient(X, y, w, b, lam)
        ndw, ndb = numeric_gradient(lambda ww, bb: lg.compute_cost(X, y, ww, bb, lam), w, b)
        assert np.allclose(dw, ndw, atol=1e-6)
        assert np.isclose(db, ndb, atol=1e-6)


def test_logistic_matches_sklearn_when_unregularised():
    X, y = datasets.two_class_2d(m=400, seed=2)
    mine = lg.LogisticRegression(alpha=0.5, num_iters=8000).fit(X, y)
    sk = SkLogistic(C=1e6, max_iter=5000).fit(X, y)
    assert np.allclose(mine.w, sk.coef_[0], atol=1e-2)
    assert np.isclose(mine.b, sk.intercept_[0], atol=1e-2)


def test_logistic_cost_decreases_and_is_finite_at_extremes():
    X, y = datasets.two_class_2d(m=200, separation=8.0, seed=13)
    clf = lg.LogisticRegression(alpha=0.5, num_iters=3000).fit(X, y)
    assert np.all(np.diff(clf.cost_history) <= 1e-12)
    assert np.isfinite(clf.cost_history[-1])


def test_threshold_trades_precision_for_recall():
    X, y = datasets.two_class_2d(m=400, separation=1.2, seed=14)
    clf = lg.LogisticRegression(alpha=0.5, num_iters=3000).fit(X, y)
    proba = clf.predict_proba(X)
    p_low = metrics.precision(y, (proba >= 0.2).astype(int))
    p_high = metrics.precision(y, (proba >= 0.8).astype(int))
    r_low = metrics.recall(y, (proba >= 0.2).astype(int))
    r_high = metrics.recall(y, (proba >= 0.8).astype(int))
    assert p_high >= p_low
    assert r_high <= r_low


def test_decision_boundary_2d_lies_on_the_boundary():
    w = np.array([2.0, -3.0])
    b = 0.5
    x0 = np.array([-1.0, 0.0, 1.0])
    x1 = lg.decision_boundary_2d(w, b, x0)
    z = w[0] * x0 + w[1] * x1 + b
    assert np.allclose(z, 0.0, atol=1e-12)


def test_decision_boundary_2d_rejects_bad_input():
    with pytest.raises(ValueError):
        lg.decision_boundary_2d(np.array([1.0, 2.0, 3.0]), 0.0, np.array([0.0]))
    with pytest.raises(ValueError):
        lg.decision_boundary_2d(np.array([1.0, 0.0]), 0.0, np.array([0.0]))


# ------------------------------------------------------------------------ metrics


def test_precision_recall_f1_match_sklearn():
    rng = np.random.default_rng(0)
    y_true = (rng.random(500) < 0.05).astype(int)
    y_pred = (rng.random(500) < 0.08).astype(int)
    assert metrics.precision(y_true, y_pred) == pytest.approx(
        sk_precision(y_true, y_pred, zero_division=0)
    )
    assert metrics.recall(y_true, y_pred) == pytest.approx(
        sk_recall(y_true, y_pred, zero_division=0)
    )
    assert metrics.f1_score(y_true, y_pred) == pytest.approx(sk_f1(y_true, y_pred, zero_division=0))


def test_confusion_matrix_matches_sklearn():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 3, 200)
    y_pred = rng.integers(0, 3, 200)
    assert np.array_equal(metrics.confusion_matrix(y_true, y_pred), sk_cm(y_true, y_pred))


def test_accuracy_is_useless_on_skewed_data_but_f1_is_not():
    """The whole point of the skewed-dataset lecture, as an executable assertion."""
    y_true = np.zeros(1000, dtype=int)
    y_true[:5] = 1                                  # 0.5% positives
    y_pred = np.zeros(1000, dtype=int)              # "always predict negative"
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(0.995)
    assert metrics.f1_score(y_true, y_pred) == 0.0


def test_metrics_handle_no_predicted_positives():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.zeros(4, dtype=int)
    assert metrics.precision(y_true, y_pred) == 0.0
    assert metrics.f1_score(y_true, y_pred) == 0.0


def test_r2_of_perfect_prediction_is_one():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert metrics.r2_score(y, y) == pytest.approx(1.0)
    assert metrics.mse(y, y) == 0.0
