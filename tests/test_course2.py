"""Course 2 — neural networks, diagnostics, decision trees, ensembles."""

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingRegressor as SkGB
from sklearn.tree import DecisionTreeClassifier as SkTree
from sklearn.tree import DecisionTreeRegressor as SkTreeReg

from mlkit import datasets, decision_tree as dt, diagnostics as dg, ensembles as ens
from mlkit import linear_regression as lr, metrics, neural_net as nn, preprocessing as pp


# ------------------------------------------------------------------ activations


def test_relu_and_linear():
    z = np.array([-2.0, 0.0, 3.0])
    assert np.allclose(nn.relu(z), [0.0, 0.0, 3.0])
    assert np.allclose(nn.linear(z), z)


def test_softmax_sums_to_one_and_survives_huge_logits():
    a = nn.softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.isclose(a.sum(), 1.0)
    assert np.all(np.isfinite(a))
    # Shifting all logits by a constant must not change the output.
    b = nn.softmax(np.array([[0.0, 1.0, -1.0]]))
    assert np.allclose(a, b)


def test_softmax_with_two_classes_reduces_to_sigmoid():
    z = 1.3
    two_class = nn.softmax(np.array([[z, 0.0]]))[0, 0]
    assert two_class == pytest.approx(nn.sigmoid(np.array([z]))[0])


def test_dense_forward_shape():
    a_in = np.zeros((7, 3))
    W = np.ones((3, 5))
    b = np.zeros(5)
    assert nn.dense_forward(a_in, W, b, "relu").shape == (7, 5)


def test_rejects_unknown_activation():
    with pytest.raises(ValueError):
        nn.Dense(3, 2, activation="mystery")


def test_layer_count_mismatch_is_rejected():
    with pytest.raises(ValueError):
        nn.NeuralNetwork([2, 3, 1], ["relu"])


# --------------------------------------------------------------- backpropagation


@pytest.mark.parametrize(
    "sizes,acts,lam,make_y",
    [
        ([4, 5, 3, 1], ["relu", "sigmoid", "sigmoid"], 0.0, lambda r, m: (r.random(m) < 0.5).astype(int)),
        ([4, 5, 3, 1], ["relu", "relu", "sigmoid"], 1.3, lambda r, m: (r.random(m) < 0.5).astype(int)),
        ([4, 6, 4], ["relu", "softmax"], 0.0, lambda r, m: r.integers(0, 4, m)),
        ([4, 6, 4], ["relu", "softmax"], 2.0, lambda r, m: r.integers(0, 4, m)),
        ([4, 5, 1], ["relu", "linear"], 0.7, lambda r, m: r.normal(size=m)),
    ],
)
def test_backprop_matches_finite_differences(sizes, acts, lam, make_y):
    """The single most important test in the repo: if this fails, nothing else means anything."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(12, sizes[0]))
    y = make_y(rng, 12)
    net = nn.NeuralNetwork(sizes, acts, lambda_=lam, seed=1)
    assert net.gradient_check(X, y) < 1e-7


def test_weights_are_not_initialised_to_zero():
    """All-zero init would make every unit in a layer identical forever."""
    net = nn.NeuralNetwork([4, 5, 1], ["relu", "sigmoid"], seed=0)
    assert np.abs(net.layers[0].W).max() > 0
    rows = net.layers[0].W.T
    assert not np.allclose(rows[0], rows[1])


# --------------------------------------------------------------------- training


def test_network_beats_logistic_regression_on_nonlinear_data():
    """Moons are not linearly separable; this is the reason hidden layers exist."""
    X, y = datasets.moons(m=400, noise=0.18, seed=3)
    Xs = pp.ZScoreScaler().fit_transform(X)
    net = nn.NeuralNetwork([2, 16, 8, 1], ["relu", "relu", "sigmoid"], seed=1)
    net.fit(Xs, y, epochs=400, alpha=0.02, batch_size=32)
    from mlkit import logistic_regression as lg

    lin = lg.LogisticRegression(alpha=0.5, num_iters=5000).fit(Xs, y)
    assert metrics.accuracy(y, net.predict(Xs)) > 0.95
    assert metrics.accuracy(y, net.predict(Xs)) > metrics.accuracy(y, lin.predict(Xs)) + 0.05


def test_softmax_network_learns_multiclass():
    X, y, _ = datasets.blobs(m=600, k=4, spread=0.7, seed=5)
    Xs = pp.ZScoreScaler().fit_transform(X)
    net = nn.NeuralNetwork([2, 12, 4], ["relu", "softmax"], seed=2)
    net.fit(Xs, y, epochs=300, alpha=0.03, batch_size=32)
    assert metrics.accuracy(y, net.predict(Xs)) > 0.95
    assert np.allclose(net.predict_proba(Xs).sum(axis=1), 1.0)


def test_linear_output_network_does_regression():
    X, y = datasets.polynomial_data(m=200, noise=2.0, seed=7)
    Xs = pp.ZScoreScaler().fit_transform(X)
    net = nn.NeuralNetwork([1, 24, 12, 1], ["relu", "relu", "linear"], seed=3)
    net.fit(Xs, y, epochs=600, alpha=0.02, batch_size=32)
    assert metrics.r2_score(y, net.predict(Xs)) > 0.9


def test_training_reduces_cost():
    X, y = datasets.two_class_2d(m=300, seed=4)
    net = nn.NeuralNetwork([2, 8, 1], ["relu", "sigmoid"], seed=0)
    net.fit(X, y, epochs=150, alpha=0.05)
    assert net.cost_history[-1] < net.cost_history[0] / 2


# ------------------------------------------------------------------ diagnostics


def test_split_sizes_and_disjointness():
    X = np.arange(300).reshape(-1, 1).astype(float)
    y = np.arange(300)
    Xtr, ytr, Xcv, ycv, Xte, yte = dg.train_cv_test_split(X, y, 0.6, 0.2, seed=0)
    assert (len(ytr), len(ycv), len(yte)) == (180, 60, 60)
    assert len(set(ytr) | set(ycv) | set(yte)) == 300      # a partition, nothing lost


def test_split_rejects_impossible_fractions():
    X = np.zeros((10, 2))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        dg.train_cv_test_split(X, y, 0.8, 0.3)


def test_bias_variance_diagnoses_each_case():
    assert "high variance" in dg.bias_variance_report(0.01, 0.30, 0.0).diagnosis
    assert "high bias" in dg.bias_variance_report(0.30, 0.31, 0.0).diagnosis
    assert "fine" in dg.bias_variance_report(0.01, 0.015, 0.0).diagnosis
    both = dg.bias_variance_report(0.30, 0.60, 0.0).diagnosis
    assert "high bias AND high variance" == both


def test_baseline_changes_the_diagnosis():
    """15% train error is a bias problem against 0, but not against a 14% human level."""
    assert "high bias" in dg.bias_variance_report(0.150, 0.155, baseline=0.0).diagnosis
    assert "fine" in dg.bias_variance_report(0.150, 0.155, baseline=0.140).diagnosis


def _poly_fit_predict(degree, Xa, ya, Xb):
    Pa = pp.polynomial_features(Xa, degree)
    scaler = pp.ZScoreScaler().fit(Pa)
    w, b = lr.normal_equation(scaler.transform(Pa), ya)
    pred_a = lr.predict(scaler.transform(Pa), w, b)
    pred_b = lr.predict(scaler.transform(pp.polynomial_features(Xb, degree)), w, b)
    return pred_a, pred_b


def test_cv_selects_the_true_polynomial_degree():
    """Data is generated from a quadratic; cv must not be fooled into a higher degree."""
    X, y = datasets.polynomial_data(m=300, noise=3.0, seed=1)
    Xtr, ytr, Xcv, ycv, _, _ = dg.train_cv_test_split(X, y, 0.6, 0.2, seed=0)
    best, j_train, j_cv = dg.select_by_cv(
        [1, 2, 3, 5, 8, 12], _poly_fit_predict, Xtr, ytr, Xcv, ycv, metrics.mse
    )
    assert best == 2
    # J_train must fall (or hold) as the model gets more flexible.
    assert np.all(np.diff(j_train) <= 1e-9)


def test_learning_curve_flattens_for_a_high_bias_model():
    X, y = datasets.polynomial_data(m=300, noise=3.0, seed=1)
    Xtr, ytr, Xcv, ycv, _, _ = dg.train_cv_test_split(X, y, 0.6, 0.2, seed=0)
    sizes, j_tr, j_cv = dg.learning_curve(
        lambda Xa, ya, Xb: _poly_fit_predict(1, Xa, ya, Xb),
        Xtr, ytr, Xcv, ycv, metrics.mse, sizes=[20, 60, 120, 180],
    )
    assert j_tr[-1] > j_tr[0]                 # train error RISES with more data
    assert abs(j_cv[-1] - j_tr[-1]) < 0.2 * j_tr[-1]   # curves converge, and stay high


def test_threshold_sweep_recall_is_non_increasing():
    rng = np.random.default_rng(0)
    y = (rng.random(600) < 0.1).astype(int)
    proba = np.clip(0.25 * y + rng.random(600) * 0.5, 0, 1)
    thr, prec, rec, f1 = dg.threshold_sweep(y, proba)
    assert np.all(np.diff(rec) <= 1e-12)       # raising the bar can only lose positives
    assert len(thr) == len(prec) == len(rec) == len(f1)


# ------------------------------------------------------------------------ trees


def test_entropy_endpoints():
    assert dt.entropy([0, 0, 0, 0]) == 0.0
    assert dt.entropy([1, 1, 1]) == 0.0
    assert dt.entropy([0, 0, 1, 1]) == pytest.approx(1.0)
    assert dt.entropy([0, 0, 0, 1]) == pytest.approx(0.8112781, abs=1e-6)


def test_gini_endpoints():
    assert dt.gini([1, 1, 1]) == 0.0
    assert dt.gini([0, 0, 1, 1]) == pytest.approx(0.5)


def test_information_gain_of_a_useless_split_is_zero():
    y = np.array([0, 0, 1, 1])
    assert dt.information_gain(y, y[:2], y[2:]) == pytest.approx(1.0)   # perfect split
    assert dt.information_gain(y, y[[0, 2]], y[[1, 3]]) == pytest.approx(0.0)


def test_information_gain_weights_children_by_size():
    """An unweighted average would rate these two splits equally; weighting must not."""
    y = np.array([0] * 8 + [1] * 8)
    tiny_pure = dt.information_gain(y, y[:1], y[1:])
    half_half = dt.information_gain(y, y[:8], y[8:])
    assert half_half > tiny_pure


def test_mushroom_root_split_reproduces_the_course_value():
    X, y, _ = datasets.mushroom_data()
    _, _, gain = dt.best_split(X, y)
    assert gain == pytest.approx(0.278, abs=1e-3)


def test_tree_classifier_matches_sklearn_accuracy():
    X, y = datasets.moons(m=400, noise=0.2, seed=1)
    mine = dt.DecisionTreeClassifier(max_depth=5, criterion="entropy").fit(X, y)
    sk = SkTree(max_depth=5, criterion="entropy", random_state=0).fit(X, y)
    assert metrics.accuracy(y, mine.predict(X)) == pytest.approx(
        metrics.accuracy(y, sk.predict(X)), abs=0.02
    )


def test_tree_regressor_matches_sklearn():
    X, y = datasets.polynomial_data(m=200, noise=2.0, seed=4)
    mine = dt.DecisionTreeRegressor(max_depth=4).fit(X, y)
    sk = SkTreeReg(max_depth=4, random_state=0).fit(X, y)
    assert metrics.r2_score(y, mine.predict(X)) == pytest.approx(
        metrics.r2_score(y, sk.predict(X)), abs=0.02
    )


def test_deeper_tree_fits_training_data_better():
    """max_depth is the bias/variance knob."""
    X, y = datasets.moons(m=300, noise=0.25, seed=6)
    shallow = dt.DecisionTreeClassifier(max_depth=1).fit(X, y)
    deep = dt.DecisionTreeClassifier(max_depth=12).fit(X, y)
    assert metrics.accuracy(y, deep.predict(X)) > metrics.accuracy(y, shallow.predict(X))


def test_tree_probabilities_are_valid():
    X, y = datasets.moons(m=200, noise=0.2, seed=2)
    tree = dt.DecisionTreeClassifier(max_depth=3).fit(X, y)
    P = tree.predict_proba(X)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P >= 0)


def test_print_tree_is_readable():
    X, y, names = datasets.mushroom_data()
    text = dt.DecisionTreeClassifier(max_depth=2).fit(X, y).print_tree(names)
    assert "predict" in text
    assert any(n in text for n in names)


def test_tree_raises_before_fit():
    with pytest.raises(RuntimeError):
        dt.DecisionTreeClassifier().predict(np.ones((2, 2)))


# -------------------------------------------------------------------- ensembles


def test_bootstrap_leaves_roughly_a_third_out_of_bag():
    rng = np.random.default_rng(0)
    m = 5000
    idx = ens.bootstrap_sample(m, rng)
    in_bag = len(np.unique(idx)) / m
    assert in_bag == pytest.approx(1 - 1 / np.e, abs=0.02)


def test_random_forest_beats_a_single_shallow_tree():
    X, y = datasets.moons(m=400, noise=0.3, seed=8)
    tree = dt.DecisionTreeClassifier(max_depth=3, seed=0).fit(X, y)
    forest = ens.RandomForestClassifier(n_estimators=40, max_depth=3, seed=0).fit(X, y)
    assert metrics.accuracy(y, forest.predict(X)) >= metrics.accuracy(y, tree.predict(X))


def test_random_forest_oob_score_is_sensible():
    X, y = datasets.moons(m=400, noise=0.2, seed=1)
    rf = ens.RandomForestClassifier(n_estimators=40, max_depth=6, seed=0).fit(X, y)
    oob = rf.oob_score(X, y)
    assert 0.85 < oob <= 1.0
    assert oob <= metrics.accuracy(y, rf.predict(X)) + 1e-9   # oob is the honest one


def test_random_forest_probabilities_are_valid():
    X, y = datasets.moons(m=200, noise=0.2, seed=3)
    rf = ens.RandomForestClassifier(n_estimators=15, max_depth=4, seed=0).fit(X, y)
    assert np.allclose(rf.predict_proba(X).sum(axis=1), 1.0)


def test_random_forest_regressor_averages_trees():
    X, y = datasets.polynomial_data(m=200, noise=2.0, seed=5)
    rf = ens.RandomForestRegressor(n_estimators=25, max_depth=5, seed=0).fit(X, y)
    assert metrics.r2_score(y, rf.predict(X)) > 0.85


def test_gradient_boosting_matches_sklearn():
    X, y = datasets.polynomial_data(m=200, noise=2.0, seed=4)
    mine = ens.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, seed=0)
    mine.fit(X, y)
    sk = SkGB(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
    assert metrics.r2_score(y, mine.predict(X)) == pytest.approx(
        metrics.r2_score(y, sk.predict(X)), abs=0.02
    )


def test_boosting_loss_decreases_every_round():
    X, y = datasets.polynomial_data(m=150, noise=2.0, seed=9)
    gb = ens.GradientBoostingRegressor(n_estimators=60, learning_rate=0.1, max_depth=2, seed=0)
    gb.fit(X, y)
    assert gb.train_loss_[-1] < gb.train_loss_[0]
    assert np.all(np.diff(gb.train_loss_) <= 1e-9)


def test_boosting_starts_from_the_mean():
    X, y = datasets.polynomial_data(m=100, noise=1.0, seed=10)
    gb = ens.GradientBoostingRegressor(n_estimators=1, learning_rate=0.0, max_depth=2).fit(X, y)
    assert gb.f0 == pytest.approx(y.mean())
    assert np.allclose(gb.predict(X), y.mean())     # lr=0 means no tree contributes
