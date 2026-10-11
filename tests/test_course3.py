"""Course 3 — clustering, anomaly detection, recommenders, PCA, reinforcement learning."""

import numpy as np
import pytest
from scipy.stats import norm
from sklearn.cluster import KMeans as SkKMeans
from sklearn.decomposition import PCA as SkPCA

from mlkit import anomaly as an, datasets, kmeans as km, metrics, pca, recommender as rec, rl


# ------------------------------------------------------------------------ K-means


def test_find_closest_centroids_picks_the_nearest():
    X = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.0]])
    centroids = np.array([[0.0, 0.0], [10.0, 10.0]])
    assert np.array_equal(km.find_closest_centroids(X, centroids), [0, 1, 0])


def test_compute_centroids_is_the_mean():
    X = np.array([[0.0, 0.0], [2.0, 4.0], [10.0, 10.0]])
    idx = np.array([0, 0, 1])
    C = km.compute_centroids(X, idx, 2)
    assert np.allclose(C[0], [1.0, 2.0])
    assert np.allclose(C[1], [10.0, 10.0])


def test_compute_centroids_reinitialises_empty_clusters():
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    C = km.compute_centroids(X, np.array([0, 0]), 3)
    assert C.shape == (3, 2)
    assert np.all(np.isfinite(C))


def test_distortion_never_increases():
    """A K-means run whose J goes up has a bug — this is the standard self-check."""
    X, _, _ = datasets.blobs(m=300, k=3, seed=2)
    _, _, _, history = km.run_kmeans(X, K=3, n_init=1, seed=0)
    assert np.all(np.diff(history) <= 1e-9)


def test_kmeans_distortion_matches_sklearn():
    X, _, _ = datasets.blobs(m=400, k=3, spread=0.6, seed=1)
    mine = km.KMeans(K=3, n_init=20, seed=0).fit(X)
    sk = SkKMeans(n_clusters=3, n_init=20, random_state=0).fit(X)
    # sklearn's inertia_ is the SUM; ours is the MEAN.
    assert mine.inertia_ == pytest.approx(sk.inertia_ / len(X), rel=1e-6)


def test_kmeans_recovers_well_separated_centres():
    X, _, centres = datasets.blobs(m=400, k=3, spread=0.5, seed=1)
    model = km.KMeans(K=3, n_init=20, seed=0).fit(X)
    found = model.centroids[np.argsort(model.centroids[:, 0])]
    truth = centres[np.argsort(centres[:, 0])]
    assert np.abs(found - truth).max() < 0.3


def test_more_restarts_never_give_a_worse_solution():
    X, _, _ = datasets.blobs(m=200, k=4, spread=1.2, seed=7)
    j_one = km.run_kmeans(X, K=4, n_init=1, seed=3)[2]
    j_many = km.run_kmeans(X, K=4, n_init=30, seed=3)[2]
    assert j_many <= j_one + 1e-12


def test_elbow_curve_decreases_monotonically_with_k():
    """Exactly why K cannot be chosen by minimising J."""
    X, _, _ = datasets.blobs(m=300, k=3, seed=4)
    _, distortions = km.elbow_curve(X, range(1, 7), n_init=5, seed=0)
    assert np.all(np.diff(distortions) <= 1e-9)


def test_kmeans_rejects_k_larger_than_m():
    with pytest.raises(ValueError):
        km.kmeans_init_centroids(np.zeros((3, 2)), K=5)


def test_kmeans_raises_before_fit():
    with pytest.raises(RuntimeError):
        km.KMeans(K=2).predict(np.zeros((2, 2)))


# -------------------------------------------------------------- anomaly detection


def test_estimate_gaussian_recovers_known_parameters():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=[3.0, -1.0], scale=[2.0, 0.5], size=(20000, 2))
    mu, var = an.estimate_gaussian(X)
    assert np.allclose(mu, [3.0, -1.0], atol=0.05)
    assert np.allclose(np.sqrt(var), [2.0, 0.5], atol=0.05)


def test_gaussian_density_matches_scipy():
    X, y = datasets.anomaly_data(200, 10, seed=1)
    mu, var = an.estimate_gaussian(X[y == 0])
    expected = norm.pdf(X[:, 0], mu[0], np.sqrt(var[0])) * norm.pdf(X[:, 1], mu[1], np.sqrt(var[1]))
    assert np.allclose(an.gaussian_density(X, mu, var), expected, atol=1e-15)


def test_gaussian_density_survives_a_constant_feature():
    X = np.column_stack([np.ones(50), np.arange(50.0)])
    mu, var = an.estimate_gaussian(X)
    assert np.all(np.isfinite(an.gaussian_density(X, mu, var)))


def test_multivariate_density_matches_independent_when_uncorrelated():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(4000, 2)) * [2.0, 0.5]
    mu, var = an.estimate_gaussian(X)
    Sigma = np.diag(var)
    assert np.allclose(
        an.multivariate_gaussian_density(X[:20], mu, Sigma),
        an.gaussian_density(X[:20], mu, var),
        rtol=1e-9,
    )


def test_multivariate_rejects_singular_covariance():
    with pytest.raises(np.linalg.LinAlgError):
        an.multivariate_gaussian_density(np.zeros((3, 2)), np.zeros(2), np.zeros((2, 2)))


def test_detector_finds_planted_anomalies():
    X, y = datasets.anomaly_data(400, 20, seed=2)
    det = an.AnomalyDetector().fit(X[y == 0][:300]).tune(X, y)
    assert det.cv_f1_ > 0.7
    assert metrics.recall(y, det.predict(X)) > 0.7


def test_select_threshold_optimises_f1_not_accuracy():
    """"Flag nothing" scores 98% accuracy but F1 = 0; the tuner must not choose it."""
    X, y = datasets.anomaly_data(400, 8, seed=5)
    det = an.AnomalyDetector().fit(X[y == 0][:300]).tune(X, y)
    assert det.predict(X).sum() > 0
    assert metrics.accuracy(y, np.zeros_like(y)) > 0.95   # the trap
    assert metrics.f1_score(y, np.zeros_like(y)) == 0.0


def test_detector_raises_before_fit():
    with pytest.raises(RuntimeError):
        an.AnomalyDetector().density(np.zeros((2, 2)))


# ---------------------------------------------------------- collaborative filtering


def _cofi_numeric_gradient(X, W, b, Y, R, lam, eps=1e-6):
    worst = 0.0
    grads = rec.cofi_gradient(X, W, b, Y, R, lam)
    for arr, g in zip((X, W, b), grads):
        it = np.nditer(arr, flags=["multi_index"])
        while not it.finished:
            ix = it.multi_index
            original = arr[ix]
            arr[ix] = original + eps
            j_plus = rec.cofi_cost(X, W, b, Y, R, lam)
            arr[ix] = original - eps
            j_minus = rec.cofi_cost(X, W, b, Y, R, lam)
            arr[ix] = original
            worst = max(worst, abs((j_plus - j_minus) / (2 * eps) - g[ix]))
            it.iternext()
    return worst


def test_cofi_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 3))
    W = rng.normal(size=(6, 3))
    b = rng.normal(size=(1, 6))
    Y = rng.normal(size=(8, 6)) * 2 + 3
    R = (rng.random((8, 6)) < 0.6).astype(float)
    assert _cofi_numeric_gradient(X, W, b, Y, R, 1.5) < 1e-6


def test_unrated_entries_contribute_nothing():
    """Multiplying the error by R is what makes missing data harmless."""
    rng = np.random.default_rng(1)
    X = rng.normal(size=(5, 2))
    W = rng.normal(size=(4, 2))
    b = np.zeros((1, 4))
    R = (rng.random((5, 4)) < 0.5).astype(float)
    Y = rng.normal(size=(5, 4))
    Y_altered = Y.copy()
    Y_altered[R == 0] += 1000.0                # garbage in the unrated cells
    assert rec.cofi_cost(X, W, b, Y, R) == pytest.approx(
        rec.cofi_cost(X, W, b, Y_altered, R)
    )


def test_mean_normalisation_gives_a_new_user_the_item_means():
    """The lecture's motivating case: without this, a new user is predicted 0 stars."""
    Y, R, _ = datasets.ratings_matrix(n_users=20, n_items=30, seed=1)
    Y = np.hstack([Y, np.zeros((30, 1))])       # a user who has rated nothing
    R = np.hstack([R, np.zeros((30, 1))])
    cf = rec.CollaborativeFiltering(n_features=4, lambda_=5.0, seed=0)
    cf.fit(Y, R, iterations=400, alpha=0.05)
    new_user_prediction = cf.predict()[:, -1]
    assert np.allclose(new_user_prediction, cf.Y_mean, atol=1e-2)


def test_normalize_ratings_uses_only_rated_entries():
    Y = np.array([[5.0, 0.0], [0.0, 0.0]])
    R = np.array([[1.0, 0.0], [0.0, 0.0]])
    _, mean = rec.normalize_ratings(Y, R)
    assert mean[0] == pytest.approx(5.0)        # not 2.5
    assert mean[1] == pytest.approx(0.0)        # no ratings at all


def test_collaborative_filtering_recovers_held_out_ratings():
    Y, R, Y_full = datasets.ratings_matrix(n_users=40, n_items=60, n_latent=3, density=0.5, seed=1)
    cf = rec.CollaborativeFiltering(n_features=5, lambda_=0.5, seed=0)
    cf.fit(Y, R, iterations=800, alpha=0.05)
    held_out = R == 0
    error = metrics.rmse(Y_full[held_out], cf.predict()[held_out])
    baseline = metrics.rmse(Y_full[held_out], np.full(held_out.sum(), Y_full.mean()))
    assert error < baseline / 3


def test_cost_decreases_during_training():
    Y, R, _ = datasets.ratings_matrix(seed=2)
    cf = rec.CollaborativeFiltering(n_features=4, lambda_=1.0, seed=0).fit(Y, R, iterations=300)
    assert cf.cost_history[-1] < cf.cost_history[0] / 2


def test_similar_items_excludes_the_query_item():
    X = np.array([[0.0, 0.0], [0.1, 0.0], [5.0, 5.0], [0.2, 0.0]])
    result = km.np.asarray(rec.similar_items(X, item=0, top_n=2))
    assert 0 not in result
    assert set(result.tolist()) == {1, 3}


def test_content_based_requires_matching_output_dimensions():
    with pytest.raises(ValueError):
        rec.content_based_score(np.ones((1, 4)), np.ones((1, 8)))
    assert rec.content_based_score(np.array([1.0, 2.0]), np.array([3.0, 4.0]))[0] == pytest.approx(11.0)


def test_recommender_raises_before_fit():
    with pytest.raises(RuntimeError):
        rec.CollaborativeFiltering().predict()


# ---------------------------------------------------------------------------- PCA


def test_pca_explained_variance_matches_sklearn():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 2))
    X = Z @ rng.normal(size=(2, 6)) + rng.normal(scale=0.05, size=(200, 6))
    mine = pca.PCA(2).fit(X)
    Xn = (X - X.mean(axis=0)) / X.std(axis=0)
    sk = SkPCA(2).fit(Xn)
    assert np.allclose(mine.explained_variance_ratio_, sk.explained_variance_ratio_, atol=1e-8)


def test_pca_components_are_orthonormal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 5))
    p = pca.PCA(3).fit(X)
    assert np.allclose(p.components_ @ p.components_.T, np.eye(3), atol=1e-10)


def test_pca_reconstruction_is_near_perfect_for_low_rank_data():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(200, 2))
    X = Z @ rng.normal(size=(2, 6))            # exactly rank 2, no noise
    assert pca.PCA(2).fit(X).reconstruction_error(X) < 1e-16


def test_pca_loses_information_when_components_are_dropped():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(200, 5))              # full rank, nothing to throw away for free
    err_2 = pca.PCA(2).fit(X).reconstruction_error(X)
    err_4 = pca.PCA(4).fit(X).reconstruction_error(X)
    assert err_4 < err_2
    assert pca.PCA(2).fit(X).explained_variance_ratio_.sum() < 1.0


def test_pca_raises_before_fit():
    with pytest.raises(RuntimeError):
        pca.PCA(2).transform(np.zeros((3, 4)))


# ----------------------------------------------------------- reinforcement learning


def test_value_iteration_satisfies_the_bellman_equation():
    env = rl.MarsRover(6, 100, 40)
    gamma = 0.9
    Q = rl.value_iteration(env, gamma=gamma)
    for s in range(1, env.n_states - 1):
        for a in range(env.n_actions):
            expected = sum(
                p * (env.rewards[sn] + (0.0 if env.terminal[sn] else gamma * Q[sn].max()))
                for p, sn in env.transitions(s, a)
            )
            assert Q[s, a] == pytest.approx(expected, abs=1e-8)


def test_discount_factor_changes_the_optimal_policy():
    """The lecture's central point about gamma, as an assertion.

    With gamma = 0.9 the distant reward of 100 is worth walking left for from every
    state. With gamma = 0.5 the state next to the smaller reward prefers to grab it.
    """
    env = rl.MarsRover(6, 100, 40)
    patient = rl.policy_from_q(rl.value_iteration(env, gamma=0.9))
    impatient = rl.policy_from_q(rl.value_iteration(env, gamma=0.5))
    assert np.all(patient[1:5] == 0)               # always left
    assert impatient[4] == 1                       # state next to the 40 goes right
    assert np.all(impatient[1:4] == 0)


def test_value_iteration_known_values():
    """Hand-checkable: from state 1, one step left collects 100 immediately."""
    env = rl.MarsRover(6, 100, 40)
    Q = rl.value_iteration(env, gamma=0.9)
    assert Q[1, 0] == pytest.approx(100.0)
    assert Q[2, 0] == pytest.approx(90.0)          # 0.9 * 100
    assert Q[4, 1] == pytest.approx(40.0)


def test_terminal_states_have_zero_value():
    env = rl.MarsRover(6, 100, 40)
    Q = rl.value_iteration(env, gamma=0.9)
    assert np.allclose(Q[0], 0.0)
    assert np.allclose(Q[-1], 0.0)


def test_q_learning_recovers_the_optimal_policy():
    """Learns from experience only — it is never told the transition probabilities."""
    env = rl.MarsRover(6, 100, 40)
    exact = rl.policy_from_q(rl.value_iteration(env, gamma=0.9))
    learned = rl.policy_from_q(rl.q_learning(env, episodes=5000, gamma=0.9, seed=0))
    assert np.array_equal(exact[1:5], learned[1:5])


def test_epsilon_greedy_is_greedy_at_zero_and_random_at_one():
    rng = np.random.default_rng(0)
    q = np.array([1.0, 5.0, 2.0])
    assert all(rl.epsilon_greedy(q, 0.0, rng) == 1 for _ in range(20))
    choices = {rl.epsilon_greedy(q, 1.0, rng) for _ in range(200)}
    assert choices == {0, 1, 2}


def test_stochastic_environment_transitions_sum_to_one():
    env = rl.MarsRover(6, 100, 40, misstep_prob=0.25)
    for s in range(1, 5):
        for a in (0, 1):
            assert sum(p for p, _ in env.transitions(s, a)) == pytest.approx(1.0)


def test_replay_buffer_is_circular_and_samples():
    buf = rl.ReplayBuffer(capacity=5)
    for i in range(12):
        buf.add(np.array([float(i)]), i % 2, float(i), np.array([float(i + 1)]), False)
    assert len(buf) == 5
    S, A, R, S2, D = buf.sample(3, np.random.default_rng(0))
    assert S.shape == (3, 1) and A.shape == (3,) and R.shape == (3,)
    assert S2.shape == (3, 1) and D.shape == (3,)


def test_dqn_converges_to_the_exact_q_values():
    """The full DQN loop — replay buffer, target network, epsilon decay — on a problem
    small enough that the analytic answer is known."""
    env = rl.MarsRover(6, 100, 40)
    agent = rl.DQNAgent(state_dim=6, n_actions=2, hidden=(32, 32), gamma=0.9, alpha=5e-3,
                        tau=0.05, seed=0)

    def encode(s):
        v = np.zeros(6)
        v[s] = 1.0
        return v

    rng = np.random.default_rng(0)
    epsilon = 1.0
    for _ in range(1500):
        s = int(rng.integers(1, 5))
        for _ in range(30):
            a = agent.act(encode(s), epsilon)
            s_next, r, done = env.step(s, a, rng)
            agent.buffer.add(encode(s), a, r, encode(s_next), done)
            agent.learn()
            if done:
                break
            s = s_next
        epsilon = max(0.02, epsilon * 0.995)

    Q_learned = np.stack([agent.q.forward(encode(s)[None, :])[0] for s in range(6)])
    Q_exact = rl.value_iteration(env, gamma=0.9)
    assert np.array_equal(np.argmax(Q_learned, axis=1)[1:5], rl.policy_from_q(Q_exact)[1:5])
    assert np.abs(Q_learned[1:5] - Q_exact[1:5]).max() < 5.0
