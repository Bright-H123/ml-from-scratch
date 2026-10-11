"""Course 3 demo: K-means + elbow, Gaussian anomaly detection, PCA, and RL.

Run:  python demos/demo_c3_unsupervised.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlkit import anomaly as an, datasets, diagnostics as dg, kmeans as km, metrics, pca, rl
from mlkit import recommender as rec

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.3))

    # --- 1. K-means ------------------------------------------------------------
    X, _, _ = datasets.blobs(m=400, k=3, spread=0.6, seed=1)
    model = km.KMeans(K=3, n_init=20, seed=0).fit(X)
    axes[0].scatter(X[:, 0], X[:, 1], c=model.labels_, s=12, cmap="viridis")
    axes[0].scatter(model.centroids[:, 0], model.centroids[:, 1], marker="X", s=200,
                    c="red", edgecolors="black")
    axes[0].set_title(f"K-means, K=3\ndistortion J = {model.inertia_:.3f}", fontsize=10)

    # --- 2. elbow --------------------------------------------------------------
    ks, distortions = km.elbow_curve(X, range(1, 9), n_init=10, seed=0)
    axes[1].plot(ks, distortions, "o-")
    axes[1].set(xlabel="K", ylabel="distortion J",
                title="Elbow method\n(J always falls — cannot minimise to pick K)")

    # --- 3. anomaly detection --------------------------------------------------
    Xa, ya = datasets.anomaly_data(400, 20, seed=2)
    det = an.AnomalyDetector().fit(Xa[ya == 0][:300]).tune(Xa, ya)
    flagged = det.predict(Xa).astype(bool)
    gx, gy = np.meshgrid(np.linspace(Xa[:, 0].min(), Xa[:, 0].max(), 200),
                         np.linspace(Xa[:, 1].min(), Xa[:, 1].max(), 200))
    p = det.density(np.column_stack([gx.ravel(), gy.ravel()])).reshape(gx.shape)
    axes[2].contour(gx, gy, p, levels=np.logspace(-8, -1, 8), linewidths=0.7, colors="grey")
    axes[2].scatter(Xa[~flagged, 0], Xa[~flagged, 1], s=10, c="tab:blue", label="normal")
    axes[2].scatter(Xa[flagged, 0], Xa[flagged, 1], s=40, facecolors="none",
                    edgecolors="red", label="flagged")
    axes[2].set_title(f"Anomaly detection\neps={det.epsilon:.2e}, cv F1={det.cv_f1_:.3f}",
                      fontsize=10)
    axes[2].legend(fontsize=8)

    # --- 4. PCA ----------------------------------------------------------------
    rng = np.random.default_rng(0)
    latent = rng.normal(size=(300, 2))
    labels = (latent[:, 0] + latent[:, 1] > 0).astype(int)
    X_high = latent @ rng.normal(size=(2, 8)) + rng.normal(scale=0.3, size=(300, 8))
    p2 = pca.PCA(2).fit(X_high)
    Z = p2.transform(X_high)
    axes[3].scatter(Z[:, 0], Z[:, 1], c=labels, s=12, cmap="coolwarm")
    axes[3].set(xlabel="PC 1", ylabel="PC 2",
                title=f"PCA 8D -> 2D\nvariance kept = {p2.explained_variance_ratio_.sum():.3f}")

    fig.tight_layout()
    path = OUT / "c3_unsupervised.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}\n")

    # --- anomaly detection vs supervised, in numbers ---------------------------
    print(f"anomaly detector: precision {metrics.precision(ya, det.predict(Xa)):.3f}  "
          f"recall {metrics.recall(ya, det.predict(Xa)):.3f}")
    print(f"'never flag anything' accuracy: {metrics.accuracy(ya, np.zeros_like(ya)):.3f} "
          f"-- and F1 {metrics.f1_score(ya, np.zeros_like(ya)):.3f}. This is why F1.\n")

    # --- collaborative filtering ----------------------------------------------
    Y, R, Y_full = datasets.ratings_matrix(n_users=40, n_items=60, n_latent=3, density=0.5, seed=1)
    cf = rec.CollaborativeFiltering(n_features=5, lambda_=0.5, seed=0).fit(Y, R, iterations=800,
                                                                          alpha=0.05)
    held = R == 0
    print(f"collaborative filtering held-out RMSE : {metrics.rmse(Y_full[held], cf.predict()[held]):.4f}")
    print(f"predict-the-global-mean baseline RMSE : "
          f"{metrics.rmse(Y_full[held], np.full(held.sum(), Y_full.mean())):.4f}")
    print(f"top 5 recommendations for user 0      : {cf.recommend(0, 5, exclude_rated=R)}\n")

    # --- reinforcement learning -----------------------------------------------
    env = rl.MarsRover(6, 100, 40)
    for gamma in (0.5, 0.9):
        Q = rl.value_iteration(env, gamma=gamma)
        policy = ["left" if a == 0 else "right" for a in rl.policy_from_q(Q)[1:5]]
        print(f"Mars rover, gamma={gamma}: policy for states 1-4 = {policy}")
    print("\nSame environment, different discount, different optimal policy.")


if __name__ == "__main__":
    main()
