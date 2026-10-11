"""Course 1 demo: why feature scaling matters, and polynomial regression.

Run:  python demos/demo_c1_regression.py
Writes demos/figures/c1_regression.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlkit import datasets, linear_regression as lr, metrics, preprocessing as pp

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- 1. learning rate: too small, right, too large -------------------------
    X, y, _, _ = datasets.linear_data(m=200, n=4, noise=0.5, seed=1)
    Xs = pp.ZScoreScaler().fit_transform(X)
    for alpha in (0.001, 0.03, 0.3):
        _, _, hist = lr.gradient_descent(Xs, y, alpha=alpha, num_iters=120)
        axes[0].plot(hist, label=f"alpha = {alpha}")
    axes[0].set(xlabel="iteration", ylabel="J(w, b)", title="Learning rate (scaled features)")
    axes[0].legend()

    # --- 2. scaling: each at the largest learning rate it can tolerate ---------
    #
    # Comparing raw and scaled at the SAME alpha would be misleading in either
    # direction: a tiny alpha barely moves the scaled features, and any alpha large
    # enough to be useful on scaled features diverges immediately on raw ones. The
    # honest comparison is what each can actually achieve. Raw features cap out at
    # alpha ~ 3e-7 (1e-6 overflows); scaled features run happily at 0.3.
    Xh, yh = datasets.house_prices()
    scaled = pp.ZScoreScaler().fit_transform(Xh)
    alpha_raw, alpha_scaled = 3e-7, 0.3
    _, _, hist_raw = lr.gradient_descent(Xh, yh, alpha=alpha_raw, num_iters=500)
    _, _, hist_scaled = lr.gradient_descent(scaled, yh, alpha=alpha_scaled, num_iters=500)
    w_opt, b_opt = lr.normal_equation(scaled, yh)
    j_opt = lr.compute_cost(scaled, yh, w_opt, b_opt)
    axes[1].plot(hist_raw, label=f"raw, alpha={alpha_raw:g} (largest stable)")
    axes[1].plot(hist_scaled, label=f"scaled, alpha={alpha_scaled:g}")
    axes[1].axhline(j_opt, color="grey", ls="--", lw=1, label=f"optimum J = {j_opt:.1f}")
    axes[1].set(xlabel="iteration", ylabel="J(w, b)", yscale="log",
                title="Feature scaling buys a usable learning rate")
    axes[1].legend(fontsize=8)

    # --- 3. polynomial degree: under / right / over ----------------------------
    Xp, yp = datasets.polynomial_data(m=60, noise=3.0, seed=0)
    grid = np.linspace(Xp.min(), Xp.max(), 300).reshape(-1, 1)
    axes[2].scatter(Xp, yp, s=14, alpha=0.6, label="data (quadratic truth)")
    for degree, style in ((1, "--"), (2, "-"), (15, ":")):
        P = pp.polynomial_features(Xp, degree)
        scaler = pp.ZScoreScaler().fit(P)
        w, b = lr.normal_equation(scaler.transform(P), yp)
        pred = lr.predict(scaler.transform(pp.polynomial_features(grid, degree)), w, b)
        r2 = metrics.r2_score(yp, lr.predict(scaler.transform(P), w, b))
        axes[2].plot(grid, pred, style, label=f"degree {degree} (train R2={r2:.3f})")
    axes[2].set(xlabel="x", ylabel="y", title="Polynomial regression: under / right / over")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = OUT / "c1_regression.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")

    print("normal equation vs sklearn agreement is asserted in tests/test_course1.py\n")
    print(f"optimum J (normal equation)                    : {j_opt:.2f}")
    print(f"raw features,    alpha={alpha_raw:g}, 500 iters : J = {hist_raw[-1]:.2f}")
    print(f"scaled features, alpha={alpha_scaled:g}, 500 iters    : J = {hist_scaled[-1]:.2f}")
    print("Raw features cannot use a larger alpha: at 1e-6 the cost overflows.")


if __name__ == "__main__":
    main()
