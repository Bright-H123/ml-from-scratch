"""Course 2 Week 3 demo: the bias/variance toolkit, end to end.

Produces the three figures the lectures draw by hand:
  * J_train / J_cv against model complexity (polynomial degree)
  * J_train / J_cv against regularisation lambda
  * learning curves for a high-bias model and a high-variance model

Run:  python demos/demo_c2_diagnostics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlkit import datasets, diagnostics as dg, linear_regression as lr, metrics
from mlkit import preprocessing as pp

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)


def poly_fit_predict(degree, X_train, y_train, X_eval, lambda_=0.0):
    P = pp.polynomial_features(X_train, degree)
    scaler = pp.ZScoreScaler().fit(P)
    w, b = lr.normal_equation(scaler.transform(P), y_train, lambda_)
    pred_train = lr.predict(scaler.transform(P), w, b)
    pred_eval = lr.predict(scaler.transform(pp.polynomial_features(X_eval, degree)), w, b)
    return pred_train, pred_eval


def main() -> None:
    X, y = datasets.polynomial_data(m=300, noise=3.0, seed=1)
    Xtr, ytr, Xcv, ycv, Xte, yte = dg.train_cv_test_split(X, y, 0.6, 0.2, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    # --- 1. complexity sweep ---------------------------------------------------
    #
    # With m = 180 and a smooth quadratic truth, every degree from 2 upwards lands
    # within ~0.7 MSE of the others: the differences are noise, not signal. Taking a
    # bare argmin of J_cv therefore picks degree 10 by a margin of 0.07, which is not
    # a real preference. Reporting both the argmin and the simplest degree within a
    # tolerance of it is the honest presentation -- and the simpler model is the one
    # to ship, since it costs nothing in J_cv and generalises on fewer assumptions.
    degrees = list(range(1, 11))
    argmin_degree, j_train, j_cv = dg.select_by_cv(
        degrees, poly_fit_predict, Xtr, ytr, Xcv, ycv, metrics.mse
    )
    tolerance = 1.05 * j_cv.min()
    simplest_degree = degrees[int(np.argmax(j_cv <= tolerance))]
    axes[0].plot(degrees, j_train, "o-", label="J_train")
    axes[0].plot(degrees, j_cv, "s-", label="J_cv")
    axes[0].axhline(tolerance, color="grey", ls=":", lw=1, label="5% above best J_cv")
    axes[0].axvline(simplest_degree, color="grey", ls="--", lw=1)
    axes[0].set(xlabel="polynomial degree", ylabel="mean squared error", yscale="log",
                title=f"Complexity: argmin is degree {argmin_degree},\n"
                      f"simplest within 5% is degree {simplest_degree} (the truth)")
    axes[0].legend(fontsize=8)

    # --- 2. lambda sweep -------------------------------------------------------
    #
    # A degree-10 fit on all 180 points is not actually overfitting, so sweeping
    # lambda there just makes things monotonically worse and shows no U at all. The
    # U-shape is a property of the OVERFITTING regime, so this panel deliberately
    # starves the model down to 20 training points, where degree 10 really does
    # overfit and regularisation really does help.
    m_small = 20
    Xs, ys = Xtr[:m_small], ytr[:m_small]
    lambdas = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
    jt, jc = [], []
    for lam in lambdas:
        pt, pc = poly_fit_predict(10, Xs, ys, Xcv, lam)
        jt.append(metrics.mse(ys, pt))
        jc.append(metrics.mse(ycv, pc))
    best_lambda = lambdas[int(np.argmin(jc))]
    axes[1].plot(range(len(lambdas)), jt, "o-", label="J_train")
    axes[1].plot(range(len(lambdas)), jc, "s-", label="J_cv")
    axes[1].axvline(lambdas.index(best_lambda), color="grey", ls="--", lw=1)
    axes[1].set_xticks(range(len(lambdas)))
    axes[1].set_xticklabels([str(l) for l in lambdas], rotation=45, fontsize=7)
    axes[1].set(xlabel="lambda", ylabel="mean squared error", yscale="log",
                title=f"Regularisation, degree 10 on m={m_small}:\n"
                      f"J_train rises, J_cv is U-shaped (best lambda={best_lambda})")
    axes[1].legend(fontsize=8)

    # --- 3. learning curves ----------------------------------------------------
    sizes = [15, 30, 60, 100, 140, 180]
    for degree, colour in ((1, "tab:red"), (2, "tab:blue")):
        s, a, b = dg.learning_curve(
            lambda Xa, ya, Xb, d=degree: poly_fit_predict(d, Xa, ya, Xb),
            Xtr, ytr, Xcv, ycv, metrics.mse, sizes=sizes,
        )
        axes[2].plot(s, a, "-", color=colour, label=f"degree {degree} J_train")
        axes[2].plot(s, b, "--", color=colour, label=f"degree {degree} J_cv")
    axes[2].set(xlabel="training set size m", ylabel="mean squared error", yscale="log",
                title="Learning curves: degree 1 = high bias (flat, high)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    path = OUT / "c2_diagnostics.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}\n")

    # --- final honest evaluation, test set touched exactly once ----------------
    pt, pe = poly_fit_predict(simplest_degree, Xtr, ytr, Xte)
    _, pc = poly_fit_predict(simplest_degree, Xtr, ytr, Xcv)
    print(dg.bias_variance_report(metrics.mse(ytr, pt), metrics.mse(ycv, pc), baseline=9.0))
    print(f"\nJ_test (touched once, after all choices were made): {metrics.mse(yte, pe):.3f}")
    print("baseline 9.0 = the noise floor: data was generated with sigma = 3.")


if __name__ == "__main__":
    main()
