"""Sensitivity of the Random Forest calibration to the Gaussian approximation.

Calibration and sharpness need prediction intervals, so the benchmark places all models
on a common scale by treating each predicted uncertainty as the standard deviation of a
Gaussian. For the Random Forest the predictive distribution is not parametric, so this
script checks whether that approximation distorts the comparison.

For each test point the predictions of the individual trees are treated as an empirical
predictive sample. Coverage at nominal level p is measured two ways:

    empirical : fraction of true y inside the central-p inter-quantile range of the sample
    Gaussian  : fraction of true y inside mean +/- z_p * sd, sd taken from the same sample

Miscalibration area MA = mean_p |observed(p) - p|, lower is better.

Result (5 folds): MA 0.157 +/- 0.006 empirical vs 0.101 +/- 0.006 Gaussian. The empirical
distribution is the worse calibrated of the two, so the approximation does not disadvantage
the Random Forest -- its overconfidence is intrinsic to the tree variance.

Usage
-----
    python nongaussian_calibration.py              # figure from the deposited calib_*.json
    python nongaussian_calibration.py --recompute  # refit the RF and recompute them

--recompute refits a 500-tree forest per fold and takes roughly an hour per fold; the
per-fold JSONs it writes are the ones shipped in LUDRP_results_data.zip.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm

from config import BASE_PATH, COLD_START_SUFFIX, SUFFIX, FIGURES_DIR

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})

RF_DIR = os.path.join(BASE_PATH, f"cv_rf{COLD_START_SUFFIX}{SUFFIX}")
NOMINAL = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
MAX_POINTS = 40000          # cap for the coverage computation only; the fit uses all data
SEED = 0
C_EMP, C_GAUSS = "#55A868", "#C44E52"


def coverage_curves(tree_preds, y):
    """Empirical and Gaussian coverage at each nominal level, from one per-tree sample."""
    mean, sd = tree_preds.mean(0), tree_preds.std(0)
    emp, gauss = [], []
    for p in NOMINAL:
        a = (1 - p) / 2
        lo, hi = np.quantile(tree_preds, a, axis=0), np.quantile(tree_preds, 1 - a, axis=0)
        emp.append(np.mean((y >= lo) & (y <= hi)))
        z = norm.ppf(1 - a)
        gauss.append(np.mean((y >= mean - z * sd) & (y <= mean + z * sd)))
    return np.array(emp), np.array(gauss)


def recompute_fold(fold):
    """Refit the RF at the reported configuration and return that fold's coverage."""
    from uadr.utils.data_preprocessing import get_preprocessed_features
    from uadr.models.random_forest import RandomForest

    kwargs = np.load(os.path.join(RF_DIR, f"best_model_kwargs_{fold}.npy"),
                     allow_pickle=True).item()
    kw = {k: kwargs[k] for k in ("n_estimators", "max_depth") if k in kwargs}
    data = get_preprocessed_features(
        current_split=str(fold), base_path=BASE_PATH,
        cross_validation_type="cell_lines_cold_start", scaling_mode=SUFFIX,
        n_bits_drug_fingerprints=128,
        data_dir=os.path.join(os.path.dirname(BASE_PATH.rstrip("/")), ".."),
    )
    model = RandomForest(**kw)
    model.fit(X_train=data["train"]["features"],
              y_train=data["train"]["targets"].response.values.astype(float))

    X = data["test"]["features"]
    y = data["test"]["targets"].response.values.astype(float)
    if len(y) > MAX_POINTS:
        idx = np.random.default_rng(SEED).choice(len(y), MAX_POINTS, replace=False)
        X, y = X[idx], y[idx]
    tree_preds = np.array([t.predict(X) for t in model.model.estimators_])
    emp, gauss = coverage_curves(tree_preds, y)
    out = {
        "fold": fold, "n": int(len(y)), "kwargs": kw,
        "MA_empirical": round(float(np.mean(np.abs(emp - NOMINAL))), 4),
        "MA_gaussian": round(float(np.mean(np.abs(gauss - NOMINAL))), 4),
        "nominal": NOMINAL.tolist(), "emp_cov": emp.tolist(), "gauss_cov": gauss.tolist(),
    }
    with open(os.path.join(RF_DIR, f"calib_{fold}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[fold {fold}] MA empirical {out['MA_empirical']:.4f} "
          f"| Gaussian {out['MA_gaussian']:.4f}")
    return out


def load_folds():
    rows = []
    for fold in range(5):
        p = os.path.join(RF_DIR, f"calib_{fold}.json")
        if os.path.exists(p):
            rows.append(json.load(open(p)))
    if not rows:
        raise SystemExit(
            f"No calib_*.json in {RF_DIR}.\n"
            "They ship in LUDRP_results_data.zip; or run with --recompute to regenerate them.")
    return sorted(rows, key=lambda r: r["fold"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recompute", action="store_true",
                    help="refit the RF per fold instead of reading the deposited JSONs")
    ap.add_argument("--folds", type=int, nargs="*", default=list(range(5)))
    args = ap.parse_args()

    rows = [recompute_fold(f) for f in args.folds] if args.recompute else load_folds()

    emp_ma = np.array([r["MA_empirical"] for r in rows])
    gauss_ma = np.array([r["MA_gaussian"] for r in rows])
    emp_cov = np.mean([r["emp_cov"] for r in rows], axis=0)
    gauss_cov = np.mean([r["gauss_cov"] for r in rows], axis=0)
    n = len(rows)
    print(f"n={n} folds | MA empirical {emp_ma.mean():.4f} +/- {emp_ma.std():.4f} "
          f"| MA Gaussian {gauss_ma.mean():.4f} +/- {gauss_ma.std():.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))

    ax1.bar([0, 1], [emp_ma.mean(), gauss_ma.mean()],
            yerr=[emp_ma.std(), gauss_ma.std()], capsize=5,
            color=[C_EMP, C_GAUSS], width=0.6)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["empirical\n(per-tree)", "Gaussian\napprox."])
    ax1.set_ylabel("Miscalibration area (lower better)")
    ax1.set_title(f"RF calibration: empirical vs Gaussian (n={n} folds)")
    for x, m, s in zip([0, 1], [emp_ma.mean(), gauss_ma.mean()], [emp_ma.std(), gauss_ma.std()]):
        ax1.text(x, m + s + 0.004, f"{m:.3f}", ha="center", fontsize=9)

    ax2.plot([0, 1], [0, 1], "k--", lw=0.9, label="ideal")
    ax2.plot(NOMINAL, emp_cov, "o-", color=C_EMP, label="empirical (per-tree)")
    ax2.plot(NOMINAL, gauss_cov, "s-", color=C_GAUSS, label="Gaussian approx.")
    ax2.set_xlabel("Nominal coverage")
    ax2.set_ylabel("Observed coverage")
    ax2.set_title("RF reliability diagram")
    ax2.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    out_dir = os.path.join(FIGURES_DIR, "appendix")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "nongaussian_calibration_rf.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=200)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
