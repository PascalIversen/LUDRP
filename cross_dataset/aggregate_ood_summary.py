"""Aggregate synthetic + cross-dataset OOD predictions into the Figure 3 / Table A3 summary.

Rebuilds ``figure_code/data/cross_dataset_ood_summary.csv.gz`` (columns: panel, model, col,
metric, value), which ``figure_code/cross_dataset_ood_figure.py`` reads to draw the AUROC heatmap
and the transfer-MSE table. Inputs:

* Synthetic OOD (panel ``synthetic``): the GDSC 5-fold perturbed predictions written by the main
  cross-validation run under
  ``examples/data/experiments/5_fold_cross_validation/results/cv_<model>_cold_z_norm/ood/
  results_<fold>_shift_<s>.csv``. For each model and shift the detection AUROC (unperturbed vs
  perturbed test instances, predicted total uncertainty as score) is computed per fold and
  averaged over the five folds.

* Cross-dataset OOD (panel ``cross``): the CTRPv2-trained transfer predictions written by
  ``run_transfer_harmonized.py`` / ``run_transfer_native.py`` / ``run_transfer_beataml.py`` under
  ``<pred_root>/{xdataset_harmonized,xdataset_native}/<model>__<train>__<test>__s<seed>.csv``.
  Each metric is computed per random cell-line split and averaged over splits (GDSC transfers use
  five splits; BeatAML uses a single split). For a scenario the AUROC separates that scenario's
  in-distribution reference (CTRPv2 held-out test in the same feature space) from the shifted
  target using the predicted total uncertainty ``unc_total``; the transfer MSE is the mean squared
  ``err`` (residual on the globally z-scored response).

Run from the repository root once the transfer runners have populated ``<pred_root>``:
    python cross_dataset/aggregate_ood_summary.py
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GDSC_RESULTS = os.path.join(REPO, "examples", "data", "experiments",
                            "5_fold_cross_validation", "results")
DEFAULT_OUT = os.path.join(REPO, "figure_code", "data", "cross_dataset_ood_summary.csv.gz")

# Model plot order and display names, matching the committed summary.
MODEL_ORDER = ["br", "rf", "mcd", "qnn", "pnne", "edl", "pnn"]
FULL_NAME = {
    "br": "Bayesian Ridge", "rf": "Random Forest", "mcd": "MC Dropout", "qnn": "Quantile NN",
    "pnne": "Gaussian NN\\nEnsemble", "edl": "Evidential DL", "pnn": "Gaussian NN",
}
# The cross-dataset transfer runners write QuantileNN predictions under its model type "qfn",
# whereas the GDSC synthetic results use "qnn". Map to the transfer-file token where they differ.
XFER_PREFIX = {"qnn": "qfn"}
SHIFTS = ["0.01", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1"]

COL_INDIST = "in-dist\\n(reference)"
COL_HARM = "CTRPv2→GDSC\\n(harmonized)"
COL_BEATAML = "CTRPv2→BeatAML\\n(diff. lab, RNA-seq)"
COL_NATIVE = "CTRPv2→GDSC\\n(native)"


def auroc(id_scores, ood_scores):
    """Detection AUROC with in-distribution=0, shifted=1; 0.5 for a constant score."""
    y = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([np.asarray(id_scores, float), np.asarray(ood_scores, float)])
    if np.allclose(s.min(), s.max()):
        return 0.5
    return float(roc_auc_score(y, s))


# --------------------------------------------------------------------------- synthetic panel
def synthetic_rows():
    rows = []
    for m in MODEL_ORDER:
        base = os.path.join(GDSC_RESULTS, f"cv_{m}_cold_z_norm")
        id_unc = pd.read_csv(os.path.join(base, "results.csv"))["y_uncertainty"].values
        for s in SHIFTS:
            fold_aurocs = []
            for fold in range(5):
                p = os.path.join(base, "ood", f"results_{fold}_shift_{s}.csv")
                ood_unc = pd.read_csv(p, index_col=0)["y_uncertainty"].values
                fold_aurocs.append(auroc(id_unc, ood_unc))
            col = "1.0" if s == "1" else s
            rows.append(("synthetic", FULL_NAME[m], col, "auroc", float(np.mean(fold_aurocs))))
    return rows


# ----------------------------------------------------------------------------- cross panel
def _by_seed(pred_root, subdir, pattern):
    """Return {seed: dataframe} for files matching <subdir>/<pattern> (pattern ends in __s*.csv)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(pred_root, subdir, pattern))):
        seed = f.split("__s")[-1].split(".")[0]
        out[seed] = pd.read_csv(f)
    return out


def _mean_over_seeds(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else np.nan


def cross_rows(pred_root):
    rows = []
    for m in MODEL_ORDER:
        name = FULL_NAME[m]
        xm = XFER_PREFIX.get(m, m)   # transfer-file token (differs from GDSC token for QuantileNN)

        # in-distribution reference (native feature space), MSE only; AUROC is 0.5 by definition.
        nat_id = _by_seed(pred_root, "xdataset_native", f"{xm}__CTRPv2__CTRPv2__s*.csv")
        harm_id = _by_seed(pred_root, "xdataset_harmonized", f"{xm}__CTRPv2__CTRPv2__s*.csv")
        mse_indist = _mean_over_seeds([np.mean(d["err"] ** 2) for d in nat_id.values()])
        rows += [("cross", name, COL_INDIST, "auroc", 0.5),
                 ("cross", name, COL_INDIST, "mse", mse_indist)]

        # harmonized CTRPv2 -> GDSC (biological/population shift only).
        harm = _by_seed(pred_root, "xdataset_harmonized", f"{xm}__CTRPv2__GDSC2__s*.csv")
        rows += _scenario(name, COL_HARM, harm_id, harm)

        # BeatAML: single split (seed 0), CTRPv2be feature space.
        be_id = _by_seed(pred_root, "xdataset_native", f"{xm}__CTRPv2be__CTRPv2be__s0.csv")
        be = _by_seed(pred_root, "xdataset_native", f"{xm}__CTRPv2be__BeatAML__s0.csv")
        rows += _scenario(name, COL_BEATAML, be_id, be)

        # native CTRPv2 -> GDSC (adds the microarray-vs-RNA-seq platform shift).
        nat = _by_seed(pred_root, "xdataset_native", f"{xm}__CTRPv2__GDSC2__s*.csv")
        rows += _scenario(name, COL_NATIVE, nat_id, nat)
    return rows


def _scenario(name, col, id_by_seed, ood_by_seed):
    """AUROC (vs same-seed in-dist) and MSE, averaged over the shifted scenario's seeds."""
    aurocs, mses = [], []
    for seed, ood in ood_by_seed.items():
        mses.append(float(np.mean(ood["err"] ** 2)))
        if seed in id_by_seed:
            aurocs.append(auroc(id_by_seed[seed]["unc_total"].values, ood["unc_total"].values))
    return [("cross", name, col, "auroc", _mean_over_seeds(aurocs)),
            ("cross", name, col, "mse", _mean_over_seeds(mses))]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-root", default=os.path.join(HERE, "results"),
                    help="directory holding xdataset_harmonized/ and xdataset_native/ prediction CSVs")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = synthetic_rows() + cross_rows(args.pred_root)
    df = pd.DataFrame(rows, columns=["panel", "model", "col", "metric", "value"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Saved {len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
