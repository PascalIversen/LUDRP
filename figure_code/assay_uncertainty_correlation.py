"""Association between predicted aleatoric uncertainty and dose-response assay quality,
before and after controlling for the response (IC50).

Tests whether the GNNE predicted aleatoric uncertainty tracks CurveCurator assay-quality
metrics (a re-fit of the GDSC dose-response curves) beyond its dependence on the response
value. Each metric is a proxy for the IC50 and the uncertainty is heteroscedastic in the
IC50, so a first-order partial Spearman controlling for the response collapses the pooled
correlations -- i.e. the model is not reading assay quality off each prediction.

Reads:  results in BASE_PATH (cv_pnne_cold_z_norm/results.csv) and the committed CurveCurator
        extract figure_code/data/gdsc_curvecurator_assay.csv.gz (regenerate via
        after_review_additions/data/make_assay_extract.py).
Writes: figures/assay_uncertainty_correlation.pdf
Run from the repository root: python figure_code/assay_uncertainty_correlation.py
"""
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from config import BASE_PATH, FIGURES_DIR, COLD_START_SUFFIX, SUFFIX, SCRIPT_DIR

ASSAY_EXTRACT = os.path.join(SCRIPT_DIR, "data", "gdsc_curvecurator_assay.csv.gz")
UNC = "y_aleatory_uncertainty"
METRICS = [
    ("R2", "Curve fit R²"),
    ("RMSE", "Curve fit RMSE"),
    ("pEC50Error", "pEC50 std. error"),
    ("FoldChange", "Dynamic range\n(FoldChange)"),
    ("max_dose_M", "Max tested conc."),
    ("AUC", "Response AUC"),
]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_joined(model="pnne"):
    """Join the model's predictions to the CurveCurator metrics by normalized name."""
    res = pd.read_csv(os.path.join(BASE_PATH, f"cv_{model}{COLD_START_SUFFIX}{SUFFIX}", "results.csv"))
    res["cl_n"] = res["cell_lines"].map(_norm)
    res["dr_n"] = res["drugs"].map(_norm)
    assay = pd.read_csv(ASSAY_EXTRACT)
    m = res.merge(assay, on=["cl_n", "dr_n"], how="left").dropna(subset=[UNC])
    for col, _ in METRICS:
        m[col] = pd.to_numeric(m[col], errors="coerce")
    return m


def main():
    m = load_joined("pnne")

    def sp(a, b):
        d = m[[a, b]].dropna()
        return spearmanr(d[a], d[b]).statistic

    def partial_response(metric):
        # first-order partial Spearman of (uncertainty, metric) controlling for the response
        r_um, r_ur, r_mr = sp(UNC, metric), sp(UNC, "response"), sp(metric, "response")
        den = np.sqrt((1 - r_ur ** 2) * (1 - r_mr ** 2))
        return (r_um - r_ur * r_mr) / den if den > 0 else np.nan

    pooled = [sp(UNC, c) for c, _ in METRICS]
    controlled = [partial_response(c) for c, _ in METRICS]

    x = np.arange(len(METRICS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.bar(x - w / 2, pooled, w, label="pooled", color="#4C72B0")
    ax.bar(x + w / 2, controlled, w, label="controlled for response (IC50)", color="#DD8452")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRICS], rotation=25, ha="right")
    ax.set_ylabel("Spearman ρ (uncertainty vs. assay metric)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "assay_uncertainty_correlation.pdf")
    fig.savefig(out)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    print(f"rho(uncertainty, IC50) = {sp(UNC, 'response'):+.3f}")
    for (c, _), p, q in zip(METRICS, pooled, controlled):
        print(f"  {c:12s} pooled {p:+.3f}  controlled {q:+.3f}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
