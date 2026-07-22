"""Active-learning (case-specific fine-tuning) summary statistics.

Reproduces the numbers reported for the uncertainty-guided fine-tuning experiment from the
per-cell-line result CSVs written by ``uadr/experiments/run_case_specific_finetuning.py``:
group mean MSEs, the paired two-sided Wilcoxon signed-rank p-value, and the matched-pairs
rank-biserial effect size r (Kerby 2014) for the uncertainty-guided selection vs. each baseline.

Two evaluation sets are reported:
  * full     -- all drugs of the held-out cell line, including the 60 fine-tuned-on drugs
                (in-sample for the fine-tuned drugs).
  * inter    -- only drugs used by neither strategy for fine-tuning (leakage-free comparison).

Run from the repository root:  python figure_code/active_learning_stats.py
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(
    SCRIPT_DIR, "..", "examples", "data", "experiments",
    "5_fold_cross_validation", "case_specific_finetune", "test_run_top60",
)


def load_results():
    """Pool the per-split case-specific fine-tuning result CSVs."""
    frames = []
    for i in range(5):
        path = os.path.join(RESULTS_DIR, f"split_{i}", "case_specific_finetune_results.csv")
        frames.append(pd.read_csv(path, index_col=0))
    return pd.concat(frames).dropna()


def rank_biserial(a, b):
    """Matched-pairs rank-biserial correlation for the Wilcoxon signed-rank test.

    Positive r means values in ``a`` are systematically smaller than in ``b`` (here: the
    uncertainty-guided MSE is lower than the baseline MSE). Zero differences are dropped and
    tied magnitudes receive average ranks, matching scipy's default handling.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[d != 0]
    if d.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    total = ranks.sum()
    # r > 0 when a < b (improvement), so orient on the negative differences.
    return float((w_neg - w_pos) / total)


def compare(df, guided, baseline, label):
    a = df[guided].to_numpy(float)
    b = df[baseline].to_numpy(float)
    _, p = stats.wilcoxon(a, b)
    r = rank_biserial(a, b)
    print(f"{label:38s} n={len(a):4d}  "
          f"mean({guided})={a.mean():.3f}  mean({baseline})={b.mean():.3f}  "
          f"p={p:.1e}  rank-biserial r={r:.2f}")
    return {"label": label, "n": len(a), "mean_guided": a.mean(),
            "mean_baseline": b.mean(), "p": p, "rank_biserial_r": r}


def main():
    df = load_results()
    rows = [
        compare(df, "mse_uncertain", "mse_random", "full: uncertainty vs random"),
        compare(df, "mse_uncertain", "mse_base", "full: uncertainty vs no-finetune"),
        compare(df, "mse_uncertain_inter", "mse_random_inter", "held-out: uncertainty vs random"),
        compare(df, "mse_uncertain_inter", "mse_base_inter", "held-out: uncertainty vs no-finetune"),
    ]
    out = os.path.join(SCRIPT_DIR, "tables", "active_learning_stats.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
