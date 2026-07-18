"""CTRPv2 replication of Tables 1 & 2, formatted IDENTICALLY to the manuscript main tables.

Reproduces the exact layout and metric recipes of the GDSC tables (figure_code/
prediction_performance_figures.py and uncertainty_performance_figures.py), computed on the CTRPv2
5-fold benchmark:
  Table 1 (tab:ctrpv2_pred): models as columns [rf, br, qfn, mcd, pnn, pnne, edl]; Drug-wise and
     Global sections, each MSE and Pearson as median +/- std (drug-wise: std over drugs; global:
     over folds). Best bold, second-best underlined.
  Table 2 (tab:ctrpv2_unc): same columns; rows AUURC, MSE@10, MSE@50, Ranking skill, r(|eps|,sigma),
     MA, each mean +/- std over the folds. Best bold, second-best underlined.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import uncertainty_toolbox as uct
from uncertainty_toolbox.metrics_calibration import miscalibration_area_from_proportions

from selective_risk import selective_risk_metrics  # co-located in figure_code/
from config import BASE_PATH, TABLES_DIR

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

# CTRPv2 5-fold benchmark results (from Zenodo), placed alongside the GDSC results under experiments/
RES = Path(BASE_PATH).parents[1] / "ctrpv2_benchmark" / "results"
OUT = Path(TABLES_DIR)

# column order + two-line headers, identical to the manuscript main tables (qnn -> qfn for CTRPv2)
MODELS = ["rf", "br", "qfn", "mcd", "pnn", "pnne", "edl"]
HEAD = {
    "rf":   r"\begin{tabular}[c]{@{}c@{}}Random\\ Forest\end{tabular}",
    "br":   r"\begin{tabular}[c]{@{}c@{}}Bayesian\\ Ridge\end{tabular}",
    "qfn":  r"\begin{tabular}[c]{@{}c@{}}Quantile\\ NN\end{tabular}",
    "mcd":  r"\begin{tabular}[c]{@{}c@{}}MC Dropout\\ NN\end{tabular}",
    "pnn":  r"\begin{tabular}[c]{@{}c@{}}Gaussian\\ NN\end{tabular}",
    "pnne": r"\begin{tabular}[c]{@{}c@{}}Gaussian\\ NN Ens.\end{tabular}",
    "edl":  r"\begin{tabular}[c]{@{}c@{}}Evidential\\ DL\end{tabular}",
}


def load(m):
    return pd.read_csv(RES / f"cv_{m}_ctrpv2_z_norm" / "results.csv", index_col=0)


def compute_auc_and_mseks(df_model, model):
    """Copied from uncertainty_performance_figures.py (AUURC, MSE@10/50, r(|err|,unc))."""
    df = df_model.rename(columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"}).copy()
    df = df[df["gt"].notna() & df["preds"].notna()]
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan
    df.sort_values("uncertainty", inplace=True)
    se = (df["gt"] - df["preds"]) ** 2
    unc = df["uncertainty"].to_numpy()
    q_mses, q_left = [], []
    for q in np.linspace(0, 1, 500):
        idx = unc < np.quantile(unc, q)
        if np.any(idx):
            q_left.append(idx.mean()); q_mses.append(se[idx].mean())
    auc = float(np.trapezoid(np.array(q_mses), np.array(q_left))) if q_left else np.nan
    mse10 = se[unc < np.quantile(unc, 0.10)].mean() if (unc < np.quantile(unc, 0.10)).any() else np.nan
    mse50 = se[unc < np.quantile(unc, 0.50)].mean() if (unc < np.quantile(unc, 0.50)).any() else np.nan
    abs_err = np.abs(df["gt"].to_numpy() - df["preds"].to_numpy())
    # Put uncertainty on a common standard-deviation scale before correlating (see std_from_unc).
    r = pearsonr(abs_err, std_from_unc(model, unc))[0] if len(unc) > 2 else np.nan
    return auc, float(mse10), float(mse50), float(r)


def std_from_unc(m, unc):
    """Gaussian-approx std for calibration, matching plot_calibration()'s get_std."""
    if m in ("pnn", "pnne", "edl", "rf"):
        return unc ** 0.5
    if m == "qfn":                    # 90% PI width -> std
        return unc / 3.29
    return unc                        # mcd, br already std-like


def ma_per_fold(m, df):
    mas = []
    for f in sorted(df["fold"].unique()):
        d = df[df["fold"] == f]
        tf, ta = plt.subplots()
        ta = uct.plot_calibration(d["y_preds"].values, std_from_unc(m, d["y_uncertainty"].values),
                                  d["response"].values, ax=ta)
        ln = ta.get_lines()
        mas.append(miscalibration_area_from_proportions(ln[-1].get_xdata(), ln[-1].get_ydata()))
        plt.close(tf)
    return np.array(mas)


def table1_stats(m):
    df = load(m)
    g_mse, g_p = [], []
    for f in sorted(df["fold"].unique()):
        d = df[df["fold"] == f]
        g_mse.append(np.mean((d["response"] - d["y_preds"]) ** 2))
        g_p.append(pearsonr(d["response"], d["y_preds"])[0])
    d_mse, d_p = [], []
    for dr in df["drugs"].unique():
        mask = df["drugs"] == dr
        if mask.sum() < 2 or df.loc[mask, "response"].nunique() < 2:
            continue
        d_mse.append(np.mean((df.loc[mask, "response"] - df.loc[mask, "y_preds"]) ** 2))
        d_p.append(pearsonr(df.loc[mask, "response"], df.loc[mask, "y_preds"])[0])
    d_p = [x for x in d_p if not np.isnan(x)]
    med = lambda a: (float(np.median(a)), float(np.std(a)))          # drug-wise/global: median +/- std (ddof=0)
    return dict(dw_mse=med(d_mse), dw_p=med(d_p), g_mse=med(g_mse), g_p=med(g_p))


def table2_stats(m):
    df = load(m)
    per = []
    for f in sorted(df["fold"].unique()):
        d = df[df["fold"] == f]
        auc, mse10, mse50, r = compute_auc_and_mseks(d, m)
        sr = selective_risk_metrics(d["y_preds"].values, d["response"].values, d["y_uncertainty"].values)
        per.append(dict(auurc=auc, mse10=mse10, mse50=mse50, r=r, skill=sr["ranking_skill"]))
    ms = lambda k: _mean_std(np.array([p[k] for p in per if not np.isnan(p[k])]))
    ma = ma_per_fold(m, df)
    return dict(auurc=ms("auurc"), mse10=ms("mse10"), mse50=ms("mse50"),
                skill=ms("skill"), r=ms("r"), ma=_mean_std(ma))


def _mean_std(a):                                                   # Table 2: mean +/- std (ddof=1)
    a = np.asarray(a, float)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0)


def row(label, stats, key, nd, higher):
    """One LaTeX row: bold best, underline second-best (by mean/median), value +/- std."""
    mus = np.array([stats[m][key][0] for m in MODELS])
    order = np.argsort(mus)
    best, second = (order[-1], order[-2]) if higher else (order[0], order[1])
    cells = [label]
    for i, m in enumerate(MODELS):
        mu, sd = stats[m][key]
        s = f"{mu:.{nd}f}"
        if i == best:
            s = r"\textbf{" + s + "}"
        elif i == second:
            s = r"\underline{" + s + "}"
        cells.append(s + r" $\pm$ " + f"{sd:.{nd}f}")
    return " & ".join(cells) + r" \\"


def build_table1(stats):
    hdr = " & ".join([" "] + [HEAD[m] for m in MODELS]) + r"\\"
    lines = [
        r"\begin{table*}[htbp]",
        r"\caption{\ac{MSE} and Pearson correlation between model predictions and experimental "
        r"logarithmic IC50 values on the CTRPv2 dataset (5-fold leave-cell-line-out). For the "
        r"drug-wise metrics, \ac{MSE} is the median over drugs; the standard "
        r"deviation represents the variability across drugs. The global metrics are calculated for "
        r"the folds, and the standard deviation is the variation over folds. Best values are shown "
        r"in bold; second-best are underlined.}",
        r"\centering",
        r"\label{tab:ctrpv2_pred}",
        r"\begin{tabular}{l" + "c" * len(MODELS) + "}",
        r"\toprule", hdr, r"\midrule",
        r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\textit{Drug-wise}} \\",
        row("MSE", stats, "dw_mse", 3, False),
        row("Pearson", stats, "dw_p", 3, True),
        r"\midrule",
        r"\multicolumn{" + str(len(MODELS) + 1) + r"}{l}{\textit{Global}} \\",
        row("MSE", stats, "g_mse", 3, False),
        row("Pearson", stats, "g_p", 3, True),
        r"\bottomrule", r"\end{tabular}%", "", r"\end{table*}",
    ]
    return "\n".join(lines)


def build_table2(stats):
    hdr = " & ".join([" "] + [HEAD[m] for m in MODELS]) + r"\\"
    lines = [
        r"\begin{table*}[htbp]",
        r"\caption{Uncertainty evaluation metrics per model on the CTRPv2 dataset (average $\pm$ "
        r"standard deviation over the cross-validation folds). AUURC: area under the "
        r"uncertainty--reduction curve (lower = better). \MSEatk{k}: MSE on the $k$\,\% most "
        r"confident predictions (lower = better). MA: miscalibration area (lower = better "
        r"calibrated). Ranking skill: normalized risk--reduction skill, independent of overall "
        r"accuracy (1 = oracle error ordering, 0 = random; higher = better). "
        r"$r(|\epsilon|,\hat{\sigma})$: Pearson correlation between absolute error and predicted "
        r"uncertainty (higher = better). Best values are shown in bold, second-best underlined.}",
        r"\centering",
        r"\label{tab:ctrpv2_unc}",
        r"\begin{tabular}{l" + "c" * len(MODELS) + "}",
        r"\toprule", hdr, r"\midrule",
        row("AUURC", stats, "auurc", 3, False),
        row(r"\MSEatk{10}", stats, "mse10", 3, False),
        row(r"\MSEatk{50}", stats, "mse50", 3, False),
        row("Ranking skill", stats, "skill", 2, True),
        row(r"$r(|\epsilon|, \hat{\sigma})$", stats, "r", 2, True),
        row("MA", stats, "ma", 2, False),
        r"\bottomrule", r"\end{tabular}%", "", r"\end{table*}",
    ]
    return "\n".join(lines)


def main():
    s1 = {m: table1_stats(m) for m in MODELS}
    s2 = {m: table2_stats(m) for m in MODELS}
    (OUT / "table1_ctrpv2_pred.tex").write_text(build_table1(s1))
    (OUT / "table2_ctrpv2_unc.tex").write_text(build_table2(s2))
    print("=== Table 1 (median +/- std) ===")
    for m in MODELS:
        st = s1[m]
        print(f"{m:5} dwMSE {st['dw_mse'][0]:.3f} dwP {st['dw_p'][0]:.3f} "
              f"gMSE {st['g_mse'][0]:.3f} gP {st['g_p'][0]:.3f}")
    print("=== Table 2 (mean +/- std over folds) ===")
    for m in MODELS:
        st = s2[m]
        print(f"{m:5} AUURC {st['auurc'][0]:.3f} MSE10 {st['mse10'][0]:.3f} MSE50 {st['mse50'][0]:.3f} "
              f"skill {st['skill'][0]:.2f} r {st['r'][0]:.2f} MA {st['ma'][0]:.2f}")
    print(f"\nwrote {OUT}/table1_ctrpv2_pred.tex and table2_ctrpv2_unc.tex")


if __name__ == "__main__":
    main()
