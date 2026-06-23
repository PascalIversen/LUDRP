import sys
sys.dont_write_bytecode = True

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from brokenaxes import brokenaxes
from scipy.stats import pearsonr, ttest_rel
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import mean_squared_error
from matplotlib.colors import ListedColormap
import uncertainty_toolbox as uct
from uncertainty_toolbox.metrics_calibration import miscalibration_area_from_proportions
from decimal import Decimal, ROUND_HALF_UP

from config import (
    MODELS, MODEL_COLORS, LINESTYLES, MODEL_TO_FULL_NAME,
    COLD_START_SUFFIX, SUFFIX, BASE_PATH, FIGURES_DIR, TABLES_DIR,
)

AXIS_LABEL_SIZE = 10 + 4  # unified axis label font size

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"

font_adder = 4

UNCERTAINTY_KIND = {
    "mcd": "std", "qnn": "quantiles", "pnn": "var",
    "pnne": "var", "edl": "var", "br": "std", "rf": "var",
}
MODEL_ZORDER = {"edl": 2, "pnn": 4, "pnne": 3}


def results_path(model):
    return os.path.join(BASE_PATH, f"cv_{model}{COLD_START_SUFFIX}{SUFFIX}", "results.csv")


def load_fold_dfs(model):
    base_file = results_path(model)
    dfs = []
    if os.path.exists(base_file):
        df_all = pd.read_csv(base_file, index_col=0)
        if "fold" in df_all.columns:
            fold_vals = [fv for fv in sorted(pd.unique(df_all["fold"])) if pd.notna(fv)]
            for fv in fold_vals:
                part = df_all[df_all["fold"] == fv].copy()
                if not part.empty:
                    dfs.append(part)
            if dfs:
                return dfs
        dfs.append(df_all)
        return dfs
    candidates = glob.glob(os.path.join(
        BASE_PATH, f"cv_{model}{COLD_START_SUFFIX}{SUFFIX}", "fold*", "results.csv"))
    for path in sorted(candidates):
        dfs.append(pd.read_csv(path, index_col=0))
    return dfs


def compute_auc_and_mseks(df_model):
    df = df_model.rename(columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"}).copy()
    df = df[df["gt"].notna() & df["preds"].notna()]
    if df.empty:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    df.sort_values("uncertainty", inplace=True)
    se = (df["gt"] - df["preds"]) ** 2
    quantiles = np.linspace(0, 1, num=500)
    q_mses, q_left = [], []
    unc = df["uncertainty"].to_numpy()
    for q in quantiles:
        thr = np.quantile(unc, q)
        idx = unc < thr
        if np.any(idx):
            q_left.append(idx.mean())
            q_mses.append(se[idx].mean())
    auc = np.trapezoid(np.array(q_mses), np.array(q_left)) if q_left else np.nan
    thr10 = np.quantile(unc, 0.10)
    thr50 = np.quantile(unc, 0.50)
    thr90 = np.quantile(unc, 0.90)
    mse10 = se[unc < thr10].mean() if (unc < thr10).any() else np.nan
    mse50 = se[unc < thr50].mean() if (unc < thr50).any() else np.nan
    mse90 = se[unc < thr90].mean() if (unc < thr90).any() else np.nan
    # Pearson correlation between absolute error and uncertainty
    abs_err = np.abs(df["gt"].to_numpy() - df["preds"].to_numpy())
    r_err_unc = pearsonr(abs_err, unc)[0] if len(unc) > 2 else np.nan
    return float(auc), float(mse10), float(mse50), float(mse90), float(r_err_unc)


def _ranking_skill(df_model):
    """Accuracy-independent uncertainty-ranking quality in [0, 1].

    1 = ordering test instances by predicted uncertainty matches the oracle ordering
    by the true absolute error; 0 = no better than random. Defined as
    1 - (AURC - AURC_oracle) / (AURC_rand - AURC_oracle), using the same risk-coverage
    curve as compute_auc_and_mseks. This isolates uncertainty quality from baseline accuracy.
    """
    df = df_model.rename(columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"}).copy()
    df = df[df["gt"].notna() & df["preds"].notna()]
    if len(df) < 3:
        return np.nan
    se = ((df["gt"] - df["preds"]) ** 2).to_numpy()
    abs_err = np.abs(df["gt"].to_numpy() - df["preds"].to_numpy())
    unc = df["uncertainty"].to_numpy()
    qs = np.linspace(0, 1, num=500)

    def _aurc(order_key):
        left, mses = [], []
        for q in qs:
            thr = np.quantile(order_key, q)
            idx = order_key < thr
            if np.any(idx):
                left.append(idx.mean())
                mses.append(se[idx].mean())
        return np.trapezoid(np.array(mses), np.array(left)) if left else np.nan

    aurc = _aurc(unc)
    aurc_oracle = _aurc(abs_err)
    aurc_rand = float(se.mean())
    denom = aurc_rand - aurc_oracle
    if not (np.isfinite(aurc) and np.isfinite(aurc_oracle)) or denom <= 1e-12:
        return np.nan
    return float(1.0 - (aurc - aurc_oracle) / denom)


def get_pi_qnn(df_model, nominal=0.9):
    if not np.isclose(nominal, 0.9):
        raise ValueError("only supports nominal=0.9")
    preds = df_model["preds"].to_numpy(float)
    unc = df_model["unc"].to_numpy(float)
    lo = preds - unc / 2
    hi = preds + unc / 2
    return lo, hi


def _to_booktabs(tex):
    lines = tex.splitlines()
    h_idx = [i for i, l in enumerate(lines) if r"\hline" in l]
    if not h_idx:
        return tex
    lines[h_idx[0]] = r"\toprule"
    for i in h_idx[1:-1]:
        lines[i] = r"\midrule"
    lines[h_idx[-1]] = r"\bottomrule"
    return "\n".join(lines)


def round_half_up(x, ndigits=2):
    q = Decimal(10) ** -ndigits
    return float(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


def _plot_auurc_curves(ax, skip_models=()):
    """Plot AUURC curves on *ax* and return metrics_per_model dict."""
    metrics_per_model = {}
    for model in MODELS:
        fold_dfs = load_fold_dfs(model)
        per_fold = []
        pooled_df = []
        for df in fold_dfs:
            auc, mse10, mse50, mse90, r_err_unc = compute_auc_and_mseks(df)
            per_fold.append({"auc": auc, "mse10": mse10, "mse50": mse50, "mse90": mse90,
                             "r_err_unc": r_err_unc, "skill": _ranking_skill(df)})
            pooled_df.append(df)
        metrics_per_model[model] = per_fold

        if not pooled_df:
            continue
        if model in skip_models:
            continue
        df_model = pd.concat(pooled_df, ignore_index=True)
        df_model = df_model.rename(columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"})
        df_model = df_model[df_model["gt"].notna() & df_model["preds"].notna()].copy()
        df_model.sort_values("uncertainty", inplace=True)
        df_model["se"] = (df_model["gt"] - df_model["preds"]) ** 2

        quantiles = np.linspace(0, 1, num=500)
        q_left, q_mses = [], []
        unc = df_model["uncertainty"].to_numpy()
        for q in quantiles:
            thr = np.quantile(unc, q)
            idx = unc < thr
            if np.any(idx):
                q_left.append(idx.mean())
                q_mses.append(df_model.loc[idx, "se"].mean())
        if q_left:
            legend_name = "Gaussian NN Ens." if model == "pnne" else MODEL_TO_FULL_NAME[model]
            ax.plot(np.array(q_left), np.array(q_mses),
                    color=MODEL_COLORS[model], linestyle=LINESTYLES[model],
                    linewidth=2, label=legend_name)
    return metrics_per_model


def _plot_coverage_panel(ax, skip_models=()):
    """Plot coverage-vs-sharpness error bars on *ax*."""
    nominal = 0.90
    z = 1.6448536269514722

    points_x, points_y, err_x, err_y, colors_pts, zorders_pts = [], [], [], [], [], []
    for model in MODELS:
        if model in skip_models:
            continue
        fold_dfs = load_fold_dfs(model)
        covs, widths = [], []
        for df_raw in fold_dfs:
            df_model = df_raw.rename(columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "unc"})
            df_model = df_model[df_model["gt"].notna() & df_model["preds"].notna()]
            if df_model.empty:
                continue
            y = df_model["gt"].to_numpy(float)
            mu = df_model["preds"].to_numpy(float)
            kind = UNCERTAINTY_KIND.get(model, "var")
            if kind == "quantiles":
                lo, hi = get_pi_qnn(df_model, nominal=nominal)
                if lo is None or hi is None:
                    std = np.sqrt(np.clip(df_model["unc"].to_numpy(float), 0, None))
                    lo = mu - z * std
                    hi = mu + z * std
            else:
                if kind == "var":
                    std = np.sqrt(np.clip(df_model["unc"].to_numpy(float), 0, None))
                else:
                    std = np.clip(df_model["unc"].to_numpy(float), 0, None)
                lo = mu - z * std
                hi = mu + z * std
            covs.append(np.mean((y >= lo) & (y <= hi)))
            widths.append(np.mean(hi - lo))

        if not covs:
            continue
        mean_cov = np.mean(covs)
        mean_width = np.mean(widths)
        sd_cov = np.std(covs, ddof=1) if len(covs) > 1 else 0.0
        sd_width = np.std(widths, ddof=1) if len(widths) > 1 else 0.0
        points_x.append(mean_cov)
        points_y.append(mean_width)
        err_x.append(sd_cov)
        err_y.append(sd_width)
        colors_pts.append(MODEL_COLORS[model])
        zorders_pts.append(MODEL_ZORDER.get(model, 1))

    for x, y, ex, ey, c, zo in zip(points_x, points_y, err_x, err_y, colors_pts, zorders_pts):
        ax.errorbar(x, y, xerr=ex, yerr=ey, fmt="+", ms=8,
                    ecolor=c, mec=c, mfc=c, capsize=2.9, capthick=1.2, zorder=zo)

    ax.axvline(x=0.9, color="red", linestyle="--", linewidth=1.1, alpha=1, zorder=1.6)
    ax.set_xlabel("Coverage at 90% nominal", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Mean PI width at 90%", fontsize=AXIS_LABEL_SIZE)
    ax.grid(True)


def _add_inset_colorbar(fig, ax, mappable):
    """Add a small horizontal inset colorbar in the lower-right of *ax*."""
    from matplotlib.ticker import FixedLocator
    cax = ax.inset_axes([0.6, 0.1, 0.35, 0.03])
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.ax.set_title("Uncertainty", fontsize=12, pad=4)
    cbar.ax.tick_params(labelsize=11)
    cbar.locator = FixedLocator([0.1, 0.3, 0.5])
    cbar.update_ticks()
    return cbar


def _plot_pnne_scatter_panels(fig, ax_c, ax_d, df_pnne, norm, cmap):
    """Plot scatter panels (c) and (d) with inset colorbars."""
    mappable_c = ax_c.scatter(df_pnne["gt"], df_pnne["preds"], c=df_pnne["uncertainty"],
                              s=3, cmap=cmap, norm=norm, rasterized=True)
    ax_c.set_xlabel("Experimental log IC50", fontsize=AXIS_LABEL_SIZE)
    ax_c.set_ylabel("Prediction", fontsize=AXIS_LABEL_SIZE)
    ax_c.set_xlim([-5, 5])
    ax_c.set_ylim([-5, 5])
    ax_c.plot([-100, 100], [-100, 100], c="grey")
    mse_full = mean_squared_error(df_pnne["gt"], df_pnne["preds"])
    ax_c.set_title("Gaussian NN Ens.: All predictions", fontsize=10 + font_adder)
    ax_c.text(0.25, 0.95, f"MSE = {mse_full:.2f}", transform=ax_c.transAxes,
              fontsize=10 + font_adder, verticalalignment="top")

    mask50 = df_pnne["uncertainty"] < df_pnne["uncertainty"].quantile(0.50)
    mappable_d = ax_d.scatter(df_pnne["gt"][mask50], df_pnne["preds"][mask50],
                              c=df_pnne["uncertainty"][mask50], s=3, cmap=cmap, norm=norm, rasterized=True)
    ax_d.set_xlabel("Experimental log IC50", fontsize=AXIS_LABEL_SIZE)
    ax_d.set_ylabel("Prediction", fontsize=AXIS_LABEL_SIZE)
    ax_d.set_xlim([-5, 5])
    ax_d.set_ylim([-5, 5])
    ax_d.plot([-100, 100], [-100, 100], c="grey")
    mse_50 = mean_squared_error(df_pnne["gt"][mask50], df_pnne["preds"][mask50])
    ax_d.text(0.25, 0.95, f"MSE = {mse_50:.2f}", transform=ax_d.transAxes,
              fontsize=10 + font_adder, verticalalignment="top")
    ax_d.set_title("Gaussian NN Ens.: Uncertainty < median", fontsize=10 + font_adder)

    _add_inset_colorbar(fig, ax_c, mappable_c)
    _add_inset_colorbar(fig, ax_d, mappable_d)


def _load_pnne_scatter_data():
    """Load and prepare PNNE data for scatter plots, return df_pnne, norm, cmap."""
    pnne_dfs = load_fold_dfs("pnne")
    results_pnne = pd.concat(pnne_dfs, ignore_index=True)
    results_pnne = results_pnne.rename(columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"})
    results_pnne = results_pnne[results_pnne["gt"].notna() & results_pnne["preds"].notna()].copy()
    results_pnne.sort_values("uncertainty", ascending=False, inplace=True)
    df_pnne = results_pnne[["gt", "preds", "uncertainty"]].copy()
    norm = mcolors.Normalize(vmin=df_pnne["uncertainty"].min(), vmax=df_pnne["uncertainty"].quantile(0.995))
    cmap = mcolors.LinearSegmentedColormap.from_list("truncated_reds", plt.cm.Reds(np.linspace(0.3, 1, 256)))
    return df_pnne, norm, cmap


# 4-panel uncertainty figure (with BR, for appendix)
def plot_four_panel():
    df_pnne, norm, cmap = _load_pnne_scatter_data()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # (a) AUURC curves with broken axes
    spec = axs[0][0].get_subplotspec()
    axs[0][0].remove()
    bax = brokenaxes(ylims=((0, 0.35), (0.55, 0.65)), hspace=0.1, subplot_spec=spec, fig=fig)

    metrics_per_model = _plot_auurc_curves(bax)

    bax.set_xlabel("Quantile of Data Left", labelpad=25 + font_adder, fontsize=AXIS_LABEL_SIZE)
    bax.set_ylabel("Cumulative MSE", labelpad=34 + font_adder, fontsize=AXIS_LABEL_SIZE)
    bax.grid(True)
    bax.invert_xaxis()
    bax.legend(loc="lower center", fontsize=10.5, handletextpad=0.4, borderaxespad=0.3, borderpad=0.3)

    # (b) Coverage vs sharpness
    ax_b = axs[0][1]
    _plot_coverage_panel(ax_b)

    # (c) and (d) scatter panels
    _plot_pnne_scatter_panels(fig, axs[1][0], axs[1][1], df_pnne, norm, cmap)

    fig.subplots_adjust(hspace=0.35)

    # subplot labels
    bax.axs[0].text(0.20, 0.98, "(a)", transform=bax.axs[0].transAxes,
                    fontsize=14 + font_adder, fontweight="bold", va="top", ha="left")
    for ax_flat, lab in zip(axs.flat[1:], ["(b)", "(c)", "(d)"]):
        ax_flat.text(0.10, 0.98, lab, transform=ax_flat.transAxes,
                     fontsize=12 + font_adder, fontweight="bold", va="top", ha="left")

    save_path = os.path.join(FIGURES_DIR, "appendix",
                             f"uncertainty_plot{COLD_START_SUFFIX}pnne_with_fold_errorbars.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=350)
    print(f"Saved: {save_path}")

    plt.close(fig)
    return metrics_per_model


# 4-panel uncertainty figure without Bayesian Ridge in panel (a)
def plot_four_panel_no_br():
    df_pnne, norm, cmap = _load_pnne_scatter_data()

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # (a) AUURC curves — no broken axes, no BR
    ax_a = axs[0][0]
    _plot_auurc_curves(ax_a, skip_models=("br",))
    ax_a.set_xlabel("Quantile of Data Left", fontsize=AXIS_LABEL_SIZE)
    ax_a.set_ylabel("Cumulative MSE", fontsize=AXIS_LABEL_SIZE)
    ax_a.grid(True)
    ax_a.invert_xaxis()
    ax_a.legend(loc="lower left", fontsize=11.7, handletextpad=0.4, borderaxespad=0.3, borderpad=0.3)

    # (b) Coverage vs sharpness
    ax_b = axs[0][1]
    _plot_coverage_panel(ax_b, skip_models=("br",))

    # (c) and (d) scatter panels
    _plot_pnne_scatter_panels(fig, axs[1][0], axs[1][1], df_pnne, norm, cmap)

    fig.subplots_adjust(hspace=0.35)

    # subplot labels
    labels = ["(a)", "(b)", "(c)", "(d)"]
    ax_a.text(0.20, 0.98, labels[0], transform=ax_a.transAxes,
              fontsize=14 + font_adder, fontweight="bold", va="top", ha="left")
    for ax_flat, lab in zip(axs.flat[1:], labels[1:]):
        ax_flat.text(0.10, 0.98, lab, transform=ax_flat.transAxes,
                     fontsize=12 + font_adder, fontweight="bold", va="top", ha="left")

    save_path = os.path.join(FIGURES_DIR, f"uncertainty_plot{COLD_START_SUFFIX}pnne_with_fold_errorbars_no_br.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=350)
    print(f"Saved: {save_path}")


# Calibration plot
def plot_calibration():
    uct.viz.set_style()
    uct.viz.update_rc("font.size", 13)
    uct.viz.update_rc("xtick.labelsize", 13)
    uct.viz.update_rc("ytick.labelsize", 13)
    plt.rcParams["text.usetex"] = False
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["figure.facecolor"] = "white"

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "-", label="Ideal", c="black", alpha=0.6, zorder=0)

    plot_order = ["pnn", "pnne", "edl", "mcd", "br", "qnn", "rf"]
    ma_per_model = {}

    for z, model in enumerate(plot_order):
        results = pd.read_csv(results_path(model), index_col=0)

        if model in ["pnn", "pnne", "edl", "rf"]:
            get_std = lambda df: df["y_uncertainty"].values ** 0.5
        elif model == "qnn":
            get_std = lambda df: df["y_uncertainty"].values / 3.29
        else:
            get_std = lambda df: df["y_uncertainty"].values

        # per-fold MA
        fold_mas = []
        for f in sorted(results["fold"].unique()):
            df_fold = results[results["fold"] == f]
            pred_mean = df_fold["y_preds"].values
            y = df_fold["response"].values
            pred_std = get_std(df_fold)

            tmpfig, tmpax = plt.subplots()
            tmpax = uct.plot_calibration(pred_mean, pred_std, y, ax=tmpax)
            lines = tmpax.get_lines()
            exp_props = lines[-1].get_xdata()
            obs_props = lines[-1].get_ydata()
            plt.close(tmpfig)
            fold_mas.append(miscalibration_area_from_proportions(exp_props, obs_props))

        fold_mas = np.array(fold_mas)
        ma_mean = round_half_up(fold_mas.mean(), 2)
        ma_std = round_half_up(fold_mas.std(), 2)
        ma_per_model[model] = fold_mas

        # pooled calibration curve
        pred_mean = results["y_preds"].values
        y = results["response"].values
        pred_std = get_std(results)

        tmpfig, tmpax = plt.subplots()
        tmpax = uct.plot_calibration(pred_mean, pred_std, y, ax=tmpax)
        lines = tmpax.get_lines()
        exp_props = lines[-1].get_xdata()
        obs_props = lines[-1].get_ydata()
        plt.close(tmpfig)

        line, = ax.plot(
            exp_props, obs_props,
            linestyle=LINESTYLES[model], color=MODEL_COLORS[model],
            label=f"{MODEL_TO_FULL_NAME[model]} (MA={ma_mean:.2f}+/-{ma_std:.2f})",
            zorder=z + 1,
        )
        ax.fill_between(exp_props, exp_props, obs_props,
                         alpha=0.15, color=line.get_color(), zorder=z)

    ax.set_xlabel("Predicted Proportion in Interval", fontsize=16)
    ax.set_ylabel("Observed Proportion in Interval", fontsize=16)
    ax.axis("square")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(framealpha=0.7, loc="upper left", fontsize=13)

    save_path = os.path.join(FIGURES_DIR, "appendix", "calibration_all_models.pdf")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")

    return ma_per_model


# PNNE uncertainty vs distance-from-mean baseline
def plot_distance_from_mean_baseline():
    results_pnne = pd.read_csv(results_path("pnne"), index_col=0)
    results_pnne = results_pnne.rename(
        columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"})
    results_pnne = results_pnne[results_pnne["gt"].notna() & results_pnne["preds"].notna()].copy()
    results_pnne["se"] = (results_pnne["gt"] - results_pnne["preds"]) ** 2
    results_pnne["dist_from_mean"] = np.abs(results_pnne["preds"] - results_pnne["preds"].mean())

    quantiles = np.linspace(0, 1, 500)
    auc_unc, auc_dist, left_frac = [], [], []

    for q in quantiles:
        thr_unc = np.quantile(results_pnne["uncertainty"], q)
        idx_unc = results_pnne["uncertainty"] < thr_unc
        if np.any(idx_unc):
            auc_unc.append(results_pnne.loc[idx_unc, "se"].mean())

        thr_dist = np.quantile(results_pnne["dist_from_mean"], q)
        idx_dist = results_pnne["dist_from_mean"] < thr_dist
        if np.any(idx_dist):
            auc_dist.append(results_pnne.loc[idx_dist, "se"].mean())
            left_frac.append(idx_dist.mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(left_frac, auc_dist, s=5, c="grey", label="Distance-from-mean baseline")
    ax.scatter(np.linspace(0, 1, len(auc_unc)), auc_unc, s=5,
               c=MODEL_COLORS["pnne"], label=f"{MODEL_TO_FULL_NAME['pnne']} uncertainty")
    ax.set_xlabel("Quantile of Data Left")
    ax.set_ylabel("Cumulative MSE")
    ax.invert_xaxis()
    ax.grid(True)
    ax.legend()

    save_path = os.path.join(FIGURES_DIR, "appendix", "pnne_uncertainty_vs_distance_from_mean_baseline.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=350)
    print(f"Saved: {save_path}")


def plot_distance_from_mean_baseline_all_models():
    """AUURC-style curves for all models plus distance-from-mean and random baselines."""
    fig, ax = plt.subplots(figsize=(8, 6))
    quantiles = np.linspace(0, 1, 500)

    # Plot each model's uncertainty curve, distance-from-mean baseline, and random baseline
    for model in MODELS:
        fold_dfs = load_fold_dfs(model)
        if not fold_dfs:
            continue
        df_model = pd.concat(fold_dfs, ignore_index=True)
        df_model = df_model.rename(
            columns={"response": "gt", "y_preds": "preds", "y_uncertainty": "uncertainty"})
        df_model = df_model[df_model["gt"].notna() & df_model["preds"].notna()].copy()
        df_model["se"] = (df_model["gt"] - df_model["preds"]) ** 2
        unc = df_model["uncertainty"].to_numpy()

        # Uncertainty curve
        q_left, q_mses = [], []
        for q in quantiles:
            thr = np.quantile(unc, q)
            idx = unc < thr
            if np.any(idx):
                q_left.append(idx.mean())
                q_mses.append(df_model.loc[idx, "se"].mean())
        if q_left:
            legend_name = "Gaussian NN Ens." if model == "pnne" else MODEL_TO_FULL_NAME[model]
            ax.plot(q_left, q_mses, color=MODEL_COLORS[model],
                    linestyle=LINESTYLES[model], linewidth=2, label=legend_name)

        # Distance-from-mean baseline (per model)
        df_model["dist_from_mean"] = np.abs(df_model["preds"] - df_model["preds"].mean())
        dist = df_model["dist_from_mean"].to_numpy()
        q_left_d, q_mses_d = [], []
        for q in quantiles:
            thr = np.quantile(dist, q)
            idx = dist < thr
            if np.any(idx):
                q_left_d.append(idx.mean())
                q_mses_d.append(df_model.loc[idx, "se"].mean())
        if q_left_d:
            ax.plot(q_left_d, q_mses_d, color=MODEL_COLORS[model],
                    linestyle=":", linewidth=1.5, alpha=0.5)

        # Random baseline (per model): flat line at overall MSE
        ax.axhline(df_model["se"].mean(), color=MODEL_COLORS[model],
                   linestyle="-.", linewidth=1, alpha=0.3)

    # Legend entries for baselines
    ax.plot([], [], color="grey", linestyle=":", linewidth=1.5, label="Dist.-from-mean baseline")
    ax.plot([], [], color="grey", linestyle="-.", linewidth=1, label="Random baseline")

    ax.set_xlabel("Quantile of Data Left", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Cumulative MSE", fontsize=AXIS_LABEL_SIZE)
    ax.invert_xaxis()
    ax.grid(True)
    ax.legend(fontsize=9)

    save_path = os.path.join(FIGURES_DIR, "appendix",
                             "all_models_vs_baselines.pdf")
    fig.savefig(save_path, bbox_inches="tight", dpi=350)
    print(f"Saved: {save_path}")
    plt.close(fig)


def _generate_uncertainty_table(metrics_per_model, ma_per_model):
    """Generate combined uncertainty metrics table (CSV + LaTeX) with MA from calibration."""

    def mean_std_safe(arr):
        if arr.size >= 2:
            return float(np.mean(arr)), float(np.std(arr, ddof=1))
        if arr.size == 1:
            return float(arr[0]), 0.0
        return np.nan, np.nan

    def fmt(mu, sd, ndigits=3):
        if np.isnan(mu):
            return ""
        return f"{mu:.{ndigits}f} +/- {sd:.{ndigits}f}"

    rows_fmt = []
    for model in MODELS:
        stats = metrics_per_model.get(model, [])
        aucs = np.array([d["auc"] for d in stats if not np.isnan(d["auc"])])
        mse10s = np.array([d["mse10"] for d in stats if not np.isnan(d["mse10"])])
        mse50s = np.array([d["mse50"] for d in stats if not np.isnan(d["mse50"])])
        mse90s = np.array([d["mse90"] for d in stats if not np.isnan(d["mse90"])])
        r_err_uncs = np.array([d["r_err_unc"] for d in stats if not np.isnan(d["r_err_unc"])])
        skills = np.array([d["skill"] for d in stats if not np.isnan(d.get("skill", np.nan))])

        mu_auc, sd_auc = mean_std_safe(aucs)
        mu_m10, sd_m10 = mean_std_safe(mse10s)
        mu_m50, sd_m50 = mean_std_safe(mse50s)
        mu_m90, sd_m90 = mean_std_safe(mse90s)
        mu_r, sd_r = mean_std_safe(r_err_uncs)
        mu_skill, sd_skill = mean_std_safe(skills)

        fold_mas = ma_per_model.get(model, np.array([]))
        mu_ma, sd_ma = mean_std_safe(fold_mas)

        rows_fmt.append([
            MODEL_TO_FULL_NAME[model],
            fmt(mu_auc, sd_auc), fmt(mu_m10, sd_m10),
            fmt(mu_m50, sd_m50), fmt(mu_m90, sd_m90),
            fmt(mu_ma, sd_ma, ndigits=2),
            fmt(mu_r, sd_r, ndigits=2),
            fmt(mu_skill, sd_skill, ndigits=2),
        ])

    metrics_df = pd.DataFrame(rows_fmt,
                               columns=["Model", "AUURC", "MSE@10", "MSE@50", "MSE@90", "MA",
                                         "Pearson(|err|,unc)", "Ranking skill"])
    metrics_df.to_csv(os.path.join(TABLES_DIR, "uncertainty_metrics.csv"), index=False)

    # --- Publication LaTeX table ---
    # Parse mean values for bolding best / underlining second-best per column
    # Lower is better for AUURC, MSE@10, MSE@50, MA; higher is better for r and ranking skill
    col_indices = [1, 2, 3, 7, 6, 5]  # AUURC, MSE@10, MSE@50, Ranking skill, r, MA

    def parse_mean(s, default=np.inf):
        if not s:
            return default
        return float(s.split(" +/- ")[0])

    higher_is_better = {6, 7}  # r(|err|, unc) and ranking skill
    best_idx, second_idx = {}, {}
    for ci in col_indices:
        default = -np.inf if ci in higher_is_better else np.inf
        vals = [parse_mean(row[ci], default) for row in rows_fmt]
        order = np.argsort(vals)[::-1] if ci in higher_is_better else np.argsort(vals)
        best_idx[ci] = int(order[0])
        second_idx[ci] = int(order[1]) if len(order) > 1 else -1

    # Build LaTeX
    header_names = {
        "rf": r"\begin{tabular}[c]{@{}c@{}}Random\\ Forest\end{tabular}",
        "br": r"\begin{tabular}[c]{@{}c@{}}Bayesian\\ Ridge\end{tabular}",
        "qnn": r"\begin{tabular}[c]{@{}c@{}}Quantile\\ NN\end{tabular}",
        "mcd": r"\begin{tabular}[c]{@{}c@{}}MC Dropout\\ NN\end{tabular}",
        "pnn": r"\begin{tabular}[c]{@{}c@{}}Gaussian\\ NN\end{tabular}",
        "pnne": r"\begin{tabular}[c]{@{}c@{}}Gaussian\\ NN Ens.\end{tabular}",
        "edl": r"\begin{tabular}[c]{@{}c@{}}Evidential\\ DL\end{tabular}",
    }
    # Table is transposed: models as columns, metrics as rows
    col_order = ["rf", "br", "qnn", "mcd", "pnn", "pnne", "edl"]
    model_to_row = {MODEL_TO_FULL_NAME[m]: i for i, m in enumerate(MODELS)}

    metric_labels = {
        1: "AUURC",
        2: r"\MSEatk{10}",
        3: r"\MSEatk{50}",
        5: "MA",
        6: r"$r(|\epsilon|, \hat{\sigma})$",
        7: "Ranking skill",
    }

    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\caption{Uncertainty evaluation metrics per model (average $\pm$ standard deviation "
                 r"over the cross-validation folds). AUURC: area under the uncertainty--reduction curve "
                 r"(lower = better). \MSEatk{k}: MSE on the $k$\,\% most confident predictions "
                 r"(lower = better). MA: miscalibration area (lower = better calibrated). "
                 r"Ranking skill: normalized risk--reduction skill, independent of overall accuracy "
                 r"(1 = oracle error ordering, 0 = random; higher = better). "
                 r"$r(|\epsilon|,\hat{\sigma})$: Pearson correlation between absolute error and predicted "
                 r"uncertainty (higher = better). Best values are shown in bold, second-best underlined.}")
    lines.append(r"\centering")
    lines.append(r"\label{tab:uncertainty_metrics}")
    ncols = len(col_order)
    lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
    lines.append(r"\toprule")
    # Header row
    header_cells = [" "] + [header_names[m] for m in col_order]
    lines.append(" & ".join(header_cells) + r"\\")
    lines.append(r"\midrule")

    # Data rows
    for ci in col_indices:
        label = metric_labels[ci]
        cells = [label]
        for m in col_order:
            ri = model_to_row[MODEL_TO_FULL_NAME[m]]
            val_str = rows_fmt[ri][ci]
            # Format: convert +/- to ± and use $\pm$
            val_str = val_str.replace(" +/- ", r" $\pm$ ")
            # Bold best, underline second-best (mean only, keep ± std normal)
            parts = val_str.split(r" $\pm$ ")
            if len(parts) == 2:
                if ri == best_idx[ci]:
                    val_str = r"\textbf{" + parts[0] + r"} $\pm$ " + parts[1]
                elif ri == second_idx[ci]:
                    val_str = r"\underline{" + parts[0] + r"} $\pm$ " + parts[1]
            cells.append(val_str)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)
    with open(os.path.join(TABLES_DIR, "uncertainty_metrics.tex"), "w") as f:
        f.write(latex_str)
    print("Saved: uncertainty_metrics.csv + .tex")


def _format_pval(p):
    if p < 1e-16:
        return "<1e-16"
    elif p < 1e-1:
        return f"{p:.0e}"
    else:
        return f"{p:.1f}"


def _run_ttests_fdr(per_model_values, metric_name, sort_by_values, alpha=0.05, ax=None):
    """Paired t-tests with FDR correction, plotted as a heatmap on *ax*."""
    mean_sort = {m: np.mean(sort_by_values[m]) for m in sort_by_values}
    sorted_models = sorted(mean_sort, key=mean_sort.get)
    n_models = len(sorted_models)

    p_matrix = np.ones((n_models, n_models))
    pairs, raw_pvals = [], []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            mi, mj = sorted_models[i], sorted_models[j]
            if mi in per_model_values and mj in per_model_values:
                A = np.array(per_model_values[mi])
                B = np.array(per_model_values[mj])
                _, p_val = ttest_rel(A, B)
                p_matrix[i, j] = p_val
                p_matrix[j, i] = p_val
                pairs.append((i, j))
                raw_pvals.append(p_val)

    _, pvals_corrected, _, _ = multipletests(raw_pvals, method="fdr_bh")
    for (i, j), p_corr in zip(pairs, pvals_corrected):
        p_matrix[i, j] = p_corr
        p_matrix[j, i] = p_corr

    sig_mask = p_matrix < alpha
    cmap = ListedColormap(["lightgrey", "gold"])

    im = ax.imshow(sig_mask, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(n_models))
    ax.set_yticks(range(n_models))
    ax.set_xticklabels([MODEL_TO_FULL_NAME[m] for m in sorted_models],
                       rotation=45, ha="right", fontsize=12)
    ax.set_yticklabels([MODEL_TO_FULL_NAME[m] for m in sorted_models], fontsize=12)

    for i in range(n_models):
        for j in range(n_models):
            label = "1.0" if i == j else _format_pval(p_matrix[i, j])
            ax.text(j, i, label, ha="center", va="center", color="black")

    ax.grid(False)
    return im, p_matrix, sorted_models


def plot_uncertainty_significance(metrics_per_model, save_path):
    """Paired t-test heatmaps for AUURC, MSE@10, and MSE@50."""
    # Extract per-fold arrays for each metric
    auurc = {m: [d["auc"] for d in stats] for m, stats in metrics_per_model.items()}
    mse10 = {m: [d["mse10"] for d in stats] for m, stats in metrics_per_model.items()}
    mse50 = {m: [d["mse50"] for d in stats] for m, stats in metrics_per_model.items()}

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=300)
    fig.subplots_adjust(wspace=0.4)

    _run_ttests_fdr(auurc, "AUURC", auurc, ax=axes[0])
    axes[0].set_title("Significance of AUURC Differences", fontsize=12)

    _run_ttests_fdr(mse10, "MSE@10", mse10, ax=axes[1])
    axes[1].set_title("Significance of MSE@10 Differences", fontsize=12)

    im, _, _ = _run_ttests_fdr(mse50, "MSE@50", mse50, ax=axes[2])
    axes[2].set_title("Significance of MSE@50 Differences", fontsize=12)

    cbar = fig.colorbar(im, ax=axes, ticks=[0, 1],
                        fraction=0.026, location="right", shrink=0.5)
    cbar.ax.set_yticklabels(["Not sig.", "Sig."], fontsize=12)
    cbar.set_label("p < 0.05", labelpad=-30, fontsize=12)

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, "appendix"), exist_ok=True)

    metrics_per_model = plot_four_panel()
    plot_four_panel_no_br()
    ma_per_model = plot_calibration()
    plot_distance_from_mean_baseline()
    plot_distance_from_mean_baseline_all_models()

    _generate_uncertainty_table(metrics_per_model, ma_per_model)

    plot_uncertainty_significance(
        metrics_per_model,
        save_path=os.path.join(FIGURES_DIR, "appendix", "ttest_heatmap_uncertainty.pdf"),
    )
