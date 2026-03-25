"""
drivers_figure.py
=================
Generate SHAP-based driver plots.

* 3 cross-plots  → FIGURES_DIR  (main figures)
* 2 beeswarm summary plots → FIGURES_DIR/appendix
"""
import sys
sys.dont_write_bytecode = True

import os
import random
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, MaxNLocator

from uadr.utils.data_preprocessing import get_preprocessed_features

from config import FIGURES_DIR

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

# matplotlib defaults
matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})

# paths / constants
XAI_DRIVERS_DIR = os.path.join(os.path.dirname(__file__), "..", "examples", "XAI_drivers")
BASE_PATH = os.path.join(XAI_DRIVERS_DIR, "..", "data", "experiments", "5_fold_cross_validation") + os.sep
CROSS_VALIDATION_TYPE = "cell_lines_cold_start"
SCALING_MODE = "drug_z_norm"
N_BITS_DRUG_FINGERPRINTS = 128
CURRENT_SPLIT = 0

SHAP_DIR = os.path.join(XAI_DRIVERS_DIR, f"shap_partial_results_{SCALING_MODE}")

#  Data loading
def load_data():
    """Load preprocessed features and metadata for the test set."""
    print(f"Current split: {CURRENT_SPLIT}", flush=True)
    out = get_preprocessed_features(
        current_split=CURRENT_SPLIT,
        base_path=BASE_PATH,
        cross_validation_type=CROSS_VALIDATION_TYPE,
        scaling_mode=SCALING_MODE,
        n_bits_drug_fingerprints=N_BITS_DRUG_FINGERPRINTS,
        data_dir=os.path.join(XAI_DRIVERS_DIR, "..", "data"),
    )

    x_test = out["test"]["features"]
    y_test = out["test"]["targets"]
    gene_list = out["gene_list"]

    # metadata
    tissues = pd.read_csv(
        os.path.join(XAI_DRIVERS_DIR, "..", "data", "GDSC", "Cell_Lines_Details_with_origin_information.csv"), index_col=0
    )[["Sample Name", "Tissue descriptor 1"]].set_index("Sample Name")
    y_test["Tissue descriptor 1"] = y_test["cell_lines"].map(tissues["Tissue descriptor 1"])

    pathways = pd.read_csv(
        os.path.join(XAI_DRIVERS_DIR, "..", "data", "GDSC", "Drug_listWed Dec  4 16_10_06 2024.csv"), index_col=0
    )[[" Name", " Target pathway"]].drop_duplicates(" Name").set_index(" Name")
    y_test["Target pathway"] = y_test["drugs"].map(pathways[" Target pathway"])

    # feature names & masks
    n_genes = len(gene_list)
    n_total = x_test.shape[1]
    feature_names = list(gene_list) + [f"drug_feature_{i}" for i in range(n_total - n_genes)]
    gene_mask = np.array([not n.startswith("drug_feature_") for n in feature_names])
    feature_names_genes = [n for n in feature_names if not n.startswith("drug_feature_")]

    # SHAP values
    shap_mean_genes = pd.read_csv(os.path.join(SHAP_DIR, "shap_mean_full.csv"))
    shap_unc_genes = pd.read_csv(os.path.join(SHAP_DIR, "shap_unc_full.csv"))

    return (x_test, y_test, gene_list, gene_mask,
            feature_names_genes, shap_mean_genes, shap_unc_genes)

# Cross-plot
def plot_shap_cross(
    shap_mean_genes,
    shap_unc_genes,
    feature_names_genes,
    save_file,
    meta=None,
    tissues="all",
    pathways="all",
    top_k_genes_unc=60,
    top_k_genes_mean=60,
    scientific_axis=True,
    line_len_px=60,
    label_pad_px=6,
    manual_dirs=None,
):
    if manual_dirs is None:
        manual_dirs = {}

    # optional filter
    if meta is not None:
        idx = np.ones(len(meta), dtype=bool)
        if tissues != "all":
            tlist = tissues if isinstance(tissues, (list, tuple, set)) else [tissues]
            idx &= meta["Tissue descriptor 1"].astype(str).str.lower().isin(
                [t.lower() for t in tlist]
            )
        if pathways != "all":
            plist = pathways if isinstance(pathways, (list, tuple, set)) else [pathways]
            idx &= meta["Target pathway"].astype(str).str.lower().isin(
                [p.lower() for p in plist]
            )
        if idx.sum() == 0:
            print("no rows matched filter")
            return None
        M_all, U_all = np.asarray(shap_mean_genes)[idx], np.asarray(shap_unc_genes)[idx]
    else:
        M_all, U_all = np.asarray(shap_mean_genes), np.asarray(shap_unc_genes)

    # top genes by |unc| and |mean| (union)
    unc_abs_all = np.abs(U_all).mean(axis=0)
    mean_abs_all = np.abs(M_all).mean(axis=0)
    k_u = int(min(top_k_genes_unc, unc_abs_all.size))
    k_m = int(min(top_k_genes_mean, mean_abs_all.size))
    top_idx = np.unique(np.concatenate([
        np.argsort(unc_abs_all)[-k_u:],
        np.argsort(mean_abs_all)[-k_m:],
    ]))

    genes = [feature_names_genes[i] for i in top_idx]
    mean_imp = M_all[:, top_idx].mean(axis=0)
    unc_imp = U_all[:, top_idx].mean(axis=0)
    df = pd.DataFrame({"gene": genes, "mean_imp": mean_imp, "unc_imp": unc_imp})
    if df.empty:
        print("nothing passed filter")
        return None

    df = df.sort_values("gene").reset_index(drop=True)

    # figure
    plt.rcParams.update({
        "font.size": 12,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "axes.linewidth": 1.0,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "savefig.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    x = df["mean_imp"].to_numpy()
    y = df["unc_imp"].to_numpy()

    ax.scatter(x, y, s=26, color="#BFC4CB", alpha=0.75, edgecolors="none")

    # important genes = top ±3 by each axis
    idx_label = (
        set(df["mean_imp"].nlargest(3).index)
        | set(df["mean_imp"].nsmallest(3).index)
        | set(df["unc_imp"].nlargest(3).index)
        | set(df["unc_imp"].nsmallest(3).index)
    )

    manual_idx = {
        df.index[df["gene"] == g][0]
        for g in manual_dirs.keys()
        if g in set(df["gene"])
    }
    idx_label |= manual_idx

    mean_top = set(df["mean_imp"].nlargest(3).index) | set(df["mean_imp"].nsmallest(3).index)
    unc_top = set(df["unc_imp"].nlargest(3).index) | set(df["unc_imp"].nsmallest(3).index)

    def mean_color(v):
        return "#2F6ADE" if v > 0 else "#D13B2E"

    def unc_color(v):
        return "#EE8C2B" if v > 0 else "#2A8C55"

    for i in idx_label:
        px, py = x[i], y[i]
        if i in mean_top:
            color = mean_color(df.loc[i, "mean_imp"])
        elif i in unc_top:
            color = unc_color(df.loc[i, "unc_imp"])
        else:
            color = "#808080"
        ax.scatter([px], [py], s=26, color=color, edgecolors="black", lw=0.6, zorder=3)
        print(i, df.loc[i, "gene"], px, py)

    # helper: fixed-pixel leader line + label
    trans = ax.transData
    inv = ax.transData.inverted()

    def place_label_with_fixed_px(px, py, text, direction="left"):
        sx, sy = trans.transform((px, py))
        if direction == "left":
            ex, ey = sx - line_len_px, sy
            tx, ty = ex - label_pad_px, ey
            ha, va = "right", "center"
        elif direction == "right":
            ex, ey = sx + line_len_px, sy
            tx, ty = ex + label_pad_px, ey
            ha, va = "left", "center"
        elif direction == "up":
            ex, ey = sx, sy + line_len_px
            tx, ty = ex, ey + label_pad_px
            ha, va = "center", "bottom"
        else:  # down
            ex, ey = sx, sy - line_len_px
            tx, ty = ex, ey - label_pad_px
            ha, va = "center", "top"

        x_end, y_end = inv.transform((ex, ey))
        x_text, y_text = inv.transform((tx, ty))

        ax.plot([px, x_end], [py, y_end], color="black", lw=0.8, zorder=3)
        ax.text(
            x_text, y_text, text, fontsize=10.3, ha=ha, va=va,
            bbox=dict(facecolor="white", alpha=0.9, edgecolor="none", pad=0.6),
            zorder=4,
        )

    for i in idx_label:
        gene = df.loc[i, "gene"]
        direction = manual_dirs.get(gene, "left")
        place_label_with_fixed_px(x[i], y[i], gene, direction)

    # zero lines & labels
    ax.axvline(0, color="black", lw=0.8, alpha=0.35)
    ax.axhline(0, color="black", lw=0.8, alpha=0.35)
    ax.tick_params(axis="both", which="major", labelsize=12)
    if scientific_axis:
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_locator(MaxNLocator(nbins=5))
            fmt = ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((-2, 2))
            axis.set_major_formatter(fmt)
        # draw once so formatters compute their scale
        fig.canvas.draw()
        # read the order of magnitude directly from each formatter
        x_oom = ax.xaxis.get_major_formatter().orderOfMagnitude
        y_oom = ax.yaxis.get_major_formatter().orderOfMagnitude
        # hide the floating offset text
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)
        def _scale_text(oom):
            if oom != 0:
                return r"$\times 10^{" + str(oom) + r"}$"
            return None
        x_scale_text = _scale_text(x_oom)
        y_scale_text = _scale_text(y_oom)
    else:
        x_scale_text = None
        y_scale_text = None

    ax.set_xlabel("Avg. Estimated Shapley Values: Mean-Output", fontsize=12, labelpad=10)
    ax.set_ylabel("Avg. Estimated Shapley Values: Uncertainty-Output", fontsize=12, labelpad=10)

    # place scale annotations with smaller font near the right/top end of each axis
    if x_scale_text:
        ax.text(0.95, -0.04, x_scale_text, fontsize=8, ha="left", va="top",
                transform=ax.transAxes)
    if y_scale_text:
        ax.text(-0.01, 1.0, y_scale_text, fontsize=8, ha="right", va="bottom",
                transform=ax.transAxes)

    # perceptually square plotting area
    fig.canvas.draw()
    pos = ax.get_position()
    side = min(pos.width, pos.height)
    ax.set_position([pos.x0, pos.y0, side, side])
    xmin, xmax = ax.get_xlim()
    pad = 0.03 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)

    plt.savefig(save_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_file}")
    return df

# Beeswarm summary plots
def plot_beeswarm_summary(shap_values, x_test, gene_mask, feature_names_genes, save_file):
    """Standard SHAP dot summary plot."""
    plt.rcParams.update({
        "font.size": 16,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
    })
    plt.figure(figsize=(6, 5))
    shap.summary_plot(
        shap_values=np.asarray(shap_values),
        features=x_test[:, gene_mask],
        feature_names=feature_names_genes,
        max_display=10,
        plot_type="dot",
        show=False,
        rng=RNG,
    )
    plt.savefig(save_file, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_file}")

#  Main
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    appendix_dir = os.path.join(FIGURES_DIR, "appendix")
    os.makedirs(appendix_dir, exist_ok=True)

    # load data
    (x_test, y_test, gene_list, gene_mask,
     feature_names_genes, shap_mean_genes, shap_unc_genes) = load_data()

    # shared kwargs for cross-plots
    common = dict(
        shap_mean_genes=shap_mean_genes,
        shap_unc_genes=shap_unc_genes,
        feature_names_genes=feature_names_genes,
        meta=y_test,
        top_k_genes_unc=1000,
        top_k_genes_mean=1000,
        scientific_axis=True,
        line_len_px=40,
        label_pad_px=6,
    )

    # 1) cross-plot: all tissues / all pathways
    plot_shap_cross(
        **common,
        tissues="all",
        pathways="all",
        manual_dirs={
            "FAM26F": "right",
            "FAM19A2": "right",
            "GPR82": "down",
            "CALCA": "up",
            "PSMA8": "right",
        },
        save_file=os.path.join(
            FIGURES_DIR, "shap_cross_plot_tissues_all_pathways_all.pdf"
        ),
    )

    # 2) cross-plot: leukemia / ERK MAPK signaling
    plot_shap_cross(
        **common,
        tissues="leukemia",
        pathways="ERK MAPK signaling",
        manual_dirs={
            "STAR": "up",
            "FAM26F": "down",
            "SREBF1": "right",
        },
        save_file=os.path.join(
            FIGURES_DIR, "shap_cross_plot_tissues_leukemia_pathways_ERK_MAPK_signaling.pdf"
        ),
    )

    # 3) cross-plot: nervous_system / ERK MAPK signaling
    plot_shap_cross(
        **common,
        tissues="nervous_system",
        pathways="ERK MAPK signaling",
        manual_dirs={
            "LRP2": "down",
            "CALCA": "down",
            "RTFDC1": "up",
            "ALLC": "right",
            "GABRB2": "down",
            "PRKCG": "right",
            "SQSTM1": "up",
        },
        save_file=os.path.join(
            FIGURES_DIR, "shap_cross_plot_tissues_nervous_system_pathways_ERK_MAPK_signaling.pdf"
        ),
    )

    # 4) beeswarm: mean SHAP
    plot_beeswarm_summary(
        shap_mean_genes, x_test, gene_mask, feature_names_genes,
        save_file=os.path.join(appendix_dir, "shap_mean.pdf"),
    )

    # 5) beeswarm: uncertainty SHAP
    plot_beeswarm_summary(
        shap_unc_genes, x_test, gene_mask, feature_names_genes,
        save_file=os.path.join(appendix_dir, "shap_unc.pdf"),
    )

    print("\nDone.")