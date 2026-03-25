import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

from config import (
    MODELS, MODEL_COLORS, LINESTYLES, MODEL_TO_FULL_NAME, SHIFTS,
    PNNE_COLORS, PNNE_LINESTYLES,
    EDL_COLORS, EDL_LINESTYLES,
    COLD_START_SUFFIX, SUFFIX, BASE_PATH, FIGURES_DIR, TABLES_DIR,
)

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})


def ood_path(model, fold, shift):
    return f"{BASE_PATH}/cv_{model}{COLD_START_SUFFIX}{SUFFIX}/ood/results_{fold}_shift_{shift}.csv"

def id_path(model):
    return f"{BASE_PATH}/cv_{model}{COLD_START_SUFFIX}{SUFFIX}/results.csv"


def build_labels_and_scores(id_scores, ood_scores):
    y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    s = np.concatenate([id_scores, ood_scores])
    return y_true, s

def auroc_from_scores(id_scores, ood_scores):
    y_true, s = build_labels_and_scores(id_scores, ood_scores)
    if np.allclose(s.min(), s.max()):
        return 0.5
    return roc_auc_score(y_true, s)

DECOMP_COLORS = {"pnne": PNNE_COLORS, "edl": EDL_COLORS}
DECOMP_LINESTYLES = {"pnne": PNNE_LINESTYLES, "edl": EDL_LINESTYLES}

def uncertainty_cols(model):
    if model in ("pnne", "edl"):
        return [("Total", "y_uncertainty"),
                ("Epistemic", "y_epistemic_uncertainty"),
                ("Aleatoric", "y_aleatory_uncertainty")]
    return [("Total", "y_uncertainty")]

def format_index(m, c):
    if m == MODEL_TO_FULL_NAME["pnne"]:
        return f"Gaussian NN\nEns. {c}"
    if c != "Total":
        return f"{m}\n{c}"
    return m


result_models = {}
for model in MODELS:
    frames = []
    for i in range(5):
        for shift in SHIFTS:
            df = pd.read_csv(ood_path(model, i, shift), index_col=0)
            df["shift"] = float(shift)
            frames.append(df)
    result_models[model] = pd.concat(frames, ignore_index=True)

unperturbed = {}
for model in MODELS:
    unperturbed[model] = pd.read_csv(id_path(model))


variation = "std"  # {"stderr", "std"}
font_adder = 6

records = []
for model in MODELS:
    id_df = unperturbed[model]
    for comp_name, col in uncertainty_cols(model):
        for shift in SHIFTS:
            fold_aurocs = []
            for i in range(5):
                df = pd.read_csv(ood_path(model, i, shift), index_col=0)
                fold_aurocs.append(auroc_from_scores(id_df[col].values, df[col].values))
            fold_aurocs = np.array(fold_aurocs)
            std_val = fold_aurocs.std(ddof=1)
            records.append({
                "model": MODEL_TO_FULL_NAME[model],
                "component": comp_name,
                "shift": float(shift),
                "mean": fold_aurocs.mean(),
                "std": std_val,
                "stderr": std_val / np.sqrt(len(fold_aurocs)),
            })

auroc_df = pd.DataFrame(records)


os.makedirs(TABLES_DIR, exist_ok=True)

table_mean = auroc_df.pivot_table(index=["model", "component"], columns="shift", values="mean")
table_mean.to_csv(os.path.join(TABLES_DIR, "ood_auroc_by_shift_mean.csv"))

table_var = auroc_df.pivot_table(index=["model", "component"], columns="shift", values=variation)
table_var.to_csv(os.path.join(TABLES_DIR, f"ood_auroc_by_shift_{variation}.csv"))

print(table_mean.round(3))


orig_cmap = matplotlib.colormaps["cividis"]
bright_cmap = mcolors.LinearSegmentedColormap.from_list(
    "bright_cividis",
    [orig_cmap(i**0.7 / 255**0.7) for i in range(256)],
)

heat_mean = auroc_df.pivot_table(index=["model", "component"], columns="shift", values="mean")
heat_mean.index = [format_index(m, c) for (m, c) in heat_mean.index]

cols_arr = np.array(heat_mean.columns, dtype=float)
sel_col = cols_arr[np.argmin(np.abs(cols_arr - 0.5))]
order = np.argsort(-heat_mean[sel_col].fillna(-np.inf).values)
heat_mean = heat_mean.iloc[order]

heat_var = auroc_df.pivot_table(index=["model", "component"], columns="shift", values=variation)
heat_var.index = [format_index(m, c) for (m, c) in heat_var.index]
heat_var = heat_var.loc[heat_mean.index, heat_mean.columns]

os.makedirs(os.path.join(FIGURES_DIR, "appendix"), exist_ok=True)


def plot_heatmap(heat_mean, heat_var, show_variation, save_path):
    vals = heat_mean.values
    fig, ax = plt.subplots(figsize=(14 if show_variation else 10, max(4, 0.6 * len(heat_mean.index))))
    im = ax.imshow(vals, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=bright_cmap)

    ax.set_yticks(np.arange(len(heat_mean.index)))
    ax.set_yticklabels(heat_mean.index, fontsize=7 + font_adder, va="center")
    ax.set_xticks(np.arange(len(heat_mean.columns)))
    ax.set_xticklabels([str(s) for s in heat_mean.columns], fontsize=7 + font_adder)
    ax.set_xlabel("Shift", fontsize=9 + font_adder)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            m = vals[i, j]
            if np.isnan(m):
                continue
            text = f"{m:.2f}±{heat_var.values[i, j]:.2f}" if show_variation else f"{m:.2f}"
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if m < 0.4 else "black", fontsize=6 + font_adder)

    fig.colorbar(im, ax=ax, label="AUROC", orientation="horizontal",
                 location="top", pad=0.03, shrink=0.6)
    fig.tight_layout(pad=3)
    fig.savefig(save_path, dpi=350)


plot_heatmap(heat_mean, heat_var, show_variation=True,
             save_path=os.path.join(FIGURES_DIR, "appendix", "ood_auroc_heatmap_with_variation.pdf"))
plot_heatmap(heat_mean, heat_var, show_variation=False,
             save_path=os.path.join(FIGURES_DIR, "ood_auroc_heatmap.pdf"))


font_adder = 8
n_cols = 4
n_rows = int(np.ceil(len(SHIFTS) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
axes = axes.flatten()

for ax_idx, sft in enumerate(SHIFTS):
    ax = axes[ax_idx]
    for model in MODELS:
        res = result_models[model].copy()
        id_df = unperturbed[model]

        res["shift"] = res["shift"].astype(str)
        res.loc[res["shift"] == "1.0", "shift"] = "1"

        if model in DECOMP_COLORS:
            for comp_name, col in uncertainty_cols(model):
                ood_scores = res.loc[res["shift"] == sft, col].values
                id_scores = id_df[col].values
                y_true, scores = build_labels_and_scores(id_scores, ood_scores)
                if np.allclose(scores.min(), scores.max()):
                    fpr, tpr = np.array([0, 1]), np.array([0, 1])
                else:
                    fpr, tpr, _ = roc_curve(y_true, scores)
                ax.plot(fpr, tpr,
                        label=f"{MODEL_TO_FULL_NAME[model]}\n ({comp_name})",
                        color=DECOMP_COLORS[model][comp_name],
                        linestyle=DECOMP_LINESTYLES[model][comp_name],
                        linewidth=3, clip_on=False)
        else:
            col = "y_uncertainty"
            ood_scores = res.loc[res["shift"] == sft, col].values
            id_scores = id_df[col].values
            y_true, scores = build_labels_and_scores(id_scores, ood_scores)
            if np.allclose(scores.min(), scores.max()):
                fpr, tpr = np.array([0, 1]), np.array([0, 1])
            else:
                fpr, tpr, _ = roc_curve(y_true, scores)
            ax.plot(fpr, tpr,
                    label=MODEL_TO_FULL_NAME[model],
                    color=MODEL_COLORS[model], linestyle=LINESTYLES[model],
                    linewidth=3, clip_on=False)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(f"Shift {sft}", fontsize=8 + font_adder)
    ax.set_xlabel("FPR", fontsize=7 + font_adder)
    ax.set_ylabel("TPR", fontsize=7 + font_adder)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="both", labelsize=6 + font_adder)

for ax_idx in range(len(SHIFTS), len(axes) - 1):
    fig.delaxes(axes[ax_idx])

legend_ax = axes[-1]
handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(handles, labels, loc="center", fontsize=7 + font_adder, ncol=1)
legend_ax.axis("off")

plt.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "appendix", "ood_roc_by_shift.pdf"), dpi=300)
