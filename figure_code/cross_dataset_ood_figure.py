"""Cross-dataset + synthetic out-of-distribution detection figure (Figure 3).

Self-contained: reads the small pre-aggregated summary in data/cross_dataset_ood_summary.csv.gz
(detection AUROC and transfer MSE per model x scenario / shift) and renders the two-panel AUROC
heatmap plus the transfer-MSE appendix table. The raw per-pair transfer predictions used to build
the summary are on Zenodo; the code that generates them is under cross_dataset/.
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from config import FIGURES_DIR, TABLES_DIR, SCRIPT_DIR

matplotlib.rcParams.update({"text.usetex": False, "font.family": "sans-serif", "font.sans-serif": ["Arial"]})
_civ = matplotlib.colormaps["cividis"]
BRIGHT_CIVIDIS = mcolors.LinearSegmentedColormap.from_list(
    "bright_cividis", [_civ(i ** 0.7 / 255 ** 0.7) for i in range(256)])

DATA = os.path.join(SCRIPT_DIR, "data", "cross_dataset_ood_summary.csv.gz")


def load():
    df = pd.read_csv(DATA)
    rows = list(dict.fromkeys(df["model"]))                     # model plot order
    syn = df[df["panel"] == "synthetic"]
    cross = df[df["panel"] == "cross"]
    shift_cols = list(dict.fromkeys(syn["col"]))
    xcols = list(dict.fromkeys(cross["col"]))
    syn_det = syn.pivot(index="model", columns="col", values="value").loc[rows, shift_cols].to_numpy(float)
    xa = cross[cross["metric"] == "auroc"].pivot(index="model", columns="col", values="value").loc[rows, xcols]
    xm = cross[cross["metric"] == "mse"].pivot(index="model", columns="col", values="value").loc[rows, xcols]
    disp = lambda s: s.replace("\\n", "\n")
    return ([disp(r) for r in rows], shift_cols, [disp(c) for c in xcols], xcols,
            syn_det, xa.to_numpy(float), xm.to_numpy(float))


def make_figure():
    rows, shift_cols, xd_labels, xcols, syn_det, xd_det, xd_mse = load()
    FS_CELL, FS_TICK, FS_LABEL, FS_TITLE = 12, 12, 14, 15

    fig = plt.figure(figsize=(16, max(5, 0.95 * len(rows))))
    gs = GridSpec(1, 2, width_ratios=[len(shift_cols), len(xcols) * 1.9], wspace=0.05)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    axL.imshow(syn_det, cmap=BRIGHT_CIVIDIS, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axL.set_title("Synthetic OOD (signed perturbation)", fontsize=FS_TITLE)
    axL.set_xlabel("Perturbation magnitude μ", fontsize=FS_LABEL)
    axL.set_xticks(range(len(shift_cols))); axL.set_xticklabels(shift_cols, fontsize=FS_TICK)
    axL.set_yticks(range(len(rows))); axL.set_yticklabels(rows, fontsize=FS_TICK, va="center")
    for i in range(syn_det.shape[0]):
        for j in range(syn_det.shape[1]):
            v = syn_det[i, j]
            axL.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                     color="white" if v < 0.4 else "black", fontweight="bold")

    imR = axR.imshow(xd_det, cmap=BRIGHT_CIVIDIS, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    axR.set_title("Cross-dataset OOD", fontsize=FS_TITLE)
    axR.set_xlabel("Transfer scenario", fontsize=FS_LABEL)
    axR.set_xticks(range(len(xcols))); axR.set_xticklabels(xd_labels, rotation=30, ha="right", fontsize=FS_TICK)
    axR.set_yticks([])
    for i in range(xd_det.shape[0]):
        for j in range(xd_det.shape[1]):
            v = xd_det[i, j]
            axR.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=FS_CELL,
                     color="white" if v < 0.4 else "black", fontweight="bold")

    axL.text(-0.06, 1.05, "a", transform=axL.transAxes, fontsize=20, fontweight="bold", va="bottom", ha="right")
    axR.text(-0.04, 1.05, "b", transform=axR.transAxes, fontsize=20, fontweight="bold", va="bottom", ha="right")
    cb = fig.colorbar(imR, ax=[axL, axR], orientation="vertical", location="right", pad=0.02, shrink=0.85)
    cb.set_label("AUROC", fontsize=FS_LABEL); cb.ax.tick_params(labelsize=FS_TICK)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    out = os.path.join(FIGURES_DIR, "cross_dataset_ood.pdf")
    fig.savefig(out, bbox_inches="tight"); fig.savefig(out.replace(".pdf", ".png"), dpi=350, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    return rows, xd_labels, xd_mse


def make_mse_table(rows, xd_labels, xd_mse):
    def short(lab):
        lab = lab.replace("\n", " ")
        if "in-dist" in lab:    return "in-dist"
        if "BeatAML" in lab:    return "CTRPv2$\\rightarrow$BeatAML"
        if "harmonized" in lab: return "CTRPv2$\\rightarrow$GDSC (harm.)"
        return "CTRPv2$\\rightarrow$GDSC (native)"
    mcol = [short(c) for c in xd_labels]
    tex = [r"\begin{table}[t]\centering",
           r"\caption{Transfer mean squared error (per-drug z-scored response) for the cross-dataset "
           r"out-of-distribution scenarios of \Cref{fig:ood}(b), ordered by increasing shift. GDSC "
           r"transfers are averaged over five random cell-line splits; BeatAML uses a single split.}",
           r"\label{tab:transfer_mse}", r"\begin{tabular}{l" + "r" * len(mcol) + "}", r"\toprule",
           "Model & " + " & ".join(mcol) + r" \\", r"\midrule"]
    for i, rn in enumerate(rows):
        vals = " & ".join(f"{xd_mse[i, j]:.2f}" if xd_mse[i, j] < 100 else f"{xd_mse[i, j]:.0f}"
                          for j in range(xd_mse.shape[1]))
        tex.append(rn.replace("\n", " ") + " & " + vals + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(TABLES_DIR, exist_ok=True)
    with open(os.path.join(TABLES_DIR, "transfer_mse_table.tex"), "w") as f:
        f.write("\n".join(tex))
    print("saved", os.path.join(TABLES_DIR, "transfer_mse_table.tex"))


if __name__ == "__main__":
    make_mse_table(*make_figure())
