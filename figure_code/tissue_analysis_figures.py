import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
from scipy.stats import kruskal
import seaborn as sns
import scikit_posthocs as sp
import matplotlib
import matplotlib.pyplot as plt
from curlyBrace import curlyBrace

from config import (
    MODEL_TO_FULL_NAME, COLD_START_SUFFIX, SUFFIX, BASE_PATH, FIGURES_DIR,
)

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})


def load_pnne_with_metadata():
    results = pd.read_csv(
        os.path.join(BASE_PATH, f"cv_pnne{COLD_START_SUFFIX}{SUFFIX}", "results.csv"), index_col=0)
    results["y_aleatory_uncertainty"] = results["y_aleatory_uncertainty"] ** 0.5
    results["y_epistemic_uncertainty"] = results["y_epistemic_uncertainty"] ** 0.5
    results["y_uncertainty"] = results["y_uncertainty"] ** 0.5

    data_dir = os.path.join(os.path.dirname(BASE_PATH), "..", "..", "GDSC")
    meta_info = pd.read_csv(
        os.path.join(data_dir, "Cell_Lines_Details_with_origin_information.csv"), index_col=0
    ).rename(columns={"Sample Name": "cell_lines"}).set_index("cell_lines")
    results = results.merge(meta_info, right_index=True, left_on="cell_lines")

    meta_drugs = pd.read_csv(os.path.join(data_dir, "Drug_listWed Dec  4 16_10_06 2024.csv"), index_col=0)
    meta_drugs = meta_drugs[[" Name", " Target pathway"]].rename(
        columns={" Name": "drugs", " Target pathway": "Target pathway"})
    results = pd.merge(results, meta_drugs, left_on="drugs", right_on="drugs", how="left")

    results["Tissue descriptor 1"] = (
        results["Tissue descriptor 1"].str.replace("_", " ").str.capitalize()
    )
    return results


def plot_uncertainty_boxplot(df, grouping, unique_id_col, xlabel, count_label,
                             save_path, dropna_group=False, drop_groups=None,
                             font_adder=9, uncertainty_type="y_uncertainty"):
    if uncertainty_type not in df.columns:
        raise ValueError(f"'{uncertainty_type}' not found in DataFrame columns.")
    df = df.copy()
    if dropna_group:
        df = df.dropna(subset=[grouping])
    if drop_groups is not None:
        df = df[~df[grouping].isin(drop_groups)]

    base_fontsize = 20 + font_adder
    plt.rcParams.update({"font.size": base_fontsize})

    unique_entities = df.drop_duplicates(subset=[unique_id_col, grouping])
    if dropna_group:
        unique_entities = unique_entities.dropna(subset=[grouping])
    if drop_groups is not None:
        unique_entities = unique_entities[~unique_entities[grouping].isin(drop_groups)]

    grouped = df.groupby(grouping)[uncertainty_type]
    group_order = list(grouped.median().sort_values().index)
    counts = unique_entities[grouping].value_counts().loc[group_order]
    positions = np.arange(len(group_order))

    groups = [df[df[grouping] == grp][uncertainty_type] for grp in group_order]
    kw_stat, kw_p = kruskal(*groups)
    print(f"Kruskal-Wallis test: statistic = {kw_stat:.3f}, p-value = {kw_p:.3e}")

    dunn_results = sp.posthoc_dunn(df, val_col=uncertainty_type, group_col=grouping, p_adjust="holm")
    dunn_results = dunn_results.loc[group_order, group_order]
    print("\nPairwise Dunn's test p-values (Holm-adjusted):\n", dunn_results)

    clusters = []
    current_cluster = [group_order[0]]
    for grp in group_order[1:]:
        all_ns = all(dunn_results.loc[prev, grp] >= 0.05 for prev in current_cluster)
        if all_ns:
            current_cluster.append(grp)
        else:
            clusters.append(current_cluster)
            current_cluster = [grp]
    clusters.append(current_cluster)
    print("\nNon-significant clusters:", clusters)

    cluster_palette = {}
    base_colors = sns.color_palette("Set2", n_colors=len(clusters))
    for idx, cluster in enumerate(clusters):
        for grp in cluster:
            cluster_palette[grp] = base_colors[idx]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(20, 20),
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.05},
    )
    rename_map = {
        "Chromatin histone methylation": "Chromatin histone\nmethylation",
        "Chromatin histone acetylation": "Chromatin histone\nacetylation",
        "Protein stability and degradation": "Protein stability\nand degradation",
    }
    plot_labels = [rename_map.get(lbl, lbl) for lbl in group_order]

    sns.boxplot(x=grouping, y=uncertainty_type, hue=grouping, data=df, ax=ax1,
                palette=cluster_palette, order=group_order, hue_order=group_order, legend=False)
    for coll in ax1.collections:
        coll.set_rasterized(True)

    ax1.set_xticklabels([])
    ax1.set_xlabel("")
    ax1.set_ylabel("Predicted Uncertainty", fontsize=22 + font_adder)
    ax1.set_xlim(-0.5, len(group_order) - 0.5)

    ax2.bar(positions, counts.values, color="gray", alpha=0.7)
    ax2.set_xlim(-0.5, len(group_order) - 0.5)
    ax2.set_ylabel(count_label, fontsize=22 + font_adder)
    ax2.set_xlabel(xlabel, fontsize=22 + font_adder)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(plot_labels, rotation=90, ha="center", fontsize=20 + font_adder)

    fig.subplots_adjust(hspace=0.05)

    group_max = {grp: df[df[grouping] == grp][uncertainty_type].max() for grp in group_order}
    y_range = df[uncertainty_type].max() - df[uncertainty_type].min()
    offset = y_range * 0.05

    for cluster in clusters:
        if len(cluster) < 2:
            continue
        x_positions = [group_order.index(grp) for grp in cluster]
        x_min, x_max = min(x_positions), max(x_positions)
        cluster_y_max = max(group_max[grp] for grp in cluster)
        y = cluster_y_max + offset
        curlyBrace(fig, ax1, p1=[x_min, y], p2=[x_max, y], k_r=0.1,
                   bool_auto=True, str_text="ns",
                   fontdict={"fontsize": 16 + font_adder, "color": "k"}, lw=2, color="k")

    ax1.set_rasterization_zorder(0)
    for child in ax1.get_children():
        try:
            child.set_rasterized(True)
        except Exception:
            pass

    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=290)
    print(f"Saved: {save_path}")


def plot_tissue_pathway_clustermap(results, save_path, uncertainty_column="y_uncertainty", font_adder=12):
    filtered = results[~results["Target pathway"].isin(["Unclassified", "Other"])].copy()
    filtered["Tissue descriptor 1"] = (
        filtered["Tissue descriptor 1"].astype(str).str.replace("_", " ").str.title()
    )
    acronym_fixes = {"Lung Nsclc": "Lung NSCLC", "Lung Sclc": "Lung SCLC"}
    filtered["Tissue descriptor 1"] = filtered["Tissue descriptor 1"].replace(acronym_fixes)

    rename_map = {
        "Chromatin histone methylation": "Chromatin histone\nmethylation",
        "Chromatin histone acetylation": "Chromatin histone\nacetylation",
        "Protein stability and degradation": "Protein stability\nand degradation",
    }
    filtered["Target pathway"] = filtered["Target pathway"].replace(rename_map)

    heatmap_data = filtered.rename({"Tissue descriptor 1": "Tissue"}, axis=1).pivot_table(
        index="Tissue", columns="Target pathway", values=uncertainty_column, aggfunc="mean")

    g = sns.clustermap(
        # Wider than tall: the manuscript constrains this figure by width, so a
        # squarer figure just costs page height. Cells stay rectangular and no
        # text shrinks.
        heatmap_data, cmap="coolwarm", figsize=(24, 15),
        cbar_kws={"label": "Mean Uncertainty", "orientation": "horizontal"},
        row_cluster=True, col_cluster=True,
        xticklabels=True, yticklabels=True,
        dendrogram_ratio=(0.12, 0.12), linewidths=0.5,
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
    g.ax_heatmap.set_xlabel("Target Pathway", fontsize=23 + font_adder)
    g.ax_heatmap.set_ylabel("Tissue", fontsize=23 + font_adder)
    g.ax_cbar.set_xlabel("Mean \nUncertainty", fontsize=18 + font_adder)

    for tick in g.ax_heatmap.get_xticklabels():
        tick.set_fontsize(16 + font_adder)
    for tick in g.ax_heatmap.get_yticklabels():
        tick.set_fontsize(16 + font_adder)

    # Park the colourbar in the empty bottom-left corner, below the row dendrogram and
    # left of the rotated column labels. Absolute figure coordinates, not offsets from
    # seaborn's default top-left slot -- that slot shrinks with the figure height and
    # the bar ended up overlapping the dendrogram.
    # Narrow enough that the right-hand tick clears the first rotated column label,
    # which starts at the heatmap's left edge (~0.13 in figure coordinates).
    g.cax.set_position([0.030, 0.100, 0.075, 0.015])
    data_min = heatmap_data.min().min()
    data_max = heatmap_data.max().max()
    g.ax_cbar.set_xticks([data_min, data_max])
    g.ax_cbar.set_xticklabels([round(data_min, 2), round(data_max, 2)])
    # rcParams["font.size"] is raised to ~29 earlier in this module, which the colourbar
    # ticks would otherwise inherit and overrun their slot.
    g.ax_cbar.tick_params(labelsize=16 + font_adder)

    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs(os.path.join(FIGURES_DIR, "appendix"), exist_ok=True)

    results = load_pnne_with_metadata()

    # main figures: total uncertainty
    plot_uncertainty_boxplot(
        results, grouping="Tissue descriptor 1", unique_id_col="cell_lines",
        xlabel="Tissue", count_label="Cell Line Count",
        save_path=os.path.join(FIGURES_DIR, "uncertainty_boxplot_tissues_annotated.pdf"))

    plot_uncertainty_boxplot(
        results, grouping="Target pathway", unique_id_col="drugs",
        xlabel="Drug Target Pathway", count_label="Drug Count",
        save_path=os.path.join(FIGURES_DIR, "uncertainty_boxplot_pathways_annotated.pdf"),
        dropna_group=True, drop_groups=["Other", "Unclassified"])

    plot_tissue_pathway_clustermap(
        results, save_path=os.path.join(FIGURES_DIR, "clustermap_tissues_pathways.pdf"))

    # appendix: epistemic
    plot_uncertainty_boxplot(
        results, grouping="Tissue descriptor 1", unique_id_col="cell_lines",
        xlabel="Tissue", count_label="Cell Line Count",
        save_path=os.path.join(FIGURES_DIR, "appendix", "uncertainty_boxplot_tissues_epistemic_annotated.pdf"),
        uncertainty_type="y_epistemic_uncertainty")

    plot_uncertainty_boxplot(
        results, grouping="Target pathway", unique_id_col="drugs",
        xlabel="Drug Target Pathway", count_label="Drug Count",
        save_path=os.path.join(FIGURES_DIR, "appendix", "uncertainty_boxplot_pathways_epistemic_annotated.pdf"),
        dropna_group=True, drop_groups=["Other", "Unclassified"],
        uncertainty_type="y_epistemic_uncertainty")

    # appendix: aleatoric
    plot_uncertainty_boxplot(
        results, grouping="Tissue descriptor 1", unique_id_col="cell_lines",
        xlabel="Tissue", count_label="Cell Line Count",
        save_path=os.path.join(FIGURES_DIR, "appendix", "uncertainty_boxplot_tissues_aleatoric_annotated.pdf"),
        uncertainty_type="y_aleatory_uncertainty")

    plot_uncertainty_boxplot(
        results, grouping="Target pathway", unique_id_col="drugs",
        xlabel="Drug Target Pathway", count_label="Drug Count",
        save_path=os.path.join(FIGURES_DIR, "appendix", "uncertainty_boxplot_pathways_aleatoric_annotated.pdf"),
        dropna_group=True, drop_groups=["Other", "Unclassified"],
        uncertainty_type="y_aleatory_uncertainty")