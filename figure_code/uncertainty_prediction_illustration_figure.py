"""
uncertainty_figures.py

Generate uncertainty-guided drug discovery figures:
* Combined figure: point predictions (top) + uncertainty-informed (bottom)
* Stratified uncertainty plot: random samples from uncertainty bins
"""
import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from scipy.stats import norm

from config import FIGURES_DIR, DATA_DIR

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})

GLOBAL_FONT_SIZE = 22

RESULTS_PATH = os.path.join(DATA_DIR, "experiments", "5_fold_cross_validation",
                            "results", "cv_pnne_cold_z_norm", "results.csv")
CELL_LINES_PATH = os.path.join(DATA_DIR, "GDSC",
                               "Cell_Lines_Details_with_origin_information.csv")
DRUG_LIST_PATH = os.path.join(DATA_DIR, "GDSC",
                              "Drug_listWed Dec  4 16_10_06 2024.csv")


def load_results():
    """Load model results with metadata and z-normalized scores."""
    results = pd.read_csv(RESULTS_PATH, index_col=0)
    results.y_aleatory_uncertainty = results.y_aleatory_uncertainty**0.5
    results.y_epistemic_uncertainty = results.y_epistemic_uncertainty**0.5
    results.y_uncertainty = results.y_uncertainty**0.5

    meta_info = (pd.read_csv(CELL_LINES_PATH, index_col=0)
                 .rename(columns={"Sample Name": "cell_lines"})
                 .set_index("cell_lines"))
    results = results.merge(meta_info, right_index=True, left_on="cell_lines")

    meta_info_drugs = pd.read_csv(DRUG_LIST_PATH, index_col=0)
    meta_info_drugs = meta_info_drugs[[" Name", " Target pathway"]].rename(
        columns={" Name": "drugs", " Target pathway": "Target pathway"})
    results = pd.merge(results, meta_info_drugs, left_on="drugs", right_on="drugs", how="left")
    results["Tissue descriptor 1"] = (results["Tissue descriptor 1"]
                                      .str.replace("_", " ").str.capitalize())

    # z-normalize per drug
    drug_stats = results.groupby("drugs").agg({"response": ["mean", "std"]}).reset_index()
    drug_stats.columns = ["drugs", "resp_mean", "resp_std"]
    results = results.merge(drug_stats, on="drugs")
    results["response_zscore"] = (results["response"] - results["resp_mean"]) / results["resp_std"]
    results["pred_zscore"] = (results["y_preds"] - results["resp_mean"]) / results["resp_std"]
    results["unc_zscore"] = results["y_uncertainty"] / results["resp_std"]

    return results


def get_example_data(results, tissue="Leukemia", drug_a_name="Sunitinib",
                     drug_b_name="FTY-720", cell_line="CESS"):
    """Extract example data for the two-drug comparison figure."""
    leukemia_data = results[results["Tissue descriptor 1"] == tissue].copy()
    print(f"Using cell line: {cell_line}")

    cell_data = leukemia_data[leukemia_data["cell_lines"] == cell_line]
    drug_a_data = cell_data[cell_data["drugs"] == drug_a_name].iloc[0]
    drug_b_data = cell_data[cell_data["drugs"] == drug_b_name].iloc[0]

    return {
        "pred_z_a": drug_a_data["pred_zscore"],
        "pred_z_b": drug_b_data["pred_zscore"],
        "unc_z_a": drug_a_data["unc_zscore"],
        "unc_z_b": drug_b_data["unc_zscore"],
        "true_z_a": drug_a_data["response_zscore"],
        "true_z_b": drug_b_data["response_zscore"],
    }


def plot_combined_figure(ex, save_path=None):
    """Point predictions (top) + uncertainty-informed (bottom)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": GLOBAL_FONT_SIZE,
        "axes.labelsize": GLOBAL_FONT_SIZE + 4,
        "xtick.labelsize": GLOBAL_FONT_SIZE + 2,
        "ytick.labelsize": GLOBAL_FONT_SIZE + 2,
        "legend.fontsize": GLOBAL_FONT_SIZE,
        "figure.dpi": 150,
        "mathtext.fontset": "dejavusans",
    })

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 16))

    threshold = -1.5
    y = np.linspace(-5, 5, 1000)

    c_mu = "#B22222"
    c_sigma = "#1565C0"
    c_mass_a = "#7B68EE"
    c_mass_b = "#20B2AA"
    c_sens = "#424242"

    mu_a, sig_a = ex["pred_z_a"], ex["unc_z_a"]
    mu_b, sig_b = ex["pred_z_b"], ex["unc_z_b"]

    # top panel: point predictions
    ax = ax1
    x_a, x_b = 0.28, 0.73

    ax.fill_between([-0.02, 1.02], threshold, -5, color=c_sens, alpha=0.06)
    ax.axhline(0, color="gray", linestyle="-", linewidth=1.5, alpha=0.3)
    ax.axhline(threshold, color=c_sens, linestyle="--", linewidth=5, alpha=0.85)

    mean_line_width = 0.03
    ax.hlines(mu_a, x_a - mean_line_width, x_a + mean_line_width,
              color=c_mu, linewidth=10, zorder=10)

    txt = ax.text(x_a + mean_line_width + 0.02, mu_a, r"$\hat{y}$",
                  fontsize=GLOBAL_FONT_SIZE + 8, color=c_mu, fontweight="bold",
                  ha="left", va="center")
    txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="white")])

    ax.scatter(x_a - 0.07, ex["true_z_a"], s=1000, marker="*",
               color="#FFD700", edgecolor="black", linewidth=3, zorder=15)

    ax.hlines(mu_b, x_b - mean_line_width, x_b + mean_line_width,
              color=c_mu, linewidth=10, zorder=10)

    txt = ax.text(x_b + mean_line_width + 0.02, mu_b, r"$\hat{y}$",
                  fontsize=GLOBAL_FONT_SIZE + 8, color=c_mu, fontweight="bold",
                  ha="left", va="center")
    txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="white")])

    ax.scatter(x_b - 0.07, ex["true_z_b"], s=1000, marker="*",
               color="#FFD700", edgecolor="black", linewidth=3, zorder=15)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-4.5, 4.0)
    ax.set_xticks([x_a, x_b])
    ax.set_xticklabels(["Sunitinib", "Fingolimod"], fontweight="bold",
                       fontsize=GLOBAL_FONT_SIZE + 2)
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_ylabel("Sensitivity (z-score)", fontweight="bold", fontsize=GLOBAL_FONT_SIZE + 4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.tick_params(width=3, length=8)
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_yticklabels([str(abs(i)) if i <= 0 else str(-i) for i in range(-4, 5, 1)])
    ax.invert_yaxis()

    txt = ax.text(0.98, threshold + 0.15, "Sensitivity\nThreshold", ha="right", va="bottom",
                  fontsize=GLOBAL_FONT_SIZE + 2, color=c_sens, fontweight="bold")
    txt.set_path_effects([path_effects.withStroke(linewidth=4, foreground="white")])

    legend_elements_top = [
        Line2D([0], [0], color=c_mu, linewidth=10, solid_capstyle="butt",
               label=r"Point prediction ($\hat{y}$)"),
        Line2D([0], [0], marker="*", color="#FFD700", linestyle="None",
               markersize=32, markeredgecolor="black", markeredgewidth=2.5,
               label="Ground truth"),
        Line2D([0], [0], color=c_sigma, linewidth=6,
               label=r"Uncertainty ($\hat{\sigma}$)"),
    ]
    legend = ax.legend(handles=legend_elements_top, loc="lower right", framealpha=0.97,
                       edgecolor="0.7", fontsize=GLOBAL_FONT_SIZE,
                       fancybox=True, shadow=False)
    legend.get_frame().set_linewidth(3)

    # bottom panel: uncertainty-informed
    ax = ax2
    x_a, x_b = 0.28, 0.73
    scale = 0.25

    prob_a = norm.cdf(threshold, mu_a, sig_a)
    prob_b = norm.cdf(threshold, mu_b, sig_b)

    pdf_a = norm.pdf(y, mu_a, sig_a) * scale
    pdf_b = norm.pdf(y, mu_b, sig_b) * scale

    sens_mask = y < threshold

    ax.fill_between([-0.02, 1.02], threshold, -5, color=c_sens, alpha=0.06)
    ax.axhline(0, color="gray", linestyle="-", linewidth=1.5, alpha=0.3)
    ax.axhline(threshold, color=c_sens, linestyle="--", linewidth=5, alpha=0.85)

    # drug A
    ax.fill_betweenx(y, x_a, x_a + pdf_a, where=~sens_mask,
                     color=c_mass_a, alpha=0.25, linewidth=0)
    ax.fill_betweenx(y, x_a, x_a + pdf_a, where=sens_mask,
                     color=c_mass_a, alpha=0.6, linewidth=0)
    ax.plot(x_a + pdf_a, y, color=c_mass_a, linewidth=6)

    ax.hlines(mu_a, x_a - mean_line_width, x_a + mean_line_width,
              color=c_mu, linewidth=10, zorder=10)
    ax.scatter(x_a - 0.16, ex["true_z_a"], s=1000, marker="*",
               color="#FFD700", edgecolor="black", linewidth=3, zorder=15)

    sig_top = mu_a + sig_a
    sig_bot = mu_a - sig_a
    bracket_x = x_a - 0.03
    ax.plot([bracket_x, bracket_x], [sig_bot, sig_top], color=c_sigma, linewidth=6, zorder=8)
    ax.plot([bracket_x - 0.015, bracket_x + 0.015], [sig_top, sig_top],
            color=c_sigma, linewidth=6, zorder=8)
    ax.plot([bracket_x - 0.015, bracket_x + 0.015], [sig_bot, sig_bot],
            color=c_sigma, linewidth=6, zorder=8)

    txt = ax.text(bracket_x - 0.02, mu_a, r"$\hat{\sigma}$",
                  fontsize=GLOBAL_FONT_SIZE + 8, color=c_sigma, fontweight="bold",
                  ha="right", va="center")
    txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="white")])

    txt = ax.text(x_a + mean_line_width + 0.02, mu_a, r"$\hat{\mu}$",
                  fontsize=GLOBAL_FONT_SIZE + 8, color=c_mu, fontweight="bold",
                  ha="left", va="center")
    txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="white")])

    # drug B
    ax.fill_betweenx(y, x_b, x_b + pdf_b, where=~sens_mask,
                     color=c_mass_b, alpha=0.25, linewidth=0)
    ax.fill_betweenx(y, x_b, x_b + pdf_b, where=sens_mask,
                     color=c_mass_b, alpha=0.6, linewidth=0)
    ax.plot(x_b + pdf_b, y, color=c_mass_b, linewidth=6)

    ax.hlines(mu_b, x_b - mean_line_width, x_b + mean_line_width,
              color=c_mu, linewidth=10, zorder=10)
    ax.scatter(x_b - 0.16, ex["true_z_b"], s=1000, marker="*",
               color="#FFD700", edgecolor="black", linewidth=3, zorder=15)

    sig_top_b = mu_b + sig_b
    sig_bot_b = mu_b - sig_b
    bracket_x_b = x_b - 0.03
    ax.plot([bracket_x_b, bracket_x_b], [sig_bot_b, sig_top_b],
            color=c_sigma, linewidth=6, zorder=8)
    ax.plot([bracket_x_b - 0.015, bracket_x_b + 0.015], [sig_top_b, sig_top_b],
            color=c_sigma, linewidth=6, zorder=8)
    ax.plot([bracket_x_b - 0.015, bracket_x_b + 0.015], [sig_bot_b, sig_bot_b],
            color=c_sigma, linewidth=6, zorder=8)

    txt = ax.text(bracket_x_b - 0.02, mu_b, r"$\hat{\sigma}$",
                  fontsize=GLOBAL_FONT_SIZE + 8, color=c_sigma, fontweight="bold",
                  ha="right", va="center")
    txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="white")])

    txt = ax.text(x_b + mean_line_width + 0.02, mu_b, r"$\hat{\mu}$",
                  fontsize=GLOBAL_FONT_SIZE + 8, color=c_mu, fontweight="bold",
                  ha="left", va="center")
    txt.set_path_effects([path_effects.withStroke(linewidth=5, foreground="white")])

    bbox_props = dict(boxstyle="round,pad=0.5", fc="white", ec="0.5",
                      linewidth=3, alpha=0.95)
    ax.text(x_a, -4.0, f"P(>1.5) = {prob_a:.0%}", ha="center",
            fontsize=GLOBAL_FONT_SIZE + 2, color=c_mass_a, fontweight="bold",
            bbox=bbox_props)
    ax.text(x_b, -4.0, f"P(>1.5) = {prob_b:.0%}", ha="center",
            fontsize=GLOBAL_FONT_SIZE + 2, color=c_mass_b, fontweight="bold",
            bbox=bbox_props)

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-4.5, 4.0)
    ax.set_xticks([x_a, x_b])
    ax.set_xticklabels(["Sunitinib", "Fingolimod"], fontweight="bold",
                       fontsize=GLOBAL_FONT_SIZE + 2)
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_ylabel("Sensitivity (z-score)", fontweight="bold", fontsize=GLOBAL_FONT_SIZE + 4)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.spines["bottom"].set_linewidth(3)
    ax.tick_params(width=3, length=8)
    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_yticklabels([str(abs(i)) if i <= 0 else str(-i) for i in range(-4, 5, 1)])
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


def plot_stratified_uncertainty(results, cell_line="CESS", n_total=20, save_path=None):
    """Stratified uncertainty plot with random samples from uncertainty bins."""
    font_size = 18
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size + 4,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size + 2,
        "legend.fontsize": font_size - 2,
        "figure.dpi": 150,
        "mathtext.fontset": "dejavusans",
    })

    # prepare per-drug z-scores for the stratified plot
    df = results[results["cell_lines"] == cell_line].copy()
    df = df.drop_duplicates(subset=["drugs"])

    df["response_z"] = df["response_zscore"]
    df["pred_z"] = df["pred_zscore"]
    df["sigma_z"] = df["unc_zscore"]

    threshold = -1.5
    df["prob_sens"] = norm.cdf(threshold, df["pred_z"], df["sigma_z"])

    df = df.dropna(subset=["sigma_z"])
    df["unc_bin"] = pd.qcut(df["sigma_z"], q=3,
                            labels=["Low Uncertainty", "Medium Uncertainty", "High Uncertainty"])

    # Always include the drugs from the first plot
    must_include = ["Sunitinib", "FTY-720"]
    forced = df[df["drugs"].isin(must_include)]

    drugs_per_bin = n_total // 3
    print(f"Randomly sampling {drugs_per_bin} drugs from each uncertainty tier...")

    selected_drugs = [forced]
    for bin_label in ["Low Uncertainty", "Medium Uncertainty", "High Uncertainty"]:
        bin_data = df[(df["unc_bin"] == bin_label) & (~df["drugs"].isin(must_include))]
        n_sample = min(len(bin_data), drugs_per_bin)
        if n_sample > 0:
            subset = bin_data.sample(n=n_sample, random_state=1)
            selected_drugs.append(subset)

    final_df = pd.concat(selected_drugs).drop_duplicates(subset=["drugs"])
    final_df = final_df.sort_values("prob_sens", ascending=False)

    n_drugs = len(final_df)
    fig_width = max(14, n_drugs * 0.7)
    fig, ax = plt.subplots(figsize=(fig_width, 9))

    y = np.linspace(-5, 5, 1000)

    c_sens = "#424242"
    bin_colors = {
        "Low Uncertainty": "#2E7D32",
        "Medium Uncertainty": "#F57C00",
        "High Uncertainty": "#C62828",
    }

    ax.fill_between([-0.5, n_drugs - 0.5], threshold, -5, color=c_sens, alpha=0.06)
    ax.axhline(0, color="gray", linestyle="-", linewidth=1.5, alpha=0.3)
    ax.axhline(threshold, color=c_sens, linestyle="--", linewidth=3, alpha=0.85)

    scale = 0.5

    for i, (_, row) in enumerate(final_df.iterrows()):
        x_pos = i
        mu = row["pred_z"]
        sigma = row["sigma_z"]
        true_val = row["response_z"]
        prob = row["prob_sens"]
        bin_label = row["unc_bin"]

        dist_color = bin_colors[bin_label]
        pdf = norm.pdf(y, mu, sigma) * scale
        sens_mask = y < threshold

        ax.fill_betweenx(y, x_pos, x_pos + pdf, where=sens_mask,
                         color=dist_color, alpha=0.6, linewidth=0)
        ax.fill_betweenx(y, x_pos, x_pos + pdf, where=~sens_mask,
                         color=dist_color, alpha=0.2, linewidth=0)
        ax.plot(x_pos + pdf, y, color="black", linewidth=0.8, alpha=0.5)

        ax.scatter(x_pos, mu, s=80, color="white", edgecolor="black", linewidth=2, zorder=10)
        ax.scatter(x_pos, true_val, s=300, marker="*",
                   color="#FFD700", edgecolor="black", linewidth=1.5, zorder=15)

        ax.text(x_pos, 4.6, f"σ={sigma:.2f}", ha="center", va="bottom",
                fontsize=font_size - 4, color=dist_color, fontweight="bold")
        ax.text(x_pos, -4.7, f"{prob:.0%}", ha="center", va="top",
                fontsize=font_size - 2, color=dist_color, fontweight="bold")

    ax.set_xlim(-0.7, n_drugs - 0.3)
    ax.set_ylim(-5.0, 5.2)

    ax.set_xticks(np.arange(n_drugs))
    drug_display_names = {"FTY-720": "Fingolimod"}
    labels = [drug_display_names.get(n, n) for n in final_df["drugs"]]
    labels = [n[:12] + ".." if len(n) > 12 else n for n in labels]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=font_size - 2)

    ax.set_yticks(np.arange(-4, 5, 1))
    ax.set_yticklabels([str(abs(i)) if i <= 0 else str(-i) for i in range(-4, 5, 1)])
    ax.set_ylabel("Sensitivity (z-score)", fontweight="bold", fontsize=font_size + 4)
    ax.invert_yaxis()

    legend_elements = [
        mpatches.Patch(facecolor=bin_colors["Low Uncertainty"], alpha=0.6,
                       label="Low uncertainty"),
        mpatches.Patch(facecolor=bin_colors["Medium Uncertainty"], alpha=0.6,
                       label="Medium uncertainty"),
        mpatches.Patch(facecolor=bin_colors["High Uncertainty"], alpha=0.6,
                       label="High uncertainty"),
        Line2D([0], [0], marker="o", color="white", markerfacecolor="white",
               markeredgecolor="black", markersize=10, label=r"Prediction ($\hat{\mu}$)"),
        Line2D([0], [0], marker="*", color="#FFD700", linestyle="None",
               markersize=18, markeredgecolor="black", markeredgewidth=1.5,
               label="Ground truth"),
    ]

    legend = ax.legend(handles=legend_elements, loc="upper right",
                       bbox_to_anchor=(0.99, 0.85), framealpha=0.95,
                       edgecolor="0.7", ncol=2)
    legend.get_frame().set_linewidth(2)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    ax.tick_params(width=2, length=6)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")

    plt.close(fig)
    return fig


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = load_results()
    example = get_example_data(results)

    plot_combined_figure(
        example,
        save_path=os.path.join(FIGURES_DIR, "uncertainty_prediction_illustration.pdf"),
    )

    appendix_dir = os.path.join(FIGURES_DIR, "appendix")
    os.makedirs(appendix_dir, exist_ok=True)

    plot_stratified_uncertainty(
        results,
        cell_line="CESS",
        n_total=35,
        save_path=os.path.join(appendix_dir, "figure_stratified_uncertainty.pdf"),
    )

    print("\nDone.")