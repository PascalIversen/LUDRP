import sys
sys.dont_write_bytecode = True
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
path_results = os.path.join(script_dir, "..", "examples", "data", "experiments",
                            "5_fold_cross_validation", "case_specific_finetune", "test_run_top60")
results = []
for i in range(5):
    r = pd.read_csv(f"{path_results}/split_{i}/case_specific_finetune_results.csv", index_col=0).dropna()
    results.append(r)
results = pd.concat(results)


def plot_pairwise_diff(results_df, col1, col2, label, p_floor=1e-16, fontsize=9, save_pdf=True, inter=None):
    """Plot difference col1 - col2 as vertical boxplot with p-value in bottom right."""
    d1 = results_df[col1]
    d2 = results_df[col2]
    diff = d1 - d2
    vals = diff.dropna()

    # stats
    mean = np.mean(vals)
    std = np.std(vals, ddof=1)
    median = np.median(vals)
    try:
        _, p = stats.wilcoxon(d1.dropna(), d2.dropna())
        p_str = f"p < {p_floor:.0e}" if p < p_floor else f"p = {p:.0e}"
    except Exception:
        p_str = "p = n/a"

    print(f"{label}: mean = {mean:.4f} ± {std:.4f}, "
          f"median = {median:.4f}, n = {len(vals)}, {p_str}")

    # styling
    plt.rcParams.update({
        "font.size": fontsize,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"]
    })

    fig, ax = plt.subplots(figsize=(3, 5), dpi=300)
    boxprops     = dict(linewidth=1.5, color="black", facecolor="#DCE6F1")
    whiskerprops = dict(linewidth=1.2, color="black")
    capprops     = dict(linewidth=1.2, color="black")
    medianprops  = dict(linewidth=1.8, color="red")
    flierprops   = dict(marker="o", markersize=4,
                        markerfacecolor="gray", alpha=0.5)

    ax.boxplot([vals],
               vert=True,
               widths=0.6,
               patch_artist=True,
               boxprops=boxprops,
               whiskerprops=whiskerprops,
               capprops=capprops,
               medianprops=medianprops,
               flierprops=flierprops)

    ax.set_xticks([1])
    ax.set_xticklabels([label], fontsize=fontsize, rotation=0, ha="center")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)

    # add p-value in bottom right corner
    if len(vals) > 0:
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()

        # Position in bottom right with some padding
        x_pos = x_max - 0.05 * (x_max - x_min)
        y_pos = y_min + 0.05 * (y_max - y_min)

        ax.text(x_pos, y_pos, p_str, ha="right", va="bottom", fontsize=fontsize,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # Save as PDF
    if save_pdf:
        if "Uncertainty – Random" in label:
            base_name = "uncertainty_vs_random"
        elif "Uncertainty – Baseline" in label:
            base_name = "uncertainty_vs_baseline"
        else:
            base_name = label.replace("ΔMSE (", "").replace(")", "").replace("–", "_").replace(" ", "_").lower()

        if inter is True:
            suffix = "_intersection"
        elif inter is False:
            suffix = "_all_drugs"
        else:
            suffix = ""

        filename = f"{base_name}{suffix}.pdf"

        # Create organized directory structure
        figures_dir = "figures"
        paper_figures_dir = os.path.join(figures_dir, "case_specific_finetuning")
        os.makedirs(paper_figures_dir, exist_ok=True)

        # Save the figure
        filepath = os.path.join(paper_figures_dir, filename)
        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=300,
                    facecolor='white', edgecolor='none')
        print(f"Saved: {filepath}")

    plt.tight_layout()
    plt.show()


# wrappers
def plot_unc_vs_random(results_df, inter=True, fontsize=12, save_pdf=True):
    if inter:
        plot_pairwise_diff(results_df,
                           "mse_uncertain_inter",
                           "mse_random_inter",
                           "ΔMSE (Uncertainty – Random)",
                           fontsize=fontsize,
                           save_pdf=save_pdf,
                           inter=True)
    else:
        plot_pairwise_diff(results_df,
                           "mse_uncertain",
                           "mse_random",
                           "ΔMSE (Uncertainty – Random)",
                           fontsize=fontsize,
                           save_pdf=save_pdf,
                           inter=False)

def plot_unc_vs_baseline(results_df, inter=True, fontsize=12, save_pdf=True):
    if inter:
        plot_pairwise_diff(results_df,
                           "mse_uncertain_inter",
                           "mse_base_inter",
                           "ΔMSE (Uncertainty – Baseline)",
                           fontsize=fontsize,
                           save_pdf=save_pdf,
                           inter=True)
    else:
        plot_pairwise_diff(results_df,
                           "mse_uncertain",
                           "mse_base",
                           "ΔMSE (Uncertainty – Baseline)",
                           fontsize=fontsize,
                           save_pdf=save_pdf,
                           inter=False)

if __name__ == "__main__":

    plot_unc_vs_baseline(results, inter=False)
    plot_unc_vs_random(results, inter=False)

    plot_unc_vs_baseline(results, inter=True)
    plot_unc_vs_random(results, inter=True)