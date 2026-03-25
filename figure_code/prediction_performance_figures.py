import sys
sys.dont_write_bytecode = True

import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel
from statsmodels.stats.multitest import multipletests
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from config import (
    MODELS, MODEL_COLORS, MODEL_TO_FULL_NAME,
    COLD_START_SUFFIX, SUFFIX, BASE_PATH, FIGURES_DIR, TABLES_DIR,
)

matplotlib.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
})


def results_path(model):
    return os.path.join(BASE_PATH, f"cv_{model}{COLD_START_SUFFIX}{SUFFIX}", "results.csv")


def compute_per_group_metrics(group):
    output_pearson = {}
    output_mse = {}
    for model in MODELS:
        try:
            results = pd.read_csv(results_path(model), index_col=0)
        except FileNotFoundError:
            print(f"Model {model} not found")
            continue
        pearsons = []
        mses = []
        for entity in results[group].unique():
            mask = results[group] == entity
            if mask.sum() < 2:
                continue
            pearsons.append(pearsonr(results.loc[mask, "response"], results.loc[mask, "y_preds"])[0])
            mses.append(np.mean((results.loc[mask, "response"] - results.loc[mask, "y_preds"]) ** 2))
        output_pearson[model] = pearsons
        output_mse[model] = mses
        print(f"{model} ({group}): median Pearson={np.median(pearsons):.3f}±{np.std(pearsons):.3f}, "
              f"median MSE={np.median(mses):.3f}±{np.std(mses):.3f}")
    return output_pearson, output_mse


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


def save_metrics_table(output_pearson, output_mse, group):
    rows = []
    for model in MODELS:
        if model not in output_pearson:
            continue
        p = np.array(output_pearson[model])
        m = np.array(output_mse[model])
        rows.append({
            "model": MODEL_TO_FULL_NAME[model],
            "pearson_median": np.median(p),
            "pearson_std": np.std(p),
            "mse_median": np.median(m),
            "mse_std": np.std(m),
            "n": len(p),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, f"performance_by_{group}.csv"), index=False)

    # formatted latex table
    rows_fmt = []
    for model in MODELS:
        if model not in output_pearson:
            continue
        p = np.array(output_pearson[model])
        m = np.array(output_mse[model])
        rows_fmt.append([
            MODEL_TO_FULL_NAME[model],
            f"{np.median(p):.3f} +/- {np.std(p):.3f}",
            f"{np.median(m):.3f} +/- {np.std(m):.3f}",
        ])
    tex_df = pd.DataFrame(rows_fmt, columns=["Model", "Pearson (median +/- std)", "MSE (median +/- std)"])
    latex_table = tex_df.to_latex(
        index=False, column_format="lcc", escape=True,
        caption=f"Prediction performance by {group} (median +/- std)",
        label=f"tab:prediction_performance_{group}",
    )
    latex_table = _to_booktabs(latex_table)
    tex_path = os.path.join(TABLES_DIR, f"performance_by_{group}.tex")
    with open(tex_path, "w") as f:
        f.write(latex_table)
    print(f"Saved: performance_by_{group}.csv + .tex")
    return df


def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1 + r) / (1 - r))


def format_pval(p):
    if p < 1e-16:
        return "<1e-16"
    elif p < 1e-1:
        return f"{p:.0e}"
    else:
        return f"{p:.1f}"


def run_ttests_fdr(output_dict, metric_name, sort_by, alpha=0.05, ax=None,
                    higher_is_better=False):
    mean_sort = {m: np.mean(sort_by[m]) for m in sort_by}
    sorted_models = sorted(mean_sort, key=mean_sort.get, reverse=higher_is_better)
    n_models = len(sorted_models)

    p_matrix = np.ones((n_models, n_models))
    pairs, raw_pvals = [], []

    for i in range(n_models):
        for j in range(i + 1, n_models):
            mi, mj = sorted_models[i], sorted_models[j]
            if mi in output_dict and mj in output_dict:
                A, B = np.array(output_dict[mi]), np.array(output_dict[mj])
                if metric_name.lower() == "pearson":
                    A, B = fisher_z(A), fisher_z(B)
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
            label = "1.0" if i == j else format_pval(p_matrix[i, j])
            ax.text(j, i, label, ha="center", va="center", color="black")

    ax.grid(False)
    return im, p_matrix, sorted_models


def plot_significance(output_pearson, output_mse, group, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.subplots_adjust(wspace=0.4)

    run_ttests_fdr(output_pearson, "Pearson", output_pearson, ax=axes[0],
                   higher_is_better=True)
    axes[0].set_title("Significance of Pearson Differences", fontsize=12)

    im2, _, _ = run_ttests_fdr(output_mse, "MSE", output_mse, ax=axes[1])
    axes[1].set_title("Significance of MSE Differences", fontsize=12)

    cbar = fig.colorbar(im2, ax=axes, ticks=[0, 1],
                        fraction=0.026, location="right", shrink=0.5)
    cbar.ax.set_yticklabels(["Not sig.", "Sig."], fontsize=12)
    cbar.set_label("p < 0.05", labelpad=-30, fontsize=12)

    fig.savefig(save_path, bbox_inches="tight")
    print(f"Saved: {save_path}")


def _generate_combined_prediction_table(metrics_by_group):
    """Generate a transposed publication LaTeX table combining drug-wise and global metrics."""

    header_names = {
        "rf": r"\begin{tabular}[c]{@{}c@{}}Random\\ Forest\end{tabular}",
        "br": r"\begin{tabular}[c]{@{}c@{}}Bayesian\\ Ridge\end{tabular}",
        "qnn": r"\begin{tabular}[c]{@{}c@{}}Quantile\\ NN\end{tabular}",
        "mcd": r"\begin{tabular}[c]{@{}c@{}}MC Dropout\\ NN\end{tabular}",
        "pnn": r"\begin{tabular}[c]{@{}c@{}}Gaussian\\ NN\end{tabular}",
        "pnne": r"\begin{tabular}[c]{@{}c@{}}Gaussian \\ NN Ens.\end{tabular}",
        "edl": r"\begin{tabular}[c]{@{}c@{}}Evidential\\ DL\end{tabular}",
    }
    col_order = ["rf", "br", "qnn", "mcd", "pnn", "pnne", "edl"]

    # Collect formatted values per model: (group, metric) -> {model: (mean, std, formatted)}
    rows_data = {}  # key: (group_label, metric_name), value: {model_key: (mean_val, fmt_str)}
    row_keys = []
    for group, (output_pearson, output_mse) in metrics_by_group.items():
        group_label = "Drug-wise" if group == "drugs" else "Global"
        for metric_name, output_dict in [("MSE", output_mse), ("Pearson", output_pearson)]:
            rk = (group_label, metric_name)
            row_keys.append(rk)
            rows_data[rk] = {}
            for model in MODELS:
                if model not in output_dict:
                    continue
                arr = np.array(output_dict[model])
                mu = np.median(arr)
                sd = np.std(arr)
                rows_data[rk][model] = (mu, f"{mu:.3f} +/- {sd:.3f}")

    # Determine best per row (lower is better for MSE, higher for Pearson)
    best_model = {}
    for rk in row_keys:
        vals = rows_data[rk]
        if not vals:
            continue
        if rk[1] == "Pearson":
            best_model[rk] = max(vals, key=lambda m: vals[m][0])
        else:
            best_model[rk] = min(vals, key=lambda m: vals[m][0])

    # Build LaTeX
    ncols = len(col_order)
    lines = []
    lines.append(r"\begin{table*}[htbp]")
    lines.append(r"\caption{\ac{MSE} and Pearson correlation between model predictions and experimental "
                 r"logarithmic IC50 values. For the drug-wise metrics, \ac{MSE} is computed for each drug "
                 r"and then averaged; the standard deviation represents the variability across drugs. "
                 r"The global metrics are calculated for the folds, and the standard deviation is the "
                 r"variation over folds. Best values are shown in bold.}")
    lines.append(r"\centering")
    lines.append(r"\label{tab:prediction_performance}")
    lines.append(r"\begin{tabular}{l" + "c" * ncols + "}")
    lines.append(r"\toprule")
    header_cells = [" "] + [header_names[m] for m in col_order]
    lines.append(" & ".join(header_cells) + r"\\")
    lines.append(r"\midrule")

    # Grouped layout: section headers via \multicolumn, short metric labels
    total_cols = ncols + 1  # label col + model cols
    sections = [
        ("Drug-wise", [("Drug-wise", "MSE"), ("Drug-wise", "Pearson")]),
        ("Global", [("Global", "MSE"), ("Global", "Pearson")]),
    ]
    metric_short = {"MSE": "MSE", "Pearson": "Pearson"}

    for si, (section_title, row_keys_section) in enumerate(sections):
        lines.append(rf"\multicolumn{{{total_cols}}}{{l}}{{\textit{{{section_title}}}}} \\")
        for rk in row_keys_section:
            if rk not in rows_data:
                continue
            label = metric_short[rk[1]]
            cells = [label]
            for m in col_order:
                if m in rows_data[rk]:
                    val_str = rows_data[rk][m][1].replace(" +/- ", r" $\pm$ ")
                    if m == best_model.get(rk):
                        parts = val_str.split(r" $\pm$ ")
                        if len(parts) == 2:
                            val_str = r"\textbf{" + parts[0] + r"}" + r" $\pm$ " + parts[1]
                else:
                    val_str = ""
                cells.append(val_str)
            lines.append(" & ".join(cells) + r" \\")
        if si < len(sections) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}%")
    lines.append(r"")
    lines.append(r"\end{table*}")

    latex_str = "\n".join(lines)
    with open(os.path.join(TABLES_DIR, "prediction_performance.tex"), "w") as f:
        f.write(latex_str)

    # Also save CSV
    csv_rows = []
    for _, row_keys_section in sections:
        for rk in row_keys_section:
            if rk not in rows_data:
                continue
            for m in col_order:
                if m in rows_data[rk]:
                    csv_rows.append({
                        "Metric": f"{rk[1]} ({rk[0].lower()})",
                        "Model": MODEL_TO_FULL_NAME[m],
                        "Value": rows_data[rk][m][1],
                    })
    pd.DataFrame(csv_rows).to_csv(os.path.join(TABLES_DIR, "prediction_performance.csv"), index=False)
    print("Saved: prediction_performance.csv + .tex")


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(os.path.join(FIGURES_DIR, "appendix"), exist_ok=True)

    metrics_by_group = {}
    for group in ["fold", "drugs"]:
        print(f"\n--- {group} ---")
        output_pearson, output_mse = compute_per_group_metrics(group)
        metrics_by_group[group] = (output_pearson, output_mse)
        save_metrics_table(output_pearson, output_mse, group)
        plot_significance(output_pearson, output_mse, group,
                          os.path.join(FIGURES_DIR, "appendix", f"ttest_heatmap_{group}.pdf"))

    _generate_combined_prediction_table(metrics_by_group)