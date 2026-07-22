"""Generate the leave-cell-line-out response-matrix splits consumed by the CV pipeline.

``uadr/utils/data_preprocessing.py`` loads precomputed per-fold response matrices from
``splits/response_matrices/drug_response_matrices_<cv_type>_<scaling_mode>_fold<k>.npy``. This
script builds those matrices from a raw GDSC (natural-log) IC50 table so that the
leave-cell-line-out partitioning and the train-only target scaling are reproducible from source
rather than only shipped as artifacts.

Each ``.npy`` holds ``{"train", "test", "validation"}``: three full ``cell_line x drug`` response
DataFrames sharing the same index/columns, where a partition keeps only its own cell lines'
measurements (the other rows are NaN). The five test folds are mutually disjoint and cover every
cell line exactly once (~185 test / ~74 validation / ~665 train cell lines per fold for GDSC).

Target scaling uses **train-only** statistics:
  * ``z_norm``      -- one global mean/std over all training measurements.
  * ``drug_z_norm`` -- per-drug mean/std from the training measurements.
The same train statistics are applied to the test and validation matrices (no leakage).

The raw GDSC IC50 export is not redistributed with this repository; download it from
https://www.cancerrxgene.org/ and pass it as a long table with cell-line, drug, and log-IC50
columns. Run from the repository root, e.g.:

    python -m uadr.utils.make_splits --responses raw_gdsc_lnic50.csv \
        --out-dir examples/data/experiments/5_fold_cross_validation/splits/response_matrices \
        --scaling-mode z_norm
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


def load_response_matrix(path, cl_col, drug_col, resp_col):
    """Read a long IC50 table and pivot to a cell_line x drug matrix of log-IC50 values."""
    df = pd.read_csv(path)
    missing = {cl_col, drug_col, resp_col} - set(df.columns)
    if missing:
        raise ValueError(f"response table is missing column(s): {sorted(missing)}")
    mat = df.pivot_table(index=cl_col, columns=drug_col, values=resp_col, aggfunc="mean")
    mat.index = mat.index.astype(str)
    return mat.sort_index()


def _scale(train_vals, mat, scaling_mode):
    """Return ``mat`` scaled with train-only statistics for the given scaling mode."""
    if scaling_mode == "z_norm":
        mu = np.nanmean(train_vals.values)
        sd = np.nanstd(train_vals.values)
        return (mat - mu) / sd
    if scaling_mode == "drug_z_norm":
        mu = train_vals.mean(axis=0)          # per-drug (column) mean over train cells
        sd = train_vals.std(axis=0)
        return (mat - mu) / sd
    raise ValueError(f"unknown scaling_mode: {scaling_mode!r}")


def _partition_matrix(full, cells):
    """Full matrix restricted to ``cells`` (other rows set to NaN), preserving index/columns."""
    out = full.copy()
    out.loc[~out.index.isin(cells)] = np.nan
    return out


def make_splits(response_matrix, out_dir, scaling_mode="z_norm",
                cv_type="cell_lines_cold_start", n_splits=5, val_fraction=0.1, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    cells = np.array(sorted(response_matrix.index))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    for fold, (train_val_idx, test_idx) in enumerate(kf.split(cells)):
        test_cells = set(cells[test_idx])
        pool = cells[train_val_idx]
        n_val = max(1, round(len(pool) * val_fraction))
        val_cells = set(rng.choice(pool, size=n_val, replace=False))
        train_cells = set(pool) - val_cells

        # Train-only scaling statistics come from the training measurements alone.
        train_vals = response_matrix.loc[response_matrix.index.isin(train_cells)]
        scaled = _scale(train_vals, response_matrix, scaling_mode)

        out = {
            "train": _partition_matrix(scaled, train_cells),
            "validation": _partition_matrix(scaled, val_cells),
            "test": _partition_matrix(scaled, test_cells),
        }
        path = os.path.join(
            out_dir, f"drug_response_matrices_{cv_type}_{scaling_mode}_fold{fold}.npy")
        np.save(path, out, allow_pickle=True)
        print(f"fold{fold}: train={len(train_cells)} val={len(val_cells)} test={len(test_cells)} -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--responses", required=True, help="long-format raw log-IC50 table (CSV)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scaling-mode", default="z_norm", choices=["z_norm", "drug_z_norm"])
    ap.add_argument("--cv-type", default="cell_lines_cold_start")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cl-col", default="cell_line")
    ap.add_argument("--drug-col", default="drug")
    ap.add_argument("--resp-col", default="response")
    a = ap.parse_args()

    mat = load_response_matrix(a.responses, a.cl_col, a.drug_col, a.resp_col)
    make_splits(mat, a.out_dir, scaling_mode=a.scaling_mode, cv_type=a.cv_type,
                n_splits=a.n_splits, val_fraction=a.val_fraction, seed=a.seed)


if __name__ == "__main__":
    main()
