"""Native-space (direct / un-harmonized) cross-dataset transfer features.

Unlike transfer_data.py (which uses the harmonized COMBINED2271 space), here each dataset is
expressed in its OWN Manica-gene space (arcsinh of its native expression). A model trained on
dataset A is applied to dataset B by taking B's native expression on the same genes and applying
A's training scaler WITHOUT refitting -> the microarray-vs-RNA-seq platform shift is preserved
(the large cross-platform shift). Same drugs/response/split machinery as transfer_data.
"""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from transfer_data import norm, load_response, build_fingerprints, DREVALPY_DATA  # noqa: E402

MANICA = Path(__file__).resolve().parents[1] / "examples" / "data" / "gene_list_paccmann_network_prop.txt"
DATASETS = ("GDSC2", "CCLE", "CTRPv2")
# native per-dataset expression; defaults to the drevalpy tree. Set NATIVE_EXPR_DIR to point at
# a slim copy containing only the common Manica genes.
EXPR_DIR = Path(os.environ.get("NATIVE_EXPR_DIR", str(DREVALPY_DATA)))


@functools.lru_cache(maxsize=1)
def common_genes():
    """Manica genes present in all three datasets' native expression matrices."""
    manica = set(pd.read_csv(MANICA, header=None)[0].astype(str))
    inter = set(manica)
    for ds in DATASETS:
        cols = pd.read_csv(EXPR_DIR / ds / "gene_expression.csv", nrows=0).columns
        inter &= set(cols)
    return sorted(inter)


@functools.lru_cache(maxsize=4)
def load_native_expr(ds):
    """arcsinh of dataset `ds`'s native expression on the common Manica genes, indexed by cl_n."""
    genes = common_genes()
    df = pd.read_csv(EXPR_DIR / ds / "gene_expression.csv")
    df = df.assign(cl_n=df["cell_line_name"].map(norm)).drop_duplicates("cl_n").set_index("cl_n")
    return np.arcsinh(df[genes].astype(np.float32))


def build_features(resp, expr_df, fp_map, scaler):
    """[scaled native expression | fingerprint] for pairs in resp; expr_df is the TEST dataset's
    native expression, scaler is the TRAINING dataset's (applied, not refit)."""
    keep = resp[resp["cl_n"].isin(expr_df.index) & resp["pubchem"].isin(fp_map)].copy()
    if keep.empty:
        return np.empty((0, len(common_genes()) + 128), np.float32), np.empty(0), keep
    gene_mat = scaler.transform(expr_df.loc[keep["cl_n"]].values).astype(np.float32)
    fp_mat = np.vstack([fp_map[p] for p in keep["pubchem"]]).astype(np.float32)
    X = np.concatenate([gene_mat, fp_mat], axis=1)
    return X, keep["y"].values.astype(np.float32), keep.reset_index(drop=True)
