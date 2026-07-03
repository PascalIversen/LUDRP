"""Harmonized cross-dataset transfer data: common gene space (COMBINED2271), Morgan drug
fingerprints, and a common response (LN_IC50_curvecurator). Lets a GNNE trained on one
dataset be evaluated on another without the microarray-vs-RNA-seq platform artifact.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DREVALPY_DATA = Path(os.environ.get("DREVALPY_DATA", "/Users/piversen/Projects/munich/drevalpy/data"))
N_BITS = 128
MORGAN_RADIUS = 2
RESPONSE = "LN_IC50_curvecurator"


def norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_harmonized_expression(data_dir: Path | None = None):
    """Return (expr_df indexed by normalized cell-line name, list of gene columns)."""
    data_dir = data_dir or DREVALPY_DATA
    df = pd.read_csv(data_dir / "COMBINED2271" / "gene_expression.csv")
    genes = [c for c in df.columns if c not in ("cell_line_name", "cellosaurus_id")]
    df = df.assign(cl_n=df["cell_line_name"].map(norm)).drop_duplicates("cl_n").set_index("cl_n")
    return df[genes].astype(np.float32), genes


def load_response(ds: str, data_dir: Path | None = None) -> pd.DataFrame:
    """Response table for a dataset: cl_n, pubchem (str), drug_name, y (LN_IC50_curvecurator)."""
    data_dir = data_dir or DREVALPY_DATA
    f = data_dir / ds / f"{ds}.csv"
    cols = pd.read_csv(f, nrows=0).columns.tolist()
    use = [c for c in ["cell_line_name", "drug_name", "pubchem_id", RESPONSE] if c in cols]
    r = pd.read_csv(f, usecols=use, low_memory=False).dropna(subset=[RESPONSE])
    r["cl_n"] = r["cell_line_name"].map(norm)
    r["pubchem"] = r["pubchem_id"].astype(str)
    r = r.rename(columns={RESPONSE: "y"})
    return r[["cl_n", "pubchem", "drug_name", "y"]]


def build_fingerprints(pubchems, datasets=("GDSC2", "CCLE", "CTRPv2"), data_dir: Path | None = None):
    """Morgan fingerprints (N_BITS) for the requested pubchem ids, from the datasets' SMILES."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    data_dir = data_dir or DREVALPY_DATA
    smiles = {}
    for ds in datasets:
        f = data_dir / ds / "drug_smiles.csv"
        if not f.exists():
            continue
        s = pd.read_csv(f, usecols=lambda c: c in ("pubchem_id", "canonical_smiles"))
        for pid, sm in zip(s["pubchem_id"].astype(str), s["canonical_smiles"]):
            smiles.setdefault(pid, sm)
    fp = {}
    for pid in set(map(str, pubchems)):
        sm = smiles.get(pid)
        if sm is None or not isinstance(sm, str):
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=N_BITS)
        fp[pid] = np.frombuffer(bytes(bv.ToBitString(), "ascii"), "u1").astype(np.float32) - ord("0")
    return fp


def build_features(resp: pd.DataFrame, expr_df, genes, fp_map, scaler):
    """Build [scaled expression | fingerprint] features for the pairs in `resp`.

    Only keeps pairs whose cell line has harmonized expression and whose drug has a fingerprint.
    `scaler` (sklearn StandardScaler, already fit on the training expression) transforms the genes.
    Returns (X float32, y_raw, meta DataFrame[cl_n, pubchem, drug_name, y]).
    """
    expr_cells = set(expr_df.index)
    keep = resp[resp["cl_n"].isin(expr_cells) & resp["pubchem"].isin(fp_map)].copy()
    if keep.empty:
        return np.empty((0, len(genes) + N_BITS), np.float32), np.empty(0), keep
    gene_mat = scaler.transform(expr_df.loc[keep["cl_n"]].values).astype(np.float32)
    fp_mat = np.vstack([fp_map[p] for p in keep["pubchem"]]).astype(np.float32)
    X = np.concatenate([gene_mat, fp_mat], axis=1)
    return X, keep["y"].values.astype(np.float32), keep.reset_index(drop=True)
