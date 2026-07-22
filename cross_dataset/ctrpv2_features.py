"""CTRPv2 feature/split loader matching uadr's get_preprocessed_features signature.

Reproduces the paper's benchmark pipeline on CTRPv2 with the *identical* method: the same
2019-gene Manica selection (applied to CTRPv2's RNA-seq, 2082/2128 genes present), arcsinh +
gene-wise standardization, 128-bit Morgan fingerprints, and a 5-fold leave-cell-line-out CV
with per-fold z-scaling of the response (LN_IC50_curvecurator). Drop-in for
run_cv.get_preprocessed_features so crossvalidate runs unchanged on CTRPv2.
"""
from __future__ import annotations

import functools
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

LUDRP_ROOT = Path(__file__).resolve().parents[1]
# Path to the drevalpy screening data (set DREVALPY_DATA to override).
DREVALPY_DATA = Path(os.environ.get("DREVALPY_DATA", "drevalpy_data"))
MANICA_GENES = LUDRP_ROOT / "examples" / "data" / "gene_list_paccmann_network_prop.txt"
RESPONSE = "LN_IC50_curvecurator"
N_BITS, MORGAN_RADIUS = 128, 2
N_SPLITS = 5


def _norm(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _morgan_by_drug(data_dir: Path) -> dict:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    s = pd.read_csv(data_dir / "CTRPv2" / "drug_smiles.csv",
                    usecols=lambda c: c in ("drug_name", "canonical_smiles"))
    fp = {}
    for name, sm in zip(s["drug_name"], s["canonical_smiles"]):
        if not isinstance(sm, str):
            continue
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, MORGAN_RADIUS, nBits=N_BITS)
        fp[_norm(name)] = np.frombuffer(bytes(bv.ToBitString(), "ascii"), "u1").astype(np.float32) - ord("0")
    return fp


@functools.lru_cache(maxsize=1)
def _load():
    """Load CTRPv2 expression (Manica genes), response, fingerprints, and the 5-fold cell split."""
    manica = pd.read_csv(MANICA_GENES, header=None)[0].astype(str).tolist()
    expr = pd.read_csv(DREVALPY_DATA / "CTRPv2" / "gene_expression.csv")
    expr = expr.assign(cl_n=expr["cell_line_name"].map(_norm)).drop_duplicates("cl_n").set_index("cl_n")
    genes = [g for g in manica if g in expr.columns]
    expr = expr[genes].astype(np.float32)

    r = pd.read_csv(DREVALPY_DATA / "CTRPv2" / "CTRPv2.csv",
                    usecols=lambda c: c in ("cell_line_name", "drug_name", RESPONSE), low_memory=False)
    r = r.dropna(subset=[RESPONSE]).rename(columns={RESPONSE: "y"})
    r["cl_n"] = r["cell_line_name"].map(_norm)
    r["dr_n"] = r["drug_name"].map(_norm)

    fp = _morgan_by_drug(DREVALPY_DATA)
    # valid cells: have expression and at least one response
    valid = sorted(set(r["cl_n"]) & set(expr.index))
    folds = [set(a) for a in np.array_split(np.random.default_rng(0).permutation(valid), N_SPLITS)]
    return expr, genes, r[["cl_n", "dr_n", "y"]], fp, folds


def _build(pairs, expr, fp, scaler):
    keep = pairs[pairs["cl_n"].isin(expr.index) & pairs["dr_n"].isin(fp)].copy()
    gene_mat = scaler.transform(np.arcsinh(expr.loc[keep["cl_n"]].values.astype(np.float32))).astype(np.float32)
    fp_mat = np.vstack([fp[d] for d in keep["dr_n"]]).astype(np.float32)
    X = np.concatenate([gene_mat, fp_mat], axis=1)
    targets = pd.DataFrame({"cell_lines": keep["cl_n"].values,
                            "drugs": keep["dr_n"].values, "response": keep["y"].values.astype(float)})
    return X, targets


def get_ctrpv2_features(current_split, base_path=None, cross_validation_type="cell_lines_cold_start",
                        scaling_mode="z_norm", n_bits_drug_fingerprints=128, data_dir=None):
    """Drop-in replacement for get_preprocessed_features, for CTRPv2."""
    expr, genes, resp, fp, folds = _load()
    k = int(current_split)
    test_cells = folds[k]
    pool = sorted(set().union(*[f for i, f in enumerate(folds) if i != k]))
    rng = np.random.default_rng(100 + k)
    pool = rng.permutation(pool)
    n_val = max(1, len(pool) // 8)                      # leave-cell-line-out validation from train pool
    val_cells, train_cells = set(pool[:n_val]), set(pool[n_val:])

    scaler = StandardScaler().fit(np.arcsinh(expr.loc[sorted(train_cells)].values.astype(np.float32)))
    Xtr, ytr = _build(resp[resp["cl_n"].isin(train_cells)], expr, fp, scaler)
    Xva, yva = _build(resp[resp["cl_n"].isin(val_cells)], expr, fp, scaler)
    Xte, yte = _build(resp[resp["cl_n"].isin(test_cells)], expr, fp, scaler)

    mu, sd = ytr["response"].mean(), ytr["response"].std()   # per-fold z-norm on train pairs
    for t in (ytr, yva, yte):
        t["response"] = (t["response"] - mu) / sd
    return {
        "train": {"features": Xtr, "targets": ytr, "scaler": scaler},
        "validation": {"features": Xva, "targets": yva},
        "test": {"features": Xte, "targets": yte},
        "gene_list": genes,
    }


if __name__ == "__main__":
    out = get_ctrpv2_features(0)
    print("genes:", len(out["gene_list"]), "feat_dim:", out["train"]["features"].shape[1])
    for k in ("train", "validation", "test"):
        t = out[k]["targets"]
        print(f"  {k:11s} pairs={len(t):>7d} cells={t.cell_lines.nunique():>4d} "
              f"resp mean={t.response.mean():+.2f} std={t.response.std():.2f}")
