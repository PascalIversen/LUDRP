"""CTRPv2 (Broad RNA-seq) -> BeatAML (OHSU RNA-seq) transfer: same-platform, different-lab shift.

BeatAML is put on the same representation as the Broad training data (log2 -> linear 2^x -> arcsinh),
and the model is trained on the genes shared by CTRPv2 and BeatAML (subset of the 2044 Manica genes).
Per (model, seed) trains one CTRPv2 model and writes, into results/xdataset_native/:
  in-dist : {m}__CTRPv2be__CTRPv2be__s{seed}.csv   (held-out CTRPv2 test, same 1666-gene model)
  transfer: {m}__CTRPv2be__BeatAML__s{seed}.csv     (BeatAML, shared drugs, unseen cells)
The "CTRPv2be" train tag keeps these separate from the 2044-gene native runs. 7 models x 5 seeds,
serialized, num_workers=0; skips a (model, seed) whose BeatAML file exists.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_transfer_data as nt                      # noqa: E402
import transfer_data as td                             # noqa: E402
from uadr.models import uncertainty_aware_model as uam  # noqa: E402
from uadr.models import quantile_NN as qfn             # noqa: E402
from uadr.models import probabilistic_NN as pnn        # noqa: E402
from uadr.models import mc_dropout_NN as mcd           # noqa: E402
from uadr.models import bayesian_regression as br      # noqa: E402
from uadr.models import random_forest as rf            # noqa: E402
from uadr.models import evidential_NN as edl           # noqa: E402

DD = Path(td.DREVALPY_DATA)
OUT = HERE / "results" / "xdataset_native"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]

CFG = {
    "pnne": (pnn.ProbabilisticFeedForwardEnsemble, dict(n_models=5, shuffle_eval=True, n_units_per_layer=[128, 32, 16], dropout_prob=0.3)),
    "pnn":  (pnn.ProbabilisticFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, importance_weighting=False)),
    "qfn":  (qfn.QuantileFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, quantiles=[0.15, 0.5, 0.85])),
    "mcd":  (mcd.MCDropoutFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, sample_size=10)),
    "edl":  (edl.EvidentialFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, reg_coeff=0.1)),
    "rf":   (rf.RandomForest, dict(n_estimators=150, max_depth=8, n_jobs=4)),
    "br":   (br.BayesianRidgeRegression, dict(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)),
}
NO_CKPT = (pnn.ProbabilisticFeedForwardEnsemble, br.BayesianRidgeRegression, rf.RandomForest)


def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_beataml_expr(genes):
    df = pd.read_csv(DD / "BeatAML2" / "gene_expression.csv")
    df = df.assign(cl_n=df["cell_line_name"].map(norm)).drop_duplicates("cl_n").set_index("cl_n")
    return np.arcsinh(np.power(2.0, df[genes].astype(np.float32)))   # log2 -> linear -> arcsinh (match Broad)


def load_beataml_resp():
    r = pd.read_csv(DD / "BeatAML2" / "BeatAML2.csv",
                    usecols=["cell_line_name", "pubchem_id", "LN_IC50_curvecurator"]).dropna(subset=["LN_IC50_curvecurator"])
    r["cl_n"] = r["cell_line_name"].map(norm)
    r["pubchem"] = r["pubchem_id"].astype(str)
    r["drug_name"] = r["pubchem"]
    return r.rename(columns={"LN_IC50_curvecurator": "y"})[["cl_n", "pubchem", "drug_name", "y"]]


def split_cells(cells, rng, ftrain=0.70, fval=0.15):
    cells = np.array(sorted(cells)); rng.shuffle(cells)
    n = len(cells); a, b = int(ftrain * n), int((ftrain + fval) * n)
    return set(cells[:a]), set(cells[a:b]), set(cells[b:])


def build(resp, expr, genes, fp, scaler):
    keep = resp[resp["cl_n"].isin(expr.index) & resp["pubchem"].isin(fp)].copy()
    if keep.empty:
        return np.empty((0, len(genes) + 128), np.float32), np.empty(0), keep
    gm = scaler.transform(expr.loc[keep["cl_n"], genes].values).astype(np.float32)
    fm = np.vstack([fp[p] for p in keep["pubchem"]]).astype(np.float32)
    return np.concatenate([gm, fm], 1), keep["y"].values.astype(np.float32), keep.reset_index(drop=True)


def write(meta, test_tag, pred, unc, mu, sd, model_name, seed, model=None, X=None):
    y_z = (meta["y"].values - mu) / sd
    df = meta.assign(model=model_name, train="CTRPv2be", test=test_tag, pred=pred, y_true_z=y_z,
                     err=np.abs(pred - y_z), unc_total=unc)
    if model is not None and isinstance(model, uam.DecomposableUncertaintyAwareModel):
        df["unc_ale"] = model.predict_aleatory_uncertainty(X)
        df["unc_epi"] = model.predict_epistemic_uncertainty(X)
    df.to_csv(OUT / f"{model_name}__CTRPv2be__{test_tag}__s{seed}.csv", index=False)
    return float(np.mean(df["err"] ** 2))


def run(model_name, seed, genes, ctrp_expr, ctrp_resp, be_expr, be_resp, fp):
    import torch
    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mc, kw = CFG[model_name]
    tr = ctrp_resp[ctrp_resp["cl_n"].isin(ctrp_expr.index)]
    trc, vac, tec = split_cells(set(tr["cl_n"]), rng)
    scaler = StandardScaler().fit(ctrp_expr.loc[sorted(trc), genes].values)
    mu, sd = tr[tr["cl_n"].isin(trc)]["y"].mean(), tr[tr["cl_n"].isin(trc)]["y"].std()
    Xtr, ytr, mtr = build(tr[tr["cl_n"].isin(trc)], ctrp_expr, genes, fp, scaler)
    Xva, yva, mva = build(tr[tr["cl_n"].isin(vac)], ctrp_expr, genes, fp, scaler)
    m = mc(n_features=Xtr.shape[1], **kw)
    m.fit(X_train=Xtr, y_train=(ytr - mu) / sd, X_eval=Xva, y_eval=(yva - mu) / sd,
          y_train_cell_line_index=mtr["cl_n"].values, y_eval_cell_line_index=mva["cl_n"].values,
          trainer_params={"progress_bar_refresh_rate": 0, "max_epochs": 300,
                          "accelerator": os.environ.get("BE_ACCEL", "auto"), "devices": 1},
          batch_size=1024, patience=5, num_workers=0,
          cross_validation_type="cell_lines_cold_start", adversarial_training=True)
    if not isinstance(m, NO_CKPT):
        m = mc.load_from_checkpoint(m.checkpoint_callback.best_model_path, n_features=Xtr.shape[1], **kw)
    m.eval()
    # in-dist CTRPv2 test
    sid = ctrp_resp[ctrp_resp["cl_n"].isin(tec) & ctrp_resp["cl_n"].isin(ctrp_expr.index)]
    Xid, yid, mid = build(sid, ctrp_expr, genes, fp, scaler)
    pid, uid = m.predict_target_and_uncertainty(Xid)
    write(mid, "CTRPv2be", pid, uid, mu, sd, model_name, seed, m, Xid)
    # BeatAML (shared drugs, all cells unseen by construction)
    shared = set(ctrp_resp["pubchem"]) & set(be_resp["pubchem"])
    sub = be_resp[be_resp["pubchem"].isin(shared)]
    Xb, yb, mb = build(sub, be_expr, genes, fp, scaler)
    pb, ub = m.predict_target_and_uncertainty(Xb)
    mse = write(mb, "BeatAML", pb, ub, mu, sd, model_name, seed, m, Xb)
    print(f"[{model_name}|s{seed}] BeatAML MSE={mse:.3f} n={len(mb)}", flush=True)


def main(which_models, which_seeds):
    be_cols = set(pd.read_csv(DD / "BeatAML2" / "gene_expression.csv", nrows=0).columns)  # read ONCE
    genes = [g for g in nt.common_genes() if g in be_cols]
    print(f"shared CTRPv2&BeatAML genes: {len(genes)}", flush=True)
    ctrp_expr = nt.load_native_expr("CTRPv2")
    ctrp_resp = td.load_response("CTRPv2")
    be_expr = load_beataml_expr(genes)
    be_resp = load_beataml_resp()
    fp = td.build_fingerprints(set(ctrp_resp["pubchem"]) | set(be_resp["pubchem"]))
    for seed in which_seeds:
        for m in which_models:
            if (OUT / f"{m}__CTRPv2be__BeatAML__s{seed}.csv").exists():
                print(f"[{m}|s{seed}] exists, skip", flush=True); continue
            try:
                run(m, seed, genes, ctrp_expr, ctrp_resp, be_expr, be_resp, fp)
            except Exception as e:
                print(f"[{m}|s{seed}] FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(CFG))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    a = ap.parse_args()
    main([m for m in a.models.split(",") if m], [int(s) for s in a.seeds.split(",") if s != ""])
