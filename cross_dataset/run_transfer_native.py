"""All-methods direct (native-space) cross-dataset transfer.

Train any of the 7 models on one dataset in its native Manica-gene space, score on all three
(diagonal = in-distribution). Saves total uncertainty (+ aleatoric/epistemic where the model
decomposes) and error per pair, so we can compute per-scenario detection (uncertainty inflation /
AUROC) and ranking (Spearman err-unc) ordered by difficulty.

    CUDA_VISIBLE_DEVICES=N python run_xdataset_native.py --model pnn --train GDSC2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import transfer_data as td                             # noqa: E402
import native_transfer_data as nt                      # noqa: E402
from uadr.models import uncertainty_aware_model as uam  # noqa: E402
from uadr.models import quantile_NN as qfn             # noqa: E402
from uadr.models import probabilistic_NN as pnn        # noqa: E402
from uadr.models import mc_dropout_NN as mcd           # noqa: E402
from uadr.models import bayesian_regression as br      # noqa: E402
from uadr.models import random_forest as rf            # noqa: E402
from uadr.models import evidential_NN as edl           # noqa: E402

OUT = HERE / "results" / "xdataset_native"
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ["GDSC2", "CCLE", "CTRPv2"]

# single fixed config per model (no tuning) -- first/default of the benchmark grids
CFG = {
    "pnne": (pnn.ProbabilisticFeedForwardEnsemble, dict(n_models=5, shuffle_eval=True, n_units_per_layer=[128, 32, 16], dropout_prob=0.3)),
    "pnn":  (pnn.ProbabilisticFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, importance_weighting=False)),
    "qfn":  (qfn.QuantileFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, quantiles=[0.15, 0.5, 0.85])),
    "mcd":  (mcd.MCDropoutFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, sample_size=10)),
    "edl":  (edl.EvidentialFeedForwardNetwork, dict(n_units_per_layer=[128, 32, 16], dropout_prob=0.3, reg_coeff=0.1)),
    "rf":   (rf.RandomForest, dict(n_estimators=500, max_depth=15, n_jobs=-1, random_state=0)),
    "br":   (br.BayesianRidgeRegression, dict(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)),
}
# models that should NOT reload from a checkpoint after fit (mirror crossvalidate)
NO_CKPT = (pnn.ProbabilisticFeedForwardEnsemble, br.BayesianRidgeRegression, rf.RandomForest)


def split_cells(cells, rng, ftrain=0.70, fval=0.15):
    cells = np.array(sorted(cells)); rng.shuffle(cells)
    n = len(cells); a, b = int(ftrain * n), int((ftrain + fval) * n)
    return set(cells[:a]), set(cells[a:b]), set(cells[b:])


def main(a):
    import torch
    np.random.seed(a.seed); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model_class, kw = CFG[a.model]
    expr = {d: nt.load_native_expr(d) for d in DATASETS}
    resp = {d: td.load_response(d) for d in DATASETS}
    drugs = {d: set(resp[d]["pubchem"]) for d in DATASETS}
    fp = td.build_fingerprints(set().union(*drugs.values()))

    tr = resp[a.train]; tr = tr[tr["cl_n"].isin(expr[a.train].index)]
    trc, vac, tec = split_cells(set(tr["cl_n"]), rng)
    seen = trc | vac
    scaler = StandardScaler().fit(expr[a.train].loc[sorted(trc)].values)
    mu, sd = tr[tr["cl_n"].isin(trc)]["y"].mean(), tr[tr["cl_n"].isin(trc)]["y"].std()

    Xtr, ytr, mtr = nt.build_features(tr[tr["cl_n"].isin(trc)], expr[a.train], fp, scaler)
    Xva, yva, mva = nt.build_features(tr[tr["cl_n"].isin(vac)], expr[a.train], fp, scaler)
    print(f"[{a.model}|{a.train}] train={len(mtr)} feat_dim={Xtr.shape[1]}", flush=True)

    model = model_class(n_features=Xtr.shape[1], **kw)
    model.fit(
        X_train=Xtr, y_train=(ytr - mu) / sd, X_eval=Xva, y_eval=(yva - mu) / sd,
        y_train_cell_line_index=mtr["cl_n"].values, y_eval_cell_line_index=mva["cl_n"].values,
        trainer_params={"progress_bar_refresh_rate": 0, "max_epochs": 300},
        batch_size=1024, patience=5, num_workers=0,
        cross_validation_type="cell_lines_cold_start", adversarial_training=True,
    )
    if not isinstance(model, NO_CKPT):
        model = model_class.load_from_checkpoint(
            model.checkpoint_callback.best_model_path, n_features=Xtr.shape[1], **kw)
    model.eval()
    decomp = isinstance(model, uam.DecomposableUncertaintyAwareModel)

    for test_ds in DATASETS:
        rt = resp[test_ds]; rt = rt[rt["cl_n"].isin(expr[test_ds].index)]
        if test_ds == a.train:
            sub = rt[rt["cl_n"].isin(tec)]
        else:
            sub = rt[rt["pubchem"].isin(drugs[a.train] & drugs[test_ds]) & ~rt["cl_n"].isin(seen)]
        X, y, meta = nt.build_features(sub, expr[test_ds], fp, scaler)
        if len(meta) == 0:
            continue
        pred, unc = model.predict_target_and_uncertainty(X)
        y_z = (y - mu) / sd
        df = meta.assign(model=a.model, train=a.train, test=test_ds, pred=pred, y_true_z=y_z,
                         err=np.abs(pred - y_z), unc_total=unc)
        if decomp:
            df["unc_ale"] = model.predict_aleatory_uncertainty(X)
            df["unc_epi"] = model.predict_epistemic_uncertainty(X)
        df.to_csv(OUT / f"{a.model}__{a.train}__{test_ds}__s{a.seed}.csv", index=False)
        print(f"[{a.model}|{a.train}->{test_ds}] n={len(df)} MSE={np.mean(df['err']**2):.3f} "
              f"med_unc={np.median(df['unc_total']):.4f}", flush=True)
    print(f"[{a.model}|{a.train}] done", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CFG))
    ap.add_argument("--train", required=True, choices=DATASETS)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
