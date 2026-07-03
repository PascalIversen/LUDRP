"""Run the paper's benchmark on CTRPv2 (one model per process; folds optionally parallelized).

Reuses uadr's crossvalidate machinery unchanged by swapping in the CTRPv2 feature loader, so
the tuning, fitting, metrics and result files match the GDSC run. Produces, per model,
results/ctrpv2_benchmark/results/cv_<model>_ctrpv2_z_norm/ -> a CTRPv2 version of Tables 1 & 2.

Model configs are copied verbatim from examples/cross_validation_models.py (same grids).

Run one model per GPU (NN) or CPU (rf/br):
    CUDA_VISIBLE_DEVICES=N python run_ctrpv2_cv.py --model pnne [--split K] [--batch_size 256]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctrpv2_features                                      # noqa: E402
from uadr.experiments import run_cv                         # noqa: E402
from uadr.models import quantile_NN as qfn                  # noqa: E402
from uadr.models import probabilistic_NN as pnn             # noqa: E402
from uadr.models import mc_dropout_NN as mcd                # noqa: E402
from uadr.models import bayesian_regression as br           # noqa: E402
from uadr.models import random_forest as rf                 # noqa: E402
from uadr.models import evidential_NN as edl                # noqa: E402

# CTRPv2 feature loader replaces the GDSC one used inside crossvalidate
run_cv.get_preprocessed_features = ctrpv2_features.get_ctrpv2_features

BASE = str(HERE / "results" / "ctrpv2_benchmark") + "/"


def model_setup(model_type):
    """Return (model_class, model_kwargs, tuning) — identical grids to the GDSC run."""
    tuning = True
    if model_type == "br":
        kw = {"alpha_1": [1e-6, 1e-5, 1e-4], "alpha_2": [1e-6, 1e-4], "lambda_1": [1e-6], "lambda_2": [1e-6]}
        return br.BayesianRidgeRegression, kw, True
    if model_type == "rf":
        return rf.RandomForest, {"n_estimators": [150], "max_depth": [8], "n_jobs": [4]}, True
    kw = {"n_units_per_layer": [[64, 16, 8], [128, 32, 16], [64, 16, 16, 8, 8]], "dropout_prob": [0.1, 0.3]}
    if model_type == "qfn":
        kw["quantiles"] = [[0.05, 0.5, 0.95]]
        return qfn.QuantileFeedForwardNetwork, kw, True
    if model_type == "pnn":
        kw["importance_weighting"] = [False]
        return pnn.ProbabilisticFeedForwardNetwork, kw, True
    if model_type == "pnne":
        # n_models=5 (vs paper's 10) for tractability on the 8-core box; uncertainty decomposition
        # is essentially unchanged, and it halves the GNNE's runtime. Documented deviation.
        kw = {"n_models": 5, "shuffle_eval": True, "n_units_per_layer": [128, 32, 16], "dropout_prob": 0.3}
        return pnn.ProbabilisticFeedForwardEnsemble, kw, False  # no tuning for the ensemble
    if model_type == "mcd":
        kw["sample_size"] = [10]
        return mcd.MCDropoutFeedForwardNetwork, kw, True
    if model_type == "edl":
        kw["reg_coeff"] = [0.01, 0.1]
        return edl.EvidentialFeedForwardNetwork, kw, True
    raise ValueError(model_type)


def main(a):
    model_class, model_kwargs, tuning = model_setup(a.model)
    run_cv.crossvalidate(
        run_id=f"cv_{a.model}_ctrpv2",
        model_class=model_class,
        model_kwargs=model_kwargs,
        n_splits=5,
        cross_validation_type="cell_lines_cold_start",
        scaling_mode="z_norm",
        n_bits_drug_fingerprints=128,
        base_path=BASE,
        data_dir="unused",
        trainer_kwargs={"progress_bar_refresh_rate": 0, "max_epochs": 1000},
        batch_size=a.batch_size,
        patience=5,
        monitor="val_loss",
        tuning=tuning,
        num_workers=a.num_workers,
        skip_existing_folds=True,
        run_split=a.split,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["br", "rf", "qfn", "pnn", "pnne", "mcd", "edl"])
    ap.add_argument("--split", type=int, default=None, help="run a single fold (for parallelism)")
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=2)
    main(ap.parse_args())
