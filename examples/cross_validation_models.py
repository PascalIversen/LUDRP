import argparse

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from uadr.models import quantile_NN as qfn
from uadr.models import probabilistic_NN as pnn
from uadr.models import mc_dropout_NN as mcd
from uadr.models import bayesian_regression as br
from uadr.models import random_forest as rf
from uadr.models import evidential_NN as edl
from uadr.experiments import run_cv

# command line arguments

parser = argparse.ArgumentParser(description="Process hpams for cross validation.")
parser.add_argument("--run_id", type=str, default="test_run")
parser.add_argument("--model_type", type=str, default="qfn")
parser.add_argument("--cv_type", type=str, default="warm_start")
parser.add_argument("--split", type=int, default=-1)
parser.add_argument("--scaling_mode", type=str, default="drug_z_norm")


args = parser.parse_args()
run_id = args.run_id
model_type = args.model_type
cv_type = args.cv_type
split = args.split if args.split != -1 else None
scaling_mode = args.scaling_mode
tuning = True
assert model_type in [
    "br",
    "rf",
    "qfn",
    "pnn",
    "pnne",
    "mcd",
    "edl",
], "Invalid model type specified."
if model_type == "br":
    model_kwargs = {
        "alpha_1": [1e-6, 1e-5, 1e-4],
        "alpha_2": [1e-6, 1e-4],
        "lambda_1": [1e-6],
        "lambda_2": [1e-6],
    }
    model_class = br.BayesianRidgeRegression
elif model_type == "rf":
    # random_forest.py additionally applies max_samples=0.85 and max_features=0.85.
    model_kwargs = {
        "n_estimators": [200],
        "max_depth": [13],
        "n_jobs": [-1],
        "random_state": [0],
    }
    model_class = rf.RandomForest
else:
    model_kwargs = {
        "n_units_per_layer": [
            [64, 16, 8],
            [128, 32, 16],
            [64, 16, 16, 8, 8],
        ],
        "dropout_prob": [0.1, 0.3],
    }

if model_type == "qfn":
    model_class = qfn.QuantileFeedForwardNetwork
    # 0.05/0.50/0.95 quantiles give a 90% central prediction interval.
    model_kwargs["quantiles"] = [[0.05, 0.5, 0.95]]
elif model_type == "pnn":
    model_class = pnn.ProbabilisticFeedForwardNetwork
    model_kwargs["importance_weighting"] = [False]
elif model_type == "pnne":
    model_class = pnn.ProbabilisticFeedForwardEnsemble
    model_kwargs = {
        "n_models": 10,
        "shuffle_eval": True,
        "n_units_per_layer": [128, 32, 16],
        "dropout_prob": 0.3,
    }
    tuning = False # No tuning for ensemble models bc of their computational cost
elif model_type == "mcd":
    model_class = mcd.MCDropoutFeedForwardNetwork
    # sample_size = number of stochastic forward passes at inference.
    model_kwargs["sample_size"] = [10]
elif model_type == "edl":
    model_class = edl.EvidentialFeedForwardNetwork
    # Evidence-regularization weight grid.
    model_kwargs["reg_coeff"] = [0.01, 0.1]

trainer_kwargs = {"progress_bar_refresh_rate": 0, "max_epochs": 1000}
print(f"max_epochs: {trainer_kwargs['max_epochs']}")
run_cv.crossvalidate(
    run_id=run_id,
    model_class=model_class,
    model_kwargs=model_kwargs,
    n_splits=5,
    cross_validation_type=cv_type,
    scaling_mode=scaling_mode,
    n_bits_drug_fingerprints=128,
    base_path="examples/data/experiments/5_fold_cross_validation/",
    data_dir="examples/data",
    trainer_kwargs=trainer_kwargs,
    batch_size=128,
    patience=5,
    monitor="val_loss",
    tuning=tuning,
    num_workers=0,
    skip_existing_folds=True,
    run_split=split
)
