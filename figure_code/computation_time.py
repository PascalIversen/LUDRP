"""
computation_time.py
===================
Benchmark training and prediction wall-clock times for every model on a single
CV split.  Produces a LaTeX table, a CSV, and a stdout summary.
"""
import sys
sys.dont_write_bytecode = True

import os
import time
import platform
import gc

import numpy as np
import pandas as pd
import torch

from uadr.utils.data_preprocessing import get_preprocessed_features
from uadr.models import probabilistic_NN as pnn
from uadr.models import mc_dropout_NN as mcd
from uadr.models import quantile_NN as qfn
from uadr.models import bayesian_regression as br
from uadr.models import random_forest as rf
from uadr.models import evidential_NN as edl

from config import TABLES_DIR

#  settings
SPLIT = 0
CV_TYPE = "cell_lines_cold_start"
SCALING_MODE = "drug_z_norm"
N_BITS = 128
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "..", "examples", "data",
                         "experiments", "5_fold_cross_validation") + os.sep
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "examples", "data")

TRAINER_KWARGS = {"progress_bar_refresh_rate": 0, "max_epochs": 1000,
                  "enable_model_summary": False}
BATCH_SIZE = 128
PATIENCE = 5
NUM_WORKERS = 0


def detect_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"CUDA ({name})"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "Apple MPS"
    return "CPU"


def count_parameters(model):
    """Return number of trainable parameters for a torch.nn.Module, else None."""
    if isinstance(model, torch.nn.Module):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return None


#  model definitions
def get_model_configs(n_features):
    """Return ordered list of (key, display_name, model_instance, init_kwargs, is_deep)."""
    common_nn = dict(n_features=n_features,
                     n_units_per_layer=[128, 32, 16],
                     dropout_prob=0.3)
    qnn_kw = dict(**common_nn, quantiles=[0.15, 0.5, 0.85])
    mcd_kw = dict(**common_nn, sample_size=10)
    edl_kw = dict(**common_nn, reg_coeff=0.01)
    pnne_kw = dict(n_features=n_features, n_models=10, shuffle_eval=True,
                   n_units_per_layer=[128, 32, 16], dropout_prob=0.3)
    return [
        ("rf",   "Random Forest",
         rf.RandomForest(n_estimators=150, max_depth=8, n_jobs=-1), {}, False),
        ("br",   "Bayesian Ridge",
         br.BayesianRidgeRegression(alpha_1=1e-6, alpha_2=1e-6,
                                    lambda_1=1e-6, lambda_2=1e-6), {}, False),
        ("qnn",  "Quantile NN",
         qfn.QuantileFeedForwardNetwork(**qnn_kw), qnn_kw, True),
        ("mcd",  "MC Dropout NN",
         mcd.MCDropoutFeedForwardNetwork(**mcd_kw), mcd_kw, True),
        ("pnn",  "Gaussian NN",
         pnn.ProbabilisticFeedForwardNetwork(**common_nn), common_nn, True),
        ("pnne", "Gaussian NN Ens.",
         pnn.ProbabilisticFeedForwardEnsemble(**pnne_kw), pnne_kw, True),
        ("edl",  "Evidential DL",
         edl.EvidentialFeedForwardNetwork(**edl_kw), edl_kw, True),
    ]


def benchmark(model, is_deep, model_init_kwargs, x_train, y_train, x_val, y_val, x_test):
    fit_kwargs = dict(
        X_train=x_train,
        y_train=y_train.response.values,
        X_eval=x_val,
        y_eval=y_val.response.values,
        y_train_cell_line_index=y_train.cell_lines.values,
        y_eval_cell_line_index=y_val.cell_lines.values,
        batch_size=BATCH_SIZE,
        patience=PATIENCE,
        num_workers=NUM_WORKERS,
        adversarial_training=True,
        cross_validation_type=CV_TYPE,
        trainer_params=TRAINER_KWARGS.copy(),
    )
    if isinstance(model, (br.BayesianRidgeRegression, rf.RandomForest)):
        # sklearn models don't accept all kwargs
        fit_kwargs = dict(
            X_train=x_train,
            y_train=y_train.response.values,
            X_eval=x_val,
            y_eval=y_val.response.values,
        )

    #  train
    t0 = time.perf_counter()
    model.fit(**fit_kwargs)
    train_time = time.perf_counter() - t0

    # epochs
    n_epochs = None
    if is_deep:
        if isinstance(model, pnn.ProbabilisticFeedForwardEnsemble):
            n_epochs = None
        elif hasattr(model, "trainer"):
            n_epochs = model.trainer.current_epoch
        elif hasattr(model, "checkpoint_callback"):
            # try to read from checkpoint
            n_epochs = None

    # load best checkpoint for PL models (not ensemble, not sklearn)
    if is_deep and not isinstance(model, pnn.ProbabilisticFeedForwardEnsemble):
        model_class = type(model)
        model = model_class.load_from_checkpoint(
            model.checkpoint_callback.best_model_path, **model_init_kwargs
        )
    model.eval()

    # parameter count
    n_params = count_parameters(model)
    if isinstance(model, pnn.ProbabilisticFeedForwardEnsemble):
        # count from one sub-model, multiply by n_models
        sub = pnn.ProbabilisticFeedForwardNetwork(
            n_features=x_train.shape[1],
            n_units_per_layer=[128, 32, 16], dropout_prob=0.3)
        n_params_one = count_parameters(sub)
        n_params = n_params_one * model.n_models if n_params_one else None
        del sub

    #  predict
    t0 = time.perf_counter()
    model.predict_target_and_uncertainty(x_test)
    predict_time = time.perf_counter() - t0

    # cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()

    return train_time, n_epochs, predict_time, n_params


def generate_latex(rows, device_str):
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\caption{Computation time for training and prediction "
                 r"(single fold, split~0). "
                 f"Device: {device_str}." + "}")
    lines.append(r"\centering")
    lines.append(r"\label{tab:computation_time}")
    lines.append(r"\begin{tabular}{lrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Model & Parameters & Train (s) & Time/Epoch (s) & Predict (s) \\")
    lines.append(r"\midrule")

    for r in rows:
        params_str = f"{r['n_params']:,}" if r["n_params"] is not None else "---"
        train_str = f"{r['train_time']:.1f}"
        if r["n_epochs"] is not None and r["n_epochs"] > 0:
            tpe = r["train_time"] / r["n_epochs"]
            epoch_str = f"{tpe:.2f}"
        else:
            epoch_str = "---"
        pred_str = f"{r['predict_time']:.2f}"
        lines.append(f"{r['name']} & {params_str} & {train_str} "
                      f"& {epoch_str} & {pred_str} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ── main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device_str = detect_device()
    print(f"Device: {device_str}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")
    print()

    # load data
    out = get_preprocessed_features(
        current_split=SPLIT,
        base_path=BASE_PATH,
        cross_validation_type=CV_TYPE,
        scaling_mode=SCALING_MODE,
        n_bits_drug_fingerprints=N_BITS,
        data_dir=DATA_DIR,
    )
    x_train = out["train"]["features"]
    y_train = out["train"]["targets"]
    x_val = out["validation"]["features"]
    y_val = out["validation"]["targets"]
    x_test = out["test"]["features"]
    y_test = out["test"]["targets"]

    n_features = x_train.shape[1]
    print(f"Train: {x_train.shape[0]:,}  Val: {x_val.shape[0]:,}  "
          f"Test: {x_test.shape[0]:,}  Features: {n_features}")
    print()

    configs = get_model_configs(n_features)
    rows = []

    for key, name, model, init_kwargs, is_deep in configs:
        print(f"{'='*60}")
        print(f"Benchmarking: {name}")
        print(f"{'='*60}")

        train_time, n_epochs, predict_time, n_params = benchmark(
            model, is_deep, init_kwargs, x_train, y_train, x_val, y_val, x_test
        )

        epoch_str = str(n_epochs) if n_epochs is not None else "—"
        tpe_str = (f"{train_time / n_epochs:.2f}s"
                   if n_epochs is not None and n_epochs > 0 else "—")
        params_str = f"{n_params:,}" if n_params is not None else "—"

        print(f"  Parameters:   {params_str}")
        print(f"  Train time:   {train_time:.1f}s  ({epoch_str} epochs, {tpe_str}/epoch)")
        print(f"  Predict time: {predict_time:.2f}s")
        print()

        rows.append(dict(key=key, name=name, train_time=train_time,
                         n_epochs=n_epochs, predict_time=predict_time,
                         n_params=n_params))

    # summary table (stdout)
    print(f"\n{'='*70}")
    print(f"{'Model':<20} {'Params':>10} {'Train (s)':>10} "
          f"{'Epochs':>7} {'s/Epoch':>8} {'Predict (s)':>12}")
    print(f"{'-'*70}")
    for r in rows:
        p = f"{r['n_params']:,}" if r['n_params'] is not None else "—"
        e = str(r['n_epochs']) if r['n_epochs'] is not None else "—"
        tpe = (f"{r['train_time']/r['n_epochs']:.2f}"
               if r['n_epochs'] is not None and r['n_epochs'] > 0 else "—")
        print(f"{r['name']:<20} {p:>10} {r['train_time']:>10.1f} "
              f"{e:>7} {tpe:>8} {r['predict_time']:>12.2f}")
    print(f"{'='*70}\n")

    # save outputs
    os.makedirs(TABLES_DIR, exist_ok=True)

    # CSV
    df = pd.DataFrame(rows)
    csv_path = os.path.join(TABLES_DIR, "computation_time.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    # LaTeX
    tex = generate_latex(rows, device_str)
    tex_path = os.path.join(TABLES_DIR, "computation_time.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Saved: {tex_path}")

    print("\nDone.")
