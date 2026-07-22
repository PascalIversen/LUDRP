import os
import itertools
import warnings
import logging
from typing import Optional, Type

import gc
import pandas as pd
import numpy as np
import torch

from uadr.utils.data_preprocessing import get_preprocessed_features
from uadr.utils.prediction_evaluation import prediction_quality_metrics
from uadr.utils.uncertainty_evaluation import uncertainty_quality_metrics
from uadr.utils.seeding import seed_everything
from uadr.models import uncertainty_aware_model as uam
from uadr.models import probabilistic_NN as pnn
from uadr.models import bayesian_regression as br
from uadr.models import random_forest as rf


def crossvalidate(
    run_id: str,
    model_class: Type[uam.UncertaintyAwareModel],
    model_kwargs: dict,
    n_splits: int = 5,
    cross_validation_type: str = "warm_start",
    scaling_mode: str = "drug_z_norm",
    n_bits_drug_fingerprints: int = 128,
    base_path: str = "examples/data/experiments/5_fold_cross_validation/",
    tuning: bool = False,
    trainer_kwargs: Optional[dict] = None,
    batch_size: int = 128,
    patience: int = 5,
    monitor: str = "val_loss",
    num_workers: int = 2,
    skip_existing_folds: bool = True,
    run_split: Optional[int] = None,
    data_dir: str = "data",
    seed: int = 0,
):
    """Run a Uncertainty Aware Model fitting and tuning procedure in a cross validation

    Args:
        run_id (str): id of the run, set by user
        model_class (uam.UncertaintyAwareModel): model to run the CV for
        model_kwargs (dict): model parameters as dict, when tuning is True it should be a dict containing lists of parameters
        n_splits (int, optional): number of splits, need tobe saved beforehand. Defaults to 5.
        cross_validation_type (str, optional): "warm_start" or "cell_lines_cold_start". Defaults to "warm_start".
        scaling_mode (str, optional): type of scaling method used. Defaults to "drug_z_norm" (z-normalization per drug).
        n_bits_drug_fingerprints (int, optional): how detailed the drug embedding should be. Defaults to 128.
        base_path (str, optional): path to the experiment data. Defaults to "examples/data/experiments/5_fold_cross_validation/".
        tuning (bool, optional): whether to tune the model. Defaults to False.
        tuning_model_params (dict, optional): parameters for the tuning. Defaults to None.
        trainer_kwargs (dict, optional): parameters for the trainer. Defaults to {"progress_bar_refresh_rate": 0, "max_epochs": 1000}.
        batch_size (int): batch size for the trainer. Defaults to 128.
        patience (int): patience for the trainer. Defaults to 5.
        monitor (str): monitor for early stopping. Defaults to "val_loss". Can also set to "train_loss".
        num_workers (int): number of workers for the trainer. Defaults to 2.
        skip_existing_folds  (bool): whether to skip folds if they exits in the results folder form previous runs. Defaults to True.
        run_split (int): if not None, only run the specified split. Defaults to None.
    """
    warnings.filterwarnings("ignore", ".*does not have many workers.*")

    seed_everything(seed)

    if trainer_kwargs is None:
        trainer_kwargs = {"progress_bar_refresh_rate": 0, "max_epochs": 1000}
    result_folder_path = f"{base_path}/results/{run_id}_{scaling_mode}/"

    os.makedirs(result_folder_path, exist_ok=True)

    results = []
    prediction_metrics_folds = []
    uncertainty_metrics_folds = []

    print(f"Start {model_class}", flush=True)
    splits = range(n_splits) if run_split is None else [run_split]
    for current_split in splits:
        print(f"Current split: {current_split}", flush=True)

        if skip_existing_folds & os.path.exists(
            f"{result_folder_path}/results_{current_split}.csv"
        ):
            print(f"Skipping split {current_split} as it already exists.")
            y_test = pd.read_csv(
                f"{result_folder_path}/results_{current_split}.csv", index_col=0
            )
            prediction_metrics = np.load(
                f"{result_folder_path}/prediction_metrics_{current_split}.npy",
                allow_pickle=True,
            ).item()
            uncertainty_metrics = np.load(
                f"{result_folder_path}/uncertainty_metrics_{current_split}.npy",
                allow_pickle=True,
            ).item()
        else:
            out = get_preprocessed_features(
                current_split=current_split,
                base_path=base_path,
                cross_validation_type=cross_validation_type,
                scaling_mode=scaling_mode,
                n_bits_drug_fingerprints=n_bits_drug_fingerprints,
                data_dir=data_dir,
            )
            x_train = out["train"]["features"]
            y_train = out["train"]["targets"]
            x_validation = out["validation"]["features"]
            y_validation = out["validation"]["targets"]
            x_test = out["test"]["features"]
            y_test = out["test"]["targets"]

            if tuning:
                model_kwargs.update({"n_features": [x_train.shape[1]]})
            else:
                model_kwargs.update({"n_features": x_train.shape[1]})

            if tuning:
                tuning_trainer_kwargs = trainer_kwargs.copy()
                tuning_trainer_kwargs.update({"enable_model_summary": False})
                best_model_kwargs = hyperparameter_tuning(
                    model_kwargs=model_kwargs,
                    model_class=model_class,
                    x_train=x_train,
                    x_validation=x_validation,
                    y_train=y_train,
                    y_validation=y_validation,
                    cross_validation_type=cross_validation_type,
                    trainer_kwargs=tuning_trainer_kwargs,
                    batch_size=batch_size,
                    patience=patience,
                    num_workers=num_workers,
                )
                model = model_class(**best_model_kwargs)
            else:
                best_model_kwargs = model_kwargs
                model = model_class(**model_kwargs)

            if monitor == "train_loss":
                # concat validation and train data
                x_train = np.concatenate([x_train, x_validation])
                y_train = pd.concat([y_train, y_validation])
                x_validation = x_train
                y_validation = y_train
            elif monitor == "val_loss":
                pass
            else:
                raise ValueError("monitor must be 'train_loss' or 'val_loss'")

            model.fit(
                X_train=x_train,
                y_train=y_train.response.values,
                X_eval=x_validation,
                y_eval=y_validation.response.values,
                y_train_cell_line_index = y_train.cell_lines.values,
                y_eval_cell_line_index = y_validation.cell_lines.values,
                trainer_params=trainer_kwargs,
                batch_size=batch_size,
                patience=patience,
                num_workers=num_workers,
                cross_validation_type=cross_validation_type,
                adversarial_training=True
            )
            if current_split == 0:
                trainer_kwargs.update({"enable_model_summary": False})
                logging.getLogger("pytorch_lightning.utilities.rank_zero").setLevel(logging.WARNING)
                logging.getLogger("pytorch_lightning.accelerators.cuda").setLevel(logging.WARNING)
            if (
                isinstance(model, pnn.ProbabilisticFeedForwardEnsemble)
                or isinstance(model, br.BayesianRidgeRegression)
                or isinstance(model, rf.RandomForest)
            ):
                pass
            elif tuning:
                model = model_class.load_from_checkpoint(
                    model.checkpoint_callback.best_model_path, **best_model_kwargs
                )
            else:
                model = model_class.load_from_checkpoint(
                    model.checkpoint_callback.best_model_path, **model_kwargs
                )
            model.eval()

            y_preds, y_uncertainty = model.predict_target_and_uncertainty(x_test)

            prediction_metrics = prediction_quality_metrics(
                y_prediction=y_preds, y_target=y_test.response.values
            )
            uncertainty_metrics = uncertainty_quality_metrics(
                y_preds, y_test.response.values, y_uncertainty
            )

            y_test["y_preds"] = y_preds
            y_test["y_uncertainty"] = y_uncertainty
            y_test["fold"] = [current_split] * len(y_test)
            if isinstance(model, uam.DecomposableUncertaintyAwareModel):
                y_aleatory_uncertainty = model.predict_aleatory_uncertainty(x_test)
                y_epistemic_uncertainty = model.predict_epistemic_uncertainty(x_test)
                y_test["y_aleatory_uncertainty"] = y_aleatory_uncertainty
                y_test["y_epistemic_uncertainty"] = y_epistemic_uncertainty
                y_test["y_aleatory_uncertainty_median_agg"] = model.predict_aleatory_uncertainty(x_test, how="median")
            y_test.to_csv(f"{result_folder_path}/results_{current_split}.csv")
            np.save(
                f"{result_folder_path}/prediction_metrics_{current_split}.npy",
                prediction_metrics,
            )
            np.save(
                f"{result_folder_path}/uncertainty_metrics_{current_split}.npy",
                uncertainty_metrics,
            )
            np.save(
                f"{result_folder_path}/best_model_kwargs_{current_split}.npy",
                best_model_kwargs,
            )

            out_of_distribution_detection(x_test=x_test.copy(), y_test=y_test, model=model, result_folder_path=result_folder_path, current_split=current_split)


        results.append(y_test)
        prediction_metrics_folds.append(prediction_metrics)
        uncertainty_metrics_folds.append(uncertainty_metrics)

    results = pd.concat(results)
    results.to_csv(f"{result_folder_path}/results.csv")

    mse = np.mean([a["mse"] for a in prediction_metrics_folds])
    pearson = np.mean([a["pearson"] for a in prediction_metrics_folds])
    print()
    print(f"MSE: {mse}", flush=True)
    print(f"Pearson: {pearson}", flush=True)
    print()
    np.save(f"{result_folder_path}/hpam_space.npy", model_kwargs)
    np.save(
        f"{result_folder_path}/prediction_metrics_folds.npy", prediction_metrics_folds
    )
    np.save(
        f"{result_folder_path}/uncertainty_metrics_folds.npy", uncertainty_metrics_folds
    )
    print(f"Done {model_class}", flush=True)

def out_of_distribution_detection(x_test, y_test, model, result_folder_path, current_split):
    os.makedirs(f"{result_folder_path}/ood/", exist_ok=True)
    y_test = y_test.copy()
    for shift in [1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1, 6e-1, 7e-1, 8e-1, 9e-1, 1]:

        shift_negative = np.random.choice([-1, 1], size=x_test.shape) #some are shifted negatively
        x_test_shifted = x_test + shift_negative * np.random.normal(shift, 1e-4, x_test.shape)

        y_preds, y_uncertainty = model.predict_target_and_uncertainty(x_test_shifted)

        y_test["y_preds"] = y_preds
        y_test["y_uncertainty"] = y_uncertainty
        y_test["fold"] = [current_split] * len(y_test)
        if isinstance(model, uam.DecomposableUncertaintyAwareModel):
            y_aleatory_uncertainty = model.predict_aleatory_uncertainty(x_test_shifted)
            y_epistemic_uncertainty = model.predict_epistemic_uncertainty(x_test_shifted)
            y_test["y_aleatory_uncertainty"] = y_aleatory_uncertainty
            y_test["y_epistemic_uncertainty"] = y_epistemic_uncertainty
            y_test["y_aleatory_uncertainty_median_agg"] = model.predict_aleatory_uncertainty(x_test_shifted, how="median")

        y_test.to_csv(f"{result_folder_path}/ood/results_{current_split}_shift_{shift}.csv")



def hyperparameter_tuning(
    model_kwargs: dict,
    model_class: Type[uam.UncertaintyAwareModel],
    x_train: np.ndarray,
    x_validation: np.ndarray,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    cross_validation_type: str = "warm_start",
    trainer_kwargs: Optional[dict] = None,
    batch_size: int = 128,
    patience: int = 5,
    num_workers: int = 2,
):
    """Tune the model parameters

    Args:
        model_kwargs (dict): model parameters (dict of lists)
        model_class (uam.UncertaintyAwareModel): model to tune
        x_train (np.ndarray): training features
        x_validation (np.ndarray): validation features
        y_train (pd.DataFrame): training targets
        y_validation (pd.DataFrame): validation targets
        cross_validation_type (str, optional): "warm_start" or "cell_lines_cold_start". Defaults to "warm_start".
        trainer_kwargs (dict, optional): parameters for the trainer. Defaults to {"progress_bar_refresh_rate": 0, "max_epochs": 1000}.
        patience (int): patience for the trainer. Defaults to 5.

    Returns:
        dict: tuned model parameters
    """
    if trainer_kwargs is None:
        trainer_kwargs = {"progress_bar_refresh_rate": 0, "max_epochs": 1000}
    parameter_space = get_parameter_combinations(model_kwargs)
    x_train = x_train.copy()
    y_train = y_train.copy()

    # split of early stopping from train set
    x_train, y_train, x_early_stopping, y_early_stopping = split_early_stopping_set(
        x_train=x_train,
        y_train=y_train,
        y_validation=y_validation,
        cross_validation_type=cross_validation_type,
    )
    best_loss = np.inf
    best_parameter_combination = None
    for parameter_combination in parameter_space:
        model = model_class(**parameter_combination)

        model.fit(
            X_train=x_train,
            y_train=y_train.response.values,
            X_eval=x_early_stopping,
            y_eval=y_early_stopping.response.values,
            y_train_cell_line_index = y_train.cell_lines.values,
            y_eval_cell_line_index = y_early_stopping.cell_lines.values,
            trainer_params=trainer_kwargs,
            batch_size=batch_size,
            patience=patience,
            num_workers=num_workers,
            cross_validation_type= cross_validation_type,
            adversarial_training=True
        )
        if not (
            isinstance(model, pnn.ProbabilisticFeedForwardEnsemble)
            or isinstance(model, br.BayesianRidgeRegression)
            or isinstance(model, rf.RandomForest)
        ):
            model = model_class.load_from_checkpoint(
                model.checkpoint_callback.best_model_path, **parameter_combination
            )
        model.eval()

        y_preds, y_uncertainty = model.predict_target_and_uncertainty(x_validation)
        assert len(y_preds) == len(y_validation.response.values), f"Predictions and targets must have the same length, but are {len(y_preds)} and {len(y_validation.response.values)}"
        mse = prediction_quality_metrics(y_preds, y_validation.response.values)["mse"]
        if mse < best_loss:
            best_loss = mse
            best_parameter_combination = parameter_combination
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
    print(f"Best parameter combination: {best_parameter_combination}")
    return best_parameter_combination


def split_early_stopping_set(
    x_train: np.array,
    y_train: pd.DataFrame,
    y_validation: pd.DataFrame,
    cross_validation_type: str,
):
    if cross_validation_type == "warm_start":
        x_early_stopping = x_train[-len(y_validation) :]
        y_early_stopping = y_train.iloc[-len(y_validation) :]
        x_train = x_train[: -len(y_validation)]
        y_train = y_train.iloc[: -len(y_validation)]

    elif cross_validation_type == "cell_lines_cold_start":
        n_validation_cell_lines = len(y_validation.cell_lines.unique())
        train_cell_lines = y_train.cell_lines.unique()
        np.random.shuffle(train_cell_lines)
        early_stopping_cell_lines = train_cell_lines[:n_validation_cell_lines]
        y_early_stopping = y_train[y_train.cell_lines.isin(early_stopping_cell_lines)]
        x_early_stopping = x_train[y_train.cell_lines.isin(early_stopping_cell_lines)]
        mask = y_train.cell_lines.isin(early_stopping_cell_lines)
        y_train = y_train[~mask]
        x_train = x_train[~mask]

    else:
        raise ValueError(
            "cross_validation_type must be 'warm_start' or 'cell_lines_cold_start'"
        )
    return x_train, y_train, x_early_stopping, y_early_stopping


def get_parameter_combinations(model_kwargs: dict):
    """Get all combinations of parameters for tuning"""

    parameter_space = []

    keys = model_kwargs.keys()

    # assert that all values are lists:
    assert all(
        [isinstance(value, list) for value in model_kwargs.values()]
    ), f"All model_kwargs values must be lists when tuning but are {model_kwargs}"

    value_combinations = list(itertools.product(*model_kwargs.values()))

    for combination in value_combinations:
        output_dict = {key: value for key, value in zip(keys, combination)}
        parameter_space.append(output_dict)
    return parameter_space


def shapley_values(model: uam.UncertaintyAwareModel, x: np.ndarray):
    """Compute the Shapley values for a model

    Args:
        model (uam.UncertaintyAwareModel): model to compute the Shapley values for
        x (np.ndarray): features

    Returns:
        np.ndarray: Shapley values
    """
    raise NotImplementedError
