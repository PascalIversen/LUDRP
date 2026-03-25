import os
import random

import pandas as pd
import numpy as np
import torch
from pytorch_lightning import seed_everything

from uadr.utils.data_preprocessing import get_preprocessed_features
from uadr.models import probabilistic_NN as pnn

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
seed_everything(SEED, workers=True)
torch.use_deterministic_algorithms(True)
SCALING_MODE = "drug_z_norm"

CHECKPOINT_DIR = "model_checkpoints" + "_" + SCALING_MODE
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
CURRENT_SPLIT = 0
CHECKPOINT_FILENAME = f"split_{CURRENT_SPLIT}_best.ckpt"

CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, CHECKPOINT_FILENAME)

def load_data(base_path: str, current_split: int, data_dir="../data"):
    """
    Load preprocessed features and targets.

    Args:
        base_path (str): Base path of dataset.
        current_split (int): Cross-validation split index.

    Returns:
        dict: Dataset dictionary with train, validation, test splits and gene list.
    """
    return get_preprocessed_features(
        current_split=current_split,
        base_path=base_path,
        cross_validation_type="cell_lines_cold_start",
        scaling_mode=SCALING_MODE,
        n_bits_drug_fingerprints=128,
        data_dir=data_dir
    )


def train_model(x_train: np.ndarray, y_train: pd.DataFrame, x_val: np.ndarray, y_val: pd.DataFrame, model_kwargs: dict):
    """
    Train and return a probabilistic feedforward network.

    Args:
        x_train (np.ndarray): Training features.
        y_train (pd.DataFrame): Training targets.
        x_val (np.ndarray): Validation features.
        y_val (pd.DataFrame): Validation targets.
        model_kwargs (dict): Model hyperparameters.

    Returns:
        pnn.ProbabilisticFeedForwardNetwork: Trained model.
    """
    model = pnn.ProbabilisticFeedForwardNetwork(**model_kwargs)

    trainer_params = {
        "progress_bar_refresh_rate": 10,
        "max_epochs": 1000,
        "deterministic": True,
    }

    model.fit(
        X_train=x_train,
        y_train=y_train.response.values,
        X_eval=x_val,
        y_eval=y_val.response.values,
        batch_size=128,
        patience=5,
        trainer_params=trainer_params,
    )
    best_path = model.checkpoint_callback.best_model_path
    os.rename(best_path, CHECKPOINT_PATH)
    return model


def load_model(model_kwargs: dict) -> pnn.ProbabilisticFeedForwardNetwork:
    """
    Load model from checkpoint.

    Args:
        model_kwargs (dict): Model hyperparameters.

    Returns:
        pnn.ProbabilisticFeedForwardNetwork: Loaded model.
    """
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CHECKPOINT_PATH}")
    model = pnn.ProbabilisticFeedForwardNetwork.load_from_checkpoint(CHECKPOINT_PATH, **model_kwargs)
    model.eval()
    model.to("cpu")
    return model


def predict(model: pnn.ProbabilisticFeedForwardNetwork, x_test: np.ndarray, y_test: pd.DataFrame):
    """
    Perform prediction and uncertainty estimation on test data.

    Args:
        model (pnn.ProbabilisticFeedForwardNetwork): Loaded model.
        x_test (np.ndarray): Test features.
        y_test (pd.DataFrame): Test targets.

    Returns:
        tuple: (uncertainty estimates (np.ndarray), squared errors (np.ndarray))
    """
    y_uncertainty = model.predict_uncertainty(x_test)
    error = np.square(y_test.response.values - model.predict_target(x_test))
    return y_uncertainty, error


if __name__ == "__main__":
    BASE_PATH = "../data/experiments/5_fold_cross_validation/"
    DATA_DIR = "../data"

    data = load_data(BASE_PATH, CURRENT_SPLIT, data_dir=DATA_DIR)

    x_train = data["train"]["features"]
    y_train = data["train"]["targets"]
    x_val = data["validation"]["features"]
    y_val = data["validation"]["targets"]
    x_test = data["test"]["features"]
    y_test = data["test"]["targets"]
    gene_list = data["gene_list"]

    model_params = {
        "n_features": x_train.shape[1],
        "n_units_per_layer": [128, 32, 16],
        "dropout_prob": 0.2,
    }

    trained_model = train_model(x_train, y_train, x_val, y_val, model_params)


