from typing import Optional, Tuple
from uadr.models.uncertainty_aware_model import (
    UncertaintyAwareModel,
    DecomposableUncertaintyAwareModel,
)
from uadr.utils.data import RegressionDataset
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
from uadr.utils.utils import remove_checkpoints
from torch import Tensor

# reimplementation of the fuassian nl loss function from torch.nn, with the addition of importance weighting and variance regularization (unused in the current setup, but may be useful for future experiments)
def gaussian_nll_loss(
    input: Tensor,
    target: Tensor,
    var: Tensor,
    full: bool = False,
    eps: float = 1e-6,
    reduction: str = "mean",
    var_scaling: float = 1.0,
    var_regularizer: float = 1.0,
    importance_weighting: bool = False,
) -> Tensor:
    r"""Gaussian negative log likelihood loss.

    See :class:`~torch.nn.GaussianNLLLoss` for details.

    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full (bool, optional): include the constant term in the loss calculation. Default: ``False``.
        eps (float, optional): value added to var, for stability. Default: 1e-6.
        reduction (str, optional): specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """

    # Check var size
    # If var.size == input.size, the case is heteroscedastic and no further checks are needed.
    # Otherwise:
    if var.size() != input.size():
        # If var is one dimension short of input, but the sizes match otherwise, then this is a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2)
        # -> unsqueeze var so that var.shape = (10, 2, 1)
        # this is done so that broadcasting can happen in the loss calculation
        if input.size()[:-1] == var.size():
            var = torch.unsqueeze(var, -1)

        # This checks if the sizes match up to the final dimension, and the final dimension of var is of size 1.
        # This is also a homoscedastic case.
        # e.g. input.size = (10, 2, 3), var.size = (10, 2, 1)
        elif (
            input.size()[:-1] == var.size()[:-1] and var.size(-1) == 1
        ):  # Heteroscedastic case
            pass

        # If none of the above pass, then the size of var is incorrect.
        else:
            raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != "none" and reduction != "mean" and reduction != "sum":
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate the loss
    loss = 0.5 * (
        var_regularizer * torch.log(var_scaling * var)
        + (input - target) ** 2 / (var_scaling * var)
    )

    if importance_weighting:
        with torch.no_grad():
            minimum = 0.7
            maximum = 1.3
            weight = var
            weight = (weight - weight.min()) / (weight.max() - weight.min())
            weight = weight * (maximum - minimum) + minimum
        loss = loss * weight

    if full:
        import math

        loss += 0.5 * math.log(2 * math.pi)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss


class ProbabilisticFeedForwardNetwork(pl.LightningModule, UncertaintyAwareModel):
    def __init__(
        self,
        n_features,
        n_units_per_layer=None,
        dropout_prob=None,
        var_scaling: float = 1.0,
        importance_weighting: bool = False,
    ) -> None:
        if n_units_per_layer is None:
            n_units_per_layer = [1024, 64]

        super().__init__()
        self.fully_connected_layers = nn.ModuleList()
        self.fully_connected_layers.append(nn.Linear(n_features, n_units_per_layer[0]))
        for i in range(1, len(n_units_per_layer)):
            self.fully_connected_layers.append(
                nn.Linear(n_units_per_layer[i - 1], n_units_per_layer[i])
            )
        self.fully_connected_layers.append(nn.Linear(n_units_per_layer[-1], 2))

        self.dropout_prob = dropout_prob
        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        self.var_scaling = var_scaling
        self.nll_loss = gaussian_nll_loss
        self.mse_loss = nn.MSELoss()
        self.mse_only = False
        self.importance_weighting = importance_weighting

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray],
        y_eval: Optional[np.ndarray],
        trainer_params: Optional[dict] = None,
        batch_size=32,
        patience=3,
        checkpoint_path: Optional[str] = None,
        adversarial_training: bool = False,
        adversarial_epsilon: float = 0.01,
        num_workers: int = 2,
        **kwargs
    ):
        self.adversarial_training = adversarial_training
        print("Adversarial training:", adversarial_training)
        self.adversarial_epsilon = adversarial_epsilon
        if trainer_params is None:
            trainer_params = {"progress_bar_refresh_rate": 0, "max_epochs": 100}

        train_dataset = RegressionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        if (X_eval is not None) and (y_eval is not None):
            val_dataset = RegressionDataset(X_eval, y_eval)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        # Train the model
        monitor = "train_loss" if ((X_eval is None) or (y_eval is None)) else "val_loss"

        early_stop_callback = EarlyStopping(
            monitor=monitor, mode="min", patience=patience
        )
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_path, monitor=monitor, mode="min", save_top_k=1
        )

        progress_bar = TQDMProgressBar(
            refresh_rate=trainer_params["progress_bar_refresh_rate"]
        )
        trainer_params_copy = trainer_params.copy()
        del trainer_params_copy["progress_bar_refresh_rate"]

        # Initialize the Lightning trainer
        trainer = pl.Trainer(
            callbacks=[early_stop_callback, self.checkpoint_callback, progress_bar],
            **trainer_params_copy
        )
        remove_checkpoints(trainer, checkpoint_path)

        trainer.fit(self, train_loader, val_loader)
        # self.load_from_checkpoint(self.checkpoint_callback.best_model_path)

    def forward(self, x):

        for layer in self.fully_connected_layers[:-2]:

            x = torch.relu(layer(x))
            if self.dropout_prob is not None:
                x = self.dropout(x)
        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)
        mean = x[:, 0]
        variance = x[:, 1]
        # enforce positive variance
        variance = torch.exp(variance)

        return mean, variance

    def adversarial_training_step(self, batch, batch_idx):
        x, y = batch
        x.requires_grad = True
        loss = self._forward_loss_and_log(x, y, "train_loss")
        loss.backward(retain_graph=True)
        x_grad = torch.sign(
            x.grad.data
        )  # calculate the sign of gradient of the loss func (with respect to input X) (adv)
        x_adversarial = x.data + self.adversarial_epsilon * x_grad
        loss_adv = self._forward_loss_and_log(x_adversarial, y, "train_loss_adv")
        return loss + loss_adv

    def _forward_loss_and_log(self, x, y, log_as: str):
        mean, variance = self.forward(x)

        if self.mse_only:
            result = self.mse_loss(mean, y)
        else:
            result = self.nll_loss(
                input=mean,
                target=y,
                var=variance,
                var_scaling=self.var_scaling,
                importance_weighting=self.importance_weighting,
            )
        self.log(log_as, result)
        return result

    def non_adversarial_training_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def training_step(self, batch, batch_idx):
        if self.adversarial_training:
            return self.adversarial_training_step(batch, batch_idx)
        else:
            return self.non_adversarial_training_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._forward_loss_and_log(x, y, "val_loss")
        return loss

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            mean, _ = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        return mean.cpu().numpy()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, variance = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        return variance.cpu().numpy()

    def predict_target_and_uncertainty(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            mean, variance = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        return mean.cpu().numpy(), variance.cpu().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
    def finetune(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray] = None,
        y_eval: Optional[np.ndarray] = None,
        trainer_params: Optional[dict] = None,
        batch_size=32,
        patience=3,
        max_epochs=1000,
        finetune_lr: float = 1e-4,
        weight_decay: float = 0.0,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 2,
        **kwargs
    ):
        self.finetune_lr = finetune_lr
        self.weight_decay = weight_decay

        train_dataset = RegressionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

        val_loader = None
        if (X_eval is not None) and (y_eval is not None):
            val_dataset = RegressionDataset(X_eval, y_eval)
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )

        early_stop_callback = EarlyStopping(
            monitor="val_loss" if val_loader is not None else "train_loss",
            mode="min",
            patience=patience
        )
        self.checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_path,
            monitor="val_loss" if val_loader is not None else "train_loss",
            mode="min",
            save_top_k=1
        )

        if trainer_params is None:
            trainer_params = {}
        trainer_params = trainer_params.copy()
        trainer_params.pop("progress_bar_refresh_rate", None)
        trainer_params["enable_progress_bar"] = False

        if trainer_params is None:
            trainer_params = {}
        trainer_params = trainer_params.copy()
        trainer_params.pop("progress_bar_refresh_rate", None)
        trainer_params.pop("max_epochs", None)  # remove max_epochs if present
        trainer_params["enable_progress_bar"] = False

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, self.checkpoint_callback],
            enable_model_summary=False,
            log_every_n_steps=9999,
            **trainer_params
        )


        trainer.fit(self, train_loader, val_loader)

class ProbabilisticFeedForwardEnsemble(DecomposableUncertaintyAwareModel):
    def __init__(
        self,
        n_features,
        n_models=10,
        n_units_per_layer=None,
        dropout_prob=None,
        var_scaling: float = 1.0,
        shuffle_eval: bool = False,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.checkpoint_paths = []
        self.model_kwargs = {
            "n_features": n_features,
            "n_units_per_layer": n_units_per_layer,
            "dropout_prob": dropout_prob,
            "var_scaling": var_scaling,
        }

        self.training = True
        self.shuffle_eval = shuffle_eval

    def _load_model(self, checkpoint_path: str) -> ProbabilisticFeedForwardNetwork:
        model = ProbabilisticFeedForwardNetwork.load_from_checkpoint(
            checkpoint_path, **self.model_kwargs
        )
        model.cpu()
        model.eval()
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray],
        y_eval: Optional[np.ndarray],
        y_train_cell_line_index: Optional[np.ndarray] = None,
        y_eval_cell_line_index: Optional[np.ndarray] = None,
        cross_validation_type: str = "warm_start",
        trainer_params: Optional[dict] = None,
        batch_size=32,
        patience=3,
        checkpoint_path: Optional[str] = None,
        adversarial_training: bool = False,
        adversarial_epsilon: float = 0.01,
        num_workers: int = 2,
        **kwargs

    ):
        self.checkpoint_paths = []
        for i in range(self.n_models):
            if self.shuffle_eval:
                if i == 0:
                    print("Shuffling eval data")
                X = np.concatenate((X_train, X_eval))
                y = np.concatenate((y_train, y_eval))
                if cross_validation_type == "warm_start":
                    len_train = len(X_train)

                    idx = np.random.permutation(len(X))
                    X = X[idx]
                    y = y[idx]
                    X_eval = X[len_train:]
                    y_eval = y[len_train:]
                    X_train = X[:len_train]
                    y_train = y[:len_train]

                elif cross_validation_type == "cell_lines_cold_start":
                    unique_cell_lines_train = np.unique(y_train_cell_line_index)
                    all_cell_lines_index = np.concatenate((y_train_cell_line_index, y_eval_cell_line_index))
                    unqiue_all_cell_lines = np.unique(all_cell_lines_index)
                    np.random.shuffle(unqiue_all_cell_lines)
                    new_unique_cell_lines_train = unqiue_all_cell_lines[:len(unique_cell_lines_train)]
                    new_unique_cell_lines_eval = unqiue_all_cell_lines[len(unique_cell_lines_train):]
                    idx_train = np.isin(all_cell_lines_index, new_unique_cell_lines_train)
                    idx_eval = np.isin(all_cell_lines_index, new_unique_cell_lines_eval)
                    X_train = X[idx_train]
                    y_train = y[idx_train]
                    X_eval = X[idx_eval]
                    y_eval = y[idx_eval]
                    assert set(new_unique_cell_lines_train).isdisjoint(set(new_unique_cell_lines_eval))
                else:
                    raise ValueError(
                        f"Cross validation type {cross_validation_type} not recognized"
                    )
            else:
                idx = np.random.permutation(len(X_train))
                X_train = X_train[idx]
                y_train = y_train[idx]

            model = ProbabilisticFeedForwardNetwork(**self.model_kwargs)
            model.fit(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                trainer_params=trainer_params,
                batch_size=batch_size,
                patience=patience,
                checkpoint_path=checkpoint_path,
                adversarial_training=adversarial_training,
                adversarial_epsilon=adversarial_epsilon,
                num_workers=num_workers,
            )
            self.checkpoint_paths.append(model.checkpoint_callback.best_model_path)
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    def eval(self):
        self.training = False

    def train(self):
        self.training = True

    def _collect_predictions(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        x_tensor = torch.from_numpy(X).float().cpu()
        all_means = []
        all_variances = []
        for ckpt_path in self.checkpoint_paths:
            model = self._load_model(ckpt_path)
            with torch.no_grad():
                mean, variance = model.forward(x_tensor)
            all_means.append(mean.numpy())
            all_variances.append(variance.numpy())
            del model
        return np.stack(all_means), np.stack(all_variances)

    def predict_target_and_uncertainty(
        self, X: np.ndarray, how: str = "mean"
    ) -> Tuple[np.ndarray, np.ndarray]:
        means, variances = self._collect_predictions(X)
        target = means.mean(axis=0)
        aleatory = variances.mean(axis=0) if how == "mean" else np.median(variances, axis=0)
        epistemic = np.var(means, axis=0)
        return target, aleatory + epistemic

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        means, _ = self._collect_predictions(X)
        return means.mean(axis=0)

    def predict_uncertainty(self, X, how: str = "mean") -> np.ndarray:
        means, variances = self._collect_predictions(X)
        aleatory = variances.mean(axis=0) if how == "mean" else np.median(variances, axis=0)
        epistemic = np.var(means, axis=0)
        return aleatory + epistemic

    def predict_aleatory_uncertainty(self, X: np.ndarray, how: str = "mean") -> np.ndarray:
        _, variances = self._collect_predictions(X)
        if how == "mean":
            return variances.mean(axis=0)
        elif how == "median":
            return np.median(variances, axis=0)
        else:
            raise ValueError("how must be either 'mean' or 'median'")

    def predict_epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        means, _ = self._collect_predictions(X)
        return np.var(means, axis=0)
