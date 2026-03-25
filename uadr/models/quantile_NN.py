from typing import Optional, Tuple
from uadr.models.uncertainty_aware_model import (
    UncertaintyAwareModel)
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


class QuantileFeedForwardNetwork(pl.LightningModule, UncertaintyAwareModel):
    def __init__(
        self, n_features, quantiles=None, n_units_per_layer=None, dropout_prob=None
    ) -> None:
        if n_units_per_layer is None:
            n_units_per_layer = [1024, 64]
        if quantiles is None:
            quantiles = [0.05, 0.5, 0.95]
        self.quantiles = quantiles

        super().__init__()
        self.fully_connected_layers = nn.ModuleList()
        self.fully_connected_layers.append(nn.Linear(n_features, n_units_per_layer[0]))
        for i in range(1, len(n_units_per_layer)):
            self.fully_connected_layers.append(
                nn.Linear(n_units_per_layer[i - 1], n_units_per_layer[i])
            )
        self.fully_connected_layers.append(nn.Linear(n_units_per_layer[-1], 3))

        self.dropout_prob = dropout_prob
        if dropout_prob is not None:
            self.dropout_layer = nn.Dropout(p=dropout_prob)

    def quantile_loss(self, y_pred, y_true, quantile):
        losses = []
        for i, q in enumerate(quantile):
            errors = y_true - y_pred[:, i]
            losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(1))
        loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray] = None,
        y_eval: Optional[np.ndarray] = None,
        trainer_params: Optional[dict] = None,
        batch_size=32,
        patience=3,
        checkpoint_path: Optional[str] = None,
        num_workers: int = 2,
        **kwargs

    ):
        if trainer_params is None:
            trainer_params = {"progress_bar_refresh_rate": 0, "max_epochs": 100}

        train_dataset = RegressionDataset(X_train, y_train)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
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
                x = self.dropout_layer(x)
        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)
        return x

    def _forward_loss_and_log(self, x, y, log_as: str):
        predicted_quantiles = self.forward(x)

        result = self.quantile_loss(predicted_quantiles, y, self.quantiles)
        self.log(log_as, result)
        return result

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        assert np.abs(self.quantiles[1] - 0.5) < 1e-5
        return self.predict_quantiles(X)[:, 1]

    def predict_quantiles(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            X = torch.from_numpy(X).float()
            X = X.to(self.device)
            quantiles = self.forward(X)
        self.train(is_training)
        return quantiles.cpu().numpy()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        # inter"quartile" range
        quantiles = self.predict_quantiles(X)
        return quantiles[:, -1] - quantiles[:, 0]

    def predict_target_and_uncertainty(self, X: np.ndarray) -> np.ndarray:
        assert (
            np.abs(self.quantiles[1] - 0.5) < 1e-5
        ), "second quantile needs to be median"
        quantiles = self.predict_quantiles(X)
        return quantiles[:, 1], quantiles[:, -1] - quantiles[:, 0]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
