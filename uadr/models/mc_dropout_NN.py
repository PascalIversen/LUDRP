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


class MCDropoutFeedForwardNetwork(pl.LightningModule, UncertaintyAwareModel):
    def __init__(
        self, n_features, n_units_per_layer=None, dropout_prob=None, sample_size=100
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
        self.fully_connected_layers.append(nn.Linear(n_units_per_layer[-1], 1))
        if dropout_prob is not None:
            self.dropout_layer = nn.Dropout(p=dropout_prob)
        else:
            self.dropout_layer = None
        self.loss = nn.MSELoss()
        self.sample_size = sample_size

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
        num_workers: int = 2,
        **kwargs

    ):
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
        for layer in self.fully_connected_layers[:-3]:
            
            x = torch.relu(layer(x))
            if self.dropout_layer is not None:
                x = self.dropout_layer(x)
        x = torch.relu(self.fully_connected_layers[-3](x))
        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)
        return x.squeeze()

    def _forward_loss_and_log(self, x, y, log_as: str):
        y_pred = self.forward(x)

        result = self.loss(y_pred, y)
        self.log(log_as, result)
        return result

    def training_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "train_loss")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        return self._forward_loss_and_log(x, y, "val_loss")

    def _get_predictions(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        self.dropout_layer.train()
        predictions = []
        for _ in range(self.sample_size):
            with torch.no_grad():
                y_pred = self.forward(torch.from_numpy(X).float().to(self.device))
            predictions.append(y_pred.cpu().numpy())
        predictions = np.stack(predictions)
        self.train(is_training)
        return predictions

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        predictions = self._get_predictions(X)
        y_pred = np.mean(predictions, axis=0)
        return y_pred

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        predictions = self._get_predictions(X)
        y_pred = np.std(predictions, axis=0)
        return y_pred

    def predict_target_and_uncertainty(self, X: np.ndarray) -> np.ndarray:
        predictions = self._get_predictions(X)
        y_pred = np.mean(predictions, axis=0)
        y_uncertainty = np.std(predictions, axis=0)
        return y_pred, y_uncertainty

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())
