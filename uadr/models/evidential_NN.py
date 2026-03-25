import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

from uadr.models.uncertainty_aware_model import DecomposableUncertaintyAwareModel
from uadr.utils.data import RegressionDataset
from uadr.utils.utils import remove_checkpoints


def nig_nll_loss(
    y: Tensor, gamma: Tensor, nu: Tensor, alpha: Tensor, beta: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """Normal-Inverse-Gamma negative log-likelihood loss.

    Args:
        y: target values
        gamma: predicted mean
        nu: virtual evidence for mean (> 0)
        alpha: shape parameter (> 1)
        beta: scale parameter (> 0)
        eps: small constant for numerical stability
    """
    nu = nu.clamp(min=eps)
    beta = beta.clamp(min=eps)
    omega = 2.0 * beta * (1.0 + nu)
    nll = (
        0.5 * torch.log(math.pi / nu)
        - alpha * torch.log(omega)
        + (alpha + 0.5) * torch.log((y - gamma) ** 2 * nu + omega)
        + torch.lgamma(alpha)
        - torch.lgamma(alpha + 0.5)
    )
    return nll.mean()


def nig_reg_loss(
    y: Tensor, gamma: Tensor, nu: Tensor, alpha: Tensor
) -> Tensor:
    """Evidence regularizer — penalizes evidence on incorrect predictions."""
    return (torch.abs(y - gamma) * (2.0 * nu + alpha)).mean()


class EvidentialFeedForwardNetwork(pl.LightningModule, DecomposableUncertaintyAwareModel):
    def __init__(
        self,
        n_features,
        n_units_per_layer=None,
        dropout_prob=None,
        reg_coeff: float = 0.01,
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
        # Output: 4 parameters (gamma, nu, alpha, beta)
        self.fully_connected_layers.append(nn.Linear(n_units_per_layer[-1], 4))

        self.dropout_prob = dropout_prob
        if dropout_prob is not None:
            self.dropout = nn.Dropout(dropout_prob)
        self.reg_coeff = reg_coeff
        self.softplus = nn.Softplus()

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

        trainer = pl.Trainer(
            callbacks=[early_stop_callback, self.checkpoint_callback, progress_bar],
            **trainer_params_copy
        )
        remove_checkpoints(trainer, checkpoint_path)

        trainer.fit(self, train_loader, val_loader)

    def forward(self, x):
        for layer in self.fully_connected_layers[:-2]:
            x = torch.relu(layer(x))
            if self.dropout_prob is not None:
                x = self.dropout(x)
        x = torch.relu(self.fully_connected_layers[-2](x))
        x = self.fully_connected_layers[-1](x)

        eps = 1e-6
        gamma = x[:, 0]
        nu = self.softplus(x[:, 1]) + eps
        alpha = self.softplus(x[:, 2]) + 1.0 + eps  # enforce > 1
        beta = self.softplus(x[:, 3]) + eps

        return gamma, nu, alpha, beta

    def _compute_loss(self, x, y):
        gamma, nu, alpha, beta = self.forward(x)
        nll = nig_nll_loss(y, gamma, nu, alpha, beta)
        reg = nig_reg_loss(y, gamma, nu, alpha)
        return nll + self.reg_coeff * reg

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self._compute_loss(x, y)
        self.log("val_loss", loss)
        return loss

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            gamma, _, _, _ = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        return gamma.cpu().numpy()

    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        aleatory = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return (aleatory + epistemic).cpu().numpy()

    def predict_aleatory_uncertainty(self, X: np.ndarray, **kwargs) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, _, alpha, beta = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        return (beta / (alpha - 1.0)).cpu().numpy()

    def predict_epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            _, nu, alpha, beta = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        return (beta / (nu * (alpha - 1.0))).cpu().numpy()

    def predict_target_and_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        is_training = self.training
        self.eval()
        with torch.no_grad():
            gamma, nu, alpha, beta = self.forward(torch.from_numpy(X).float().to(self.device))
        self.train(is_training)
        aleatory = beta / (alpha - 1.0)
        epistemic = beta / (nu * (alpha - 1.0))
        return gamma.cpu().numpy(), (aleatory + epistemic).cpu().numpy()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def on_before_optimizer_step(self, optimizer):
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
