import numpy as np
from sklearn.linear_model import BayesianRidge
from uadr.models.uncertainty_aware_model import UncertaintyAwareModel
from typing import Optional


class BayesianRidgeRegression(UncertaintyAwareModel):
    def __init__(
        self,
        lambda_1: float = 1e-06,
        lambda_2: float = 1e-06,
        alpha_1: float = 1e-06,
        alpha_2: float = 1e-06,
        **kwargs
    ) -> None:
        self.model = BayesianRidge(
            max_iter=500,
            alpha_1=alpha_1,
            alpha_2=alpha_2,
            lambda_1=lambda_1,
            lambda_2=lambda_2,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray] = None,
        y_eval: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        X = X_train.copy()
        y = y_train.copy()
        if X_eval is not None and y_eval is not None:
            X = np.concatenate((X, X_eval))
            y = np.concatenate((y, y_eval))
        self.model.fit(X, y)

    def eval(self):
        pass

    def predict_target(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_uncertainty(self, X):
        return self.model.predict(X, return_std=True)[1]

    def predict_target_and_uncertainty(self, X):
        return self.model.predict(X, return_std=True)
