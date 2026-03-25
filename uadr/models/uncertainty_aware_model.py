from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class UncertaintyAwareModel(ABC):

    """
    Abstract class for uncertainty aware models.

    """

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: Optional[np.ndarray],
        y_eval: Optional[np.ndarray],
        **kwargs
    ) -> None:
        pass

    @abstractmethod
    def predict_target(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_uncertainty(self, X: np.ndarray) -> np.ndarray:
        pass


class DecomposableUncertaintyAwareModel(UncertaintyAwareModel):

    """
    An abstract class for uncertainty aware models where the uncertainty can be decomposed into aleatory and epistemic components.
    """

    @abstractmethod
    def predict_aleatory_uncertainty(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_epistemic_uncertainty(self, X: np.ndarray) -> np.ndarray:
        pass
