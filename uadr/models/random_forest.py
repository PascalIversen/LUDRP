from sklearn.ensemble import RandomForestRegressor
import numpy as np
from .uncertainty_aware_model import UncertaintyAwareModel


class RandomForest(UncertaintyAwareModel):
    def __init__(self, n_estimators=150, max_depth=None, random_state=None, **kwargs):
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           random_state=random_state, max_samples=0.85, n_jobs=-1, max_features=0.85, verbose=2)

    def fit(self, X_train, y_train, **kwargs):
        self.model.fit(X_train, y_train)

    def predict_target(self, X):
        return self.model.predict(X)

    def predict_uncertainty(self, X):
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        return np.var(predictions, axis=0)

    def predict_target_and_uncertainty(self, X):
        mean_prediction = self.predict_target(X)
        uncertainty = self.predict_uncertainty(X)
        return mean_prediction, uncertainty
    def eval(self):
        pass