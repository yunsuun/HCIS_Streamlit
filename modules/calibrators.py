import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

RANDOM_STATE = 42


class BaseCalibrator:
    name = "base"
    def fit(self, oof_pred, y_true):
        raise NotImplementedError
    def predict(self, pred):
        raise NotImplementedError


class NoneCalibrator(BaseCalibrator):
    name = "none"
    def fit(self, oof_pred, y_true):
        return self
    def predict(self, pred):
        return np.clip(np.asarray(pred), 1e-15, 1 - 1e-15)


class PlattCalibrator(BaseCalibrator):
    name = "platt"
    def __init__(self):
        self.lr = LogisticRegression(
            solver="lbfgs",
            random_state=RANDOM_STATE,
            max_iter=2000
        )
    def fit(self, oof_pred, y_true):
        self.lr.fit(
            np.asarray(oof_pred).reshape(-1, 1),
            np.asarray(y_true).astype(int)
        )
        return self
    def predict(self, pred):
        return np.clip(
            self.lr.predict_proba(
                np.asarray(pred).reshape(-1, 1)
            )[:, 1],
            1e-15,
            1 - 1e-15
        )


class IsotonicCalibrator(BaseCalibrator):
    name = "isotonic"
    def __init__(self):
        self.iso = IsotonicRegression(out_of_bounds="clip")
    def fit(self, oof_pred, y_true):
        self.iso.fit(
            np.asarray(oof_pred).astype(float),
            np.asarray(y_true).astype(int)
        )
        return self
    def predict(self, pred):
        return np.clip(
            self.iso.predict(np.asarray(pred).astype(float)),
            1e-15,
            1 - 1e-15
        )