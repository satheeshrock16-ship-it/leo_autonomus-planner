"""Regression model for predicting required collision-avoidance delta-v."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None


@dataclass
class RegressorMetrics:
    mae: float
    rmse: float
    r2: float


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if denom <= 1e-12:
        return 0.0
    return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))


class DeltaVRegressor:
    def __init__(self) -> None:
        self.model: Any = None
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.feature_mean = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0) + 1e-12
        if lgb is not None:
            dataset = lgb.Dataset(X, label=y)
            self.model = lgb.train(
                {
                    "objective": "regression",
                    "metric": "l2",
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "verbosity": -1,
                },
                dataset,
                num_boost_round=120,
            )
            return

        # Fallback linear model via least squares.
        Xn = (X - self.feature_mean) / self.feature_std
        Xb = np.hstack([np.ones((Xn.shape[0], 1), dtype=float), Xn])
        coef, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        self.model = {"coef": coef.tolist()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeltaVRegressor is not fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if lgb is not None and hasattr(self.model, "predict"):
            return np.asarray(self.model.predict(X), dtype=float)
        coef = np.asarray(self.model["coef"], dtype=float)
        Xn = (X - self.feature_mean) / self.feature_std
        Xb = np.hstack([np.ones((Xn.shape[0], 1), dtype=float), Xn])
        return Xb @ coef

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> RegressorMetrics:
        preds = self.predict(X)
        y = np.asarray(y_true, dtype=float)
        return RegressorMetrics(mae=_mae(y, preds), rmse=_rmse(y, preds), r2=_r2(y, preds))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if lgb is not None and hasattr(self.model, "save_model"):
            self.model.save_model(str(p))
            return
        payload = {
            "backend": "linear_fallback",
            "model": self.model,
            "feature_mean": None if self.feature_mean is None else self.feature_mean.tolist(),
            "feature_std": None if self.feature_std is None else self.feature_std.tolist(),
        }
        with p.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "DeltaVRegressor":
        p = Path(path)
        reg = cls()
        if lgb is not None:
            try:
                reg.model = lgb.Booster(model_file=str(p))
                return reg
            except Exception:
                pass
        with p.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        reg.model = payload.get("model")
        mean = payload.get("feature_mean")
        std = payload.get("feature_std")
        reg.feature_mean = None if mean is None else np.asarray(mean, dtype=float)
        reg.feature_std = None if std is None else np.asarray(std, dtype=float)
        return reg
