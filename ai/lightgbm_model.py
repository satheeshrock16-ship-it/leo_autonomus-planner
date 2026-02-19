"""Burn/no-burn decision layer using LightGBM when available."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

try:
    import lightgbm as lgb
except Exception:  # Keep system runnable without optional dependency.
    lgb = None


@dataclass
class BurnDecision:
    should_burn: bool
    confidence: float


class BurnDecisionModel:
    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if lgb is None:
            self.model = {"threshold": 0.5}
            return
        train = lgb.Dataset(X, label=y)
        self.model = lgb.train({"objective": "binary", "verbosity": -1}, train, num_boost_round=30)

    def predict(self, features: np.ndarray) -> BurnDecision:
        if self.model is None:
            raise RuntimeError("BurnDecisionModel is not fitted.")
        if lgb is None:
            score = float(np.clip(features[0], 0.0, 1.0))
        else:
            score = float(self.model.predict(features.reshape(1, -1))[0])
        return BurnDecision(should_burn=score > 0.5, confidence=score)
