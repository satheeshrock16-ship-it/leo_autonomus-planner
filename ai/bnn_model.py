"""Bayesian uncertainty estimator for collision probability calibration."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class BNNPrediction:
    mean_probability: float
    epistemic_std: float


class MonteCarloDropoutBNN:
    """Simple Monte-Carlo surrogate standing in for a full Bayesian NN."""

    def __init__(self, samples: int = 50, dropout_scale: float = 0.05):
        self.samples = samples
        self.dropout_scale = dropout_scale

    def predict(self, base_probability: float) -> BNNPrediction:
        draws = np.clip(
            np.random.normal(base_probability, self.dropout_scale, size=self.samples),
            0.0,
            1.0,
        )
        return BNNPrediction(float(draws.mean()), float(draws.std(ddof=1)))
