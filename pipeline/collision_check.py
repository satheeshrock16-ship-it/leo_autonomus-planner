"""Collision screening and probabilistic risk analysis."""
from __future__ import annotations

import numpy as np

from physics.collision_probability import collision_probability


def run(relative_position_km: np.ndarray, covariance_km2: np.ndarray) -> float:
    return collision_probability(relative_position_km, covariance_km2)
