"""Collision probability via Mahalanobis distance approximation."""
from __future__ import annotations

import numpy as np


def mahalanobis_distance(relative_position_km: np.ndarray, covariance_km2: np.ndarray) -> float:
    r = np.asarray(relative_position_km).reshape(3, 1)
    p_inv = np.linalg.pinv(covariance_km2)
    d2 = float((r.T @ p_inv @ r)[0, 0])
    return float(np.sqrt(max(d2, 0.0)))


def collision_probability(relative_position_km: np.ndarray, covariance_km2: np.ndarray) -> float:
    """Gaussian encounter proxy: Pc = exp(-0.5 * D^2)."""
    d = mahalanobis_distance(relative_position_km, covariance_km2)
    return float(np.exp(-0.5 * d**2))
