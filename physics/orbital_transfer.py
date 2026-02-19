"""Return-to-original-orbit correction burn utilities."""
from __future__ import annotations

import numpy as np

from config import EARTH_MU_KM3_S2


def correction_burn_for_return(
    current_radius_km: float,
    nominal_radius_km: float,
    mu: float = EARTH_MU_KM3_S2,
) -> float:
    """Compute tangential burn magnitude to restore circular nominal orbit."""
    v_current = np.sqrt(mu / current_radius_km)
    v_nominal = np.sqrt(mu / nominal_radius_km)
    return float(v_nominal - v_current)


def apply_instantaneous_delta_v(velocity_eci: np.ndarray, delta_v_eci: np.ndarray) -> np.ndarray:
    return np.asarray(velocity_eci) + np.asarray(delta_v_eci)
