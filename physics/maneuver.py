"""Delta-V maneuver planning primitives."""
from __future__ import annotations

import numpy as np

from config import EARTH_MU_KM3_S2


BURN_AXES = {
    "tangential": np.array([0.0, 1.0, 0.0]),
    "radial": np.array([1.0, 0.0, 0.0]),
    "normal": np.array([0.0, 0.0, 1.0]),
}


def vis_viva_speed(radius_km: float, semi_major_axis_km: float, mu: float = EARTH_MU_KM3_S2) -> float:
    return float(np.sqrt(mu * (2 / radius_km - 1 / semi_major_axis_km)))


def estimate_tangential_delta_v_for_altitude_offset(
    radius_km: float,
    desired_delta_radius_km: float,
    mu: float = EARTH_MU_KM3_S2,
) -> float:
    """Small impulse approximation for near-circular orbit change."""
    v_circ = np.sqrt(mu / radius_km)
    return float(0.5 * v_circ * desired_delta_radius_km / radius_km)


def build_thrust_vector(delta_v_km_s: float, burn_type: str) -> np.ndarray:
    if burn_type not in BURN_AXES:
        raise ValueError(f"Unknown burn type '{burn_type}'.")
    return BURN_AXES[burn_type] * delta_v_km_s


def burn_duration_ms(delta_v_km_s: float, accel_m_s2: float = 0.5) -> int:
    delta_v_m_s = delta_v_km_s * 1000.0
    return int(max(50, 1000.0 * delta_v_m_s / accel_m_s2))
