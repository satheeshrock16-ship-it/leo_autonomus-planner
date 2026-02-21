"""Pure-physics burn-time and delta-v conversions."""
from __future__ import annotations

from typing import Any

import numpy as np


SATELLITE_MASS_KG = 500.0
THRUST_N = 5.0


def delta_v_from_burn_time_km_s(
    burn_time_seconds: float,
    mass_kg: float = SATELLITE_MASS_KG,
    force_newtons: float = THRUST_N,
) -> float:
    accel_m_s2 = float(force_newtons) / max(float(mass_kg), 1e-9)
    delta_v_m_s = accel_m_s2 * max(float(burn_time_seconds), 0.0)
    return float(delta_v_m_s / 1000.0)


def burn_time_from_delta_v_seconds(
    delta_v_km_s: float,
    mass_kg: float = SATELLITE_MASS_KG,
    force_newtons: float = THRUST_N,
) -> float:
    delta_v_m_s = max(float(delta_v_km_s), 0.0) * 1000.0
    return float((delta_v_m_s * float(mass_kg)) / max(float(force_newtons), 1e-9))


def solve_burn_from_delta_v_vector(
    delta_v_vector_km_s: np.ndarray,
    mass_kg: float = SATELLITE_MASS_KG,
    force_newtons: float = THRUST_N,
) -> dict[str, Any]:
    delta_v = np.asarray(delta_v_vector_km_s, dtype=float)
    delta_v_mag = float(np.linalg.norm(delta_v))
    burn_time_s = burn_time_from_delta_v_seconds(
        delta_v_km_s=delta_v_mag,
        mass_kg=mass_kg,
        force_newtons=force_newtons,
    )
    return {
        "delta_v_vector": delta_v,
        "delta_v_magnitude_km_s": delta_v_mag,
        "burn_time_seconds": burn_time_s,
        "delta_v_mps": float(delta_v_mag * 1000.0),
    }
