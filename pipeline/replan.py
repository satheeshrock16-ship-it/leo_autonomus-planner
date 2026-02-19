"""Maneuver re-planning stage."""
from __future__ import annotations

import numpy as np

from physics.maneuver import (
    estimate_tangential_delta_v_for_altitude_offset,
    build_thrust_vector,
    burn_duration_ms,
)
from physics.orbital_transfer import correction_burn_for_return


def run(current_radius_km: float, nominal_radius_km: float, burn_type: str = "tangential"):
    delta_r = 1.0
    delta_v = estimate_tangential_delta_v_for_altitude_offset(current_radius_km, delta_r)
    thrust = build_thrust_vector(abs(delta_v), burn_type)
    return_delta_v = correction_burn_for_return(current_radius_km + delta_r, nominal_radius_km)
    return {
        "avoidance_delta_v_km_s": float(delta_v),
        "thrust_vector": thrust,
        "duration_ms": burn_duration_ms(abs(delta_v)),
        "return_delta_v_km_s": float(return_delta_v),
    }
