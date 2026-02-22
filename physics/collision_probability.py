"""Scalar collision probability fallback based on Gaussian hard-body integration."""
from __future__ import annotations

from physics.collision_probability_3d import collision_probability_3d_alfano


def collision_probability(closest_approach_km: float, sigma_m: float = 100.0, hard_body_radius_m: float = 10.0) -> float:
    """Fallback scalar Pc for callers without full state vectors."""
    res = collision_probability_3d_alfano(
        rel_pos_rtn_km=[float(closest_approach_km), 0.0, 0.0],
        sigma_rtn_m=(float(sigma_m), float(sigma_m), float(sigma_m)),
        hard_body_radius_m=float(hard_body_radius_m),
        integration_points_rho=40,
        integration_points_theta=96,
    )
    return float(res["Pc"])
