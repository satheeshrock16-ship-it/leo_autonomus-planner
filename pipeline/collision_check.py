"""Collision screening and probabilistic risk analysis."""
from __future__ import annotations

from typing import Any

import numpy as np

from config import COVARIANCE_CONFIG
from physics.collision_probability import collision_probability
from physics.collision_probability_3d import collision_probability_3d_alfano, relative_state_rtn


def _sigma_tuple() -> tuple[float, float, float]:
    return (
        float(COVARIANCE_CONFIG.get("sigma_r_m", 120.0)),
        float(COVARIANCE_CONFIG.get("sigma_t_m", 180.0)),
        float(COVARIANCE_CONFIG.get("sigma_n_m", 150.0)),
    )


def run(closest_approach_km: float, **kwargs: Any) -> float:
    return float(run_detailed(closest_approach_km, **kwargs)["Pc"])


def run_detailed(closest_approach_km: float, **kwargs: Any) -> dict[str, float]:
    if (
        "sat_r_eci_km" in kwargs
        and "sat_v_eci_km_s" in kwargs
        and "rel_r_eci_km" in kwargs
    ):
        rtn_state = relative_state_rtn(
            sat_r_eci_km=np.asarray(kwargs["sat_r_eci_km"], dtype=float),
            sat_v_eci_km_s=np.asarray(kwargs["sat_v_eci_km_s"], dtype=float),
            rel_r_eci_km=np.asarray(kwargs["rel_r_eci_km"], dtype=float),
            rel_v_eci_km_s=np.asarray(kwargs.get("rel_v_eci_km_s", np.zeros(3)), dtype=float),
        )
        result = collision_probability_3d_alfano(
            rel_pos_rtn_km=rtn_state["position_rtn_km"],
            sigma_rtn_m=_sigma_tuple(),
            hard_body_radius_m=float(COVARIANCE_CONFIG.get("hard_body_radius_m", 10.0)),
            integration_points_z=int(COVARIANCE_CONFIG.get("integration_points_z", 81)),
            integration_points_rho=int(COVARIANCE_CONFIG.get("integration_points_rho", 36)),
            integration_points_theta=int(COVARIANCE_CONFIG.get("integration_points_theta", 72)),
        )
        return {"Pc": float(result["Pc"]), "confidence_metric": float(result["confidence_metric"])}
    return {"Pc": float(collision_probability(closest_approach_km)), "confidence_metric": 0.5}
