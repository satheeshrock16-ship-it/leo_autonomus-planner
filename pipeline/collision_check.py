"""Collision screening and covariance-aware probabilistic risk analysis."""
from __future__ import annotations

from typing import Any

import numpy as np

from config import COVARIANCE_CONFIG
from physics.collision_probability import collision_probability
from physics.collision_probability_3d import (
    build_full_covariance_rtn_m2,
    collision_probability_3d_alfano,
    relative_state_rtn,
)
from physics.constants import MU
from validation.monte_carlo_pc import monte_carlo_validate_pc


def _covariance_block(name: str) -> dict[str, Any]:
    block = COVARIANCE_CONFIG.get(name, {})
    if isinstance(block, dict):
        return block
    return {}


def _position_covariance_from_block(block: dict[str, Any], fallback: dict[str, Any]) -> np.ndarray:
    src = fallback.copy()
    src.update(block)
    return build_full_covariance_rtn_m2(
        sigma_r_m=float(src.get("sigma_r_m", 120.0)),
        sigma_t_m=float(src.get("sigma_t_m", 180.0)),
        sigma_n_m=float(src.get("sigma_n_m", 150.0)),
        rho_rt=float(src.get("rho_rt", 0.0)),
        rho_rn=float(src.get("rho_rn", 0.0)),
        rho_tn=float(src.get("rho_tn", 0.0)),
    )


def _velocity_sigmas_from_block(block: dict[str, Any], fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        float(block.get("sigma_vr_m_s", fallback[0])),
        float(block.get("sigma_vt_m_s", fallback[1])),
        float(block.get("sigma_vn_m_s", fallback[2])),
    )


def run(closest_approach_km: float, **kwargs: Any) -> float:
    return float(run_detailed(closest_approach_km, **kwargs)["Pc"])


def run_detailed(closest_approach_km: float, **kwargs: Any) -> dict[str, float]:
    if (
        "sat_r_eci_km" in kwargs
        and "sat_v_eci_km_s" in kwargs
        and "rel_r_eci_km" in kwargs
    ):
        sat_r_eci_km = np.asarray(kwargs["sat_r_eci_km"], dtype=float)
        sat_v_eci_km_s = np.asarray(kwargs["sat_v_eci_km_s"], dtype=float)
        rel_r_eci_km = np.asarray(kwargs["rel_r_eci_km"], dtype=float)
        rel_v_eci_km_s = np.asarray(kwargs.get("rel_v_eci_km_s", np.zeros(3)), dtype=float)
        rtn_state = relative_state_rtn(
            sat_r_eci_km=sat_r_eci_km,
            sat_v_eci_km_s=sat_v_eci_km_s,
            rel_r_eci_km=rel_r_eci_km,
            rel_v_eci_km_s=rel_v_eci_km_s,
        )

        fallback_src = {
            "sigma_r_m": float(COVARIANCE_CONFIG.get("sigma_r_m", 120.0)),
            "sigma_t_m": float(COVARIANCE_CONFIG.get("sigma_t_m", 180.0)),
            "sigma_n_m": float(COVARIANCE_CONFIG.get("sigma_n_m", 150.0)),
            "rho_rt": float(COVARIANCE_CONFIG.get("rho_rt", 0.0)),
            "rho_rn": float(COVARIANCE_CONFIG.get("rho_rn", 0.0)),
            "rho_tn": float(COVARIANCE_CONFIG.get("rho_tn", 0.0)),
        }
        sat_block = _covariance_block("satellite")
        deb_block = _covariance_block("debris")
        sat_cov_rtn_m2 = _position_covariance_from_block(sat_block, fallback=fallback_src)
        deb_cov_rtn_m2 = _position_covariance_from_block(deb_block, fallback=fallback_src)
        sat_sigma_v = _velocity_sigmas_from_block(sat_block, fallback=(0.02, 0.03, 0.02))
        deb_sigma_v = _velocity_sigmas_from_block(deb_block, fallback=(0.03, 0.04, 0.03))
        n_rho = int(kwargs.get("integration_points_rho", COVARIANCE_CONFIG.get("integration_points_rho", 48)))
        n_theta = int(kwargs.get("integration_points_theta", COVARIANCE_CONFIG.get("integration_points_theta", 144)))
        if bool(kwargs.get("fast_mode", False)):
            n_rho = max(min(n_rho, 20), 12)
            n_theta = max(min(n_theta, 60), 24)

        mean_motion = float(kwargs.get("mean_motion_rad_s", 0.0))
        if mean_motion <= 0.0:
            sat_radius = float(np.linalg.norm(sat_r_eci_km))
            mean_motion = float(np.sqrt(MU / max(sat_radius**3, 1e-9)))

        pc_result = collision_probability_3d_alfano(
            rel_pos_rtn_km=rtn_state["position_rtn_km"],
            rel_vel_rtn_km_s=rtn_state.get("velocity_rtn_km_s", np.zeros(3)),
            hard_body_radius_m=float(COVARIANCE_CONFIG.get("hard_body_radius_m", 10.0)),
            integration_points_rho=n_rho,
            integration_points_theta=n_theta,
            sat_cov_rtn_m2=sat_cov_rtn_m2,
            debris_cov_rtn_m2=deb_cov_rtn_m2,
            sat_sigma_v_rtn_m_s=sat_sigma_v,
            debris_sigma_v_rtn_m_s=deb_sigma_v,
            covariance_dt_s=float(kwargs.get("covariance_dt_s", 0.0)),
            mean_motion_rad_s=mean_motion,
        )

        out: dict[str, float] = {
            "Pc": float(pc_result["Pc"]),
            "confidence_metric": float(pc_result["confidence_metric"]),
            "relative_cov_rr_m2": float(pc_result["relative_covariance_rtn_m2"][0, 0]),
            "relative_cov_tt_m2": float(pc_result["relative_covariance_rtn_m2"][1, 1]),
            "relative_cov_nn_m2": float(pc_result["relative_covariance_rtn_m2"][2, 2]),
        }

        mc_enabled = bool(kwargs.get("run_monte_carlo", COVARIANCE_CONFIG.get("monte_carlo_enabled", True)))
        if mc_enabled:
            mc = monte_carlo_validate_pc(
                analytical_pc=float(pc_result["Pc"]),
                mean_state_rtn=np.hstack(
                    [
                        np.asarray(rtn_state["position_rtn_km"], dtype=float),
                        np.asarray(rtn_state.get("velocity_rtn_km_s", np.zeros(3)), dtype=float),
                    ]
                ),
                covariance_rtn=np.asarray(pc_result["relative_state_covariance_rtn_km"], dtype=float),
                hard_body_radius_m=float(COVARIANCE_CONFIG.get("hard_body_radius_m", 10.0)),
                samples=int(kwargs.get("monte_carlo_samples", COVARIANCE_CONFIG.get("monte_carlo_samples", 5000))),
                mean_motion_rad_s=0.0,
                dt_s=0.0,
                seed=int(COVARIANCE_CONFIG.get("monte_carlo_seed", 42)),
            )
            out["Pc_mc"] = float(mc["monte_carlo_pc"])
            out["pc_abs_error"] = float(mc["absolute_error"])
            out["pc_relative_error"] = float(mc["relative_error"])
        else:
            out["Pc_mc"] = float("nan")
            out["pc_abs_error"] = float("nan")
            out["pc_relative_error"] = float("nan")
        return out

    return {
        "Pc": float(collision_probability(closest_approach_km)),
        "confidence_metric": 0.5,
        "Pc_mc": float("nan"),
        "pc_abs_error": float("nan"),
        "pc_relative_error": float("nan"),
    }
