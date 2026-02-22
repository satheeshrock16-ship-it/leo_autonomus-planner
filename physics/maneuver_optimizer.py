"""Burn timing optimization for conjunction avoidance."""
from __future__ import annotations

from typing import Any

import numpy as np

from physics.burn_physics import propellant_used_kg
from physics.constants import G0, ISP_SECONDS, SATELLITE_INITIAL_MASS_KG, SAFE_DISTANCE_KM


def analytical_required_delta_v_km_s(
    miss_distance_km: float,
    separation_constraint_km: float,
    lead_time_s: float,
    max_delta_v_km_s: float = 0.02,
) -> float:
    required_displacement_km = max(float(separation_constraint_km) - float(miss_distance_km), 0.0)
    if required_displacement_km <= 0.0:
        return 0.0
    if lead_time_s <= 1e-6:
        return float(max_delta_v_km_s)
    dv = required_displacement_km / float(lead_time_s)
    return float(min(max(dv, 0.0), max_delta_v_km_s))


def _predict_post_burn_pc(
    base_pc: float,
    miss_before_km: float,
    miss_after_km: float,
    decay_scale_km: float,
) -> float:
    pc0 = float(max(base_pc, 0.0))
    if pc0 <= 0.0:
        return 0.0
    growth_km = max(float(miss_after_km) - float(miss_before_km), 0.0)
    scale = max(float(decay_scale_km), 1e-6)
    return float(pc0 * np.exp(-0.5 * (growth_km / scale) ** 2))


def optimize_burn_timing(
    tca_time_s: float,
    orbit_period_s: float,
    miss_distance_km: float,
    separation_constraint_km: float,
    lead_orbits_min: float = 0.1,
    lead_orbits_max: float = 2.0,
    sweep_points: int = 24,
    max_delta_v_km_s: float = 0.02,
    collision_probability: float = 1.0,
    collision_probability_threshold: float = 0.1,
    lambda_fuel_penalty: float = 0.0,
    current_mass_kg: float = SATELLITE_INITIAL_MASS_KG,
    isp_seconds: float = ISP_SECONDS,
    g0_m_s2: float = G0,
    safe_distance_km: float = SAFE_DISTANCE_KM,
) -> dict[str, Any]:
    leads = np.linspace(float(lead_orbits_min), float(lead_orbits_max), max(int(sweep_points), 4), dtype=float)
    candidates: list[dict[str, float]] = []
    target_separation_km = max(float(separation_constraint_km), float(safe_distance_km))

    for lead_orbits in leads:
        lead_s = float(lead_orbits * orbit_period_s)
        burn_time_s = float(tca_time_s - lead_s)
        if burn_time_s < 0.0:
            continue
        dv = analytical_required_delta_v_km_s(
            miss_distance_km=miss_distance_km,
            separation_constraint_km=target_separation_km,
            lead_time_s=lead_s,
            max_delta_v_km_s=max_delta_v_km_s,
        )
        miss_after = float(miss_distance_km + (dv * lead_s))
        predicted_pc = _predict_post_burn_pc(
            base_pc=collision_probability,
            miss_before_km=miss_distance_km,
            miss_after_km=miss_after,
            decay_scale_km=max(target_separation_km, 1e-3),
        )
        sep_ok = bool(miss_after >= target_separation_km)
        pc_ok = bool(predicted_pc < float(collision_probability_threshold))
        feasible = bool(sep_ok and pc_ok)

        fuel_used = float(
            propellant_used_kg(
                delta_v_km_s=dv,
                initial_mass_kg=current_mass_kg,
                isp_seconds=isp_seconds,
                g0_m_s2=g0_m_s2,
            )
        )
        fuel_penalty = float(fuel_used / max(float(current_mass_kg), 1e-9))
        objective = float(dv + float(lambda_fuel_penalty) * fuel_penalty)
        candidates.append(
            {
                "lead_orbits": float(lead_orbits),
                "lead_time_s": lead_s,
                "burn_time_s": burn_time_s,
                "required_delta_v_km_s": dv,
                "simulated_miss_distance_km": miss_after,
                "predicted_collision_probability": predicted_pc,
                "fuel_penalty": fuel_penalty,
                "objective": objective,
                "feasible": feasible,
            }
        )

    if not candidates:
        fallback_dv = float(max_delta_v_km_s)
        fallback_miss = float(miss_distance_km)
        return {
            "burn_time_s": float(max(tca_time_s - 0.1 * orbit_period_s, 0.0)),
            "required_delta_v_km_s": fallback_dv,
            "simulated_miss_distance_km": fallback_miss,
            "predicted_collision_probability": float(collision_probability),
            "fuel_penalty": float(
                propellant_used_kg(
                    delta_v_km_s=fallback_dv,
                    initial_mass_kg=current_mass_kg,
                    isp_seconds=isp_seconds,
                    g0_m_s2=g0_m_s2,
                )
                / max(float(current_mass_kg), 1e-9)
            ),
            "objective": fallback_dv,
            "feasible": False,
            "candidates": [],
        }

    feasible_candidates = [c for c in candidates if c["feasible"]]
    pool = feasible_candidates if feasible_candidates else candidates
    best = min(pool, key=lambda row: row["objective"])
    return {
        "burn_time_s": best["burn_time_s"],
        "required_delta_v_km_s": best["required_delta_v_km_s"],
        "simulated_miss_distance_km": best["simulated_miss_distance_km"],
        "predicted_collision_probability": best["predicted_collision_probability"],
        "fuel_penalty": best["fuel_penalty"],
        "objective": best["objective"],
        "feasible": bool(best["feasible"]),
        "candidates": candidates,
    }


def plan_collision_avoidance_delta_v(
    min_distance_km: float,
    relative_velocity_km_s: float,
    desired_separation_km: float = 120.0,
    lead_time_s: float = 600.0,
    max_delta_v_km_s: float = 0.05,
) -> float:
    _ = max(float(relative_velocity_km_s), 0.0)  # explicit API parameter for future coupling.
    gap_km = max(float(desired_separation_km) - float(min_distance_km), 0.0)
    if gap_km <= 0.0:
        return 0.0
    dv = gap_km / max(float(lead_time_s), 1e-6)
    return float(min(max(dv, 0.0), float(max_delta_v_km_s)))


def tangential_delta_v_from_miss_distance(
    predicted_miss_km: float,
    sat_position_norm_km: float,
    mu_km3_s2: float = 398600.4418,
    safe_distance_km: float = SAFE_DISTANCE_KM,
) -> float:
    r_km = float(max(sat_position_norm_km, 1e-9))
    v_circ = float(np.sqrt(float(mu_km3_s2) / r_km))
    delta_r = float(safe_distance_km) - float(predicted_miss_km)
    if delta_r <= 0.0:
        return 0.0
    return float((v_circ / r_km) * delta_r)
