"""Burn timing optimization for conjunction avoidance."""
from __future__ import annotations

from typing import Any

import numpy as np


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


def optimize_burn_timing(
    tca_time_s: float,
    orbit_period_s: float,
    miss_distance_km: float,
    separation_constraint_km: float,
    lead_orbits_min: float = 0.1,
    lead_orbits_max: float = 2.0,
    sweep_points: int = 24,
    max_delta_v_km_s: float = 0.02,
) -> dict[str, Any]:
    leads = np.linspace(float(lead_orbits_min), float(lead_orbits_max), max(int(sweep_points), 4), dtype=float)
    candidates: list[dict[str, float]] = []
    for lead_orbits in leads:
        lead_s = float(lead_orbits * orbit_period_s)
        burn_time_s = float(tca_time_s - lead_s)
        if burn_time_s < 0.0:
            continue
        dv = analytical_required_delta_v_km_s(
            miss_distance_km=miss_distance_km,
            separation_constraint_km=separation_constraint_km,
            lead_time_s=lead_s,
            max_delta_v_km_s=max_delta_v_km_s,
        )
        miss_after = float(miss_distance_km + (dv * lead_s))
        feasible = bool(miss_after >= separation_constraint_km)
        candidates.append(
            {
                "lead_orbits": float(lead_orbits),
                "lead_time_s": lead_s,
                "burn_time_s": burn_time_s,
                "required_delta_v_km_s": dv,
                "simulated_miss_distance_km": miss_after,
                "feasible": feasible,
            }
        )
    if not candidates:
        return {
            "burn_time_s": float(max(tca_time_s - 0.1 * orbit_period_s, 0.0)),
            "required_delta_v_km_s": float(max_delta_v_km_s),
            "simulated_miss_distance_km": float(miss_distance_km),
            "feasible": False,
            "candidates": [],
        }
    feasible = [c for c in candidates if c["feasible"]]
    pool = feasible if feasible else candidates
    best = min(pool, key=lambda row: row["required_delta_v_km_s"])
    return {
        "burn_time_s": best["burn_time_s"],
        "required_delta_v_km_s": best["required_delta_v_km_s"],
        "simulated_miss_distance_km": best["simulated_miss_distance_km"],
        "feasible": best["feasible"],
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
    safe_distance_km: float = 5.0,
) -> float:
    r_km = float(max(sat_position_norm_km, 1e-9))
    v_circ = float(np.sqrt(float(mu_km3_s2) / r_km))
    delta_r = float(safe_distance_km) - float(predicted_miss_km)
    if delta_r <= 0.0:
        return 0.0
    return float((v_circ / r_km) * delta_r)
