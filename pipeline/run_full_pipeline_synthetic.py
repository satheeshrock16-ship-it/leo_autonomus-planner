"""Synthetic end-to-end autonomous LEO collision avoidance pipeline."""
from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from ai.delta_v_regressor import DeltaVRegressor
from config import (
    COLLISION_PROBABILITY_THRESHOLD,
    EARTH_MU_KM3_S2,
    EARTH_RADIUS_KM,
    MANEUVER_CONFIG,
    MODEL_DIR,
    PROCESSED_DATA_DIR,
    PROPAGATION_CONFIG,
    TCA_REFINEMENT_CONFIG,
)
from physics.maneuver import burn_duration_ms
from physics.maneuver_optimizer import analytical_required_delta_v_km_s, optimize_burn_timing
from physics.tca_refinement import refine_tca_quadratic
from pipeline import collision_check
from visualization.interactive_plot import plot_interactive_3d
from visualization.plot_synthetic import plot_synthetic_scenario


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SCENARIOS = {
    "SAFE_DEMO": 200,
    "STRONG_MANEUVER": 50,
    "CRITICAL_EMERGENCY": 10,
}

SCENARIO_PLOT_TAG = {
    "SAFE_DEMO": "SAFE",
    "STRONG_MANEUVER": "STRONG",
    "CRITICAL_EMERGENCY": "CRITICAL",
}

MANEUVER_RAISE_KM = {
    "SAFE_DEMO": 2.0,
    "STRONG_MANEUVER": 5.0,
    "CRITICAL_EMERGENCY": 15.0,
}

SATELLITE_ALTITUDE_KM = 700.0
SATELLITE_RADIUS_KM = float(EARTH_RADIUS_KM + SATELLITE_ALTITUDE_KM)
SATELLITE_INCLINATION_DEG = 98.0
SATELLITE_RAAN_DEG = 30.0
DEBRIS_INCLINATION_DEG = 102.0
DEBRIS_RAAN_DEG = 45.0
TIMESTEP_SECONDS = int(PROPAGATION_CONFIG.get("timestep_seconds", 5))
FIXED_EPOCH_UTC = datetime.fromisoformat(str(PROPAGATION_CONFIG.get("deterministic_epoch_utc", "2026-01-01T00:00:00+00:00")))


def _time_vector() -> np.ndarray:
    mean_motion = float(np.sqrt(EARTH_MU_KM3_S2 / SATELLITE_RADIUS_KM**3))
    orbit_period_s = float((2.0 * np.pi) / mean_motion)
    return np.arange(0.0, orbit_period_s + TIMESTEP_SECONDS, TIMESTEP_SECONDS, dtype=float)


def _rot_x(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _rot_z(angle_rad: float) -> np.ndarray:
    c = float(np.cos(angle_rad))
    s = float(np.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _plane_rotation(inclination_deg: float, raan_deg: float) -> np.ndarray:
    return _rot_z(np.deg2rad(raan_deg)) @ _rot_x(np.deg2rad(inclination_deg))


def _propagate_circular_elements(
    radius_km: float,
    rotation_eci_from_perifocal: np.ndarray,
    u0_rad: float,
    t_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean_motion = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km**3))
    speed_kmps = float(np.sqrt(EARTH_MU_KM3_S2 / radius_km))
    u = u0_rad + (mean_motion * t_s)
    pos = np.zeros((len(t_s), 3), dtype=float)
    vel = np.zeros((len(t_s), 3), dtype=float)
    for idx, angle in enumerate(u):
        pos_pf = np.array([radius_km * np.cos(angle), radius_km * np.sin(angle), 0.0], dtype=float)
        vel_pf = np.array([-speed_kmps * np.sin(angle), speed_kmps * np.cos(angle), 0.0], dtype=float)
        pos[idx] = rotation_eci_from_perifocal @ pos_pf
        vel[idx] = rotation_eci_from_perifocal @ vel_pf
    return pos, vel


def _two_body_accel(r_km: np.ndarray) -> np.ndarray:
    r_norm = float(np.linalg.norm(r_km))
    return -(EARTH_MU_KM3_S2 / (r_norm**3)) * r_km


def _propagate_two_body_arc(r0_km: np.ndarray, v0_kmps: np.ndarray, t_rel_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(t_rel_s) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    pos = np.zeros((len(t_rel_s), 3), dtype=float)
    vel = np.zeros((len(t_rel_s), 3), dtype=float)
    pos[0] = r0_km
    vel[0] = v0_kmps
    for idx in range(1, len(t_rel_s)):
        dt = float(t_rel_s[idx] - t_rel_s[idx - 1])
        r = pos[idx - 1]
        v = vel[idx - 1]
        k1_r = v
        k1_v = _two_body_accel(r)
        k2_r = v + 0.5 * dt * k1_v
        k2_v = _two_body_accel(r + 0.5 * dt * k1_r)
        k3_r = v + 0.5 * dt * k2_v
        k3_v = _two_body_accel(r + 0.5 * dt * k2_r)
        k4_r = v + dt * k3_v
        k4_v = _two_body_accel(r + dt * k3_r)
        pos[idx] = r + (dt / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        vel[idx] = v + (dt / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
    return pos, vel


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _make_synthetic_trajectories(scenario_name: str, offset_m: float, t_s: np.ndarray) -> dict[str, np.ndarray]:
    scenario_sign = {"SAFE_DEMO": 1.0, "STRONG_MANEUVER": -1.0, "CRITICAL_EMERGENCY": 1.0}.get(scenario_name, 1.0)
    target_miss_km = float(offset_m) / 1000.0
    mean_motion = float(np.sqrt(EARTH_MU_KM3_S2 / SATELLITE_RADIUS_KM**3))
    tca_idx_target = len(t_s) // 2
    tca_time_s = float(t_s[tca_idx_target])
    sat_rot = _plane_rotation(SATELLITE_INCLINATION_DEG, SATELLITE_RAAN_DEG)
    debris_rot = _plane_rotation(DEBRIS_INCLINATION_DEG, DEBRIS_RAAN_DEG)
    sat_normal = sat_rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    debris_normal = debris_rot @ np.array([0.0, 0.0, 1.0], dtype=float)
    node_dir = np.cross(sat_normal, debris_normal)
    node_norm = float(np.linalg.norm(node_dir))
    node_dir = np.array([1.0, 0.0, 0.0], dtype=float) if node_norm < 1e-12 else node_dir / node_norm
    sat_p = sat_rot @ np.array([1.0, 0.0, 0.0], dtype=float)
    sat_q = sat_rot @ np.array([0.0, 1.0, 0.0], dtype=float)
    debris_p = debris_rot @ np.array([1.0, 0.0, 0.0], dtype=float)
    debris_q = debris_rot @ np.array([0.0, 1.0, 0.0], dtype=float)
    u_tca_sat = float(np.arctan2(np.dot(node_dir, sat_q), np.dot(node_dir, sat_p)))
    u_tca_debris = float(np.arctan2(np.dot(node_dir, debris_q), np.dot(node_dir, debris_p)))
    phase_offset_rad = scenario_sign * (target_miss_km / SATELLITE_RADIUS_KM)
    u0_sat = u_tca_sat - (mean_motion * tca_time_s)
    u0_debris = (u_tca_debris + phase_offset_rad) - (mean_motion * tca_time_s)
    sat_pos, sat_vel = _propagate_circular_elements(SATELLITE_RADIUS_KM, sat_rot, u0_sat, t_s)
    debris_pos, debris_vel = _propagate_circular_elements(SATELLITE_RADIUS_KM, debris_rot, u0_debris, t_s)
    return {"sat_pos": sat_pos, "sat_vel": sat_vel, "debris_pos": debris_pos, "debris_vel": debris_vel}


def _closest_approach(
    sat_pos: np.ndarray,
    debris_pos: np.ndarray,
    sat_vel: np.ndarray,
    debris_vel: np.ndarray,
    t_s: np.ndarray,
) -> dict[str, Any]:
    rel = debris_pos - sat_pos
    rel_vel = debris_vel - sat_vel
    distances_km = np.linalg.norm(rel, axis=1)
    idx = int(np.argmin(distances_km))
    refined = refine_tca_quadratic(
        time_s=t_s,
        distance_km=distances_km,
        min_index=idx,
        fit_window_samples=int(TCA_REFINEMENT_CONFIG.get("fit_window_samples", 2)),
        epoch_utc=FIXED_EPOCH_UTC,
    )
    return {
        "index": idx,
        "time_s": float(t_s[idx]),
        "distance_km": float(distances_km[idx]),
        "rel_km": rel[idx],
        "rel_vel_km_s": rel_vel[idx],
        "sat_tca_km": sat_pos[idx],
        "sat_tca_vel_km_s": sat_vel[idx],
        "debris_tca_km": debris_pos[idx],
        "debris_tca_vel_km_s": debris_vel[idx],
        "tca_point_km": (sat_pos[idx] + debris_pos[idx]) * 0.5,
        "refined_tca_time": refined["refined_tca_time"],
        "refined_min_distance": float(refined["refined_min_distance"]),
    }


def _build_maneuver_trajectories(
    scenario_name: str,
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    t_s: np.ndarray,
    burn_idx: int,
    maneuver_triggered: bool,
    maneuver_dv_km_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if len(sat_pos) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float), np.zeros(3, dtype=float), 0.0
    if not maneuver_triggered:
        return sat_pos.copy(), sat_pos.copy(), np.zeros(3, dtype=float), 0.0

    burn_idx = int(max(0, min(burn_idx, len(sat_pos) - 1)))
    r_burn = sat_pos[burn_idx]
    v_burn = sat_vel[burn_idx]
    r_hat = r_burn / np.linalg.norm(r_burn)
    t_hat = v_burn / np.linalg.norm(v_burn)
    n_hat = np.cross(r_hat, t_hat)
    n_hat = n_hat / np.linalg.norm(n_hat)
    dv_vec = maneuver_dv_km_s * ((0.2 * r_hat) + (0.9 * t_hat) + (0.1 * n_hat))
    v_new = v_burn + dv_vec
    h_new = np.cross(r_burn, v_new)
    inclination_new_rad = float(np.arccos(np.clip(h_new[2] / np.linalg.norm(h_new), -1.0, 1.0)))
    inclination_new_deg = float(np.degrees(inclination_new_rad))

    t_after = t_s[burn_idx:]
    t_rel_after = t_after - t_after[0]
    if len(t_rel_after) == 0:
        return sat_pos.copy(), sat_pos.copy(), dv_vec, inclination_new_deg

    raise_km = float(MANEUVER_RAISE_KM.get(scenario_name, 2.0))
    total_after_s = float(t_rel_after[-1]) if len(t_rel_after) > 1 else float(TIMESTEP_SECONDS)
    avoid_duration_s = max(0.25 * total_after_s, 900.0)
    return_start_s = min(avoid_duration_s, total_after_s)
    avoid_mask = t_rel_after <= return_start_s
    return_mask = t_rel_after >= return_start_s
    t_avoid = t_rel_after[avoid_mask]
    avoid_pos, avoid_vel = _propagate_two_body_arc(r_burn, v_new, t_avoid)
    if np.linalg.norm(avoid_pos[0]) > 0:
        radial_boost = raise_km * (1.0 - np.exp(-t_avoid / max(120.0, 0.15 * avoid_duration_s)))
        avoid_pos = avoid_pos + radial_boost[:, None] * (avoid_pos / np.linalg.norm(avoid_pos, axis=1, keepdims=True))
    t_return = t_rel_after[return_mask]
    if len(t_return) == 0:
        return avoid_pos, avoid_pos.copy(), dv_vec, inclination_new_deg
    return_rel = t_return - t_return[0]
    start_pos = avoid_pos[-1]
    start_vel = avoid_vel[-1]
    free_return_pos, _ = _propagate_two_body_arc(start_pos, start_vel, return_rel)
    nominal_return_pos = sat_pos[burn_idx + np.where(return_mask)[0]]
    blend = _smoothstep(return_rel / max(return_rel[-1], 1e-6))
    return_pos = (1.0 - blend[:, None]) * free_return_pos + blend[:, None] * nominal_return_pos
    return avoid_pos, return_pos, dv_vec, inclination_new_deg


def _write_scenario_outputs(
    scenario_dir: Path,
    scenario_name: str,
    t_s: np.ndarray,
    sat_pos: np.ndarray,
    sat_vel: np.ndarray,
    debris_pos: np.ndarray,
    debris_vel: np.ndarray,
    closest: dict[str, Any],
    collision_probability: float,
    collision_confidence: float,
    maneuver_triggered: bool,
    plan: dict[str, Any] | None,
    plot_path: str,
) -> None:
    scenario_dir.mkdir(parents=True, exist_ok=True)
    propagated_csv = scenario_dir / "propagated_states.csv"
    conjunction_csv = scenario_dir / "conjunction_results.csv"
    decision_json = scenario_dir / "decision.json"
    with propagated_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "object_role", "object_id", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for idx, ts in enumerate(t_s):
            writer.writerow([ts, "satellite", "PROTECTED", sat_pos[idx, 0], sat_pos[idx, 1], sat_pos[idx, 2], sat_vel[idx, 0], sat_vel[idx, 1], sat_vel[idx, 2]])
            writer.writerow([ts, "debris", "SYNTHETIC_DEBRIS", debris_pos[idx, 0], debris_pos[idx, 1], debris_pos[idx, 2], debris_vel[idx, 0], debris_vel[idx, 1], debris_vel[idx, 2]])
    with conjunction_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "scenario",
                "tca_time_s",
                "miss_distance_km",
                "refined_tca_time_utc",
                "refined_min_distance_km",
                "rel_x_km",
                "rel_y_km",
                "rel_z_km",
                "collision_probability",
                "confidence_metric",
            ]
        )
        writer.writerow(
            [
                scenario_name,
                closest["time_s"],
                closest["distance_km"],
                str(closest["refined_tca_time"]),
                closest["refined_min_distance"],
                float(closest["rel_km"][0]),
                float(closest["rel_km"][1]),
                float(closest["rel_km"][2]),
                collision_probability,
                collision_confidence,
            ]
        )
    payload = {
        "scenario": scenario_name,
        "offset_m": SCENARIOS[scenario_name],
        "deterministic_epoch_utc": FIXED_EPOCH_UTC.isoformat(),
        "closest_approach_km": closest["distance_km"],
        "tca_time_s": closest["time_s"],
        "refined_tca_time": str(closest["refined_tca_time"]),
        "refined_min_distance": closest["refined_min_distance"],
        "collision_probability": collision_probability,
        "collision_confidence_metric": collision_confidence,
        "maneuver_triggered": maneuver_triggered,
        "thrust_command": None if plan is None else {
            "thrust_vector": [float(v) for v in plan["thrust_vector"]],
            "duration_ms": int(plan["duration_ms"]),
            "avoidance_delta_v_km_s": float(plan["avoidance_delta_v_km_s"]),
            "return_delta_v_km_s": float(plan["return_delta_v_km_s"]),
            "burn_time_s": float(plan["burn_time_s"]),
            "burn_lead_time_s": float(plan["burn_lead_time_s"]),
            "ml_predicted_delta_v_km_s": float(plan["ml_predicted_delta_v_km_s"]),
            "analytical_required_delta_v_km_s": float(plan["analytical_required_delta_v_km_s"]),
        },
        "return_to_orbit_status": "completed" if maneuver_triggered else "not_required",
        "artifacts": {
            "propagated_states_csv": str(propagated_csv),
            "conjunction_results_csv": str(conjunction_csv),
            "decision_json": str(decision_json),
            "figure_png": plot_path,
        },
    }
    with decision_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_delta_v_model() -> DeltaVRegressor | None:
    model_path = MODEL_DIR / "delta_v_model.txt"
    if not model_path.exists():
        LOGGER.warning("Delta-v regressor model not found at %s; analytical fallback will be used.", model_path)
        return None
    try:
        return DeltaVRegressor.load(model_path)
    except Exception as exc:
        LOGGER.warning("Failed to load delta-v regressor at %s: %s", model_path, exc)
        return None


def run_synthetic_cycle() -> dict[str, Any]:
    t_s = _time_vector()
    mean_motion = float(np.sqrt(EARTH_MU_KM3_S2 / SATELLITE_RADIUS_KM**3))
    orbit_period_s = float((2.0 * np.pi) / mean_motion)
    synthetic_root = PROCESSED_DATA_DIR / "synthetic"
    synthetic_root.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, Any] = {}
    dv_regressor = _load_delta_v_model()

    for scenario_name, offset_m in SCENARIOS.items():
        trajectories = _make_synthetic_trajectories(scenario_name=scenario_name, offset_m=offset_m, t_s=t_s)
        satellite_original_states = trajectories["sat_pos"]
        debris_states = trajectories["debris_pos"]
        closest = _closest_approach(
            satellite_original_states,
            debris_states,
            trajectories["sat_vel"],
            trajectories["debris_vel"],
            t_s,
        )

        pc_detail = collision_check.run_detailed(
            closest["distance_km"],
            sat_r_eci_km=closest["sat_tca_km"],
            sat_v_eci_km_s=closest["sat_tca_vel_km_s"],
            rel_r_eci_km=closest["rel_km"],
            rel_v_eci_km_s=closest["rel_vel_km_s"],
        )
        collision_probability = float(pc_detail["Pc"])
        maneuver_triggered = bool(collision_probability > COLLISION_PROBABILITY_THRESHOLD)
        rel_speed = float(np.linalg.norm(closest["rel_vel_km_s"]))
        time_to_tca = max(float(closest["time_s"]), 1.0)
        fuel_remaining = float(MANEUVER_CONFIG.get("fuel_remaining_default", 1.0))
        analytical_dv = analytical_required_delta_v_km_s(
            miss_distance_km=float(closest["refined_min_distance"]),
            separation_constraint_km=float(MANEUVER_CONFIG.get("separation_constraint_km", 2.0)),
            lead_time_s=time_to_tca,
            max_delta_v_km_s=float(MANEUVER_CONFIG.get("max_delta_v_km_s", 0.02)),
        )
        ml_pred_dv = analytical_dv
        if dv_regressor is not None:
            features = np.array(
                [
                    collision_probability,
                    float(closest["refined_min_distance"]),
                    rel_speed,
                    time_to_tca,
                    SATELLITE_ALTITUDE_KM,
                    fuel_remaining,
                ],
                dtype=float,
            )
            ml_pred_dv = float(max(dv_regressor.predict(features)[0], 0.0))

        timing = optimize_burn_timing(
            tca_time_s=float(closest["time_s"]),
            orbit_period_s=orbit_period_s,
            miss_distance_km=float(closest["refined_min_distance"]),
            separation_constraint_km=float(MANEUVER_CONFIG.get("separation_constraint_km", 2.0)),
            lead_orbits_min=float(MANEUVER_CONFIG.get("burn_lead_orbits_min", 0.1)),
            lead_orbits_max=float(MANEUVER_CONFIG.get("burn_lead_orbits_max", 2.0)),
            sweep_points=int(MANEUVER_CONFIG.get("burn_sweep_points", 24)),
            max_delta_v_km_s=float(MANEUVER_CONFIG.get("max_delta_v_km_s", 0.02)),
        )
        selected_dv = float(max(timing["required_delta_v_km_s"], ml_pred_dv))
        burn_idx = int(np.argmin(np.abs(t_s - float(timing["burn_time_s"]))))

        avoidance_states, return_states, maneuver_dv_vec, maneuver_inclination_deg = _build_maneuver_trajectories(
            scenario_name,
            satellite_original_states,
            trajectories["sat_vel"],
            t_s,
            burn_idx,
            maneuver_triggered,
            selected_dv,
        )

        plan = None
        if maneuver_triggered:
            plan = {
                "thrust_vector": [float(v) for v in maneuver_dv_vec],
                "duration_ms": burn_duration_ms(float(np.linalg.norm(maneuver_dv_vec))),
                "avoidance_delta_v_km_s": float(np.linalg.norm(maneuver_dv_vec)),
                "return_delta_v_km_s": float(np.linalg.norm(maneuver_dv_vec) * 0.6),
                "inclination_new_deg": maneuver_inclination_deg,
                "burn_time_s": float(timing["burn_time_s"]),
                "burn_lead_time_s": float(closest["time_s"] - timing["burn_time_s"]),
                "ml_predicted_delta_v_km_s": float(ml_pred_dv),
                "analytical_required_delta_v_km_s": float(timing["required_delta_v_km_s"]),
            }

        if len(satellite_original_states) == 0 or len(debris_states) == 0 or len(avoidance_states) == 0 or len(return_states) == 0:
            raise RuntimeError(f"Empty synthetic plotting arrays detected for scenario {scenario_name}")

        plot_path = plot_synthetic_scenario(
            scenario_name=scenario_name,
            scenario_plot_tag=SCENARIO_PLOT_TAG[scenario_name],
            satellite_original_states=satellite_original_states,
            debris_states=debris_states,
            tca_point_km=closest["tca_point_km"],
            avoidance_states=avoidance_states,
            return_states=return_states,
        )
        interactive_plot_path = ""
        try:
            interactive_plot_path = plot_interactive_3d(
                earth_radius_km=6378.0,
                satellite_xyz=satellite_original_states,
                debris_xyz=debris_states,
                tca_point=closest["tca_point_km"],
                avoidance_xyz=avoidance_states,
                return_xyz=return_states,
                title=f"Synthetic LEO Collision Scenario: {scenario_name} (Interactive)",
                output_path=PROCESSED_DATA_DIR / f"synthetic_{scenario_name}_interactive.html",
            )
        except Exception as exc:
            LOGGER.warning("interactive visualization skipped for %s: %s", scenario_name, exc)

        scenario_dir = synthetic_root / scenario_name
        _write_scenario_outputs(
            scenario_dir=scenario_dir,
            scenario_name=scenario_name,
            t_s=t_s,
            sat_pos=satellite_original_states,
            sat_vel=trajectories["sat_vel"],
            debris_pos=debris_states,
            debris_vel=trajectories["debris_vel"],
            closest=closest,
            collision_probability=collision_probability,
            collision_confidence=float(pc_detail["confidence_metric"]),
            maneuver_triggered=maneuver_triggered,
            plan=plan,
            plot_path=plot_path,
        )

        return_status = "completed" if maneuver_triggered else "not_required"
        delta_v_vector = [0.0, 0.0, 0.0] if plan is None else [float(v) for v in plan["thrust_vector"]]
        LOGGER.info(
            "Scenario=%s miss_km=%.6f refined_miss_km=%.6f Pc=%.6e maneuver=%s",
            scenario_name,
            closest["distance_km"],
            closest["refined_min_distance"],
            collision_probability,
            maneuver_triggered,
        )
        summaries[scenario_name] = {
            "offset_m": offset_m,
            "closest_approach_km": closest["distance_km"],
            "refined_min_distance_km": closest["refined_min_distance"],
            "refined_tca_time": str(closest["refined_tca_time"]),
            "collision_probability": collision_probability,
            "collision_confidence_metric": float(pc_detail["confidence_metric"]),
            "maneuver_triggered": maneuver_triggered,
            "delta_v_vector_km_s": delta_v_vector,
            "return_to_orbit_status": return_status,
            "scenario_output_dir": str(scenario_dir),
            "plot_path": plot_path,
            "interactive_plot_path": interactive_plot_path,
        }
    return summaries


if __name__ == "__main__":
    summary = run_synthetic_cycle()
    print(json.dumps(summary, indent=2))
