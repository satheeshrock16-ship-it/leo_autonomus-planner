"""Synthetic end-to-end autonomous LEO collision avoidance pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from config import COLLISION_PROBABILITY_THRESHOLD, PROCESSED_DATA_DIR
from pipeline import collision_check, replan
from visualization.plot_synthetic import plot_synthetic_scenario


SCENARIOS = {
    "SAFE_DEMO": 200,
    "STRONG_MANEUVER": 50,
    "CRITICAL_EMERGENCY": 10,
}

SATELLITE_INITIAL_STATE = {
    "x_km": 7000.0,
    "y_km": 0.0,
    "z_km": 0.0,
    "vx_kmps": 0.0,
    "vy_kmps": 7.5,
    "vz_kmps": 0.0,
}

ORBIT_WINDOW_SECONDS = 5400
TIMESTEP_SECONDS = 5


def _time_vector() -> np.ndarray:
    return np.arange(0.0, ORBIT_WINDOW_SECONDS + TIMESTEP_SECONDS, TIMESTEP_SECONDS, dtype=float)


def _propagate_circular(position0_km: np.ndarray, velocity0_kmps: np.ndarray, t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius_km = float(np.linalg.norm(position0_km))
    speed_kmps = float(np.linalg.norm(velocity0_kmps))
    omega = speed_kmps / radius_km

    theta0 = float(np.arctan2(position0_km[1], position0_km[0]))
    theta = theta0 + omega * t_s

    pos = np.column_stack(
        [
            radius_km * np.cos(theta),
            radius_km * np.sin(theta),
            np.full_like(theta, position0_km[2]),
        ]
    )
    vel = np.column_stack(
        [
            -radius_km * omega * np.sin(theta),
            radius_km * omega * np.cos(theta),
            np.zeros_like(theta),
        ]
    )
    return pos, vel


def _make_synthetic_trajectories(offset_m: float, t_s: np.ndarray) -> dict[str, np.ndarray]:
    sat_pos0 = np.array(
        [
            SATELLITE_INITIAL_STATE["x_km"],
            SATELLITE_INITIAL_STATE["y_km"],
            SATELLITE_INITIAL_STATE["z_km"],
        ],
        dtype=float,
    )
    sat_vel0 = np.array(
        [
            SATELLITE_INITIAL_STATE["vx_kmps"],
            SATELLITE_INITIAL_STATE["vy_kmps"],
            SATELLITE_INITIAL_STATE["vz_kmps"],
        ],
        dtype=float,
    )

    sat_pos, sat_vel = _propagate_circular(sat_pos0, sat_vel0, t_s)

    radial_hat = sat_pos0 / np.linalg.norm(sat_pos0)
    debris_pos0 = sat_pos0 + radial_hat * (offset_m / 1000.0)
    debris_vel0 = sat_vel0.copy()
    debris_pos, debris_vel = _propagate_circular(debris_pos0, debris_vel0, t_s)

    return {
        "sat_pos": sat_pos,
        "sat_vel": sat_vel,
        "debris_pos": debris_pos,
        "debris_vel": debris_vel,
    }


def _closest_approach(sat_pos: np.ndarray, debris_pos: np.ndarray, t_s: np.ndarray) -> dict[str, Any]:
    rel = debris_pos - sat_pos
    distances_km = np.linalg.norm(rel, axis=1)
    idx = int(np.argmin(distances_km))
    return {
        "index": idx,
        "time_s": float(t_s[idx]),
        "distance_km": float(distances_km[idx]),
        "rel_km": rel[idx],
        "sat_tca_km": sat_pos[idx],
        "debris_tca_km": debris_pos[idx],
        "tca_point_km": (sat_pos[idx] + debris_pos[idx]) * 0.5,
    }


def _build_maneuver_trajectories(
    sat_pos: np.ndarray,
    tca_idx: int,
    maneuver_triggered: bool,
    thrust_vector_kmps: np.ndarray,
    return_delta_v_km_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    avoidance_orbit = sat_pos.copy()
    return_orbit = sat_pos.copy()

    if not maneuver_triggered:
        return avoidance_orbit, return_orbit

    viz_scale = 900.0
    offset_vec = thrust_vector_kmps * viz_scale
    avoidance_orbit[tca_idx:] = sat_pos[tca_idx:] + offset_vec

    return_shift = np.array([0.0, -return_delta_v_km_s * viz_scale, 0.0], dtype=float)
    return_orbit[tca_idx:] = avoidance_orbit[tca_idx:] + return_shift

    return avoidance_orbit, return_orbit


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
        writer.writerow(["scenario", "tca_time_s", "miss_distance_km", "rel_x_km", "rel_y_km", "rel_z_km", "collision_probability"])
        writer.writerow([
            scenario_name,
            closest["time_s"],
            closest["distance_km"],
            float(closest["rel_km"][0]),
            float(closest["rel_km"][1]),
            float(closest["rel_km"][2]),
            collision_probability,
        ])

    payload = {
        "scenario": scenario_name,
        "offset_m": SCENARIOS[scenario_name],
        "closest_approach_km": closest["distance_km"],
        "tca_time_s": closest["time_s"],
        "collision_probability": collision_probability,
        "maneuver_triggered": maneuver_triggered,
        "thrust_command": None if plan is None else {
            "thrust_vector": [float(v) for v in plan["thrust_vector"]],
            "duration_ms": int(plan["duration_ms"]),
            "avoidance_delta_v_km_s": float(plan["avoidance_delta_v_km_s"]),
            "return_delta_v_km_s": float(plan["return_delta_v_km_s"]),
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


def run_synthetic_cycle() -> dict[str, Any]:
    t_s = _time_vector()
    synthetic_root = PROCESSED_DATA_DIR / "synthetic"
    synthetic_root.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}

    for scenario_name, offset_m in SCENARIOS.items():
        trajectories = _make_synthetic_trajectories(offset_m=offset_m, t_s=t_s)
        closest = _closest_approach(trajectories["sat_pos"], trajectories["debris_pos"], t_s)

        collision_probability = float(collision_check.run(closest["distance_km"]))
        maneuver_triggered = bool(collision_probability > COLLISION_PROBABILITY_THRESHOLD)

        plan = None
        thrust_vector = np.zeros(3, dtype=float)
        return_delta_v_km_s = 0.0
        if maneuver_triggered:
            plan = replan.run(current_radius_km=7000.0, nominal_radius_km=7000.0)
            thrust_vector = np.asarray(plan["thrust_vector"], dtype=float)
            return_delta_v_km_s = float(plan["return_delta_v_km_s"])

        avoidance_orbit, return_orbit = _build_maneuver_trajectories(
            trajectories["sat_pos"],
            closest["index"],
            maneuver_triggered,
            thrust_vector,
            return_delta_v_km_s,
        )

        plot_path = plot_synthetic_scenario(
            scenario_name=scenario_name,
            satellite_orbit_km=trajectories["sat_pos"],
            debris_orbit_km=trajectories["debris_pos"],
            tca_point_km=closest["tca_point_km"],
            avoidance_orbit_km=avoidance_orbit,
            return_orbit_km=return_orbit,
        )

        scenario_dir = synthetic_root / scenario_name
        _write_scenario_outputs(
            scenario_dir=scenario_dir,
            scenario_name=scenario_name,
            t_s=t_s,
            sat_pos=trajectories["sat_pos"],
            sat_vel=trajectories["sat_vel"],
            debris_pos=trajectories["debris_pos"],
            debris_vel=trajectories["debris_vel"],
            closest=closest,
            collision_probability=collision_probability,
            maneuver_triggered=maneuver_triggered,
            plan=plan,
            plot_path=plot_path,
        )

        return_status = "completed" if maneuver_triggered else "not_required"
        delta_v_vector = [0.0, 0.0, 0.0] if plan is None else [float(v) for v in plan["thrust_vector"]]

        print(f"Scenario: {scenario_name}")
        print(f"Closest approach (km): {closest['distance_km']:.6f}")
        print(f"Collision probability: {collision_probability:.6f}")
        print(f"Maneuver triggered: {maneuver_triggered}")
        print(f"Delta-V vector (km/s): {delta_v_vector}")
        print(f"Return-to-orbit status: {return_status}")
        print("-")

        summaries[scenario_name] = {
            "offset_m": offset_m,
            "closest_approach_km": closest["distance_km"],
            "collision_probability": collision_probability,
            "maneuver_triggered": maneuver_triggered,
            "delta_v_vector_km_s": delta_v_vector,
            "return_to_orbit_status": return_status,
            "scenario_output_dir": str(scenario_dir),
            "plot_path": plot_path,
        }

    return summaries


if __name__ == "__main__":
    summary = run_synthetic_cycle()
    print(json.dumps(summary, indent=2))
