"""Shared deterministic engine for synthetic SAFE/STRONG/CRITICAL cases."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from config import PROCESSED_DATA_DIR
from hardware.propulsion_controller import send_propulsion_command
from physics.burn_physics import SATELLITE_MASS_KG, THRUST_N, solve_burn_from_delta_v_vector
from physics.maneuver_optimizer import tangential_delta_v_from_miss_distance
from physics.orbit_intersection import OrbitalElements, align_orbits_at_node, orbit_period_seconds, propagate_orbit
from visualization.plot_3d_encounter import plot_3d_encounter


SAFE_DISTANCE_KM = 5.0
MU = 398600.4418  # km^3/s^2


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    satellite: OrbitalElements
    debris: OrbitalElements
    target_miss_km: float
    phase_offset_mode: Literal["target_over_sat_node_radius", "none"] = "target_over_sat_node_radius"
    expected_min_km: float | None = None
    expected_max_km: float | None = None
    expected_maneuver_triggered: bool | None = None


def _two_body_accel(r_km: np.ndarray) -> np.ndarray:
    r_norm = float(np.linalg.norm(r_km))
    return -(MU / max(r_norm**3, 1e-12)) * r_km


def _propagate_two_body_arc(r0_km: np.ndarray, v0_km_s: np.ndarray, t_rel_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t_rel_s, dtype=float)
    if len(t) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)
    pos = np.zeros((len(t), 3), dtype=float)
    vel = np.zeros((len(t), 3), dtype=float)
    pos[0] = r0_km
    vel[0] = v0_km_s
    for idx in range(1, len(t)):
        dt = float(t[idx] - t[idx - 1])
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


def _compute_collision_probability(min_distance_km: float) -> float:
    sigma = SAFE_DISTANCE_KM
    return float(np.exp(-0.5 * (float(min_distance_km) / sigma) ** 2))


def _build_avoidance_and_return_arcs(
    sat_xyz: np.ndarray,
    sat_vel: np.ndarray,
    t_s: np.ndarray,
    tca_idx: int,
    delta_v_vector_km_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if float(np.linalg.norm(delta_v_vector_km_s)) <= 0.0:
        pre = max(tca_idx - 60, 0)
        post = min(tca_idx + 160, len(sat_xyz))
        avoid = sat_xyz[pre : min(pre + 80, post)]
        ret = sat_xyz[min(pre + 79, post - 1) : post]
        if len(ret) == 0:
            ret = avoid[-1:].copy()
        return avoid, ret

    burn_lead_s = 300.0
    tca_time = float(t_s[tca_idx])
    burn_time = max(tca_time - burn_lead_s, float(t_s[0]))
    burn_idx = int(np.argmin(np.abs(t_s - burn_time)))
    t_future = t_s[burn_idx:] - t_s[burn_idx]
    avoid_pos, avoid_vel = _propagate_two_body_arc(
        r0_km=sat_xyz[burn_idx],
        v0_km_s=sat_vel[burn_idx] + delta_v_vector_km_s,
        t_rel_s=t_future,
    )
    mid = max(len(avoid_pos) // 2, 2)
    t_return = t_future[mid:] - t_future[mid]
    ret_raw, _ = _propagate_two_body_arc(
        r0_km=avoid_pos[mid],
        v0_km_s=avoid_vel[mid] - delta_v_vector_km_s,
        t_rel_s=t_return,
    )
    nominal_tail = sat_xyz[burn_idx + mid : burn_idx + mid + len(ret_raw)]
    if len(ret_raw) == len(nominal_tail) and len(ret_raw) > 1:
        blend = np.linspace(0.0, 1.0, len(ret_raw), dtype=float)
        ret = (1.0 - blend[:, None]) * ret_raw + blend[:, None] * nominal_tail
    else:
        ret = ret_raw
    return avoid_pos[: mid + 1], ret


def _scenario_time_grid(period_sat_s: float, period_deb_s: float) -> tuple[np.ndarray, float]:
    period = max(float(period_sat_s), float(period_deb_s))
    dt = 1.0
    align_t = 10_000.0
    half_window = float(int(min(2200.0, 0.35 * period)))
    start = align_t - half_window
    stop = align_t + half_window
    t_s = np.arange(start, stop + dt, dt, dtype=float)
    return t_s, align_t


def run_synthetic_scenario(scenario: SyntheticScenario) -> dict[str, Any]:
    period_sat = orbit_period_seconds(scenario.satellite)
    period_deb = orbit_period_seconds(scenario.debris)
    t_s, align_t = _scenario_time_grid(period_sat, period_deb)

    base_alignment = align_orbits_at_node(
        satellite_elements=scenario.satellite,
        debris_elements=scenario.debris,
        phase_offset_rad=0.0,
        alignment_time_s=align_t,
    )
    sat_node_xyz, sat_node_vel = propagate_orbit(
        scenario.satellite,
        np.array([align_t], dtype=float),
        mean_anomaly_at_epoch_rad=float(base_alignment["satellite_mean_anomaly_epoch_rad"]),
    )
    sat_radius_at_node_km = float(np.linalg.norm(sat_node_xyz[0]))

    if scenario.phase_offset_mode == "none":
        phase_offset_rad = 0.0
    else:
        phase_offset_rad = float(scenario.target_miss_km) / max(sat_radius_at_node_km, 1e-9)

    alignment = align_orbits_at_node(
        satellite_elements=scenario.satellite,
        debris_elements=scenario.debris,
        phase_offset_rad=phase_offset_rad,
        alignment_time_s=align_t,
    )

    sat_xyz, sat_vel = propagate_orbit(
        scenario.satellite,
        t_s,
        mean_anomaly_at_epoch_rad=float(alignment["satellite_mean_anomaly_epoch_rad"]),
    )
    deb_xyz, deb_vel = propagate_orbit(
        scenario.debris,
        t_s,
        mean_anomaly_at_epoch_rad=float(alignment["debris_mean_anomaly_epoch_rad"]),
    )

    rel_xyz = deb_xyz - sat_xyz
    rel_vel = deb_vel - sat_vel
    dist = np.linalg.norm(rel_xyz, axis=1)
    tca_idx = int(np.argmin(dist))
    min_distance_km = float(dist[tca_idx])
    relative_velocity_km_s = float(np.linalg.norm(rel_vel[tca_idx]))
    tca_point_km = (sat_xyz[tca_idx] + deb_xyz[tca_idx]) * 0.5

    if scenario.expected_min_km is not None:
        assert min_distance_km > float(scenario.expected_min_km), (
            f"{scenario.name}: min_distance={min_distance_km:.6f} is below {scenario.expected_min_km:.6f}"
        )
    if scenario.expected_max_km is not None:
        assert min_distance_km < float(scenario.expected_max_km), (
            f"{scenario.name}: min_distance={min_distance_km:.6f} is above {scenario.expected_max_km:.6f}"
        )

    r_km = float(np.linalg.norm(sat_xyz[tca_idx]))
    v_vec = sat_vel[tca_idx]
    v_hat = v_vec / max(float(np.linalg.norm(v_vec)), 1e-12)
    delta_v_mag_km_s = tangential_delta_v_from_miss_distance(
        predicted_miss_km=min_distance_km,
        sat_position_norm_km=r_km,
        mu_km3_s2=MU,
        safe_distance_km=SAFE_DISTANCE_KM,
    )
    delta_v_vector_km_s = v_hat * delta_v_mag_km_s
    burn = solve_burn_from_delta_v_vector(delta_v_vector_km_s, mass_kg=SATELLITE_MASS_KG, force_newtons=THRUST_N)
    burn_time_s = float(burn["burn_time_seconds"])
    maneuver_triggered = bool(delta_v_mag_km_s > 0.0)
    collision_probability = _compute_collision_probability(min_distance_km)
    if scenario.expected_maneuver_triggered is not None:
        assert maneuver_triggered == bool(scenario.expected_maneuver_triggered), (
            f"{scenario.name}: maneuver_triggered={maneuver_triggered} does not match expected "
            f"{scenario.expected_maneuver_triggered}"
        )

    avoid_arc, return_arc = _build_avoidance_and_return_arcs(
        sat_xyz=sat_xyz,
        sat_vel=sat_vel,
        t_s=t_s,
        tca_idx=tca_idx,
        delta_v_vector_km_s=delta_v_vector_km_s,
    )
    viz_orbit_time = np.linspace(align_t - period_sat, align_t + period_sat, 900, dtype=float)
    sat_orbit_full, _ = propagate_orbit(
        scenario.satellite,
        viz_orbit_time,
        mean_anomaly_at_epoch_rad=float(alignment["satellite_mean_anomaly_epoch_rad"]),
    )
    deb_orbit_full, _ = propagate_orbit(
        scenario.debris,
        viz_orbit_time,
        mean_anomaly_at_epoch_rad=float(alignment["debris_mean_anomaly_epoch_rad"]),
    )
    plot_paths = plot_3d_encounter(
        scenario_name=scenario.name,
        satellite_orbit_km=sat_orbit_full,
        debris_orbit_km=deb_orbit_full,
        tca_point_km=tca_point_km,
        delta_v_vector_km_s=delta_v_vector_km_s,
        avoidance_arc_km=avoid_arc,
        return_arc_km=return_arc,
    )

    hw = send_propulsion_command(delta_v_vector_km_s, burn_time_s)

    print(f"Scenario: {scenario.name}")
    print(f"Minimum distance (km): {min_distance_km:.6f}")
    print(f"Relative velocity (km/s): {relative_velocity_km_s:.6f}")
    print(f"Collision probability: {collision_probability:.6f}")
    print(f"Maneuver triggered: {maneuver_triggered}")
    print(f"Delta-v magnitude (km/s): {delta_v_mag_km_s:.6f}")
    print(f"Burn time (s): {burn_time_s:.6f}")
    print(f"Servo yaw (deg): {hw.yaw_deg:.2f}")
    print(f"Servo pitch (deg): {hw.pitch_deg:.2f}")

    case_dir = PROCESSED_DATA_DIR / "synthetic_cases" / scenario.name.lower()
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario": scenario.name,
        "safe_distance_km": SAFE_DISTANCE_KM,
        "satellite_mass_kg": SATELLITE_MASS_KG,
        "thrust_n": THRUST_N,
        "mu_km3_s2": MU,
        "phase_offset_rad": phase_offset_rad,
        "minimum_distance_km": min_distance_km,
        "relative_velocity_km_s": relative_velocity_km_s,
        "collision_probability": collision_probability,
        "maneuver_triggered": maneuver_triggered,
        "delta_v_vector_km_s": [float(v) for v in delta_v_vector_km_s],
        "delta_v_magnitude_km_s": delta_v_mag_km_s,
        "burn_time_seconds": burn_time_s,
        "servo_yaw_deg": float(hw.yaw_deg),
        "servo_pitch_deg": float(hw.pitch_deg),
        "hardware_connected": bool(hw.connected),
        "serial_command": hw.command_string,
        "tca_time_seconds": float(t_s[tca_idx]),
        "interactive_html": plot_paths.get("interactive_html", ""),
        "snapshot_png": plot_paths.get("snapshot_png", ""),
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
