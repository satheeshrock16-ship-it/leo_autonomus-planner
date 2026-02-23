"""Shared deterministic engine for synthetic SAFE/STRONG/CRITICAL cases."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import Any, Literal

import numpy as np

from config import PROCESSED_DATA_DIR
from hardware.propulsion_controller import PropulsionCommandResult, execute_burn
from physics.burn_physics import FuelState, SATELLITE_MASS_KG, THRUST_N, solve_burn_from_delta_v_vector
from physics.constants import SAFE_DISTANCE_KM
from physics.maneuver_optimizer import tangential_delta_v_from_miss_distance
from physics.orbit_intersection import OrbitalElements, align_orbits_at_node, orbit_period_seconds, propagate_orbit
from physics.orbital_elements import compute_orbital_elements
from physics.tca_refinement import refine_tca_analytic
from physics.two_body_propagation import propagate_universal_trajectory
from pipeline import collision_check
from visualization.plot_3d_encounter import plot_3d_encounter


MU = np.float64(398600.4418)  # km^3/s^2


class ManeuverState(str, Enum):
    IDLE = "IDLE"
    BURN1 = "BURN1"
    COAST = "COAST"
    BURN2 = "BURN2"
    COMPLETE = "COMPLETE"


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


def _scenario_time_grid(period_sat_s: float, period_deb_s: float) -> tuple[np.ndarray, float]:
    period = np.float64(max(float(period_sat_s), float(period_deb_s)))
    dt = np.float64(1.0)
    align_t = np.float64(10_000.0)
    half_window = np.float64(int(min(2200.0, 0.35 * float(period))))
    start = align_t - half_window
    stop = align_t + half_window
    t_s = np.arange(start, stop + dt, dt, dtype=np.float64)
    return t_s, float(align_t)


def _two_body_accel(r_km: np.ndarray) -> np.ndarray:
    r = np.asarray(r_km, dtype=np.float64)
    r_norm = np.float64(np.linalg.norm(r))
    return -(MU / max(float(r_norm**3), 1e-12)) * r


def _rk4_step(r_km: np.ndarray, v_km_s: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    dt = np.float64(dt_s)
    r = np.asarray(r_km, dtype=np.float64)
    v = np.asarray(v_km_s, dtype=np.float64)

    k1_r = v
    k1_v = _two_body_accel(r)
    k2_r = v + np.float64(0.5) * dt * k1_v
    k2_v = _two_body_accel(r + np.float64(0.5) * dt * k1_r)
    k3_r = v + np.float64(0.5) * dt * k2_v
    k3_v = _two_body_accel(r + np.float64(0.5) * dt * k2_r)
    k4_r = v + dt * k3_v
    k4_v = _two_body_accel(r + dt * k3_r)

    r_next = r + (dt / np.float64(6.0)) * (k1_r + np.float64(2.0) * k2_r + np.float64(2.0) * k3_r + k4_r)
    v_next = v + (dt / np.float64(6.0)) * (k1_v + np.float64(2.0) * k2_v + np.float64(2.0) * k3_v + k4_v)
    return r_next.astype(np.float64), v_next.astype(np.float64)


def _propagate_arc_chunked(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    t_rel_s: np.ndarray,
    max_step_s: float = 300.0,
) -> tuple[np.ndarray, np.ndarray]:
    t_rel = np.asarray(t_rel_s, dtype=np.float64)
    if len(t_rel) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    pos = np.zeros((len(t_rel), 3), dtype=np.float64)
    vel = np.zeros((len(t_rel), 3), dtype=np.float64)
    pos[0] = np.asarray(r0_km, dtype=np.float64)
    vel[0] = np.asarray(v0_km_s, dtype=np.float64)

    for idx in range(1, len(t_rel)):
        dt_total = float(t_rel[idx] - t_rel[idx - 1])
        steps = max(1, int(np.ceil(abs(dt_total) / max(float(max_step_s), 1e-6))))
        dt_step = float(dt_total / steps)
        r_step = np.asarray(pos[idx - 1], dtype=np.float64)
        v_step = np.asarray(vel[idx - 1], dtype=np.float64)
        for _ in range(steps):
            try:
                seg_t = np.array([0.0, dt_step], dtype=np.float64)
                seg_pos, seg_vel = propagate_universal_trajectory(
                    r0_km=r_step,
                    v0_km_s=v_step,
                    t_rel_s=seg_t,
                    mu_km3_s2=float(MU),
                )
                r_step = np.asarray(seg_pos[-1], dtype=np.float64)
                v_step = np.asarray(seg_vel[-1], dtype=np.float64)
            except Exception:
                r_step, v_step = _rk4_step(r_step, v_step, dt_step)
        pos[idx] = r_step
        vel[idx] = v_step

    return pos, vel


def _wrap_to_pi(angle_rad: float) -> float:
    angle = np.float64(angle_rad)
    return float((angle + np.float64(np.pi)) % (np.float64(2.0) * np.float64(np.pi)) - np.float64(np.pi))


def _wrap_to_2pi(angle_rad: float) -> float:
    angle = np.float64(angle_rad) % (np.float64(2.0) * np.float64(np.pi))
    if angle < np.float64(0.0):
        angle += np.float64(2.0) * np.float64(np.pi)
    return float(angle)


def _mean_anomaly_from_true_anomaly(true_anomaly_rad: float, eccentricity: float) -> float:
    e = float(max(eccentricity, 0.0))
    f = float(true_anomaly_rad)
    if e < 1e-10:
        return _wrap_to_pi(f)
    sqrt_1me = float(np.sqrt(max(1.0 - e, 0.0)))
    sqrt_1pe = float(np.sqrt(max(1.0 + e, 1e-12)))
    sin_half = float(np.sin(0.5 * f))
    cos_half = float(np.cos(0.5 * f))
    ecc_anomaly = float(2.0 * np.arctan2(sqrt_1me * sin_half, sqrt_1pe * cos_half))
    mean_anomaly = float(ecc_anomaly - (e * np.sin(ecc_anomaly)))
    return _wrap_to_pi(mean_anomaly)


def _classical_elements_from_state(r_km: np.ndarray, v_km_s: np.ndarray, mu_km3_s2: float) -> dict[str, float]:
    r_vec = np.asarray(r_km, dtype=np.float64).reshape(3)
    v_vec = np.asarray(v_km_s, dtype=np.float64).reshape(3)
    invariant = compute_orbital_elements(r_vec, v_vec, mu=float(mu_km3_s2))

    a = float(invariant["semi_major_axis"])
    e = float(invariant["eccentricity"])
    e_vec = np.asarray(invariant["eccentricity_vector"], dtype=np.float64)
    h_vec = np.asarray(invariant["specific_angular_momentum_vector"], dtype=np.float64)
    h_norm = float(np.linalg.norm(h_vec))
    r_norm = float(np.linalg.norm(r_vec))
    if h_norm <= 0.0 or r_norm <= 0.0:
        raise RuntimeError("Invalid state for classical element conversion.")

    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n_vec = np.cross(k_hat, h_vec).astype(np.float64)
    n_norm = float(np.linalg.norm(n_vec))

    inc = float(np.arccos(np.clip(h_vec[2] / h_norm, -1.0, 1.0)))
    raan = float(np.arctan2(n_vec[1], n_vec[0])) if n_norm > 1e-12 else 0.0
    raan = _wrap_to_pi(raan)

    if e > 1e-12 and n_norm > 1e-12:
        argp_num = float(np.dot(np.cross(n_vec, e_vec), h_vec) / (n_norm * h_norm * e))
        argp_den = float(np.dot(n_vec, e_vec) / (n_norm * e))
        argp = float(np.arctan2(argp_num, argp_den))
    elif e > 1e-12:
        argp = float(np.arctan2(e_vec[1], e_vec[0]))
    else:
        argp = 0.0
    argp = _wrap_to_pi(argp)

    if e > 1e-12:
        f_num = float(np.dot(np.cross(e_vec, r_vec), h_vec) / (e * h_norm * r_norm))
        f_den = float(np.dot(e_vec, r_vec) / (e * r_norm))
        true_anomaly = float(np.arctan2(f_num, f_den))
    elif n_norm > 1e-12:
        f_num = float(np.dot(np.cross(n_vec, r_vec), h_vec) / (n_norm * h_norm * r_norm))
        f_den = float(np.dot(n_vec, r_vec) / (n_norm * r_norm))
        true_anomaly = float(np.arctan2(f_num, f_den))
    else:
        true_anomaly = float(np.arctan2(r_vec[1], r_vec[0]))
    true_anomaly = _wrap_to_pi(true_anomaly)

    mean_anomaly = _mean_anomaly_from_true_anomaly(true_anomaly, e)
    mean_motion = float(np.sqrt(float(mu_km3_s2) / max(a**3, 1e-12))) if a > 0.0 else 0.0
    return {
        "a": a,
        "e": e,
        "i": inc,
        "raan": raan,
        "argp": argp,
        "f": true_anomaly,
        "M": mean_anomaly,
        "n": mean_motion,
    }


def _transition_maneuver_state(
    current_state: ManeuverState,
    *,
    maneuver_triggered: bool = False,
    burn1_complete: bool = False,
    at_restoration_time: bool = False,
    burn2_complete: bool = False,
    cycle_complete: bool = False,
) -> ManeuverState:
    if current_state == ManeuverState.IDLE and maneuver_triggered:
        return ManeuverState.BURN1
    if current_state == ManeuverState.BURN1 and burn1_complete:
        return ManeuverState.COAST
    if current_state == ManeuverState.COAST and at_restoration_time:
        return ManeuverState.BURN2
    if current_state == ManeuverState.BURN2 and burn2_complete:
        return ManeuverState.COMPLETE
    if current_state == ManeuverState.COMPLETE and cycle_complete:
        return ManeuverState.IDLE
    return current_state


def _print_two_impulse_log(
    burn1_hw: PropulsionCommandResult | None,
    burn2_hw: PropulsionCommandResult | None,
    burn1_time_s: float,
    burn2_time_s: float,
    coast_duration_s: float,
    total_delta_v_km_s: float,
) -> None:
    burn1_yaw = float(burn1_hw.yaw_deg) if burn1_hw is not None else 0.0
    burn1_pitch = float(burn1_hw.pitch_deg) if burn1_hw is not None else 0.0
    burn2_yaw = float(burn2_hw.yaw_deg) if burn2_hw is not None else 0.0
    burn2_pitch = float(burn2_hw.pitch_deg) if burn2_hw is not None else 0.0
    total_burn_time_s = float(max(burn1_time_s, 0.0) + max(burn2_time_s, 0.0))

    print("--- BURN 1 ---")
    print(f"Yaw: {burn1_yaw:.2f}")
    print(f"Pitch: {burn1_pitch:.2f}")
    print(f"Burn time: {float(max(burn1_time_s, 0.0)):.6f} s")
    print("")
    print("--- COAST PHASE ---")
    print(f"Duration: {float(max(coast_duration_s, 0.0)):.6f} s")
    print("")
    print("--- BURN 2 ---")
    print(f"Yaw: {burn2_yaw:.2f}")
    print(f"Pitch: {burn2_pitch:.2f}")
    print(f"Burn time: {float(max(burn2_time_s, 0.0)):.6f} s")
    print("")
    print(f"Total Delta-v: {float(max(total_delta_v_km_s, 0.0)):.12f} km/s")
    print(f"Total Burn Time: {total_burn_time_s:.6f} s")


def _run_two_impulse_state_machine(
    sat_xyz: np.ndarray,
    sat_vel: np.ndarray,
    t_s: np.ndarray,
    tca_idx: int,
    delta_v_mag_km_s: float,
    fuel_state: FuelState,
    m_tca_rad: float,
    f_tca_rad: float,
    epoch_tca_s: float,
    r_tca_vector_km: np.ndarray,
) -> dict[str, Any]:
    sat_pos = np.asarray(sat_xyz, dtype=np.float64)
    sat_vel_arr = np.asarray(sat_vel, dtype=np.float64)
    times = np.asarray(t_s, dtype=np.float64)
    dv_mag = np.float64(max(float(delta_v_mag_km_s), 0.0))

    burn_lead_s = np.float64(300.0)
    tca_time = np.float64(times[tca_idx])
    burn1_time = np.float64(max(float(tca_time - burn_lead_s), float(times[0])))
    burn1_idx = int(np.argmin(np.abs(times - burn1_time)))

    r0 = np.asarray(sat_pos[burn1_idx], dtype=np.float64)
    v0 = np.asarray(sat_vel_arr[burn1_idx], dtype=np.float64)
    elements0 = compute_orbital_elements(r0, v0, mu=float(MU))
    classical0 = _classical_elements_from_state(r0, v0, float(MU))
    h0_norm = np.float64(np.linalg.norm(np.asarray(elements0["specific_angular_momentum_vector"], dtype=np.float64)))

    state = ManeuverState.IDLE
    state_history = [state.value]
    maneuver_triggered = bool(dv_mag > np.float64(0.0))

    state = _transition_maneuver_state(state, maneuver_triggered=maneuver_triggered)
    if state != ManeuverState.BURN1:
        pre = max(tca_idx - 80, 0)
        post = min(tca_idx + 220, len(sat_pos))
        avoidance_arc = np.asarray(sat_pos[pre : min(pre + 120, post)], dtype=np.float64)
        restored_arc = np.asarray(sat_pos[min(pre + 119, post - 1) : post], dtype=np.float64)
        if restored_arc.size == 0:
            restored_arc = avoidance_arc[-1:].copy()
        zeros = np.zeros(3, dtype=np.float64)
        burn1_solution = solve_burn_from_delta_v_vector(zeros, mass_kg=fuel_state.current_mass_kg, force_newtons=THRUST_N)
        burn2_solution = solve_burn_from_delta_v_vector(zeros, mass_kg=fuel_state.current_mass_kg, force_newtons=THRUST_N)
        return {
            "maneuver_triggered": False,
            "state_history": state_history,
            "final_state": state.value,
            "burn1_time_s": float(burn1_time),
            "burn2_time_s": float(burn1_time),
            "restoration_time_s": float(burn1_time),
            "coast_duration_s": 0.0,
            "delta_M_at_restore_rad": 0.0,
            "angular_separation_deg": 0.0,
            "theta_safe_deg": 5.0,
            "epoch_tca_s": float(epoch_tca_s),
            "f_tca_rad": float(f_tca_rad),
            "M_tca_rad": float(m_tca_rad),
            "r_tca_vector_km": np.asarray(r_tca_vector_km, dtype=np.float64),
            "burn1_true_anomaly_rad": float(classical0["f"]),
            "burn2_true_anomaly_rad": float(classical0["f"]),
            "burn1_true_anomaly_unwrapped_rad": float(classical0["f"]),
            "burn2_true_anomaly_unwrapped_rad": float(classical0["f"]),
            "tca_true_anomaly_unwrapped_rad": float(f_tca_rad),
            "a1_km": float(elements0["semi_major_axis"]),
            "e1": float(elements0["eccentricity"]),
            "i1_rad": float(classical0["i"]),
            "raan1_rad": float(classical0["raan"]),
            "argp1_rad": float(classical0["argp"]),
            "M0_1_rad": float(classical0["M"]),
            "n1_rad_s": float(classical0["n"]),
            "burn1_position_km": r0,
            "burn2_position_km": r0,
            "delta_v1_vector_km_s": zeros,
            "delta_v2_vector_km_s": zeros,
            "burn1_solution": burn1_solution,
            "burn2_solution": burn2_solution,
            "hardware_burn1": None,
            "hardware_burn2": None,
            "avoidance_arc_km": avoidance_arc,
            "restored_orbit_segment_km": restored_arc,
            "elements0": elements0,
            "elements1": elements0,
            "elements2": elements0,
            "a_deviation": np.float64(0.0),
            "epsilon_deviation": np.float64(0.0),
            "h_norm_deviation": np.float64(0.0),
            "h0_norm": h0_norm,
        }

    state_history.append(state.value)

    v0_norm = np.float64(np.linalg.norm(v0))
    if v0_norm <= np.float64(0.0):
        raise RuntimeError("Invalid zero velocity at Burn 1 state.")
    t_hat0 = v0 / v0_norm
    delta_v1 = (t_hat0 * dv_mag).astype(np.float64)
    v1 = (v0 + delta_v1).astype(np.float64)
    elements1 = compute_orbital_elements(r0, v1, mu=float(MU))
    classical1 = _classical_elements_from_state(r0, v1, float(MU))
    a1 = float(classical1["a"])
    e1 = float(classical1["e"])
    i1 = float(classical1["i"])
    raan1 = float(classical1["raan"])
    argp1 = float(classical1["argp"])
    m0_1 = float(classical1["M"])
    n1 = float(classical1["n"])

    burn1_solution = solve_burn_from_delta_v_vector(delta_v1, mass_kg=fuel_state.current_mass_kg, force_newtons=THRUST_N)
    hw_burn1 = execute_burn(delta_v1, float(burn1_solution["burn_time_seconds"]))
    if float(np.linalg.norm(delta_v1)) > 0.0:
        fuel_state.apply_delta_v_vector(delta_v1)

    state = _transition_maneuver_state(state, burn1_complete=True)
    state_history.append(state.value)

    a0 = float(elements0["semi_major_axis"])
    if a0 <= 0.0 or a1 <= 0.0 or n1 <= 0.0:
        raise RuntimeError("Burn 1 produced a non-elliptic orbit; restoration period is undefined.")

    orbital_period_1 = float((2.0 * np.pi) / n1)
    max_propagation_time = float(max(1e-3, 0.75 * orbital_period_1))
    scan_dt_s = float(max(1.0, min(5.0, max_propagation_time / 300.0)))

    def _search_restoration(theta_safe_deg_local: float) -> dict[str, float | bool]:
        theta_safe_rad_local = float(np.deg2rad(theta_safe_deg_local))
        t_rel_s = 0.0
        crossed_tca = bool(_wrap_to_pi(m0_1 - m_tca_rad) > 0.0)
        delta_m_last = float(_wrap_to_pi(m0_1 - m_tca_rad))
        while t_rel_s < max_propagation_time:
            t_rel_s = float(min(t_rel_s + scan_dt_s, max_propagation_time))
            m_current = float(_wrap_to_pi(m0_1 + (n1 * t_rel_s)))
            delta_m = float(_wrap_to_pi(m_current - m_tca_rad))
            if delta_m > 0.0:
                crossed_tca = True
            if delta_m > theta_safe_rad_local:
                return {
                    "triggered": True,
                    "t_restore_s": t_rel_s,
                    "delta_m_rad": delta_m,
                    "theta_safe_deg": theta_safe_deg_local,
                    "crossed_tca": crossed_tca,
                }
            delta_m_last = delta_m
        return {
            "triggered": False,
            "t_restore_s": max_propagation_time,
            "delta_m_rad": delta_m_last,
            "theta_safe_deg": theta_safe_deg_local,
            "crossed_tca": crossed_tca,
        }

    theta_safe_deg = 5.0
    search = _search_restoration(theta_safe_deg)
    if not bool(search["crossed_tca"]):
        print("Temporary orbit never crossed M_TCA \u2013 increasing theta_safe to 10 deg")
        theta_safe_deg = 10.0
        search = _search_restoration(theta_safe_deg)

    adjust_attempts = 0
    while not bool(search["triggered"]) and adjust_attempts < 3:
        print("Restoration window exceeded \u2013 adjusting safety angle")
        theta_safe_deg = max(theta_safe_deg * 0.5, 0.5)
        search = _search_restoration(theta_safe_deg)
        adjust_attempts += 1

    t_restore = float(search["t_restore_s"])
    delta_m_at_restore = float(search["delta_m_rad"])
    if not bool(search["triggered"]):
        m_restore = float(_wrap_to_pi(m0_1 + (n1 * t_restore)))
        delta_m_at_restore = float(_wrap_to_pi(m_restore - m_tca_rad))
    restoration_time = float(burn1_time + t_restore)
    angular_separation_deg = float(np.degrees(delta_m_at_restore))

    avoid_samples = int(np.clip(np.ceil(t_restore / 5.0) + 1.0, 120.0, 12000.0))
    t_avoid_rel = np.linspace(np.float64(0.0), np.float64(t_restore), avoid_samples, dtype=np.float64)
    avoid_pos, avoid_vel = _propagate_arc_chunked(
        r0_km=r0,
        v0_km_s=v1,
        t_rel_s=t_avoid_rel,
    )

    state = _transition_maneuver_state(state, at_restoration_time=True)
    state_history.append(state.value)

    r2 = np.asarray(avoid_pos[-1], dtype=np.float64)
    v2_current = np.asarray(avoid_vel[-1], dtype=np.float64)
    burn2_classical_pre = _classical_elements_from_state(r2, v2_current, float(MU))
    f_burn1 = float(classical1["f"])
    f_burn2 = float(burn2_classical_pre["f"])
    f_tca = float(f_tca_rad)

    f_tca_unwrapped = float(f_burn1 + _wrap_to_pi(f_tca - f_burn1))
    if f_tca_unwrapped <= f_burn1:
        f_tca_unwrapped += float(2.0 * np.pi)
    f_burn2_unwrapped = float(f_burn1 + _wrap_to_pi(f_burn2 - f_burn1))
    if f_burn2_unwrapped <= f_burn1:
        f_burn2_unwrapped += float(2.0 * np.pi)
    if f_burn2_unwrapped <= f_tca_unwrapped:
        f_burn2_unwrapped += float(2.0 * np.pi)

    print("--- ANGULAR RESTORATION LOG ---")
    print(f"TCA anomaly (deg): {float(np.degrees(f_tca_unwrapped)):.6f}")
    print(f"Burn1 anomaly (deg): {float(np.degrees(f_burn1)):.6f}")
    print(f"Burn2 anomaly (deg): {float(np.degrees(f_burn2_unwrapped)):.6f}")
    print(f"Angular separation (deg): {angular_separation_deg:.6f}")

    epsilon0 = np.float64(-MU / (np.float64(2.0) * np.float64(a0)))
    r2_norm = np.float64(np.linalg.norm(r2))
    v2_speed = np.float64(np.linalg.norm(v2_current))
    if r2_norm <= np.float64(0.0) or v2_speed <= np.float64(0.0):
        raise RuntimeError("Invalid Burn 2 state.")

    v_required = np.float64(np.sqrt(max(float(np.float64(2.0) * (epsilon0 + MU / r2_norm)), 0.0)))
    t_hat2 = v2_current / v2_speed
    delta_v2_mag = np.float64(v_required - v2_speed)
    delta_v2 = (delta_v2_mag * t_hat2).astype(np.float64)

    burn2_solution = solve_burn_from_delta_v_vector(delta_v2, mass_kg=fuel_state.current_mass_kg, force_newtons=THRUST_N)
    hw_burn2 = execute_burn(delta_v2, float(burn2_solution["burn_time_seconds"]))
    if float(np.linalg.norm(delta_v2)) > 0.0:
        fuel_state.apply_delta_v_vector(delta_v2)

    state = _transition_maneuver_state(state, burn2_complete=True)
    state_history.append(state.value)

    state = _transition_maneuver_state(state, cycle_complete=True)
    state_history.append(state.value)

    v2_new = (v2_current + delta_v2).astype(np.float64)
    elements2 = compute_orbital_elements(r2, v2_new, mu=float(MU))
    h0_vec = np.asarray(elements0["specific_angular_momentum_vector"], dtype=np.float64)
    h2_vec = np.asarray(elements2["specific_angular_momentum_vector"], dtype=np.float64)
    h0_norm = np.float64(np.linalg.norm(h0_vec))
    h2_norm = np.float64(np.linalg.norm(h2_vec))
    a_deviation = np.float64(np.abs(np.float64(elements2["semi_major_axis"]) - a0))
    epsilon_deviation = np.float64(np.abs(np.float64(elements2["specific_orbital_energy"]) - epsilon0))
    h0_unit = h0_vec / max(float(h0_norm), 1e-12)
    h2_unit = h2_vec / max(float(h2_norm), 1e-12)
    h_norm_deviation = np.float64(np.linalg.norm(h2_unit - h0_unit))

    assert a_deviation < np.float64(1e-8), f"Restoration semi-major axis mismatch: {a_deviation:.3e}"
    assert epsilon_deviation < np.float64(1e-8), f"Restoration energy mismatch: {epsilon_deviation:.3e}"
    assert h_norm_deviation < np.float64(1e-8), f"Restoration angular momentum mismatch: {h_norm_deviation:.3e}"

    period0 = np.float64((2.0 * np.pi) / np.sqrt(MU / (np.float64(a0) ** np.float64(3.0))))
    restore_duration = np.float64(min(float(period0 * np.float64(0.35)), 2400.0))
    restore_samples = max(220, int(np.ceil(float(restore_duration) / 5.0)) + 1)
    t_restore_rel = np.linspace(np.float64(0.0), restore_duration, restore_samples, dtype=np.float64)
    restored_arc, _ = _propagate_arc_chunked(
        r0_km=r2,
        v0_km_s=v2_new,
        t_rel_s=t_restore_rel,
    )

    return {
        "maneuver_triggered": True,
        "state_history": state_history,
        "final_state": state.value,
        "burn1_time_s": float(burn1_time),
        "burn2_time_s": float(restoration_time),
        "restoration_time_s": float(restoration_time),
        "coast_duration_s": float(t_restore),
        "delta_M_at_restore_rad": float(delta_m_at_restore),
        "angular_separation_deg": float(angular_separation_deg),
        "theta_safe_deg": float(search["theta_safe_deg"]),
        "epoch_tca_s": float(epoch_tca_s),
        "f_tca_rad": float(f_tca_rad),
        "M_tca_rad": float(m_tca_rad),
        "r_tca_vector_km": np.asarray(r_tca_vector_km, dtype=np.float64),
        "burn1_true_anomaly_rad": float(f_burn1),
        "burn2_true_anomaly_rad": float(f_burn2),
        "burn1_true_anomaly_unwrapped_rad": float(f_burn1),
        "burn2_true_anomaly_unwrapped_rad": float(f_burn2_unwrapped),
        "tca_true_anomaly_unwrapped_rad": float(f_tca_unwrapped),
        "a1_km": float(a1),
        "e1": float(e1),
        "i1_rad": float(i1),
        "raan1_rad": float(raan1),
        "argp1_rad": float(argp1),
        "M0_1_rad": float(m0_1),
        "n1_rad_s": float(n1),
        "burn1_position_km": r0,
        "burn2_position_km": r2,
        "delta_v1_vector_km_s": delta_v1,
        "delta_v2_vector_km_s": delta_v2,
        "burn1_solution": burn1_solution,
        "burn2_solution": burn2_solution,
        "hardware_burn1": hw_burn1,
        "hardware_burn2": hw_burn2,
        "avoidance_arc_km": avoid_pos,
        "restored_orbit_segment_km": restored_arc,
        "elements0": elements0,
        "elements1": elements1,
        "elements2": elements2,
        "a_deviation": a_deviation,
        "epsilon_deviation": epsilon_deviation,
        "h_norm_deviation": h_norm_deviation,
        "h0_norm": h0_norm,
    }


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
    sat_node_xyz, _ = propagate_orbit(
        scenario.satellite,
        np.array([align_t], dtype=np.float64),
        mean_anomaly_at_epoch_rad=float(base_alignment["satellite_mean_anomaly_epoch_rad"]),
    )
    sat_radius_at_node_km = float(np.linalg.norm(np.asarray(sat_node_xyz[0], dtype=np.float64)))

    if scenario.phase_offset_mode == "none":
        phase_offset_rad = 0.0
    else:
        phase_offset_rad = float(np.float64(scenario.target_miss_km) / max(np.float64(sat_radius_at_node_km), np.float64(1e-9)))

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

    sat_xyz = np.asarray(sat_xyz, dtype=np.float64)
    sat_vel = np.asarray(sat_vel, dtype=np.float64)
    deb_xyz = np.asarray(deb_xyz, dtype=np.float64)
    deb_vel = np.asarray(deb_vel, dtype=np.float64)

    rel_xyz = deb_xyz - sat_xyz
    dist = np.linalg.norm(rel_xyz, axis=1)
    tca_idx = int(np.argmin(dist))
    refined = refine_tca_analytic(
        time_s=t_s,
        sat_pos_km=sat_xyz,
        sat_vel_km_s=sat_vel,
        debris_pos_km=deb_xyz,
        debris_vel_km_s=deb_vel,
        min_index=tca_idx,
        search_window_s=1.0,
    )
    min_distance_km = float(refined["refined_min_distance"])
    relative_velocity_km_s = float(np.linalg.norm(np.asarray(refined["rel_vel_tca_km_s"], dtype=np.float64)))
    tca_point_km = (
        np.asarray(refined["sat_tca_km"], dtype=np.float64) + np.asarray(refined["debris_tca_km"], dtype=np.float64)
    ) * np.float64(0.5)
    r_tca_vector = np.asarray(refined["sat_tca_km"], dtype=np.float64)
    v_tca_vector = np.asarray(refined["sat_tca_vel_km_s"], dtype=np.float64)
    tca_elements = _classical_elements_from_state(r_tca_vector, v_tca_vector, float(MU))
    f_tca_rad = float(tca_elements["f"])
    m_tca_rad = float(tca_elements["M"])
    epoch_tca_s = float(refined["refined_time_s"])

    if scenario.expected_min_km is not None:
        assert min_distance_km > float(scenario.expected_min_km), (
            f"{scenario.name}: min_distance={min_distance_km:.6f} is below {scenario.expected_min_km:.6f}"
        )
    if scenario.expected_max_km is not None:
        assert min_distance_km < float(scenario.expected_max_km), (
            f"{scenario.name}: min_distance={min_distance_km:.6f} is above {scenario.expected_max_km:.6f}"
        )

    r_km = float(np.linalg.norm(np.asarray(refined["sat_tca_km"], dtype=np.float64)))
    delta_v_mag_km_s = tangential_delta_v_from_miss_distance(
        predicted_miss_km=min_distance_km,
        sat_position_norm_km=r_km,
        mu_km3_s2=float(MU),
        safe_distance_km=SAFE_DISTANCE_KM,
    )

    fuel_state = FuelState(initial_mass_kg=SATELLITE_MASS_KG, propellant_fraction=0.3, force_newtons=THRUST_N)
    two_impulse = _run_two_impulse_state_machine(
        sat_xyz=sat_xyz,
        sat_vel=sat_vel,
        t_s=t_s,
        tca_idx=tca_idx,
        delta_v_mag_km_s=delta_v_mag_km_s,
        fuel_state=fuel_state,
        m_tca_rad=m_tca_rad,
        f_tca_rad=f_tca_rad,
        epoch_tca_s=epoch_tca_s,
        r_tca_vector_km=r_tca_vector,
    )
    maneuver_triggered = bool(two_impulse["maneuver_triggered"])
    delta_v1_vector_km_s = np.asarray(two_impulse["delta_v1_vector_km_s"], dtype=np.float64)
    delta_v2_vector_km_s = np.asarray(two_impulse["delta_v2_vector_km_s"], dtype=np.float64)
    burn1 = two_impulse["burn1_solution"]
    burn2 = two_impulse["burn2_solution"]
    hw_burn1 = two_impulse["hardware_burn1"]
    hw_burn2 = two_impulse["hardware_burn2"]

    delta_v1_mag_km_s = float(np.linalg.norm(delta_v1_vector_km_s))
    delta_v2_mag_km_s = float(np.linalg.norm(delta_v2_vector_km_s))
    total_delta_v_km_s = float(delta_v1_mag_km_s + delta_v2_mag_km_s)
    burn_time_s = float(np.float64(burn1["burn_time_seconds"]) + np.float64(burn2["burn_time_seconds"]))

    pc_detail = collision_check.run_detailed(
        min_distance_km,
        sat_r_eci_km=np.asarray(refined["sat_tca_km"], dtype=np.float64),
        sat_v_eci_km_s=np.asarray(refined["sat_tca_vel_km_s"], dtype=np.float64),
        rel_r_eci_km=np.asarray(refined["rel_tca_km"], dtype=np.float64),
        rel_v_eci_km_s=np.asarray(refined["rel_vel_tca_km_s"], dtype=np.float64),
        covariance_dt_s=float(refined.get("coarse_to_refined_dt_s", 0.0)),
    )
    collision_probability = float(pc_detail["Pc"])
    if scenario.expected_maneuver_triggered is not None:
        assert maneuver_triggered == bool(scenario.expected_maneuver_triggered), (
            f"{scenario.name}: maneuver_triggered={maneuver_triggered} does not match expected "
            f"{scenario.expected_maneuver_triggered}"
        )

    burn1_epoch = float(two_impulse["burn1_time_s"])
    restoration_epoch = float(two_impulse["restoration_time_s"])
    window_before_s = max(600.0, float(0.2 * period_sat))
    window_after_s = max(900.0, float(min(0.5 * period_sat, 3600.0)))
    viz_start = float(min(burn1_epoch - window_before_s, epoch_tca_s - float(0.1 * period_sat)))
    viz_stop = float(max(restoration_epoch + window_after_s, epoch_tca_s + float(0.1 * period_sat)))
    if viz_stop <= viz_start:
        viz_stop = viz_start + max(1200.0, float(period_sat))
    viz_orbit_time = np.linspace(np.float64(viz_start), np.float64(viz_stop), 900, dtype=np.float64)
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
        satellite_orbit_km=np.asarray(sat_orbit_full, dtype=np.float64),
        debris_orbit_km=np.asarray(deb_orbit_full, dtype=np.float64),
        tca_point_km=tca_point_km,
        avoidance_arc_km=np.asarray(two_impulse["avoidance_arc_km"], dtype=np.float64),
        restored_orbit_segment_km=np.asarray(two_impulse["restored_orbit_segment_km"], dtype=np.float64),
        burn1_point_km=np.asarray(two_impulse["burn1_position_km"], dtype=np.float64),
        burn2_point_km=np.asarray(two_impulse["burn2_position_km"], dtype=np.float64),
        delta_v1_vector_km_s=delta_v1_vector_km_s,
        delta_v2_vector_km_s=delta_v2_vector_km_s,
    )

    _print_two_impulse_log(
        burn1_hw=hw_burn1,
        burn2_hw=hw_burn2,
        burn1_time_s=float(burn1["burn_time_seconds"]),
        burn2_time_s=float(burn2["burn_time_seconds"]),
        coast_duration_s=float(two_impulse["coast_duration_s"]),
        total_delta_v_km_s=total_delta_v_km_s,
    )

    elements0 = two_impulse["elements0"]
    elements1 = two_impulse["elements1"]
    elements2 = two_impulse["elements2"]

    print(f"Scenario: {scenario.name}")
    print(f"Minimum distance (km): {min_distance_km:.6f}")
    print(f"Relative velocity (km/s): {relative_velocity_km_s:.6f}")
    print(f"Collision probability: {collision_probability:.6f}")
    print(f"Monte Carlo Pc: {float(pc_detail.get('Pc_mc', float('nan'))):.6f}")
    print(f"Maneuver triggered: {maneuver_triggered}")
    print(f"Burn time total (s): {burn_time_s:.6f}")
    print(f"Mass after burn(s) (kg): {fuel_state.current_mass_kg:.6f}")
    print(f"Original semi-major axis: {float(elements0['semi_major_axis']):.12f}")
    print(f"Post-burn1 semi-major axis: {float(elements1['semi_major_axis']):.12f}")
    print(f"Restored semi-major axis: {float(elements2['semi_major_axis']):.12f}")
    print(f"Deviation in a: {float(two_impulse['a_deviation']):.12e}")
    print(f"Total Delta-v1: {delta_v1_mag_km_s:.12f} km/s")
    print(f"Total Delta-v2: {delta_v2_mag_km_s:.12f} km/s")
    print(f"Total Delta-v: {total_delta_v_km_s:.12f} km/s")
    if hw_burn1 is not None:
        print(f"Burn 1 Servo yaw (deg): {hw_burn1.yaw_deg:.2f}")
        print(f"Burn 1 Servo pitch (deg): {hw_burn1.pitch_deg:.2f}")
    if hw_burn2 is not None:
        print(f"Burn 2 Servo yaw (deg): {hw_burn2.yaw_deg:.2f}")
        print(f"Burn 2 Servo pitch (deg): {hw_burn2.pitch_deg:.2f}")

    case_dir = PROCESSED_DATA_DIR / "synthetic_cases" / scenario.name.lower()
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scenario": scenario.name,
        "safe_distance_km": SAFE_DISTANCE_KM,
        "satellite_mass_kg": SATELLITE_MASS_KG,
        "thrust_n": THRUST_N,
        "mu_km3_s2": float(MU),
        "phase_offset_rad": phase_offset_rad,
        "epoch_tca_s": float(two_impulse.get("epoch_tca_s", epoch_tca_s)),
        "f_tca_rad": float(two_impulse.get("f_tca_rad", f_tca_rad)),
        "M_tca_rad": float(two_impulse.get("M_tca_rad", m_tca_rad)),
        "r_tca_vector_km": [float(v) for v in np.asarray(two_impulse.get("r_tca_vector_km", r_tca_vector), dtype=np.float64)],
        "delta_M_at_restore_rad": float(two_impulse.get("delta_M_at_restore_rad", 0.0)),
        "angular_separation_deg": float(two_impulse.get("angular_separation_deg", 0.0)),
        "theta_safe_deg": float(two_impulse.get("theta_safe_deg", 5.0)),
        "minimum_distance_km": min_distance_km,
        "relative_velocity_km_s": relative_velocity_km_s,
        "collision_probability": collision_probability,
        "collision_probability_monte_carlo": float(pc_detail.get("Pc_mc", float("nan"))),
        "collision_probability_abs_error": float(pc_detail.get("pc_abs_error", float("nan"))),
        "collision_probability_relative_error": float(pc_detail.get("pc_relative_error", float("nan"))),
        "maneuver_triggered": maneuver_triggered,
        "maneuver_state_history": list(two_impulse["state_history"]),
        "final_maneuver_state": str(two_impulse["final_state"]),
        "delta_v_vector_km_s": [float(v) for v in delta_v1_vector_km_s],
        "delta_v_restoration_vector_km_s": [float(v) for v in delta_v2_vector_km_s],
        "delta_v1_magnitude_km_s": delta_v1_mag_km_s,
        "delta_v2_magnitude_km_s": delta_v2_mag_km_s,
        "delta_v_total_km_s": total_delta_v_km_s,
        "burn_time_seconds": burn_time_s,
        "burn1_time_seconds": float(burn1["burn_time_seconds"]),
        "burn2_time_seconds": float(burn2["burn_time_seconds"]),
        "coast_duration_seconds": float(two_impulse["coast_duration_s"]),
        "burn1_epoch_seconds": float(two_impulse["burn1_time_s"]),
        "burn2_epoch_seconds": float(two_impulse["burn2_time_s"]),
        "restoration_time_seconds": float(two_impulse["restoration_time_s"]),
        "burn1_true_anomaly_deg": float(np.degrees(float(two_impulse.get("burn1_true_anomaly_rad", 0.0)))),
        "burn2_true_anomaly_deg": float(np.degrees(float(two_impulse.get("burn2_true_anomaly_rad", 0.0)))),
        "tca_true_anomaly_deg": float(np.degrees(float(two_impulse.get("f_tca_rad", f_tca_rad)))),
        "burn1_true_anomaly_unwrapped_deg": float(
            np.degrees(float(two_impulse.get("burn1_true_anomaly_unwrapped_rad", 0.0)))
        ),
        "burn2_true_anomaly_unwrapped_deg": float(
            np.degrees(float(two_impulse.get("burn2_true_anomaly_unwrapped_rad", 0.0)))
        ),
        "tca_true_anomaly_unwrapped_deg": float(
            np.degrees(float(two_impulse.get("tca_true_anomaly_unwrapped_rad", f_tca_rad)))
        ),
        "temporary_orbit_a1_km": float(two_impulse.get("a1_km", 0.0)),
        "temporary_orbit_e1": float(two_impulse.get("e1", 0.0)),
        "temporary_orbit_i1_rad": float(two_impulse.get("i1_rad", 0.0)),
        "temporary_orbit_raan1_rad": float(two_impulse.get("raan1_rad", 0.0)),
        "temporary_orbit_argp1_rad": float(two_impulse.get("argp1_rad", 0.0)),
        "temporary_orbit_M0_1_rad": float(two_impulse.get("M0_1_rad", 0.0)),
        "temporary_orbit_n1_rad_s": float(two_impulse.get("n1_rad_s", 0.0)),
        "mass_before_kg": float(SATELLITE_MASS_KG),
        "mass_after_kg": float(fuel_state.current_mass_kg),
        "propellant_used_kg": float(max(SATELLITE_MASS_KG - fuel_state.current_mass_kg, 0.0)),
        "remaining_propellant_kg": float(fuel_state.remaining_propellant_kg),
        "burn_count": int(fuel_state.burn_count),
        "total_delta_v_km_s": float(fuel_state.total_delta_v_km_s),
        "mass_history_kg": [float(v) for v in fuel_state.mass_history_kg],
        "burn1_position_km": [float(v) for v in np.asarray(two_impulse["burn1_position_km"], dtype=np.float64)],
        "burn2_position_km": [float(v) for v in np.asarray(two_impulse["burn2_position_km"], dtype=np.float64)],
        "original_orbit": {
            "semi_major_axis": float(elements0["semi_major_axis"]),
            "eccentricity": float(elements0["eccentricity"]),
            "eccentricity_vector": [float(v) for v in np.asarray(elements0["eccentricity_vector"], dtype=np.float64)],
            "specific_angular_momentum_vector": [
                float(v) for v in np.asarray(elements0["specific_angular_momentum_vector"], dtype=np.float64)
            ],
            "specific_orbital_energy": float(elements0["specific_orbital_energy"]),
        },
        "post_burn1_orbit": {
            "semi_major_axis": float(elements1["semi_major_axis"]),
            "eccentricity": float(elements1["eccentricity"]),
            "eccentricity_vector": [float(v) for v in np.asarray(elements1["eccentricity_vector"], dtype=np.float64)],
            "specific_angular_momentum_vector": [
                float(v) for v in np.asarray(elements1["specific_angular_momentum_vector"], dtype=np.float64)
            ],
            "specific_orbital_energy": float(elements1["specific_orbital_energy"]),
        },
        "restored_orbit": {
            "semi_major_axis": float(elements2["semi_major_axis"]),
            "eccentricity": float(elements2["eccentricity"]),
            "eccentricity_vector": [float(v) for v in np.asarray(elements2["eccentricity_vector"], dtype=np.float64)],
            "specific_angular_momentum_vector": [
                float(v) for v in np.asarray(elements2["specific_angular_momentum_vector"], dtype=np.float64)
            ],
            "specific_orbital_energy": float(elements2["specific_orbital_energy"]),
        },
        "restoration_validation": {
            "a_deviation": float(two_impulse["a_deviation"]),
            "epsilon_deviation": float(two_impulse["epsilon_deviation"]),
            "h_norm_deviation": float(two_impulse["h_norm_deviation"]),
        },
        "hardware_connected_burn1": bool(hw_burn1.connected) if hw_burn1 is not None else False,
        "hardware_connected_burn2": bool(hw_burn2.connected) if hw_burn2 is not None else False,
        "serial_command_burn1": hw_burn1.command_string if hw_burn1 is not None else "",
        "serial_command_burn2": hw_burn2.command_string if hw_burn2 is not None else "",
        "tca_time_seconds": float(refined["refined_time_s"]),
        "interactive_html": plot_paths.get("interactive_html", ""),
        "snapshot_png": plot_paths.get("snapshot_png", ""),
    }
    with (case_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary
