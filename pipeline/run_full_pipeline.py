"""End-to-end autonomous LEO collision avoidance orchestration."""
from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from sgp4.api import Satrec, jday

from ai.bnn_model import MonteCarloDropoutBNN
from ai.delta_v_regressor import DeltaVRegressor
from ai.pinn_model import CWPhysicsInformedValidator
from config import (
    COLLISION_PROBABILITY_THRESHOLD,
    DEBRIS_DATA_DIR,
    LOGS_DIR,
    MANEUVER_CONFIG,
    MODEL_DIR,
    PERFORMANCE_CONFIG,
    PROCESSED_DATA_DIR,
    PROPAGATION_CONFIG,
    SATELLITE_DATA_DIR,
    SAFE_DISTANCE_KM,
    TCA_REFINEMENT_CONFIG,
)
from physics.burn_physics import FuelState, solve_burn_from_delta_v_vector
from physics.constants import SATELLITE_INITIAL_MASS_KG
from physics.maneuver import build_thrust_vector
from physics.maneuver_optimizer import analytical_required_delta_v_km_s, optimize_burn_timing
from physics.tca_refinement import refine_tca_analytic
from pipeline import collision_check, fetch_data
from pipeline.propulsion_interface import ThrustCommand
from results.performance_benchmark import plot_runtime_scaling, write_performance_metrics
from visualization.interactive_plot import plot_interactive_3d
from visualization.plot_avoidance import plot_avoidance_3d
from visualization.plot_orbit import plot_orbits


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)
CONJUNCTION_LOG = LOGS_DIR / "conjunction_analysis.log"
_FILE_HANDLER = logging.FileHandler(CONJUNCTION_LOG, encoding="utf-8")
_FILE_HANDLER.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
_ROOT_LOGGER = logging.getLogger()
if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == str(CONJUNCTION_LOG) for h in _ROOT_LOGGER.handlers):
    _ROOT_LOGGER.addHandler(_FILE_HANDLER)

SATELLITE_TLE_PATH = SATELLITE_DATA_DIR / "satellite_tles.json"
DEBRIS_TLE_PATH = DEBRIS_DATA_DIR / "debris_tles.json"


def _log_structured(payload: dict[str, Any]) -> None:
    LOGGER.info("CONJUNCTION_ANALYSIS %s", json.dumps(payload, sort_keys=True))


@dataclass
class DebrisPropagation:
    norad_id: str
    pos_km: np.ndarray
    vel_km_s: np.ndarray
    valid_mask: np.ndarray


def _load_tle_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"TLE file must contain a JSON list: {path}")
    return records


def _build_satrec(record: dict[str, Any]) -> Satrec:
    line1 = record.get("TLE_LINE1")
    line2 = record.get("TLE_LINE2")
    if not line1 or not line2:
        raise ValueError("Missing TLE lines")
    return Satrec.twoline2rv(line1, line2)


def _sgp4_state_from_jd_fr(sat: Satrec, jd: float, fr: float) -> tuple[np.ndarray, np.ndarray]:
    err, r_eci_km, v_eci_km_s = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code: {err}")
    return np.asarray(r_eci_km, dtype=float), np.asarray(v_eci_km_s, dtype=float)


def _to_jd_fr(epoch: datetime) -> tuple[float, float]:
    sec = epoch.second + (epoch.microsecond / 1_000_000.0)
    return jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, sec)


def _propagate_satellite(sat: Satrec, jd_fr: list[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos = np.full((len(jd_fr), 3), np.nan, dtype=float)
    vel = np.full((len(jd_fr), 3), np.nan, dtype=float)
    valid = np.zeros(len(jd_fr), dtype=bool)
    for i, (jd, fr) in enumerate(jd_fr):
        try:
            r, v = _sgp4_state_from_jd_fr(sat, jd, fr)
        except Exception:
            continue
        pos[i] = r
        vel[i] = v
        valid[i] = True
    return pos, vel, valid


def _propagate_debris_worker(task: tuple[str, str, str, list[tuple[float, float]]]) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    norad_id, line1, line2, jd_fr = task
    sat = Satrec.twoline2rv(line1, line2)
    pos = np.full((len(jd_fr), 3), np.nan, dtype=float)
    vel = np.full((len(jd_fr), 3), np.nan, dtype=float)
    valid = np.zeros(len(jd_fr), dtype=bool)
    for i, (jd, fr) in enumerate(jd_fr):
        err, r, v = sat.sgp4(jd, fr)
        if err != 0:
            continue
        pos[i] = np.asarray(r, dtype=float)
        vel[i] = np.asarray(v, dtype=float)
        valid[i] = True
    return norad_id, pos, vel, valid


def _propagate_debris_records(
    debris_records: list[dict[str, Any]],
    jd_fr: list[tuple[float, float]],
    workers: int,
) -> list[DebrisPropagation]:
    tasks: list[tuple[str, str, str, list[tuple[float, float]]]] = []
    for rec in debris_records:
        line1 = rec.get("TLE_LINE1")
        line2 = rec.get("TLE_LINE2")
        if not line1 or not line2:
            continue
        tasks.append((str(rec.get("NORAD_CAT_ID", "UNKNOWN")), line1, line2, jd_fr))
    if not tasks:
        return []

    if workers > 1:
        with mp.Pool(processes=workers) as pool:
            raw = pool.map(_propagate_debris_worker, tasks)
    else:
        raw = [_propagate_debris_worker(t) for t in tasks]
    return [DebrisPropagation(norad_id=nid, pos_km=pos, vel_km_s=vel, valid_mask=valid) for nid, pos, vel, valid in raw]


def _epoch_schedule() -> tuple[datetime, list[datetime], np.ndarray]:
    start_iso = str(PROPAGATION_CONFIG.get("deterministic_epoch_utc", "2026-01-01T00:00:00+00:00"))
    start_time = datetime.fromisoformat(start_iso)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    window_h = int(PROPAGATION_CONFIG.get("screening_window_hours", 72))
    step_s = int(PROPAGATION_CONFIG.get("timestep_seconds", 60))
    duration_seconds = window_h * 3600
    time_offsets_s = np.arange(0, duration_seconds + step_s, step_s, dtype=float)
    epochs = [start_time + timedelta(seconds=float(dt)) for dt in time_offsets_s]
    return start_time, epochs, time_offsets_s


def _load_delta_v_model() -> DeltaVRegressor | None:
    model_path = MODEL_DIR / "delta_v_model.txt"
    if not model_path.exists():
        LOGGER.warning("Delta-v model not found at %s; analytical fallback will be used.", model_path)
        return None
    try:
        return DeltaVRegressor.load(model_path)
    except Exception as exc:
        LOGGER.warning("Could not load delta-v model: %s", exc)
        return None


def _run_performance_benchmark(sat_record: dict[str, Any], debris_records: list[dict[str, Any]]) -> dict[str, Any]:
    bench_counts = [int(v) for v in PERFORMANCE_CONFIG.get("benchmark_counts", [100, 1000, 5000])]
    workers = int(PERFORMANCE_CONFIG.get("parallel_workers", 4))
    start_iso = str(PROPAGATION_CONFIG.get("deterministic_epoch_utc", "2026-01-01T00:00:00+00:00"))
    start_time = datetime.fromisoformat(start_iso)
    if start_time.tzinfo is None:
        start_time = start_time.replace(tzinfo=timezone.utc)
    window_h = int(PROPAGATION_CONFIG.get("benchmark_window_hours", 2))
    step_s = int(PROPAGATION_CONFIG.get("benchmark_timestep_seconds", 120))
    time_offsets = np.arange(0, window_h * 3600 + step_s, step_s, dtype=float)
    epochs = [start_time + timedelta(seconds=float(s)) for s in time_offsets]
    jd_fr = [_to_jd_fr(ep) for ep in epochs]
    satrec = _build_satrec(sat_record)
    sat_pos, _, sat_valid = _propagate_satellite(satrec, jd_fr)

    timings: list[dict[str, float]] = []
    base_debris = [r for r in debris_records if r.get("TLE_LINE1") and r.get("TLE_LINE2")]
    for count in bench_counts:
        if not base_debris:
            break
        repeats = int(np.ceil(count / max(len(base_debris), 1)))
        use_records = (base_debris * repeats)[:count]
        t0 = perf_counter()
        debris_results = _propagate_debris_records(use_records, jd_fr, workers=workers)
        min_dist = float("inf")
        for deb in debris_results:
            valid = deb.valid_mask & sat_valid
            if not np.any(valid):
                continue
            dist = np.linalg.norm(deb.pos_km[valid] - sat_pos[valid], axis=1)
            min_dist = min(min_dist, float(np.min(dist)))
        dt = perf_counter() - t0
        timings.append({"debris_count": float(count), "runtime_seconds": float(dt), "min_distance_km": min_dist})
        _log_structured(
            {
                "pipeline": "benchmark",
                "debris_count": int(count),
                "runtime_seconds": float(dt),
                "min_distance_km": float(min_dist),
            }
        )

    metrics_rows = [{"debris_count": row["debris_count"], "runtime_seconds": row["runtime_seconds"]} for row in timings]
    csv_path = write_performance_metrics(metrics_rows)
    plot_path = plot_runtime_scaling(metrics_rows)
    return {"metrics": timings, "performance_metrics_csv": str(csv_path), "runtime_scaling_plot": str(plot_path)}


def run_autonomous_cycle(fetch_live_data: bool = False, benchmark_mode: bool = False) -> dict[str, Any]:
    cycle_t0 = perf_counter()
    if fetch_live_data:
        fetch_data.run(limit=100)

    satellite_records = _load_tle_records(SATELLITE_TLE_PATH)
    debris_records = _load_tle_records(DEBRIS_TLE_PATH)
    if not satellite_records:
        raise RuntimeError(f"No satellite records found in {SATELLITE_TLE_PATH}")
    if not debris_records:
        raise RuntimeError(f"No debris records found in {DEBRIS_TLE_PATH}")
    protected_satellite = satellite_records[0]
    protected_sat_norad = str(protected_satellite.get("NORAD_CAT_ID", "UNKNOWN"))
    start_time, epochs, time_offsets_s = _epoch_schedule()
    jd_fr = [_to_jd_fr(ep) for ep in epochs]
    workers = int(PERFORMANCE_CONFIG.get("parallel_workers", 4))

    satrec_sat = _build_satrec(protected_satellite)
    sat_pos, sat_vel, sat_valid = _propagate_satellite(satrec_sat, jd_fr)
    debris_propagations = _propagate_debris_records(debris_records, jd_fr, workers=workers)
    if not debris_propagations:
        raise RuntimeError("No valid debris TLE lines available for SGP4 propagation.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    propagated_csv = PROCESSED_DATA_DIR / "propagated_states.csv"
    conjunction_csv = PROCESSED_DATA_DIR / "conjunction_results.csv"
    decision_json = PROCESSED_DATA_DIR / "decision.json"

    with propagated_csv.open("w", newline="", encoding="utf-8") as f_states:
        writer = csv.writer(f_states)
        writer.writerow(["epoch_utc", "object_role", "norad_cat_id", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])
        for i, epoch in enumerate(epochs):
            if sat_valid[i]:
                writer.writerow([epoch.isoformat(), "satellite", protected_sat_norad, sat_pos[i, 0], sat_pos[i, 1], sat_pos[i, 2], sat_vel[i, 0], sat_vel[i, 1], sat_vel[i, 2]])
            for deb in debris_propagations:
                if deb.valid_mask[i]:
                    writer.writerow([epoch.isoformat(), "debris", deb.norad_id, deb.pos_km[i, 0], deb.pos_km[i, 1], deb.pos_km[i, 2], deb.vel_km_s[i, 0], deb.vel_km_s[i, 1], deb.vel_km_s[i, 2]])

    conjunction_rows: list[dict[str, Any]] = []
    fuel_state = FuelState(
        initial_mass_kg=float(MANEUVER_CONFIG.get("initial_mass_kg", SATELLITE_INITIAL_MASS_KG)),
        propellant_fraction=float(MANEUVER_CONFIG.get("propellant_fraction_default", 0.3)),
    )
    sat_rel_positions: list[np.ndarray] = []
    sat_rel_velocities: list[np.ndarray] = []
    previous_best_rel = None
    for i in range(len(epochs)):
        best_dist = float("inf")
        best_rel = None
        for deb in debris_propagations:
            if not (sat_valid[i] and deb.valid_mask[i]):
                continue
            rel = deb.pos_km[i] - sat_pos[i]
            dist = float(np.linalg.norm(rel))
            if dist < best_dist:
                best_dist = dist
                best_rel = rel
        if best_rel is None:
            sat_rel_positions.append(np.zeros(3))
            sat_rel_velocities.append(np.zeros(3))
        else:
            sat_rel_positions.append(best_rel)
            if previous_best_rel is None:
                sat_rel_velocities.append(np.zeros(3))
            else:
                sat_rel_velocities.append((best_rel - previous_best_rel) / float(PROPAGATION_CONFIG.get("timestep_seconds", 60)))
            previous_best_rel = best_rel

    for deb in debris_propagations:
        valid = sat_valid & deb.valid_mask
        if not np.any(valid):
            continue
        rel = deb.pos_km - sat_pos
        rel_vel = deb.vel_km_s - sat_vel
        dist = np.linalg.norm(rel, axis=1)
        masked_dist = np.where(valid, dist, np.inf)
        idx = int(np.argmin(masked_dist))
        if not np.isfinite(masked_dist[idx]):
            continue
        refined = refine_tca_analytic(
            time_s=time_offsets_s,
            sat_pos_km=sat_pos,
            sat_vel_km_s=sat_vel,
            debris_pos_km=deb.pos_km,
            debris_vel_km_s=deb.vel_km_s,
            min_index=idx,
            search_window_s=float(PROPAGATION_CONFIG.get("timestep_seconds", 60))
            * float(TCA_REFINEMENT_CONFIG.get("search_window_multiplier", 1.0)),
            epoch_utc=start_time,
        )
        pc_detail = collision_check.run_detailed(
            float(refined["refined_min_distance"]),
            sat_r_eci_km=refined["sat_tca_km"],
            sat_v_eci_km_s=refined["sat_tca_vel_km_s"],
            rel_r_eci_km=refined["rel_tca_km"],
            rel_v_eci_km_s=refined["rel_vel_tca_km_s"],
            covariance_dt_s=float(refined.get("coarse_to_refined_dt_s", 0.0)),
            run_monte_carlo=False,
            fast_mode=True,
            integration_points_rho=16,
            integration_points_theta=48,
        )
        conjunction_rows.append(
            {
                "debris_norad_id": deb.norad_id,
                "tca_utc": epochs[idx].isoformat(),
                "refined_tca_utc": str(refined["refined_tca_time"]),
                "miss_distance_km": float(masked_dist[idx]),
                "refined_min_distance_km": float(refined["refined_min_distance"]),
                "rel_x_km": float(refined["rel_tca_km"][0]),
                "rel_y_km": float(refined["rel_tca_km"][1]),
                "rel_z_km": float(refined["rel_tca_km"][2]),
                "collision_probability": float(pc_detail["Pc"]),
                "collision_probability_mc": float(pc_detail.get("Pc_mc", float("nan"))),
                "pc_abs_error": float(pc_detail.get("pc_abs_error", float("nan"))),
                "pc_relative_error": float(pc_detail.get("pc_relative_error", float("nan"))),
                "confidence_metric": float(pc_detail["confidence_metric"]),
                "relative_cov_rr_m2": float(pc_detail.get("relative_cov_rr_m2", 0.0)),
                "relative_cov_tt_m2": float(pc_detail.get("relative_cov_tt_m2", 0.0)),
                "relative_cov_nn_m2": float(pc_detail.get("relative_cov_nn_m2", 0.0)),
                "sat_tca_km": np.asarray(refined["sat_tca_km"], dtype=float).copy(),
                "debris_tca_km": np.asarray(refined["debris_tca_km"], dtype=float).copy(),
                "sat_tca_vel_km_s": np.asarray(refined["sat_tca_vel_km_s"], dtype=float).copy(),
                "debris_tca_vel_km_s": np.asarray(refined["debris_tca_vel_km_s"], dtype=float).copy(),
                "refined_time_s": float(refined["refined_time_s"]),
                "covariance_dt_s": float(refined.get("coarse_to_refined_dt_s", 0.0)),
            }
        )
    conjunction_rows.sort(key=lambda row: row["miss_distance_km"])

    with conjunction_csv.open("w", newline="", encoding="utf-8") as f_conj:
        writer = csv.writer(f_conj)
        writer.writerow(
            [
                "debris_norad_id",
                "tca_utc",
                "refined_tca_utc",
                "miss_distance_km",
                "refined_min_distance_km",
                "rel_x_km",
                "rel_y_km",
                "rel_z_km",
                "collision_probability",
                "collision_probability_mc",
                "pc_abs_error",
                "pc_relative_error",
                "confidence_metric",
            ]
        )
        for row in conjunction_rows:
            writer.writerow(
                [
                    row["debris_norad_id"],
                    row["tca_utc"],
                    row["refined_tca_utc"],
                    row["miss_distance_km"],
                    row["refined_min_distance_km"],
                    row["rel_x_km"],
                    row["rel_y_km"],
                    row["rel_z_km"],
                    row["collision_probability"],
                    row["collision_probability_mc"],
                    row["pc_abs_error"],
                    row["pc_relative_error"],
                    row["confidence_metric"],
                ]
            )

    if not conjunction_rows:
        raise RuntimeError("No conjunction results produced from propagated states.")
    highest_risk = max(conjunction_rows, key=lambda row: row["collision_probability"])
    highest_risk_detail = collision_check.run_detailed(
        float(highest_risk["refined_min_distance_km"]),
        sat_r_eci_km=np.asarray(highest_risk["sat_tca_km"], dtype=float),
        sat_v_eci_km_s=np.asarray(highest_risk["sat_tca_vel_km_s"], dtype=float),
        rel_r_eci_km=np.array([highest_risk["rel_x_km"], highest_risk["rel_y_km"], highest_risk["rel_z_km"]], dtype=float),
        rel_v_eci_km_s=np.asarray(highest_risk["debris_tca_vel_km_s"], dtype=float) - np.asarray(highest_risk["sat_tca_vel_km_s"], dtype=float),
        covariance_dt_s=float(highest_risk.get("covariance_dt_s", 0.0)),
        run_monte_carlo=True,
    )
    highest_risk["collision_probability"] = float(highest_risk_detail["Pc"])
    highest_risk["collision_probability_mc"] = float(highest_risk_detail.get("Pc_mc", float("nan")))
    highest_risk["pc_abs_error"] = float(highest_risk_detail.get("pc_abs_error", float("nan")))
    highest_risk["pc_relative_error"] = float(highest_risk_detail.get("pc_relative_error", float("nan")))
    highest_risk["confidence_metric"] = float(highest_risk_detail["confidence_metric"])
    highest_risk["relative_cov_rr_m2"] = float(highest_risk_detail.get("relative_cov_rr_m2", 0.0))
    highest_risk["relative_cov_tt_m2"] = float(highest_risk_detail.get("relative_cov_tt_m2", 0.0))
    highest_risk["relative_cov_nn_m2"] = float(highest_risk_detail.get("relative_cov_nn_m2", 0.0))
    pc = float(highest_risk["collision_probability"])
    pc_mc = float(highest_risk["collision_probability_mc"])
    closest_miss_km = float(highest_risk["miss_distance_km"])
    refined_miss_km = float(highest_risk["refined_min_distance_km"])

    protected_orbit = sat_pos[sat_valid]
    sat_tca = np.asarray(highest_risk["sat_tca_km"], dtype=float)
    debris_tca = np.asarray(highest_risk["debris_tca_km"], dtype=float)
    highest_risk_norad = str(highest_risk["debris_norad_id"])
    risk_debris = next((d for d in debris_propagations if d.norad_id == highest_risk_norad), None)
    highest_risk_debris_orbit = np.asarray(risk_debris.pos_km[risk_debris.valid_mask], dtype=float) if risk_debris is not None else np.empty((0, 3), dtype=float)
    if protected_orbit.ndim == 2 and protected_orbit.shape[0] > 0 and highest_risk_debris_orbit.ndim == 2 and highest_risk_debris_orbit.shape[0] > 0:
        plot_orbits(protected_orbit, highest_risk_debris_orbit, title="LEO Encounter: Protected vs Highest-Risk Debris", tca_satellite_km=sat_tca, tca_debris_km=debris_tca)

    rel_pos = np.asarray(sat_rel_positions, dtype=float)
    rel_vel = np.asarray(sat_rel_velocities, dtype=float)
    if rel_pos.size == 0:
        rel_pos = np.zeros((2, 3), dtype=float)
        rel_vel = np.zeros((2, 3), dtype=float)
    elif rel_pos.shape[0] == 1:
        rel_pos = np.vstack([rel_pos, rel_pos])
        rel_vel = np.vstack([rel_vel, rel_vel])
    pinn_report = CWPhysicsInformedValidator().evaluate(rel_pos, rel_vel, n=0.0011, dt=float(PROPAGATION_CONFIG.get("timestep_seconds", 60)))
    uncertainty = MonteCarloDropoutBNN().predict(pc)

    dv_model = _load_delta_v_model()
    tca_dt = datetime.fromisoformat(str(highest_risk["refined_tca_utc"]))
    time_to_tca = max((tca_dt - start_time).total_seconds(), 1.0)
    initial_propellant = max(float(fuel_state.initial_mass_kg - fuel_state.dry_mass_kg), 1e-9)
    fuel_remaining = float(fuel_state.remaining_propellant_kg / initial_propellant)
    analytical_dv = analytical_required_delta_v_km_s(
        miss_distance_km=refined_miss_km,
        separation_constraint_km=float(MANEUVER_CONFIG.get("separation_constraint_km", SAFE_DISTANCE_KM)),
        lead_time_s=time_to_tca,
        max_delta_v_km_s=float(MANEUVER_CONFIG.get("max_delta_v_km_s", 0.02)),
    )
    ml_pred_dv = analytical_dv
    if dv_model is not None:
        rel_speed = float(np.linalg.norm(debris_tca - sat_tca) / max(float(PROPAGATION_CONFIG.get("timestep_seconds", 60)), 1.0))
        features = np.array([pc, refined_miss_km, rel_speed, time_to_tca, 700.0, fuel_remaining], dtype=float)
        ml_pred_dv = float(max(dv_model.predict(features)[0], 0.0))
    timing = optimize_burn_timing(
        tca_time_s=time_to_tca,
        orbit_period_s=5400.0,
        miss_distance_km=refined_miss_km,
        separation_constraint_km=float(MANEUVER_CONFIG.get("separation_constraint_km", SAFE_DISTANCE_KM)),
        lead_orbits_min=float(MANEUVER_CONFIG.get("burn_lead_orbits_min", 0.1)),
        lead_orbits_max=float(MANEUVER_CONFIG.get("burn_lead_orbits_max", 2.0)),
        sweep_points=int(MANEUVER_CONFIG.get("burn_sweep_points", 24)),
        max_delta_v_km_s=float(MANEUVER_CONFIG.get("max_delta_v_km_s", 0.02)),
        collision_probability=pc,
        collision_probability_threshold=float(COLLISION_PROBABILITY_THRESHOLD),
        lambda_fuel_penalty=float(MANEUVER_CONFIG.get("fuel_penalty_weight", 0.0)),
        current_mass_kg=float(fuel_state.current_mass_kg),
        safe_distance_km=float(SAFE_DISTANCE_KM),
    )
    selected_dv = float(max(timing["required_delta_v_km_s"], ml_pred_dv))
    should_burn = bool((pc > COLLISION_PROBABILITY_THRESHOLD or refined_miss_km < SAFE_DISTANCE_KM) and selected_dv > 0.0)

    output: dict[str, Any] = {
        "satellite_tle_path": str(SATELLITE_TLE_PATH),
        "debris_tle_path": str(DEBRIS_TLE_PATH),
        "debris_count": len(debris_propagations),
        "protected_satellite_norad_id": protected_sat_norad,
        "window_hours": int(PROPAGATION_CONFIG.get("screening_window_hours", 72)),
        "timestep_seconds": int(PROPAGATION_CONFIG.get("timestep_seconds", 60)),
        "deterministic_epoch_utc": start_time.isoformat(),
        "collision_probability": pc,
        "collision_probability_monte_carlo": pc_mc,
        "collision_probability_abs_error": float(highest_risk.get("pc_abs_error", float("nan"))),
        "collision_probability_relative_error": float(highest_risk.get("pc_relative_error", float("nan"))),
        "collision_confidence": float(highest_risk["confidence_metric"]),
        "closest_approach_km": closest_miss_km,
        "refined_min_distance_km": refined_miss_km,
        "highest_risk_debris_norad_id": highest_risk["debris_norad_id"],
        "highest_risk_tca_utc": highest_risk["tca_utc"],
        "highest_risk_refined_tca_utc": highest_risk["refined_tca_utc"],
        "pinn_physics_mse": pinn_report.mse_physics,
        "covariance_diagonal_km2": [
            float(highest_risk.get("relative_cov_rr_m2", 0.0) / 1_000_000.0),
            float(highest_risk.get("relative_cov_tt_m2", 0.0) / 1_000_000.0),
            float(highest_risk.get("relative_cov_nn_m2", 0.0) / 1_000_000.0),
        ],
        "decision": should_burn,
        "decision_confidence": float(max(min(1.0 - uncertainty.epistemic_std, 1.0), 0.0)),
        "predicted_delta_v_km_s": selected_dv,
        "ml_predicted_delta_v_km_s": ml_pred_dv,
        "analytical_required_delta_v_km_s": float(timing["required_delta_v_km_s"]),
        "conjunction_log": str(CONJUNCTION_LOG),
        "fuel_state": {
            "current_mass_kg": float(fuel_state.current_mass_kg),
            "dry_mass_kg": float(fuel_state.dry_mass_kg),
            "remaining_propellant_kg": float(fuel_state.remaining_propellant_kg),
            "burn_count": int(fuel_state.burn_count),
            "total_delta_v_km_s": float(fuel_state.total_delta_v_km_s),
            "mass_history_kg": [float(v) for v in fuel_state.mass_history_kg],
        },
        "outputs": {"propagated_states_csv": str(propagated_csv), "conjunction_results_csv": str(conjunction_csv), "decision_json": str(decision_json)},
    }

    interactive_avoidance_xyz = None
    interactive_return_xyz = None
    if should_burn:
        thrust_vector = build_thrust_vector(selected_dv, "tangential")
        burn_solution = solve_burn_from_delta_v_vector(
            delta_v_vector_km_s=np.asarray(thrust_vector, dtype=float),
            mass_kg=float(fuel_state.current_mass_kg),
        )
        fuel_update = fuel_state.apply_delta_v_vector(np.asarray(thrust_vector, dtype=float))
        cmd = ThrustCommand(
            thrust_vector=[float(v) for v in thrust_vector],
            duration_ms=int(max(50, round(float(burn_solution["burn_time_seconds"]) * 1000.0))),
            burn_type="tangential",
        )
        output["thrust_command"] = {
            "thrust_vector": cmd.thrust_vector,
            "duration_ms": cmd.duration_ms,
            "burn_type": cmd.burn_type,
            "burn_time_s": float(timing["burn_time_s"]),
            "burn_lead_time_s": float(time_to_tca - timing["burn_time_s"]),
            "burn_execution_time_s": float(burn_solution["burn_time_seconds"]),
            "mass_before_kg": float(burn_solution["mass_before_kg"]),
            "mass_after_kg": float(burn_solution["mass_after_kg"]),
            "propellant_used_kg": float(burn_solution["propellant_used_kg"]),
            "remaining_propellant_kg": float(fuel_update["remaining_propellant_kg"]),
        }
        if protected_orbit.ndim == 2 and protected_orbit.shape[0] > 0:
            tca_idx = int(np.argmin(np.linalg.norm(protected_orbit - sat_tca.reshape(1, 3), axis=1)))
            base_leg = protected_orbit[max(tca_idx - 1, 0):]
            if base_leg.size > 0:
                viz_scale = 1800.0
                interactive_avoidance_xyz = base_leg + (np.asarray(thrust_vector, dtype=float) * viz_scale)
                interactive_return_xyz = interactive_avoidance_xyz + np.array([0.0, -selected_dv * viz_scale * 0.6, 0.0], dtype=float)
        plot_avoidance_3d(rel_pos, thrust_vector, return_delta_v=selected_dv * 0.6)
    else:
        output["thrust_command"] = None

    tca_point = (sat_tca + debris_tca) * 0.5
    if protected_orbit.ndim == 2 and protected_orbit.shape[0] > 0 and highest_risk_debris_orbit.ndim == 2 and highest_risk_debris_orbit.shape[0] > 0:
        try:
            plot_interactive_3d(
                earth_radius_km=6378.0,
                satellite_xyz=protected_orbit,
                debris_xyz=highest_risk_debris_orbit,
                tca_point=tca_point,
                avoidance_xyz=interactive_avoidance_xyz,
                return_xyz=interactive_return_xyz,
                title="LEO Encounter: Protected vs Highest-Risk Debris (Interactive)",
                output_path=PROCESSED_DATA_DIR / "real_encounter_interactive.html",
            )
        except Exception as exc:
            LOGGER.warning("interactive visualization skipped: %s", exc)

    output["fuel_state"] = {
        "current_mass_kg": float(fuel_state.current_mass_kg),
        "dry_mass_kg": float(fuel_state.dry_mass_kg),
        "remaining_propellant_kg": float(fuel_state.remaining_propellant_kg),
        "burn_count": int(fuel_state.burn_count),
        "total_delta_v_km_s": float(fuel_state.total_delta_v_km_s),
        "mass_history_kg": [float(v) for v in fuel_state.mass_history_kg],
        "burn_history": fuel_state.burn_history,
    }

    if benchmark_mode:
        output["performance_benchmark"] = _run_performance_benchmark(protected_satellite, debris_records)

    runtime_seconds = float(perf_counter() - cycle_t0)
    output["runtime_seconds"] = runtime_seconds

    # Collect structured data for API
    objects = []
    # Satellite
    if sat_valid.any():
        last_valid_idx = np.where(sat_valid)[0]
        if len(last_valid_idx) > 0:
            idx = last_valid_idx[-1]
            pos = sat_pos[idx]
            vel = sat_vel[idx]
            alt = np.linalg.norm(pos) - 6371.0
            objects.append({
                "id": protected_sat_norad,
                "type": "satellite",
                "position": pos.tolist(),
                "velocity": vel.tolist(),
                "altitude": float(alt)
            })
    # Debris
    for deb in debris_propagations:
        valid_idx = np.where(deb.valid_mask)[0]
        if len(valid_idx) > 0:
            idx = valid_idx[-1]
            pos = deb.pos_km[idx]
            vel = deb.vel_km_s[idx]
            alt = np.linalg.norm(pos) - 6371.0
            objects.append({
                "id": deb.norad_id,
                "type": "debris",
                "position": pos.tolist(),
                "velocity": vel.tolist(),
                "altitude": float(alt)
            })
    maneuvers = {
        "delta_v": output.get("predicted_delta_v_km_s", 0.0),
        "burn_time": output.get("thrust_command", {}).get("burn_execution_time_s") if output.get("thrust_command") else None,
        "direction": output.get("thrust_command", {}).get("thrust_vector") if output.get("thrust_command") else None,
        "fuel_estimate": output.get("thrust_command", {}).get("propellant_used_kg") if output.get("thrust_command") else None
    }
    output["objects"] = objects
    output["collisions"] = conjunction_rows
    output["maneuvers"] = maneuvers

    _log_structured(
        {
            "pipeline": "real",
            "tca_time_utc": str(highest_risk["refined_tca_utc"]),
            "miss_distance_km": refined_miss_km,
            "analytical_pc": pc,
            "monte_carlo_pc": pc_mc,
            "delta_v_km_s": selected_dv if should_burn else 0.0,
            "remaining_mass_kg": float(fuel_state.current_mass_kg),
            "runtime_seconds": runtime_seconds,
            "burn_count": int(fuel_state.burn_count),
        }
    )

    def make_json_safe(obj):
        if isinstance(obj, dict):
            return {k: make_json_safe(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_safe(i) for i in obj]
        elif isinstance(obj, tuple):
            return [make_json_safe(i) for i in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        elif hasattr(obj, "__dict__"):
            return make_json_safe(vars(obj))
        else:
            return obj

    safe_output = make_json_safe(output)
    with decision_json.open("w", encoding="utf-8") as f_decision:
        json.dump(safe_output, f_decision, indent=2)
    return {
        "analysis": {
            "collision_probability": pc,
            "miss_distance_km": refined_miss_km,
            "delta_v_km_s": selected_dv if should_burn else 0.0,
            "runtime_seconds": runtime_seconds,
            "tca_time": str(highest_risk["refined_tca_utc"])
        }
    }


if __name__ == "__main__":
    summary = run_autonomous_cycle(fetch_live_data=False, benchmark_mode=False)
    print(summary)
