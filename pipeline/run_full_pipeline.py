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
    MANEUVER_CONFIG,
    MODEL_DIR,
    PERFORMANCE_CONFIG,
    PROCESSED_DATA_DIR,
    PROPAGATION_CONFIG,
    SATELLITE_DATA_DIR,
)
from physics.maneuver import build_thrust_vector, burn_duration_ms
from physics.maneuver_optimizer import analytical_required_delta_v_km_s, optimize_burn_timing
from physics.tca_refinement import refine_tca_quadratic
from pipeline import collision_check, fetch_data
from pipeline.propulsion_interface import ThrustCommand
from results.performance_benchmark import plot_runtime_scaling, write_performance_metrics
from visualization.interactive_plot import plot_interactive_3d
from visualization.plot_avoidance import plot_avoidance_3d
from visualization.plot_orbit import plot_orbits


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)

SATELLITE_TLE_PATH = SATELLITE_DATA_DIR / "satellite_tles.json"
DEBRIS_TLE_PATH = DEBRIS_DATA_DIR / "debris_tles.json"


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

    metrics_rows = [{"debris_count": row["debris_count"], "runtime_seconds": row["runtime_seconds"]} for row in timings]
    csv_path = write_performance_metrics(metrics_rows)
    plot_path = plot_runtime_scaling(metrics_rows)
    return {"metrics": timings, "performance_metrics_csv": str(csv_path), "runtime_scaling_plot": str(plot_path)}


def run_autonomous_cycle(fetch_live_data: bool = False, benchmark_mode: bool = False) -> dict[str, Any]:
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
        refined = refine_tca_quadratic(
            time_s=time_offsets_s[valid],
            distance_km=dist[valid],
            fit_window_samples=2,
            epoch_utc=start_time,
        )
        pc_detail = collision_check.run_detailed(
            float(masked_dist[idx]),
            sat_r_eci_km=sat_pos[idx],
            sat_v_eci_km_s=sat_vel[idx],
            rel_r_eci_km=rel[idx],
            rel_v_eci_km_s=rel_vel[idx],
        )
        conjunction_rows.append(
            {
                "debris_norad_id": deb.norad_id,
                "tca_utc": epochs[idx].isoformat(),
                "refined_tca_utc": str(refined["refined_tca_time"]),
                "miss_distance_km": float(masked_dist[idx]),
                "refined_min_distance_km": float(refined["refined_min_distance"]),
                "rel_x_km": float(rel[idx, 0]),
                "rel_y_km": float(rel[idx, 1]),
                "rel_z_km": float(rel[idx, 2]),
                "collision_probability": float(pc_detail["Pc"]),
                "confidence_metric": float(pc_detail["confidence_metric"]),
                "sat_tca_km": sat_pos[idx].copy(),
                "debris_tca_km": deb.pos_km[idx].copy(),
            }
        )
    conjunction_rows.sort(key=lambda row: row["miss_distance_km"])

    with conjunction_csv.open("w", newline="", encoding="utf-8") as f_conj:
        writer = csv.writer(f_conj)
        writer.writerow(["debris_norad_id", "tca_utc", "refined_tca_utc", "miss_distance_km", "refined_min_distance_km", "rel_x_km", "rel_y_km", "rel_z_km", "collision_probability", "confidence_metric"])
        for row in conjunction_rows:
            writer.writerow([row["debris_norad_id"], row["tca_utc"], row["refined_tca_utc"], row["miss_distance_km"], row["refined_min_distance_km"], row["rel_x_km"], row["rel_y_km"], row["rel_z_km"], row["collision_probability"], row["confidence_metric"]])

    if not conjunction_rows:
        raise RuntimeError("No conjunction results produced from propagated states.")
    highest_risk = max(conjunction_rows, key=lambda row: row["collision_probability"])
    pc = float(highest_risk["collision_probability"])
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
    tca_dt = datetime.fromisoformat(str(highest_risk["tca_utc"]))
    time_to_tca = max((tca_dt - start_time).total_seconds(), 1.0)
    fuel_remaining = float(MANEUVER_CONFIG.get("fuel_remaining_default", 1.0))
    analytical_dv = analytical_required_delta_v_km_s(
        miss_distance_km=refined_miss_km,
        separation_constraint_km=float(MANEUVER_CONFIG.get("separation_constraint_km", 2.0)),
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
        separation_constraint_km=float(MANEUVER_CONFIG.get("separation_constraint_km", 2.0)),
        lead_orbits_min=float(MANEUVER_CONFIG.get("burn_lead_orbits_min", 0.1)),
        lead_orbits_max=float(MANEUVER_CONFIG.get("burn_lead_orbits_max", 2.0)),
        sweep_points=int(MANEUVER_CONFIG.get("burn_sweep_points", 24)),
        max_delta_v_km_s=float(MANEUVER_CONFIG.get("max_delta_v_km_s", 0.02)),
    )
    selected_dv = float(max(timing["required_delta_v_km_s"], ml_pred_dv))
    should_burn = bool(pc > COLLISION_PROBABILITY_THRESHOLD and selected_dv > 0.0)

    output: dict[str, Any] = {
        "satellite_tle_path": str(SATELLITE_TLE_PATH),
        "debris_tle_path": str(DEBRIS_TLE_PATH),
        "debris_count": len(debris_propagations),
        "protected_satellite_norad_id": protected_sat_norad,
        "window_hours": int(PROPAGATION_CONFIG.get("screening_window_hours", 72)),
        "timestep_seconds": int(PROPAGATION_CONFIG.get("timestep_seconds", 60)),
        "deterministic_epoch_utc": start_time.isoformat(),
        "collision_probability": pc,
        "collision_confidence": float(highest_risk["confidence_metric"]),
        "closest_approach_km": closest_miss_km,
        "refined_min_distance_km": refined_miss_km,
        "highest_risk_debris_norad_id": highest_risk["debris_norad_id"],
        "highest_risk_tca_utc": highest_risk["tca_utc"],
        "highest_risk_refined_tca_utc": highest_risk["refined_tca_utc"],
        "pinn_physics_mse": pinn_report.mse_physics,
        "covariance_diagonal_km2": [],
        "decision": should_burn,
        "decision_confidence": float(max(min(1.0 - uncertainty.epistemic_std, 1.0), 0.0)),
        "predicted_delta_v_km_s": selected_dv,
        "ml_predicted_delta_v_km_s": ml_pred_dv,
        "analytical_required_delta_v_km_s": float(timing["required_delta_v_km_s"]),
        "outputs": {"propagated_states_csv": str(propagated_csv), "conjunction_results_csv": str(conjunction_csv), "decision_json": str(decision_json)},
    }

    interactive_avoidance_xyz = None
    interactive_return_xyz = None
    if should_burn:
        thrust_vector = build_thrust_vector(selected_dv, "tangential")
        cmd = ThrustCommand(thrust_vector=[float(v) for v in thrust_vector], duration_ms=burn_duration_ms(selected_dv), burn_type="tangential")
        output["thrust_command"] = {
            "thrust_vector": cmd.thrust_vector,
            "duration_ms": cmd.duration_ms,
            "burn_type": cmd.burn_type,
            "burn_time_s": float(timing["burn_time_s"]),
            "burn_lead_time_s": float(time_to_tca - timing["burn_time_s"]),
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

    if benchmark_mode:
        output["performance_benchmark"] = _run_performance_benchmark(protected_satellite, debris_records)

    with decision_json.open("w", encoding="utf-8") as f_decision:
        json.dump(output, f_decision, indent=2)
    return output


if __name__ == "__main__":
    summary = run_autonomous_cycle(fetch_live_data=False, benchmark_mode=False)
    print(summary)
