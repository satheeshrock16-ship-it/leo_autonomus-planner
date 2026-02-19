"""End-to-end autonomous LEO collision avoidance orchestration."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sgp4.api import Satrec, jday

from ai.bnn_model import MonteCarloDropoutBNN
from ai.lightgbm_model import BurnDecisionModel
from ai.pinn_model import CWPhysicsInformedValidator
from config import (
    COLLISION_PROBABILITY_THRESHOLD,
    DEBRIS_DATA_DIR,
    PROCESSED_DATA_DIR,
    SATELLITE_DATA_DIR,
)
from pipeline import collision_check, fetch_data, replan
from pipeline.propulsion_interface import ThrustCommand
from visualization.plot_avoidance import plot_avoidance_3d
from visualization.plot_orbit import plot_orbits


SATELLITE_TLE_PATH = SATELLITE_DATA_DIR / "satellite_tles.json"
DEBRIS_TLE_PATH = DEBRIS_DATA_DIR / "debris_tles.json"
WINDOW_HOURS = 72
TIMESTEP_SECONDS = 60


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


def _sgp4_state(sat: Satrec, epoch: datetime) -> tuple[np.ndarray, np.ndarray]:
    sec = epoch.second + (epoch.microsecond / 1_000_000.0)
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, sec)
    err, r_eci_km, v_eci_km_s = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error code: {err}")
    return np.asarray(r_eci_km, dtype=float), np.asarray(v_eci_km_s, dtype=float)


def run_autonomous_cycle(fetch_live_data: bool = False) -> dict[str, Any]:
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
    first_debris_norad = str(debris_records[0].get("NORAD_CAT_ID", "UNKNOWN"))

    start_time = datetime.now(timezone.utc)
    duration_seconds = WINDOW_HOURS * 3600
    time_offsets_s = np.arange(0, duration_seconds + TIMESTEP_SECONDS, TIMESTEP_SECONDS, dtype=int)
    epochs = [start_time + timedelta(seconds=int(dt)) for dt in time_offsets_s]

    print(f"[DEBUG] Debris objects loaded: {len(debris_records)}")
    print(f"[DEBUG] First debris NORAD ID: {first_debris_norad}")
    print(f"[DEBUG] Protected satellite NORAD ID: {protected_sat_norad}")
    print(
        "[DEBUG] TLE source mode: "
        f"fetch_live_data={fetch_live_data}, "
        f"satellite_tle_path={SATELLITE_TLE_PATH}, "
        f"debris_tle_path={DEBRIS_TLE_PATH}"
    )
    print(
        "[DEBUG] SGP4 TLE source check: "
        f"sat_line1={protected_satellite.get('TLE_LINE1', '')[:25]}..., "
        f"debris_line1={debris_records[0].get('TLE_LINE1', '')[:25]}..."
    )
    print(
        "[DEBUG] Time window params: "
        f"start_utc={start_time.isoformat()}, "
        f"window_hours={WINDOW_HOURS}, "
        f"timestep_seconds={TIMESTEP_SECONDS}, "
        f"samples={len(epochs)}"
    )

    satrec_sat = _build_satrec(protected_satellite)
    debris_satrecs: list[tuple[str, Satrec]] = []
    for rec in debris_records:
        try:
            debris_satrecs.append((str(rec.get("NORAD_CAT_ID", "UNKNOWN")), _build_satrec(rec)))
        except Exception:
            continue
    if not debris_satrecs:
        raise RuntimeError("No valid debris TLE lines available for SGP4 propagation.")

    print(f"[DEBUG] Debris objects accepted for SGP4: {len(debris_satrecs)}")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    propagated_csv = PROCESSED_DATA_DIR / "propagated_states.csv"
    conjunction_csv = PROCESSED_DATA_DIR / "conjunction_results.csv"
    decision_json = PROCESSED_DATA_DIR / "decision.json"
    print(
        "[DEBUG] Processed outputs overwrite targets: "
        f"{propagated_csv}, {conjunction_csv}, {decision_json}"
    )

    min_distance_by_debris: dict[str, float] = {}
    tca_by_debris: dict[str, datetime] = {}
    rel_at_tca_by_debris: dict[str, np.ndarray] = {}
    sat_at_tca_by_debris: dict[str, np.ndarray] = {}
    debris_at_tca_by_debris: dict[str, np.ndarray] = {}
    sat_trajectory_km: list[np.ndarray] = []
    debris_trajectories_km: dict[str, list[np.ndarray]] = {}
    sat_rel_positions: list[np.ndarray] = []
    sat_rel_velocities: list[np.ndarray] = []

    previous_best_rel = None
    with propagated_csv.open("w", newline="", encoding="utf-8") as f_states:
        writer = csv.writer(f_states)
        writer.writerow(["epoch_utc", "object_role", "norad_cat_id", "x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"])

        for epoch in epochs:
            sat_r, sat_v = _sgp4_state(satrec_sat, epoch)
            sat_trajectory_km.append(sat_r.copy())
            writer.writerow(
                [
                    epoch.isoformat(),
                    "satellite",
                    protected_sat_norad,
                    sat_r[0],
                    sat_r[1],
                    sat_r[2],
                    sat_v[0],
                    sat_v[1],
                    sat_v[2],
                ]
            )

            best_rel_now = None
            best_dist_now = float("inf")
            for debris_norad, debris_sat in debris_satrecs:
                try:
                    deb_r, deb_v = _sgp4_state(debris_sat, epoch)
                except Exception:
                    continue

                writer.writerow(
                    [
                        epoch.isoformat(),
                        "debris",
                        debris_norad,
                        deb_r[0],
                        deb_r[1],
                        deb_r[2],
                        deb_v[0],
                        deb_v[1],
                        deb_v[2],
                    ]
                )

                rel = deb_r - sat_r
                distance = float(np.linalg.norm(rel))
                debris_trajectories_km.setdefault(debris_norad, []).append(deb_r.copy())
                prior = min_distance_by_debris.get(debris_norad, float("inf"))
                if distance < prior:
                    min_distance_by_debris[debris_norad] = distance
                    tca_by_debris[debris_norad] = epoch
                    rel_at_tca_by_debris[debris_norad] = rel
                    sat_at_tca_by_debris[debris_norad] = sat_r.copy()
                    debris_at_tca_by_debris[debris_norad] = deb_r.copy()

                if distance < best_dist_now:
                    best_dist_now = distance
                    best_rel_now = rel

            if best_rel_now is not None:
                sat_rel_positions.append(best_rel_now)
                if previous_best_rel is None:
                    sat_rel_velocities.append(np.zeros(3))
                else:
                    sat_rel_velocities.append((best_rel_now - previous_best_rel) / TIMESTEP_SECONDS)
                previous_best_rel = best_rel_now
            else:
                sat_rel_positions.append(np.zeros(3))
                sat_rel_velocities.append(np.zeros(3))

    conjunction_rows: list[dict[str, Any]] = []
    for norad_id, miss_distance in min_distance_by_debris.items():
        rel_tca = rel_at_tca_by_debris[norad_id]
        pc_each = collision_check.run(miss_distance)
        conjunction_rows.append(
            {
                "debris_norad_id": norad_id,
                "tca_utc": tca_by_debris[norad_id].isoformat(),
                "miss_distance_km": miss_distance,
                "rel_x_km": float(rel_tca[0]),
                "rel_y_km": float(rel_tca[1]),
                "rel_z_km": float(rel_tca[2]),
                "collision_probability": float(pc_each),
            }
        )
    conjunction_rows.sort(key=lambda row: row["miss_distance_km"])

    with conjunction_csv.open("w", newline="", encoding="utf-8") as f_conj:
        writer = csv.writer(f_conj)
        writer.writerow(
            [
                "debris_norad_id",
                "tca_utc",
                "miss_distance_km",
                "rel_x_km",
                "rel_y_km",
                "rel_z_km",
                "collision_probability",
            ]
        )
        for row in conjunction_rows:
            writer.writerow(
                [
                    row["debris_norad_id"],
                    row["tca_utc"],
                    row["miss_distance_km"],
                    row["rel_x_km"],
                    row["rel_y_km"],
                    row["rel_z_km"],
                    row["collision_probability"],
                ]
            )

    if not conjunction_rows:
        raise RuntimeError("No conjunction results produced from propagated states.")

    highest_risk = max(conjunction_rows, key=lambda row: row["collision_probability"])
    pc = float(highest_risk["collision_probability"])
    highest_risk_norad = str(highest_risk["debris_norad_id"])

    protected_orbit = np.asarray(sat_trajectory_km, dtype=float)
    highest_risk_debris_orbit = np.asarray(debris_trajectories_km.get(highest_risk_norad, []), dtype=float)
    sat_tca = sat_at_tca_by_debris.get(highest_risk_norad)
    debris_tca = debris_at_tca_by_debris.get(highest_risk_norad)
    if (
        protected_orbit.ndim == 2
        and protected_orbit.shape[0] > 0
        and highest_risk_debris_orbit.ndim == 2
        and highest_risk_debris_orbit.shape[0] > 0
        and sat_tca is not None
        and debris_tca is not None
    ):
        plot_orbits(
            protected_orbit,
            highest_risk_debris_orbit,
            title="LEO Encounter: Protected vs Highest-Risk Debris",
            tca_satellite_km=sat_tca,
            tca_debris_km=debris_tca,
        )

    rel_pos = np.asarray(sat_rel_positions, dtype=float)
    rel_vel = np.asarray(sat_rel_velocities, dtype=float)
    if rel_pos.size == 0:
        rel_pos = np.zeros((2, 3), dtype=float)
        rel_vel = np.zeros((2, 3), dtype=float)
    elif rel_pos.shape[0] == 1:
        rel_pos = np.vstack([rel_pos, rel_pos])
        rel_vel = np.vstack([rel_vel, rel_vel])
    dt = float(TIMESTEP_SECONDS)
    n_est = 0.0011
    pinn_report = CWPhysicsInformedValidator().evaluate(rel_pos, rel_vel, n=n_est, dt=dt)

    bnn = MonteCarloDropoutBNN()
    uncertainty = bnn.predict(pc)

    decision_model = BurnDecisionModel()
    X = np.array([[0.1, 0.2, 0.2, 0.3], [0.8, 0.7, 0.3, 0.5], [0.9, 0.1, 0.1, 0.1]])
    y = np.array([0, 1, 1])
    decision_model.fit(X, y)
    closest_miss_km = float(highest_risk["miss_distance_km"])
    decision = decision_model.predict(np.array([pc, uncertainty.epistemic_std, closest_miss_km, 0.3]))

    output: dict[str, Any] = {
        "satellite_tle_path": str(SATELLITE_TLE_PATH),
        "debris_tle_path": str(DEBRIS_TLE_PATH),
        "debris_count": len(debris_satrecs),
        "protected_satellite_norad_id": protected_sat_norad,
        "first_debris_norad_id": first_debris_norad,
        "window_hours": WINDOW_HOURS,
        "timestep_seconds": TIMESTEP_SECONDS,
        "collision_probability": pc,
        "closest_approach_km": closest_miss_km,
        "highest_risk_debris_norad_id": highest_risk["debris_norad_id"],
        "highest_risk_tca_utc": highest_risk["tca_utc"],
        "pinn_physics_mse": pinn_report.mse_physics,
        "covariance_diagonal_km2": [],
        "decision": decision.should_burn,
        "decision_confidence": decision.confidence,
        "outputs": {
            "propagated_states_csv": str(propagated_csv),
            "conjunction_results_csv": str(conjunction_csv),
            "decision_json": str(decision_json),
        },
    }

    if pc > COLLISION_PROBABILITY_THRESHOLD and decision.should_burn:
        plan = replan.run(current_radius_km=6878.0, nominal_radius_km=6878.0)
        cmd = ThrustCommand(
            thrust_vector=[float(v) for v in plan["thrust_vector"]],
            duration_ms=plan["duration_ms"],
            burn_type="tangential",
        )
        output["thrust_command"] = {
            "thrust_vector": cmd.thrust_vector,
            "duration_ms": cmd.duration_ms,
            "burn_type": cmd.burn_type,
        }
        plot_avoidance_3d(rel_pos, plan["thrust_vector"], return_delta_v=plan["return_delta_v_km_s"])
    else:
        output["thrust_command"] = None

    with decision_json.open("w", encoding="utf-8") as f_decision:
        json.dump(output, f_decision, indent=2)

    return output


if __name__ == "__main__":
    summary = run_autonomous_cycle(fetch_live_data=False)
    print(summary)
