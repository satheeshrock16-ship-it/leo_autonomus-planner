"""3D orbit plotting utilities."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR, PROCESSED_DATA_DIR


EARTH_RADIUS_KM = 6378.0


def _get_first(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _plot_earth(ax, radius_km: float = EARTH_RADIUS_KM) -> Any:
    u = np.linspace(0.0, 2.0 * np.pi, 60)
    v = np.linspace(0.0, np.pi, 30)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightsteelblue", alpha=0.5, linewidth=0.0)
    return plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightsteelblue", markersize=10, label="Earth")


def _set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        span = EARTH_RADIUS_KM * 1.2
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(-span, span)
        ax.set_box_aspect((1, 1, 1))
        return

    x_vals = points[:, 0]
    y_vals = points[:, 1]
    z_vals = points[:, 2]

    x_range = float(x_vals.max() - x_vals.min())
    y_range = float(y_vals.max() - y_vals.min())
    z_range = float(z_vals.max() - z_vals.min())
    max_range = max(x_range, y_range, z_range, 2.0 * EARTH_RADIUS_KM)

    x_mid = float((x_vals.max() + x_vals.min()) * 0.5)
    y_mid = float((y_vals.max() + y_vals.min()) * 0.5)
    z_mid = float((z_vals.max() + z_vals.min()) * 0.5)

    half = max_range * 0.5
    ax.set_xlim(x_mid - half, x_mid + half)
    ax.set_ylim(y_mid - half, y_mid + half)
    ax.set_zlim(z_mid - half, z_mid + half)
    ax.set_box_aspect((1, 1, 1))


def _load_propagated_states(path: Path) -> tuple[np.ndarray, dict[str, np.ndarray], list[str], dict[str, dict[str, np.ndarray]]]:
    satellite_points: list[list[float]] = []
    satellite_times: list[str] = []
    debris_points_by_id: dict[str, list[list[float]]] = {}
    debris_times_by_id: dict[str, list[str]] = {}

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            object_type = str(_get_first(row, ["object_type", "object_role"], "")).strip().lower()
            obj_id = str(_get_first(row, ["norad_id", "norad_cat_id", "object_id"], "UNKNOWN")).strip()
            time_tag = str(_get_first(row, ["time", "epoch_utc", "time_s"], "")).strip()

            try:
                point = [
                    float(_get_first(row, ["x_km"])),
                    float(_get_first(row, ["y_km"])),
                    float(_get_first(row, ["z_km"])),
                ]
            except (TypeError, ValueError):
                continue

            if object_type == "satellite":
                satellite_points.append(point)
                satellite_times.append(time_tag)
            elif object_type == "debris":
                debris_points_by_id.setdefault(obj_id, []).append(point)
                debris_times_by_id.setdefault(obj_id, []).append(time_tag)

    sat = np.asarray(satellite_points, dtype=float) if satellite_points else np.empty((0, 3), dtype=float)
    debris = {
        obj_id: np.asarray(points, dtype=float)
        for obj_id, points in debris_points_by_id.items()
        if points
    }
    debris_time_maps: dict[str, dict[str, np.ndarray]] = {}
    for obj_id, points in debris.items():
        times = debris_times_by_id.get(obj_id, [])
        debris_time_maps[obj_id] = {times[i]: points[i] for i in range(min(len(times), len(points)))}

    return sat, debris, satellite_times, debris_time_maps


def _parse_conjunction(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "highest_risk_id": None,
        "tca_time": None,
        "rel_vec": None,
    }

    if not path.exists():
        return result

    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            result["highest_risk_id"] = str(payload.get("highest_risk_debris_norad_id") or payload.get("debris_norad_id") or "") or None
            result["tca_time"] = payload.get("highest_risk_tca_utc") or payload.get("tca_utc") or payload.get("time")
            rx = payload.get("rel_x_km")
            ry = payload.get("rel_y_km")
            rz = payload.get("rel_z_km")
            if rx is not None and ry is not None and rz is not None:
                result["rel_vec"] = np.array([float(rx), float(ry), float(rz)], dtype=float)
        return result

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                miss_km = float(_get_first(row, ["miss_distance_km"], "inf"))
            except ValueError:
                continue
            rows.append({
                "miss_distance_km": miss_km,
                "highest_risk_id": str(_get_first(row, ["debris_norad_id", "norad_id", "object_id"], "") or "") or None,
                "tca_time": _get_first(row, ["tca_utc", "time", "tca_time_s"]),
                "rel_x_km": _get_first(row, ["rel_x_km"]),
                "rel_y_km": _get_first(row, ["rel_y_km"]),
                "rel_z_km": _get_first(row, ["rel_z_km"]),
            })

    if not rows:
        return result

    best = min(rows, key=lambda item: item["miss_distance_km"])
    result["highest_risk_id"] = best["highest_risk_id"]
    result["tca_time"] = best["tca_time"]
    if best["rel_x_km"] is not None and best["rel_y_km"] is not None and best["rel_z_km"] is not None:
        result["rel_vec"] = np.array([float(best["rel_x_km"]), float(best["rel_y_km"]), float(best["rel_z_km"])], dtype=float)
    return result


def _compute_tca_and_index(
    sat: np.ndarray,
    sat_times: list[str],
    debris_track: np.ndarray,
    debris_time_map: dict[str, np.ndarray],
    conj_meta: dict[str, Any],
) -> tuple[np.ndarray | None, int]:
    tca_time = conj_meta.get("tca_time")
    rel_vec = conj_meta.get("rel_vec")

    sat_time_map = {sat_times[i]: sat[i] for i in range(min(len(sat_times), len(sat)))}
    if tca_time is not None:
        tca_time = str(tca_time)
        sat_at_tca = sat_time_map.get(tca_time)
        deb_at_tca = debris_time_map.get(tca_time)
        if sat_at_tca is not None:
            if rel_vec is not None:
                return sat_at_tca + (0.5 * rel_vec), sat_times.index(tca_time) if tca_time in sat_times else len(sat) // 2
            if deb_at_tca is not None:
                return (sat_at_tca + deb_at_tca) * 0.5, sat_times.index(tca_time) if tca_time in sat_times else len(sat) // 2

    n = min(len(sat), len(debris_track))
    if n == 0:
        return None, 0
    rel = debris_track[:n] - sat[:n]
    idx = int(np.argmin(np.linalg.norm(rel, axis=1)))
    return (sat[idx] + debris_track[idx]) * 0.5, idx


def _load_maneuver_overlay(conjunction_results_path: Path) -> tuple[bool, np.ndarray, float]:
    decision_path = conjunction_results_path.with_name("decision.json")
    if not decision_path.exists():
        return False, np.zeros(3, dtype=float), 0.0

    try:
        with decision_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return False, np.zeros(3, dtype=float), 0.0

    thrust = payload.get("thrust_command") if isinstance(payload, dict) else None
    if not thrust:
        return False, np.zeros(3, dtype=float), 0.0

    thrust_vector = np.asarray(thrust.get("thrust_vector", [0.0, 0.0, 0.0]), dtype=float)
    return_delta_v = float(thrust.get("return_delta_v_km_s", 0.0))
    return True, thrust_vector, return_delta_v


def plot_real_encounter_full(
    propagated_states_path: str | Path,
    conjunction_results_path: str | Path,
    output_path: str | Path,
) -> str:
    propagated_states_path = Path(propagated_states_path)
    conjunction_results_path = Path(conjunction_results_path)
    output_path = Path(output_path)

    sat, debris_by_id, sat_times, debris_time_maps = _load_propagated_states(propagated_states_path)
    if sat.size == 0:
        raise RuntimeError(f"No protected satellite trajectory found in {propagated_states_path}")

    conj_meta = _parse_conjunction(conjunction_results_path)
    highest_risk_id = conj_meta.get("highest_risk_id")

    if highest_risk_id and highest_risk_id in debris_by_id:
        highest_risk_track = debris_by_id[highest_risk_id]
        highest_risk_time_map = debris_time_maps.get(highest_risk_id, {})
    elif debris_by_id:
        first_id = next(iter(debris_by_id.keys()))
        highest_risk_track = debris_by_id[first_id]
        highest_risk_time_map = debris_time_maps.get(first_id, {})
    else:
        highest_risk_track = np.empty((0, 3), dtype=float)
        highest_risk_time_map = {}

    tca_point, tca_idx = _compute_tca_and_index(sat, sat_times, highest_risk_track, highest_risk_time_map, conj_meta)

    maneuver_triggered, thrust_vector, return_delta_v_km_s = _load_maneuver_overlay(conjunction_results_path)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    earth_handle = _plot_earth(ax)

    sat_handle, = ax.plot(
        sat[:, 0],
        sat[:, 1],
        sat[:, 2],
        color="blue",
        linewidth=4.0,
        label="Protected Orbit",
    )

    debris_handle = None
    debris_tracks = [track for track in debris_by_id.values() if track.size > 0]
    if debris_tracks:
        debris_points = np.vstack(debris_tracks)
        debris_handle = ax.scatter(
            debris_points[:, 0],
            debris_points[:, 1],
            debris_points[:, 2],
            color="red",
            s=9,
            alpha=0.7,
            label="Debris",
        )

    tca_handle = None
    if tca_point is not None:
        tca_handle = ax.scatter(
            tca_point[0],
            tca_point[1],
            tca_point[2],
            color="red",
            marker="x",
            s=110,
            linewidths=2.2,
            label="TCA",
        )

    avoidance_handle = None
    return_handle = None
    if maneuver_triggered and len(sat) > 1:
        tca_idx = int(max(0, min(tca_idx, len(sat) - 1)))
        scale = 900.0
        avoidance = sat.copy()
        avoidance[tca_idx:] = sat[tca_idx:] + (thrust_vector * scale)
        return_track = avoidance.copy()
        return_track[tca_idx:] = avoidance[tca_idx:] + np.array([0.0, -return_delta_v_km_s * scale, 0.0], dtype=float)

        avoidance_handle, = ax.plot(
            avoidance[tca_idx:, 0],
            avoidance[tca_idx:, 1],
            avoidance[tca_idx:, 2],
            color="green",
            linewidth=2.0,
            label="Avoidance Arc",
        )
        return_handle, = ax.plot(
            return_track[tca_idx:, 0],
            return_track[tca_idx:, 1],
            return_track[tca_idx:, 2],
            color="purple",
            linewidth=2.0,
            label="Return-to-Orbit",
        )

    point_sets = [sat, np.array([[0.0, 0.0, 0.0]], dtype=float)]
    point_sets.extend([track for track in debris_by_id.values() if track.size > 0])
    if highest_risk_track.size > 0:
        point_sets.append(highest_risk_track)
    if tca_point is not None:
        point_sets.append(tca_point.reshape(1, 3))
    all_points = np.vstack(point_sets)
    _set_equal_axes(ax, all_points)

    ax.set_title("LEO Real Encounter - Full Propagated View")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    handles = [earth_handle, sat_handle]
    if debris_handle is not None:
        handles.append(debris_handle)
    if tca_handle is not None:
        handles.append(tca_handle)
    if avoidance_handle is not None:
        handles.append(avoidance_handle)
    if return_handle is not None:
        handles.append(return_handle)
    ax.legend(handles=handles, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170)
    plt.close(fig)
    return str(output_path)


def plot_orbits(
    protected_orbit: np.ndarray,
    debris_orbit: np.ndarray,
    title: str = "LEO Encounter",
    tca_satellite_km: np.ndarray | None = None,
    tca_debris_km: np.ndarray | None = None,
) -> str:
    del protected_orbit, debris_orbit, title, tca_satellite_km, tca_debris_km
    return plot_real_encounter_full(
        propagated_states_path=PROCESSED_DATA_DIR / "propagated_states.csv",
        conjunction_results_path=PROCESSED_DATA_DIR / "conjunction_results.csv",
        output_path=PROCESSED_DATA_DIR / "real_encounter_full.png",
    )
