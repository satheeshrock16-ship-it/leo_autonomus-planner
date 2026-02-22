"""Sub-step TCA refinement utilities."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np

from physics.constants import MU


def _two_body_accel(r_km: np.ndarray, mu_km3_s2: float = MU) -> np.ndarray:
    r = np.asarray(r_km, dtype=float)
    r_norm = float(np.linalg.norm(r))
    return -(float(mu_km3_s2) / max(r_norm**3, 1e-12)) * r


def propagate_two_body_state(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    dt_s: float,
    mu_km3_s2: float = MU,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one state with fixed-step RK4 for short refinement windows."""
    r = np.asarray(r0_km, dtype=float).copy()
    v = np.asarray(v0_km_s, dtype=float).copy()
    dt_total = float(dt_s)
    if abs(dt_total) <= 1e-12:
        return r, v
    sign = 1.0 if dt_total >= 0.0 else -1.0
    remaining = abs(dt_total)
    step = min(2.0, remaining)
    while remaining > 1e-12:
        h = sign * min(step, remaining)
        k1_r = v
        k1_v = _two_body_accel(r, mu_km3_s2=mu_km3_s2)
        k2_r = v + 0.5 * h * k1_v
        k2_v = _two_body_accel(r + 0.5 * h * k1_r, mu_km3_s2=mu_km3_s2)
        k3_r = v + 0.5 * h * k2_v
        k3_v = _two_body_accel(r + 0.5 * h * k2_r, mu_km3_s2=mu_km3_s2)
        k4_r = v + h * k3_v
        k4_v = _two_body_accel(r + h * k3_r, mu_km3_s2=mu_km3_s2)
        r = r + (h / 6.0) * (k1_r + 2.0 * k2_r + 2.0 * k3_r + k4_r)
        v = v + (h / 6.0) * (k1_v + 2.0 * k2_v + 2.0 * k3_v + k4_v)
        remaining -= abs(h)
    return r, v


def refine_tca_analytic(
    time_s: np.ndarray,
    sat_pos_km: np.ndarray,
    sat_vel_km_s: np.ndarray,
    debris_pos_km: np.ndarray,
    debris_vel_km_s: np.ndarray,
    min_index: int | None = None,
    search_window_s: float | None = None,
    epoch_utc: datetime | None = None,
) -> dict[str, Any]:
    """Refine TCA from coarse screening using analytical relative-motion minimizer.

    For coarse relative state (r0, v0) this computes:
        t* = - (r0 . v0) / (v0 . v0)
    then bounds to [0, search_window] and performs precise two-body propagation.
    """
    t = np.asarray(time_s, dtype=float)
    sat_pos = np.asarray(sat_pos_km, dtype=float)
    sat_vel = np.asarray(sat_vel_km_s, dtype=float)
    deb_pos = np.asarray(debris_pos_km, dtype=float)
    deb_vel = np.asarray(debris_vel_km_s, dtype=float)

    if t.ndim != 1:
        raise ValueError("time_s must be a 1D array.")
    if sat_pos.shape != deb_pos.shape or sat_vel.shape != deb_vel.shape:
        raise ValueError("satellite/debris position and velocity arrays must have matching shapes.")
    if sat_pos.shape != (len(t), 3) or sat_vel.shape != (len(t), 3):
        raise ValueError("sat_pos/sat_vel/debris_pos/debris_vel must be shaped (N, 3).")

    rel = deb_pos - sat_pos
    dist = np.linalg.norm(rel, axis=1)
    idx = int(np.argmin(dist)) if min_index is None else int(min_index)
    idx = max(0, min(idx, len(t) - 1))

    if len(t) == 1:
        rel_v = deb_vel[idx] - sat_vel[idx]
        out_time = float(t[idx]) if epoch_utc is None else epoch_utc + timedelta(seconds=float(t[idx]))
        return {
            "method": "analytic",
            "coarse_index": idx,
            "refined_index": idx,
            "coarse_time_s": float(t[idx]),
            "refined_time_s": float(t[idx]),
            "coarse_to_refined_dt_s": 0.0,
            "refined_tca_time": out_time,
            "refined_min_distance": float(dist[idx]),
            "sat_tca_km": sat_pos[idx].copy(),
            "sat_tca_vel_km_s": sat_vel[idx].copy(),
            "debris_tca_km": deb_pos[idx].copy(),
            "debris_tca_vel_km_s": deb_vel[idx].copy(),
            "rel_tca_km": rel[idx].copy(),
            "rel_vel_tca_km_s": rel_v.copy(),
        }

    if search_window_s is None:
        if idx >= len(t) - 1:
            dt_grid = max(float(t[idx] - t[idx - 1]), 0.0)
        else:
            dt_grid = max(float(t[idx + 1] - t[idx]), 0.0)
    else:
        dt_grid = max(float(search_window_s), 0.0)
    if dt_grid <= 1e-12:
        dt_grid = 0.0

    rel_r0 = rel[idx]
    rel_v0 = deb_vel[idx] - sat_vel[idx]
    vv = float(np.dot(rel_v0, rel_v0))
    t_star = 0.0 if vv <= 1e-18 else float(-(np.dot(rel_r0, rel_v0)) / vv)
    if not np.isfinite(t_star):
        t_star = 0.0
    in_window = bool(0.0 <= t_star <= dt_grid)
    dt_refine = t_star if in_window else 0.0

    sat_r_ref, sat_v_ref = propagate_two_body_state(
        r0_km=sat_pos[idx],
        v0_km_s=sat_vel[idx],
        dt_s=dt_refine,
        mu_km3_s2=MU,
    )
    deb_r_ref, deb_v_ref = propagate_two_body_state(
        r0_km=deb_pos[idx],
        v0_km_s=deb_vel[idx],
        dt_s=dt_refine,
        mu_km3_s2=MU,
    )
    rel_ref = deb_r_ref - sat_r_ref
    rel_vel_ref = deb_v_ref - sat_v_ref
    refined_time_s = float(t[idx] + dt_refine)
    out_time = refined_time_s if epoch_utc is None else epoch_utc + timedelta(seconds=refined_time_s)

    return {
        "method": "analytic",
        "coarse_index": idx,
        "refined_index": idx,
        "coarse_time_s": float(t[idx]),
        "refined_time_s": refined_time_s,
        "coarse_to_refined_dt_s": float(dt_refine),
        "t_star_s": float(t_star),
        "search_window_s": float(dt_grid),
        "in_window": in_window,
        "refined_tca_time": out_time,
        "refined_min_distance": float(np.linalg.norm(rel_ref)),
        "sat_tca_km": sat_r_ref,
        "sat_tca_vel_km_s": sat_v_ref,
        "debris_tca_km": deb_r_ref,
        "debris_tca_vel_km_s": deb_v_ref,
        "rel_tca_km": rel_ref,
        "rel_vel_tca_km_s": rel_vel_ref,
    }


def refine_tca_quadratic(
    time_s: np.ndarray,
    distance_km: np.ndarray,
    min_index: int | None = None,
    fit_window_samples: int = 2,
    epoch_utc: datetime | None = None,
) -> dict[str, Any]:
    """Legacy quadratic fit refinement kept for compatibility/tests."""
    t = np.asarray(time_s, dtype=float)
    d = np.asarray(distance_km, dtype=float)
    if t.ndim != 1 or d.ndim != 1 or len(t) != len(d):
        raise ValueError("time_s and distance_km must be 1D arrays with equal length.")
    if len(t) < 3:
        idx = int(np.argmin(d))
        refined_time_s = float(t[idx])
        out_time = refined_time_s if epoch_utc is None else epoch_utc + timedelta(seconds=refined_time_s)
        return {"refined_tca_time": out_time, "refined_min_distance": float(d[idx])}

    idx_min = int(np.argmin(d)) if min_index is None else int(min_index)
    idx_min = max(0, min(idx_min, len(t) - 1))
    i0 = max(0, idx_min - int(fit_window_samples))
    i1 = min(len(t), idx_min + int(fit_window_samples) + 1)
    if i1 - i0 < 3:
        i0 = max(0, idx_min - 1)
        i1 = min(len(t), idx_min + 2)

    tw = t[i0:i1]
    d2w = np.square(d[i0:i1])
    coeffs = np.polyfit(tw, d2w, deg=2)
    a, b, c = [float(v) for v in coeffs]
    if abs(a) < 1e-14:
        t_star = float(t[idx_min])
    else:
        t_star = float(-b / (2.0 * a))
    t_star = float(np.clip(t_star, float(tw[0]), float(tw[-1])))
    d2_star = float((a * t_star * t_star) + (b * t_star) + c)
    d_star = float(np.sqrt(max(d2_star, 0.0)))
    local_min = float(np.min(d[i0:i1]))
    if not np.isfinite(d_star) or d_star > (local_min * 1.5 + 1e-9):
        t_star = float(t[idx_min])
        d_star = float(d[idx_min])

    out_time = t_star if epoch_utc is None else epoch_utc + timedelta(seconds=t_star)
    return {"refined_tca_time": out_time, "refined_min_distance": d_star}
