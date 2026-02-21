"""Sub-second TCA refinement using quadratic fit of squared miss distance."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import numpy as np


def refine_tca_quadratic(
    time_s: np.ndarray,
    distance_km: np.ndarray,
    min_index: int | None = None,
    fit_window_samples: int = 2,
    epoch_utc: datetime | None = None,
) -> dict[str, Any]:
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
