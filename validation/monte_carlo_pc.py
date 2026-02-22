"""Monte Carlo validation for analytical conjunction collision probability."""
from __future__ import annotations

from typing import Any

import numpy as np

from physics.cw_equation import cw_state_transition_matrix


def _stable_covariance(cov: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    c = np.asarray(cov, dtype=float)
    c = 0.5 * (c + c.T)
    w, v = np.linalg.eigh(c)
    w = np.clip(w, floor, None)
    return v @ np.diag(w) @ v.T


def monte_carlo_validate_pc(
    analytical_pc: float,
    mean_state_rtn: np.ndarray,
    covariance_rtn: np.ndarray,
    hard_body_radius_m: float,
    samples: int = 5000,
    mean_motion_rad_s: float = 0.0,
    dt_s: float = 0.0,
    seed: int | None = None,
) -> dict[str, Any]:
    """Validate analytical Pc with random sampling.

    `mean_state_rtn` and `covariance_rtn` may be 3D position-only (km, km^2) or
    6D state ([km, km/s], covariance in matching units).
    """
    mean = np.asarray(mean_state_rtn, dtype=float).reshape(-1)
    cov = _stable_covariance(np.asarray(covariance_rtn, dtype=float))
    dim = int(mean.shape[0])
    if cov.shape != (dim, dim):
        raise ValueError("covariance shape must match mean_state_rtn dimension.")

    n_samples = max(int(samples), 100)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=mean, cov=cov, size=n_samples)

    if dim == 6 and abs(float(dt_s)) > 1e-12:
        phi = cw_state_transition_matrix(n=float(mean_motion_rad_s), t_s=float(dt_s))
        draws = (phi @ draws.T).T
    elif dim not in (3, 6):
        raise ValueError("mean_state_rtn must have dimension 3 or 6.")

    pos = draws[:, :3]
    hard_body_radius_km = float(max(hard_body_radius_m, 0.0) / 1000.0)
    hits = int(np.count_nonzero(np.linalg.norm(pos, axis=1) <= hard_body_radius_km))
    pc_mc = float(hits / n_samples)

    analytical = float(np.clip(analytical_pc, 0.0, 1.0))
    abs_error = float(abs(pc_mc - analytical))
    rel_error = float(abs_error / max(abs(analytical), 1e-12))
    return {
        "analytical_pc": analytical,
        "monte_carlo_pc": pc_mc,
        "absolute_error": abs_error,
        "relative_error": rel_error,
        "samples": n_samples,
        "hits": hits,
    }
