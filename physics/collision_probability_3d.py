"""Approximate 3D Gaussian collision probability in RTN frame."""
from __future__ import annotations

import math
from typing import Any

import numpy as np


def eci_to_rtn_rotation(r_eci_km: np.ndarray, v_eci_km_s: np.ndarray) -> np.ndarray:
    r = np.asarray(r_eci_km, dtype=float)
    v = np.asarray(v_eci_km_s, dtype=float)
    r_hat = r / max(float(np.linalg.norm(r)), 1e-12)
    h = np.cross(r, v)
    n_hat = h / max(float(np.linalg.norm(h)), 1e-12)
    t_hat = np.cross(n_hat, r_hat)
    return np.vstack([r_hat, t_hat, n_hat])


def relative_state_rtn(
    sat_r_eci_km: np.ndarray,
    sat_v_eci_km_s: np.ndarray,
    rel_r_eci_km: np.ndarray,
    rel_v_eci_km_s: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    rot = eci_to_rtn_rotation(sat_r_eci_km, sat_v_eci_km_s)
    rel_r = np.asarray(rel_r_eci_km, dtype=float)
    rel_r_rtn = rot @ rel_r
    out = {"position_rtn_km": rel_r_rtn}
    if rel_v_eci_km_s is not None:
        out["velocity_rtn_km_s"] = rot @ np.asarray(rel_v_eci_km_s, dtype=float)
    return out


def _gaussian_diag_pdf_2d(x: float, y: float, mux: float, muy: float, sx: float, sy: float) -> float:
    nx = ((x - mux) / max(sx, 1e-9)) ** 2
    ny = ((y - muy) / max(sy, 1e-9)) ** 2
    denom = 2.0 * math.pi * max(sx * sy, 1e-12)
    return math.exp(-0.5 * (nx + ny)) / denom


def _gaussian_pdf_1d(z: float, muz: float, sz: float) -> float:
    denom = max(sz, 1e-9) * math.sqrt(2.0 * math.pi)
    nz = ((z - muz) / max(sz, 1e-9)) ** 2
    return math.exp(-0.5 * nz) / denom


def _bivariate_circle_probability(
    mux: float,
    muy: float,
    sx: float,
    sy: float,
    radius_m: float,
    n_rho: int,
    n_theta: int,
) -> float:
    if radius_m <= 0.0:
        return 0.0
    rhos = np.linspace(0.0, radius_m, max(n_rho, 8), dtype=float)
    thetas = np.linspace(0.0, 2.0 * math.pi, max(n_theta, 16), endpoint=False, dtype=float)

    total = 0.0
    d_theta = (2.0 * math.pi) / len(thetas)
    for theta in thetas:
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        vals = []
        for rho in rhos:
            x = float(rho * c)
            y = float(rho * s)
            vals.append(_gaussian_diag_pdf_2d(x, y, mux, muy, sx, sy) * float(rho))
        total += float(np.trapz(np.asarray(vals, dtype=float), rhos)) * d_theta
    return max(0.0, min(total, 1.0))


def collision_probability_3d_alfano(
    rel_pos_rtn_km: np.ndarray,
    sigma_rtn_m: tuple[float, float, float],
    hard_body_radius_m: float,
    integration_points_z: int = 81,
    integration_points_rho: int = 36,
    integration_points_theta: int = 72,
) -> dict[str, Any]:
    mu = np.asarray(rel_pos_rtn_km, dtype=float) * 1000.0
    sig_r, sig_t, sig_n = [max(float(s), 1e-3) for s in sigma_rtn_m]
    radius = float(max(hard_body_radius_m, 0.0))

    if radius == 0.0:
        return {"Pc": 0.0, "confidence_metric": 1.0}

    z_grid = np.linspace(-radius, radius, max(int(integration_points_z), 21), dtype=float)
    pz_vals = np.array([_gaussian_pdf_1d(float(z), float(mu[2]), sig_n) for z in z_grid], dtype=float)
    inner_probs = np.zeros_like(z_grid)
    for i, z in enumerate(z_grid):
        rho_max = math.sqrt(max(radius * radius - float(z) * float(z), 0.0))
        inner_probs[i] = _bivariate_circle_probability(
            mux=float(mu[0]),
            muy=float(mu[1]),
            sx=sig_r,
            sy=sig_t,
            radius_m=rho_max,
            n_rho=int(integration_points_rho),
            n_theta=int(integration_points_theta),
        )
    pc_fine = float(np.trapz(pz_vals * inner_probs, z_grid))

    z_coarse = np.linspace(-radius, radius, max(int(integration_points_z // 2), 11), dtype=float)
    pz_coarse = np.array([_gaussian_pdf_1d(float(z), float(mu[2]), sig_n) for z in z_coarse], dtype=float)
    inner_coarse = np.zeros_like(z_coarse)
    for i, z in enumerate(z_coarse):
        rho_max = math.sqrt(max(radius * radius - float(z) * float(z), 0.0))
        inner_coarse[i] = _bivariate_circle_probability(
            mux=float(mu[0]),
            muy=float(mu[1]),
            sx=sig_r,
            sy=sig_t,
            radius_m=rho_max,
            n_rho=max(int(integration_points_rho // 2), 10),
            n_theta=max(int(integration_points_theta // 2), 20),
        )
    pc_coarse = float(np.trapz(pz_coarse * inner_coarse, z_coarse))
    error_est = abs(pc_fine - pc_coarse)
    confidence = float(np.clip(1.0 - min(error_est / max(abs(pc_fine), 1e-8), 1.0), 0.0, 1.0))
    return {"Pc": max(0.0, min(pc_fine, 1.0)), "confidence_metric": confidence}
