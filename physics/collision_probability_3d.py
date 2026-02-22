"""Covariance-aware 3D collision probability in RTN with encounter-plane integration."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from physics.cw_equation import propagate_covariance_cw


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


def build_full_covariance_rtn_m2(
    sigma_r_m: float,
    sigma_t_m: float,
    sigma_n_m: float,
    rho_rt: float = 0.0,
    rho_rn: float = 0.0,
    rho_tn: float = 0.0,
) -> np.ndarray:
    sr = max(float(sigma_r_m), 1e-6)
    st = max(float(sigma_t_m), 1e-6)
    sn = max(float(sigma_n_m), 1e-6)
    rrt = float(np.clip(rho_rt, -0.99, 0.99))
    rrn = float(np.clip(rho_rn, -0.99, 0.99))
    rtn = float(np.clip(rho_tn, -0.99, 0.99))
    cov = np.array(
        [
            [sr * sr, rrt * sr * st, rrn * sr * sn],
            [rrt * sr * st, st * st, rtn * st * sn],
            [rrn * sr * sn, rtn * st * sn, sn * sn],
        ],
        dtype=float,
    )
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-6, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def build_state_covariance_rtn_km_units(
    position_cov_rtn_m2: np.ndarray,
    sigma_v_rtn_m_s: tuple[float, float, float],
) -> np.ndarray:
    p_pos_m2 = np.asarray(position_cov_rtn_m2, dtype=float)
    if p_pos_m2.shape != (3, 3):
        raise ValueError("position_cov_rtn_m2 must be shape (3, 3).")
    sv = np.maximum(np.asarray(sigma_v_rtn_m_s, dtype=float), 1e-6)
    p_pos_km2 = p_pos_m2 / 1_000_000.0
    p_vel_km2_s2 = np.diag((sv / 1000.0) ** 2)
    out = np.zeros((6, 6), dtype=float)
    out[:3, :3] = p_pos_km2
    out[3:, 3:] = p_vel_km2_s2
    out = 0.5 * (out + out.T)
    return out


def propagate_relative_covariance_rtn_km2(
    sat_cov_rtn_m2: np.ndarray,
    debris_cov_rtn_m2: np.ndarray,
    sat_sigma_v_rtn_m_s: tuple[float, float, float],
    debris_sigma_v_rtn_m_s: tuple[float, float, float],
    mean_motion_rad_s: float,
    dt_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_sat = build_state_covariance_rtn_km_units(sat_cov_rtn_m2, sat_sigma_v_rtn_m_s)
    p_deb = build_state_covariance_rtn_km_units(debris_cov_rtn_m2, debris_sigma_v_rtn_m_s)
    if abs(float(dt_s)) > 1e-12:
        p_sat = propagate_covariance_cw(p_sat, n=float(mean_motion_rad_s), t_s=float(dt_s))
        p_deb = propagate_covariance_cw(p_deb, n=float(mean_motion_rad_s), t_s=float(dt_s))
    p_rel = p_sat + p_deb
    p_rel = 0.5 * (p_rel + p_rel.T)
    return p_sat, p_deb, p_rel


def encounter_plane_basis(rel_velocity_vec: np.ndarray) -> np.ndarray:
    v = np.asarray(rel_velocity_vec, dtype=float)
    v_norm = float(np.linalg.norm(v))
    if v_norm <= 1e-12:
        v_hat = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        v_hat = v / v_norm
    ref = np.array([1.0, 0.0, 0.0], dtype=float) if abs(float(np.dot(v_hat, [1.0, 0.0, 0.0]))) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=float)
    n1 = np.cross(v_hat, ref)
    n1_norm = float(np.linalg.norm(n1))
    if n1_norm <= 1e-12:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        n1 = np.cross(v_hat, ref)
        n1_norm = max(float(np.linalg.norm(n1)), 1e-12)
    n1 = n1 / n1_norm
    n2 = np.cross(v_hat, n1)
    n2 = n2 / max(float(np.linalg.norm(n2)), 1e-12)
    return np.vstack([n1, n2, v_hat])


def _stable_2x2_cov(cov_2x2: np.ndarray) -> np.ndarray:
    cov = np.asarray(cov_2x2, dtype=float)
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-9, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def _integrate_gaussian_circle(
    mean_xy_m: np.ndarray,
    cov_xy_m2: np.ndarray,
    radius_m: float,
    n_rho: int,
    n_theta: int,
) -> float:
    if radius_m <= 0.0:
        return 0.0
    mu = np.asarray(mean_xy_m, dtype=float).reshape(2)
    cov = _stable_2x2_cov(cov_xy_m2)
    det = float(np.linalg.det(cov))
    inv = np.linalg.inv(cov)
    norm = 1.0 / (2.0 * math.pi * math.sqrt(max(det, 1e-18)))
    rhos = np.linspace(0.0, float(radius_m), max(int(n_rho), 16), dtype=float)
    thetas = np.linspace(0.0, 2.0 * math.pi, max(int(n_theta), 48), endpoint=False, dtype=float)
    d_theta = (2.0 * math.pi) / len(thetas)
    total = 0.0
    for theta in thetas:
        c = math.cos(float(theta))
        s = math.sin(float(theta))
        points = np.column_stack([rhos * c, rhos * s])
        delta = points - mu.reshape(1, 2)
        expo = -0.5 * np.sum((delta @ inv) * delta, axis=1)
        pdf = norm * np.exp(np.clip(expo, -700.0, 20.0))
        total += float(np.trapz(pdf * rhos, rhos)) * d_theta
    return float(np.clip(total, 0.0, 1.0))


def collision_probability_3d_alfano(
    rel_pos_rtn_km: np.ndarray,
    sigma_rtn_m: tuple[float, float, float] | None = None,
    hard_body_radius_m: float = 10.0,
    integration_points_rho: int = 48,
    integration_points_theta: int = 144,
    covariance_rtn_m2: np.ndarray | None = None,
    rel_vel_rtn_km_s: np.ndarray | None = None,
    covariance_dt_s: float = 0.0,
    mean_motion_rad_s: float = 0.0,
    sat_cov_rtn_m2: np.ndarray | None = None,
    debris_cov_rtn_m2: np.ndarray | None = None,
    sat_sigma_v_rtn_m_s: tuple[float, float, float] = (0.02, 0.03, 0.02),
    debris_sigma_v_rtn_m_s: tuple[float, float, float] = (0.03, 0.04, 0.03),
) -> dict[str, Any]:
    rel_pos_km = np.asarray(rel_pos_rtn_km, dtype=float).reshape(3)
    rel_vel_km_s = np.asarray(rel_vel_rtn_km_s if rel_vel_rtn_km_s is not None else np.array([1.0, 0.0, 0.0]), dtype=float).reshape(3)

    if covariance_rtn_m2 is not None:
        p_rel_pos_m2 = np.asarray(covariance_rtn_m2, dtype=float)
        p_rel_state_km = np.zeros((6, 6), dtype=float)
        p_rel_state_km[:3, :3] = p_rel_pos_m2 / 1_000_000.0
    elif sat_cov_rtn_m2 is not None and debris_cov_rtn_m2 is not None:
        _, _, p_rel_state_km = propagate_relative_covariance_rtn_km2(
            sat_cov_rtn_m2=np.asarray(sat_cov_rtn_m2, dtype=float),
            debris_cov_rtn_m2=np.asarray(debris_cov_rtn_m2, dtype=float),
            sat_sigma_v_rtn_m_s=sat_sigma_v_rtn_m_s,
            debris_sigma_v_rtn_m_s=debris_sigma_v_rtn_m_s,
            mean_motion_rad_s=float(mean_motion_rad_s),
            dt_s=float(covariance_dt_s),
        )
        p_rel_pos_m2 = p_rel_state_km[:3, :3] * 1_000_000.0
    else:
        if sigma_rtn_m is None:
            sigma_rtn_m = (120.0, 180.0, 150.0)
        sr, st, sn = [max(float(v), 1e-6) for v in sigma_rtn_m]
        p_rel_pos_m2 = np.diag([sr * sr, st * st, sn * sn])
        p_rel_state_km = np.zeros((6, 6), dtype=float)
        p_rel_state_km[:3, :3] = p_rel_pos_m2 / 1_000_000.0

    basis = encounter_plane_basis(rel_velocity_vec=rel_vel_km_s)
    rel_pos_m = rel_pos_km * 1000.0
    mean_enc_m = basis @ rel_pos_m
    p_enc_m2 = basis @ p_rel_pos_m2 @ basis.T
    p2_m2 = _stable_2x2_cov(p_enc_m2[:2, :2])
    mu2_m = mean_enc_m[:2]

    radius_m = float(max(hard_body_radius_m, 0.0))
    if radius_m <= 0.0:
        return {
            "Pc": 0.0,
            "confidence_metric": 1.0,
            "encounter_covariance_2d_m2": p2_m2,
            "relative_covariance_rtn_m2": p_rel_pos_m2,
            "relative_state_covariance_rtn_km": p_rel_state_km,
        }

    pc_fine = _integrate_gaussian_circle(
        mean_xy_m=mu2_m,
        cov_xy_m2=p2_m2,
        radius_m=radius_m,
        n_rho=max(int(integration_points_rho), 16),
        n_theta=max(int(integration_points_theta), 48),
    )
    pc_coarse = _integrate_gaussian_circle(
        mean_xy_m=mu2_m,
        cov_xy_m2=p2_m2,
        radius_m=radius_m,
        n_rho=max(int(integration_points_rho // 2), 10),
        n_theta=max(int(integration_points_theta // 2), 24),
    )
    err_est = abs(pc_fine - pc_coarse)
    conf = float(np.clip(1.0 - min(err_est / max(abs(pc_fine), 1e-12), 1.0), 0.0, 1.0))
    return {
        "Pc": float(np.clip(pc_fine, 0.0, 1.0)),
        "confidence_metric": conf,
        "encounter_covariance_2d_m2": p2_m2,
        "encounter_basis": basis,
        "encounter_mean_xy_m": mu2_m,
        "relative_covariance_rtn_m2": p_rel_pos_m2,
        "relative_state_covariance_rtn_km": p_rel_state_km,
    }
