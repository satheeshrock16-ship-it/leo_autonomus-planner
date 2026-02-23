"""Exact two-body universal-variable propagation utilities."""
from __future__ import annotations

import numpy as np


def _stumpff_c(z: np.float64) -> np.float64:
    z_val = np.float64(z)
    if z_val > np.float64(1e-8):
        root = np.sqrt(z_val)
        return np.float64((1.0 - np.cos(root)) / z_val)
    if z_val < np.float64(-1e-8):
        root = np.sqrt(-z_val)
        return np.float64((np.cosh(root) - 1.0) / (-z_val))
    return np.float64(
        0.5
        - z_val / 24.0
        + (z_val * z_val) / 720.0
        - (z_val * z_val * z_val) / 40320.0
    )


def _stumpff_s(z: np.float64) -> np.float64:
    z_val = np.float64(z)
    if z_val > np.float64(1e-8):
        root = np.sqrt(z_val)
        return np.float64((root - np.sin(root)) / (root * root * root))
    if z_val < np.float64(-1e-8):
        root = np.sqrt(-z_val)
        return np.float64((np.sinh(root) - root) / (root * root * root))
    return np.float64(
        (1.0 / 6.0)
        - z_val / 120.0
        + (z_val * z_val) / 5040.0
        - (z_val * z_val * z_val) / 362880.0
    )


def propagate_universal_variable(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    dt_s: float,
    mu_km3_s2: float,
    tolerance: float = 1e-13,
    max_iter: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate one two-body state using universal variables."""

    r0 = np.asarray(r0_km, dtype=np.float64).reshape(3)
    v0 = np.asarray(v0_km_s, dtype=np.float64).reshape(3)
    dt = np.float64(dt_s)
    mu = np.float64(mu_km3_s2)
    tol = np.float64(tolerance)

    r0_norm = np.float64(np.linalg.norm(r0))
    v0_sq = np.float64(np.dot(v0, v0))
    if r0_norm <= np.float64(0.0):
        raise ValueError("r0 norm must be positive.")
    if mu <= np.float64(0.0):
        raise ValueError("mu must be positive.")
    if dt == np.float64(0.0):
        return r0.copy(), v0.copy()

    sqrt_mu = np.float64(np.sqrt(mu))
    vr0 = np.float64(np.dot(r0, v0) / r0_norm)
    alpha = np.float64(np.float64(2.0) / r0_norm - v0_sq / mu)

    if np.abs(alpha) > np.float64(1e-12):
        chi = np.float64(sqrt_mu * dt * alpha)
    else:
        chi = np.float64(sqrt_mu * dt / r0_norm)
    if chi == np.float64(0.0):
        chi = np.float64(np.sign(dt) * np.sqrt(mu) * np.abs(dt) / r0_norm)

    converged = False
    for _ in range(max_iter):
        z = np.float64(alpha * chi * chi)
        c_val = _stumpff_c(z)
        s_val = _stumpff_s(z)

        term1 = np.float64(r0_norm * vr0 / sqrt_mu) * chi * chi * c_val
        term2 = np.float64(1.0 - alpha * r0_norm) * chi * chi * chi * s_val
        f_val = np.float64(term1 + term2 + r0_norm * chi - sqrt_mu * dt)

        fp1 = np.float64(r0_norm * vr0 / sqrt_mu) * chi * np.float64(1.0 - z * s_val)
        fp2 = np.float64(1.0 - alpha * r0_norm) * chi * chi * c_val
        fp_val = np.float64(fp1 + fp2 + r0_norm)
        if np.abs(fp_val) <= np.float64(1e-18):
            raise RuntimeError("Universal-variable propagation derivative underflow.")

        ratio = np.float64(f_val / fp_val)
        chi = np.float64(chi - ratio)
        if np.abs(ratio) < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError("Universal-variable propagation did not converge.")

    z = np.float64(alpha * chi * chi)
    c_val = _stumpff_c(z)
    s_val = _stumpff_s(z)

    f_lagrange = np.float64(np.float64(1.0) - (chi * chi / r0_norm) * c_val)
    g_lagrange = np.float64(dt - (chi * chi * chi / sqrt_mu) * s_val)
    r_vec = (f_lagrange * r0 + g_lagrange * v0).astype(np.float64)
    r_norm = np.float64(np.linalg.norm(r_vec))
    if r_norm <= np.float64(0.0):
        raise RuntimeError("Propagated radius norm collapsed to zero.")

    fdot = np.float64((sqrt_mu / (r_norm * r0_norm)) * (alpha * chi * chi * chi * s_val - chi))
    gdot = np.float64(np.float64(1.0) - (chi * chi / r_norm) * c_val)
    v_vec = (fdot * r0 + gdot * v0).astype(np.float64)
    return r_vec, v_vec


def propagate_universal_trajectory(
    r0_km: np.ndarray,
    v0_km_s: np.ndarray,
    t_rel_s: np.ndarray,
    mu_km3_s2: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate multiple epochs relative to a common initial state."""

    t_rel = np.asarray(t_rel_s, dtype=np.float64)
    if t_rel.ndim != 1:
        raise ValueError("t_rel_s must be a 1-D array.")
    if t_rel.size == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 3), dtype=np.float64)

    positions = np.zeros((t_rel.size, 3), dtype=np.float64)
    velocities = np.zeros((t_rel.size, 3), dtype=np.float64)
    for idx, dt in enumerate(t_rel):
        r_vec, v_vec = propagate_universal_variable(
            r0_km=r0_km,
            v0_km_s=v0_km_s,
            dt_s=float(dt),
            mu_km3_s2=float(mu_km3_s2),
        )
        positions[idx] = r_vec
        velocities[idx] = v_vec
    return positions, velocities
