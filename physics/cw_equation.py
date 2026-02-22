"""Clohessy-Wiltshire relative motion propagation in Hill frame."""
from __future__ import annotations

import numpy as np


def mean_motion(mu: float, semi_major_axis_km: float) -> float:
    return float(np.sqrt(mu / semi_major_axis_km**3))


def cw_state_transition_matrix(n: float, t_s: float) -> np.ndarray:
    """Return the 6x6 CW state transition matrix for [r, v] in RTN."""
    n = float(n)
    t_s = float(t_s)
    if abs(n) <= 1e-12:
        phi = np.eye(6, dtype=float)
        phi[0:3, 3:6] = np.eye(3, dtype=float) * t_s
        return phi
    nt = n * t_s
    s = float(np.sin(nt))
    c = float(np.cos(nt))

    phi_rr = np.array(
        [
            [4.0 - (3.0 * c), 0.0, 0.0],
            [6.0 * (s - nt), 1.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )
    phi_rv = np.array(
        [
            [s / n, (2.0 * (1.0 - c)) / n, 0.0],
            [(-2.0 * (1.0 - c)) / n, (4.0 * s - 3.0 * nt) / n, 0.0],
            [0.0, 0.0, s / n],
        ],
        dtype=float,
    )
    phi_vr = np.array(
        [
            [3.0 * n * s, 0.0, 0.0],
            [6.0 * n * (c - 1.0), 0.0, 0.0],
            [0.0, 0.0, -n * s],
        ],
        dtype=float,
    )
    phi_vv = np.array(
        [
            [c, 2.0 * s, 0.0],
            [-2.0 * s, 4.0 * c - 3.0, 0.0],
            [0.0, 0.0, c],
        ],
        dtype=float,
    )
    return np.block([[phi_rr, phi_rv], [phi_vr, phi_vv]])


def propagate_covariance_cw(state_cov: np.ndarray, n: float, t_s: float) -> np.ndarray:
    """Propagate a 6x6 covariance matrix with CW STM."""
    p0 = np.asarray(state_cov, dtype=float)
    if p0.shape != (6, 6):
        raise ValueError("state_cov must be shape (6, 6).")
    phi = cw_state_transition_matrix(n=n, t_s=t_s)
    out = phi @ p0 @ phi.T
    out = 0.5 * (out + out.T)
    return out


def propagate_cw(
    x0: np.ndarray,
    v0: np.ndarray,
    n: float,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearized CW dynamics:
    x¨ - 2n y˙ - 3n²x = 0
    y¨ + 2n x˙ = 0
    z¨ + n²z = 0
    """
    x0 = np.asarray(x0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    x = np.zeros((len(t), 3))
    v = np.zeros((len(t), 3))

    for i, ti in enumerate(t):
        phi = cw_state_transition_matrix(n=n, t_s=float(ti))
        state = phi @ np.hstack([x0, v0])
        x[i] = state[:3]
        v[i] = state[3:]

    return x, v
