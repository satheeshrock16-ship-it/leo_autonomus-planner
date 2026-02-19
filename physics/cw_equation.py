"""Clohessy-Wiltshire relative motion propagation in Hill frame."""
from __future__ import annotations

import numpy as np


def mean_motion(mu: float, semi_major_axis_km: float) -> float:
    return float(np.sqrt(mu / semi_major_axis_km**3))


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
    x0, v0 = np.asarray(x0), np.asarray(v0)
    x = np.zeros((len(t), 3))
    v = np.zeros((len(t), 3))

    for i, ti in enumerate(t):
        nt = n * ti
        s, c = np.sin(nt), np.cos(nt)

        x[i, 0] = (4 - 3 * c) * x0[0] + s / n * v0[0] + 2 * (1 - c) / n * v0[1]
        x[i, 1] = 6 * (s - nt) * x0[0] + x0[1] - 2 * (1 - c) / n * v0[0] + (4 * s - 3 * nt) / n * v0[1]
        x[i, 2] = c * x0[2] + s / n * v0[2]

        v[i, 0] = 3 * n * s * x0[0] + c * v0[0] + 2 * s * v0[1]
        v[i, 1] = -6 * n * (1 - c) * x0[0] - 2 * s * v0[0] + (4 * c - 3) * v0[1]
        v[i, 2] = -n * s * x0[2] + c * v0[2]

    return x, v
