"""Orbital element reconstruction from Cartesian state vectors."""
from __future__ import annotations

from typing import Any

import numpy as np


def compute_orbital_elements(r: np.ndarray, v: np.ndarray, mu: float) -> dict[str, Any]:
    """Compute exact two-body orbital element invariants from state vectors.

    Parameters
    ----------
    r : np.ndarray
        Position vector in km.
    v : np.ndarray
        Velocity vector in km/s.
    mu : float
        Gravitational parameter in km^3/s^2.
    """

    r_vec = np.asarray(r, dtype=np.float64).reshape(3)
    v_vec = np.asarray(v, dtype=np.float64).reshape(3)
    mu_val = np.float64(mu)

    r_norm = np.float64(np.linalg.norm(r_vec))
    v_norm = np.float64(np.linalg.norm(v_vec))
    if r_norm <= np.float64(0.0):
        raise ValueError("Position norm must be positive.")
    if mu_val <= np.float64(0.0):
        raise ValueError("Gravitational parameter mu must be positive.")

    h_vec = np.cross(r_vec, v_vec).astype(np.float64)
    epsilon = np.float64((v_norm * v_norm) * np.float64(0.5) - mu_val / r_norm)
    if np.isclose(epsilon, np.float64(0.0), atol=np.float64(0.0), rtol=np.float64(0.0)):
        raise ValueError("Parabolic orbit detected (epsilon ~= 0); semi-major axis is undefined.")

    semi_major_axis = np.float64(-mu_val / (np.float64(2.0) * epsilon))
    e_vec = (np.cross(v_vec, h_vec) / mu_val - (r_vec / r_norm)).astype(np.float64)
    eccentricity = np.float64(np.linalg.norm(e_vec))

    return {
        "semi_major_axis": semi_major_axis,
        "eccentricity_vector": e_vec,
        "eccentricity": eccentricity,
        "specific_angular_momentum_vector": h_vec,
        "specific_orbital_energy": epsilon,
    }
