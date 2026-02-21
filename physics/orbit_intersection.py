"""Deterministic plane intersection and node-based orbit alignment utilities."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from config import EARTH_MU_KM3_S2


def _rot_z(angle_rad: float) -> np.ndarray:
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def _rot_x(angle_rad: float) -> np.ndarray:
    c = float(math.cos(angle_rad))
    s = float(math.sin(angle_rad))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=float)


def _wrap_angle_rad(angle_rad: float) -> float:
    return float((angle_rad + math.pi) % (2.0 * math.pi) - math.pi)


def _rotation_eci_from_perifocal(inclination_deg: float, raan_deg: float, argp_deg: float) -> np.ndarray:
    return _rot_z(math.radians(raan_deg)) @ _rot_x(math.radians(inclination_deg)) @ _rot_z(math.radians(argp_deg))


@dataclass(frozen=True)
class OrbitalElements:
    semi_major_axis_km: float
    eccentricity: float
    inclination_deg: float
    raan_deg: float
    argp_deg: float


def compute_angular_momentum_vector(inclination_deg: float, raan_deg: float) -> np.ndarray:
    inc = math.radians(float(inclination_deg))
    raan = math.radians(float(raan_deg))
    h = np.array(
        [
            math.sin(inc) * math.sin(raan),
            -math.sin(inc) * math.cos(raan),
            math.cos(inc),
        ],
        dtype=float,
    )
    return h / max(float(np.linalg.norm(h)), 1e-12)


def compute_plane_intersection_line(h1_eci: np.ndarray, h2_eci: np.ndarray) -> np.ndarray:
    line = np.cross(np.asarray(h1_eci, dtype=float), np.asarray(h2_eci, dtype=float))
    norm = float(np.linalg.norm(line))
    if norm <= 1e-12:
        raise ValueError("Orbital planes are nearly parallel; intersection line is undefined.")
    return line / norm


def compute_node_true_anomaly(elements: OrbitalElements, node_direction_eci: np.ndarray) -> float:
    node_hat = np.asarray(node_direction_eci, dtype=float)
    node_hat = node_hat / max(float(np.linalg.norm(node_hat)), 1e-12)
    rot = _rotation_eci_from_perifocal(elements.inclination_deg, elements.raan_deg, elements.argp_deg)
    p_hat = rot[:, 0]
    q_hat = rot[:, 1]
    return _wrap_angle_rad(math.atan2(float(np.dot(node_hat, q_hat)), float(np.dot(node_hat, p_hat))))


def _eccentric_anomaly_from_true_anomaly(nu_rad: float, eccentricity: float) -> float:
    e = float(eccentricity)
    beta_num = math.sqrt(max(1.0 - e, 0.0)) * math.sin(0.5 * nu_rad)
    beta_den = math.sqrt(1.0 + e) * math.cos(0.5 * nu_rad)
    return float(2.0 * math.atan2(beta_num, beta_den))


def _mean_anomaly_from_true_anomaly(nu_rad: float, eccentricity: float) -> float:
    e = float(eccentricity)
    ecc_anomaly = _eccentric_anomaly_from_true_anomaly(nu_rad, e)
    return _wrap_angle_rad(ecc_anomaly - e * math.sin(ecc_anomaly))


def _solve_kepler(mean_anomaly: np.ndarray, eccentricity: float, max_iter: int = 18) -> np.ndarray:
    e = float(eccentricity)
    E = np.asarray(mean_anomaly, dtype=float).copy()
    for _ in range(max_iter):
        f = E - e * np.sin(E) - mean_anomaly
        fp = 1.0 - e * np.cos(E)
        E = E - (f / np.clip(fp, 1e-12, None))
    return E


def orbit_period_seconds(elements: OrbitalElements, mu_km3_s2: float = EARTH_MU_KM3_S2) -> float:
    a = float(elements.semi_major_axis_km)
    return float(2.0 * math.pi * math.sqrt((a**3) / mu_km3_s2))


def propagate_orbit(
    elements: OrbitalElements,
    time_s: np.ndarray,
    mean_anomaly_at_epoch_rad: float,
    mu_km3_s2: float = EARTH_MU_KM3_S2,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=float)
    a = float(elements.semi_major_axis_km)
    e = float(elements.eccentricity)
    n = math.sqrt(mu_km3_s2 / (a**3))
    mean_anomaly = mean_anomaly_at_epoch_rad + (n * t)
    ecc_anomaly = _solve_kepler(mean_anomaly, e)

    cosE = np.cos(ecc_anomaly)
    sinE = np.sin(ecc_anomaly)
    r_pf = np.column_stack([a * (cosE - e), a * math.sqrt(1.0 - e**2) * sinE, np.zeros_like(t)])
    denom = np.clip(1.0 - e * cosE, 1e-10, None)
    vel_scale = math.sqrt(mu_km3_s2 * a) / (a * denom)
    v_pf = np.column_stack(
        [
            -vel_scale * sinE,
            vel_scale * math.sqrt(1.0 - e**2) * cosE,
            np.zeros_like(t),
        ]
    )

    rot = _rotation_eci_from_perifocal(elements.inclination_deg, elements.raan_deg, elements.argp_deg)
    return r_pf @ rot.T, v_pf @ rot.T


def align_orbits_at_node(
    satellite_elements: OrbitalElements,
    debris_elements: OrbitalElements,
    phase_offset_rad: float = 0.0,
    alignment_time_s: float = 0.0,
    prefer_negative_node: bool = False,
) -> dict[str, Any]:
    h_sat = compute_angular_momentum_vector(satellite_elements.inclination_deg, satellite_elements.raan_deg)
    h_deb = compute_angular_momentum_vector(debris_elements.inclination_deg, debris_elements.raan_deg)
    node_line = compute_plane_intersection_line(h_sat, h_deb)
    if prefer_negative_node:
        node_line = -node_line

    nu_sat = compute_node_true_anomaly(satellite_elements, node_line)
    nu_deb = _wrap_angle_rad(compute_node_true_anomaly(debris_elements, node_line) + float(phase_offset_rad))

    M_sat_at_node = _mean_anomaly_from_true_anomaly(nu_sat, satellite_elements.eccentricity)
    M_deb_at_node = _mean_anomaly_from_true_anomaly(nu_deb, debris_elements.eccentricity)
    n_sat = math.sqrt(EARTH_MU_KM3_S2 / (float(satellite_elements.semi_major_axis_km) ** 3))
    n_deb = math.sqrt(EARTH_MU_KM3_S2 / (float(debris_elements.semi_major_axis_km) ** 3))

    M_sat_epoch = _wrap_angle_rad(M_sat_at_node - n_sat * float(alignment_time_s))
    M_deb_epoch = _wrap_angle_rad(M_deb_at_node - n_deb * float(alignment_time_s))
    return {
        "node_direction_eci": node_line,
        "satellite_true_anomaly_rad": nu_sat,
        "debris_true_anomaly_rad": nu_deb,
        "satellite_mean_anomaly_epoch_rad": M_sat_epoch,
        "debris_mean_anomaly_epoch_rad": M_deb_epoch,
    }
