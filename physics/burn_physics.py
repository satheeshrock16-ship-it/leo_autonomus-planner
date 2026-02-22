"""Burn-time, mass, and propellant models using the rocket equation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from physics.constants import G0, ISP_SECONDS, SATELLITE_INITIAL_MASS_KG


SATELLITE_MASS_KG = SATELLITE_INITIAL_MASS_KG
THRUST_N = 5.0


def mass_after_delta_v_kg(
    delta_v_km_s: float,
    initial_mass_kg: float = SATELLITE_MASS_KG,
    isp_seconds: float = ISP_SECONDS,
    g0_m_s2: float = G0,
) -> float:
    m0 = max(float(initial_mass_kg), 1e-9)
    dv_m_s = max(float(delta_v_km_s), 0.0) * 1000.0
    if dv_m_s <= 0.0:
        return m0
    denom = max(float(isp_seconds) * float(g0_m_s2), 1e-9)
    return float(m0 / np.exp(dv_m_s / denom))


def propellant_used_kg(
    delta_v_km_s: float,
    initial_mass_kg: float = SATELLITE_MASS_KG,
    isp_seconds: float = ISP_SECONDS,
    g0_m_s2: float = G0,
) -> float:
    m0 = max(float(initial_mass_kg), 1e-9)
    mf = mass_after_delta_v_kg(
        delta_v_km_s=delta_v_km_s,
        initial_mass_kg=m0,
        isp_seconds=isp_seconds,
        g0_m_s2=g0_m_s2,
    )
    return float(max(m0 - mf, 0.0))


def delta_v_from_burn_time_km_s(
    burn_time_seconds: float,
    mass_kg: float = SATELLITE_MASS_KG,
    force_newtons: float = THRUST_N,
    isp_seconds: float = ISP_SECONDS,
    g0_m_s2: float = G0,
) -> float:
    m0 = max(float(mass_kg), 1e-9)
    thrust = max(float(force_newtons), 1e-9)
    mdot = thrust / max(float(isp_seconds) * float(g0_m_s2), 1e-9)
    burn_time = max(float(burn_time_seconds), 0.0)
    mf = max(m0 - (mdot * burn_time), 1e-9)
    dv_m_s = max(float(isp_seconds) * float(g0_m_s2) * float(np.log(m0 / mf)), 0.0)
    return float(dv_m_s / 1000.0)


def burn_time_from_delta_v_seconds(
    delta_v_km_s: float,
    mass_kg: float = SATELLITE_MASS_KG,
    force_newtons: float = THRUST_N,
    isp_seconds: float = ISP_SECONDS,
    g0_m_s2: float = G0,
) -> float:
    m0 = max(float(mass_kg), 1e-9)
    mf = mass_after_delta_v_kg(
        delta_v_km_s=delta_v_km_s,
        initial_mass_kg=m0,
        isp_seconds=isp_seconds,
        g0_m_s2=g0_m_s2,
    )
    propellant = max(m0 - mf, 0.0)
    mdot = max(float(force_newtons), 1e-9) / max(float(isp_seconds) * float(g0_m_s2), 1e-9)
    return float(propellant / max(mdot, 1e-9))


def solve_burn_from_delta_v_vector(
    delta_v_vector_km_s: np.ndarray,
    mass_kg: float = SATELLITE_MASS_KG,
    force_newtons: float = THRUST_N,
    isp_seconds: float = ISP_SECONDS,
    g0_m_s2: float = G0,
) -> dict[str, Any]:
    delta_v = np.asarray(delta_v_vector_km_s, dtype=float)
    delta_v_mag = float(np.linalg.norm(delta_v))
    m_before = float(max(mass_kg, 1e-9))
    m_after = mass_after_delta_v_kg(
        delta_v_km_s=delta_v_mag,
        initial_mass_kg=m_before,
        isp_seconds=isp_seconds,
        g0_m_s2=g0_m_s2,
    )
    burn_time_s = burn_time_from_delta_v_seconds(
        delta_v_km_s=delta_v_mag,
        mass_kg=m_before,
        force_newtons=force_newtons,
        isp_seconds=isp_seconds,
        g0_m_s2=g0_m_s2,
    )
    propellant = float(max(m_before - m_after, 0.0))
    return {
        "delta_v_vector": delta_v,
        "delta_v_magnitude_km_s": delta_v_mag,
        "burn_time_seconds": burn_time_s,
        "delta_v_mps": float(delta_v_mag * 1000.0),
        "mass_before_kg": m_before,
        "mass_after_kg": float(m_after),
        "propellant_used_kg": propellant,
        "isp_seconds": float(isp_seconds),
        "g0_m_s2": float(g0_m_s2),
    }


@dataclass
class FuelState:
    initial_mass_kg: float = SATELLITE_MASS_KG
    propellant_fraction: float = 0.3
    isp_seconds: float = ISP_SECONDS
    g0_m_s2: float = G0
    force_newtons: float = THRUST_N
    current_mass_kg: float = field(init=False)
    dry_mass_kg: float = field(init=False)
    burn_count: int = field(default=0, init=False)
    total_delta_v_km_s: float = field(default=0.0, init=False)
    mass_history_kg: list[float] = field(default_factory=list, init=False)
    burn_history: list[dict[str, float]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        m0 = float(max(self.initial_mass_kg, 1e-6))
        prop_fraction = float(np.clip(self.propellant_fraction, 0.0, 0.95))
        self.current_mass_kg = m0
        self.dry_mass_kg = float(max(m0 * (1.0 - prop_fraction), 1e-6))
        self.mass_history_kg.append(self.current_mass_kg)

    @property
    def remaining_propellant_kg(self) -> float:
        return float(max(self.current_mass_kg - self.dry_mass_kg, 0.0))

    def apply_delta_v_vector(self, delta_v_vector_km_s: np.ndarray) -> dict[str, float]:
        delta_v_vec = np.asarray(delta_v_vector_km_s, dtype=float)
        burn = solve_burn_from_delta_v_vector(
            delta_v_vector_km_s=delta_v_vec,
            mass_kg=self.current_mass_kg,
            force_newtons=self.force_newtons,
            isp_seconds=self.isp_seconds,
            g0_m_s2=self.g0_m_s2,
        )
        desired_mass_after = float(burn["mass_after_kg"])
        clamped_mass_after = float(max(desired_mass_after, self.dry_mass_kg))
        used_propellant = float(max(self.current_mass_kg - clamped_mass_after, 0.0))
        self.current_mass_kg = clamped_mass_after
        self.mass_history_kg.append(self.current_mass_kg)
        dv_mag = float(np.linalg.norm(delta_v_vec))
        self.total_delta_v_km_s += dv_mag
        self.burn_count += 1

        entry = {
            "delta_v_km_s": dv_mag,
            "burn_time_seconds": float(burn["burn_time_seconds"]),
            "mass_before_kg": float(burn["mass_before_kg"]),
            "mass_after_kg": self.current_mass_kg,
            "propellant_used_kg": used_propellant,
            "remaining_propellant_kg": self.remaining_propellant_kg,
            "burn_count": float(self.burn_count),
            "total_delta_v_km_s": self.total_delta_v_km_s,
        }
        self.burn_history.append(entry)
        return entry
