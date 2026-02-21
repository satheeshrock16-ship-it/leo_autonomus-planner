"""Collision probability via a simple Gaussian miss-distance model."""
from __future__ import annotations

import logging
import math

LOGGER = logging.getLogger(__name__)


def collision_probability(closest_approach_km: float, sigma_m: float = 100.0) -> float:
    """Compute collision probability from scalar miss distance with strict unit handling.

    Orbital propagation distances are in kilometers.
    Probability math is performed in meters.
    """
    closest_approach_km = float(closest_approach_km)
    miss_distance_m = closest_approach_km * 1000.0
    sigma = float(sigma_m)

    if closest_approach_km > 1.0:
        pc = 0.0
    else:
        pc = float(math.exp(-(miss_distance_m**2) / (2.0 * sigma**2)))

    LOGGER.debug(
        "Scalar Pc fallback used: miss_km=%.6f miss_m=%.3f sigma_m=%.3f pc=%.6e",
        closest_approach_km,
        miss_distance_m,
        sigma,
        pc,
    )
    return pc
