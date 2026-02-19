"""Collision screening and probabilistic risk analysis."""
from __future__ import annotations

from physics.collision_probability import collision_probability


def run(closest_approach_km: float) -> float:
    return collision_probability(closest_approach_km)
