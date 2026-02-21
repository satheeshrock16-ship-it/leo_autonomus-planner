from __future__ import annotations

from physics.maneuver_optimizer import optimize_burn_timing


def test_optimizer_produces_non_negative_dv():
    result = optimize_burn_timing(
        tca_time_s=7200.0,
        orbit_period_s=5400.0,
        miss_distance_km=0.3,
        separation_constraint_km=2.0,
        lead_orbits_min=0.1,
        lead_orbits_max=1.5,
        sweep_points=20,
        max_delta_v_km_s=0.03,
    )
    assert result["required_delta_v_km_s"] >= 0.0
    assert result["burn_time_s"] >= 0.0
