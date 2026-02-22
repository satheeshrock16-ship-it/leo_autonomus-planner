from __future__ import annotations

import numpy as np

from physics.burn_physics import FuelState, mass_after_delta_v_kg, solve_burn_from_delta_v_vector


def test_mass_after_delta_v_decreases():
    m0 = 500.0
    m1 = mass_after_delta_v_kg(delta_v_km_s=0.02, initial_mass_kg=m0)
    assert m1 < m0


def test_fuel_state_tracks_history():
    fuel = FuelState(initial_mass_kg=500.0, propellant_fraction=0.3)
    burn = solve_burn_from_delta_v_vector(np.array([0.0, 0.01, 0.0], dtype=float), mass_kg=fuel.current_mass_kg)
    fuel.apply_delta_v_vector(np.array([0.0, 0.01, 0.0], dtype=float))
    assert burn["mass_after_kg"] < burn["mass_before_kg"]
    assert fuel.burn_count == 1
    assert fuel.total_delta_v_km_s > 0.0
    assert len(fuel.mass_history_kg) >= 2
