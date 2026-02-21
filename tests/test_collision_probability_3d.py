from __future__ import annotations

import numpy as np

from physics.collision_probability_3d import collision_probability_3d_alfano


def test_collision_probability_decreases_with_miss_distance():
    near = collision_probability_3d_alfano(
        rel_pos_rtn_km=np.array([0.001, 0.001, 0.001]),
        sigma_rtn_m=(100.0, 100.0, 100.0),
        hard_body_radius_m=10.0,
    )["Pc"]
    far = collision_probability_3d_alfano(
        rel_pos_rtn_km=np.array([1.0, 1.0, 1.0]),
        sigma_rtn_m=(100.0, 100.0, 100.0),
        hard_body_radius_m=10.0,
    )["Pc"]
    assert near > far
