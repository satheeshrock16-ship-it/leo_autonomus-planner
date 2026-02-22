from __future__ import annotations

import numpy as np

from physics.tca_refinement import refine_tca_analytic, refine_tca_quadratic


def test_refine_tca_quadratic_recovers_vertex():
    t = np.linspace(0.0, 10.0, 11)
    d = np.sqrt((t - 4.2) ** 2 + 0.25)
    coarse_idx = int(np.argmin(d))
    refined = refine_tca_quadratic(t, d, min_index=coarse_idx, fit_window_samples=2)
    assert abs(float(refined["refined_tca_time"]) - 4.2) < 0.2
    assert abs(float(refined["refined_min_distance"]) - 0.5) < 0.1


def test_refine_tca_analytic_refines_forward_window():
    t = np.array([0.0, 1.0, 2.0], dtype=float)
    sat_pos = np.array([[7000.0, 0.0, 0.0], [7000.0, 7.5, 0.0], [7000.0, 15.0, 0.0]], dtype=float)
    sat_vel = np.array([[0.0, 7.5, 0.0], [0.0, 7.5, 0.0], [0.0, 7.5, 0.0]], dtype=float)
    deb_pos = sat_pos + np.array([[1.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.6, 0.0, 0.0]], dtype=float)
    deb_vel = sat_vel + np.array([[-0.8, 0.0, 0.0], [-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]], dtype=float)
    refined = refine_tca_analytic(
        time_s=t,
        sat_pos_km=sat_pos,
        sat_vel_km_s=sat_vel,
        debris_pos_km=deb_pos,
        debris_vel_km_s=deb_vel,
        min_index=1,
        search_window_s=1.0,
    )
    assert 0.0 <= refined["coarse_to_refined_dt_s"] <= 1.0
    assert float(refined["refined_min_distance"]) <= 0.25
