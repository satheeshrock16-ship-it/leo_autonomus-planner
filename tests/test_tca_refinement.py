from __future__ import annotations

import numpy as np

from physics.tca_refinement import refine_tca_quadratic


def test_refine_tca_quadratic_recovers_vertex():
    t = np.linspace(0.0, 10.0, 11)
    d = np.sqrt((t - 4.2) ** 2 + 0.25)
    coarse_idx = int(np.argmin(d))
    refined = refine_tca_quadratic(t, d, min_index=coarse_idx, fit_window_samples=2)
    assert abs(float(refined["refined_tca_time"]) - 4.2) < 0.2
    assert abs(float(refined["refined_min_distance"]) - 0.5) < 0.1
