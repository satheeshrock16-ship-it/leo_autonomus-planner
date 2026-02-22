from __future__ import annotations

import numpy as np

from physics.collision_probability_3d import build_full_covariance_rtn_m2, collision_probability_3d_alfano


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


def test_collision_probability_accepts_full_covariance():
    sat_cov = build_full_covariance_rtn_m2(120.0, 180.0, 150.0, rho_rt=0.1, rho_rn=0.05, rho_tn=-0.02)
    deb_cov = build_full_covariance_rtn_m2(160.0, 220.0, 180.0, rho_rt=0.08, rho_rn=0.03, rho_tn=0.04)
    out = collision_probability_3d_alfano(
        rel_pos_rtn_km=np.array([0.02, -0.01, 0.0]),
        rel_vel_rtn_km_s=np.array([0.0, 11.0, 1.5]),
        sat_cov_rtn_m2=sat_cov,
        debris_cov_rtn_m2=deb_cov,
        covariance_dt_s=0.5,
        mean_motion_rad_s=0.0011,
        hard_body_radius_m=10.0,
    )
    assert 0.0 <= float(out["Pc"]) <= 1.0
