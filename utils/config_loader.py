"""Centralized configuration loading for YAML-based runtime settings."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_CONFIG: dict[str, Any] = {
    "propagation": {
        "timestep_seconds": 60,
        "screening_window_hours": 72,
        "deterministic_epoch_utc": "2026-01-01T00:00:00+00:00",
        "benchmark_timestep_seconds": 120,
        "benchmark_window_hours": 2,
    },
    "covariance": {
        "sigma_r_m": 120.0,
        "sigma_t_m": 180.0,
        "sigma_n_m": 150.0,
        "rho_rt": 0.0,
        "rho_rn": 0.0,
        "rho_tn": 0.0,
        "satellite": {
            "sigma_r_m": 120.0,
            "sigma_t_m": 180.0,
            "sigma_n_m": 150.0,
            "rho_rt": 0.05,
            "rho_rn": 0.02,
            "rho_tn": 0.03,
            "sigma_vr_m_s": 0.02,
            "sigma_vt_m_s": 0.03,
            "sigma_vn_m_s": 0.02,
        },
        "debris": {
            "sigma_r_m": 160.0,
            "sigma_t_m": 220.0,
            "sigma_n_m": 180.0,
            "rho_rt": 0.08,
            "rho_rn": 0.03,
            "rho_tn": 0.05,
            "sigma_vr_m_s": 0.03,
            "sigma_vt_m_s": 0.04,
            "sigma_vn_m_s": 0.03,
        },
        "hard_body_radius_m": 10.0,
        "integration_points_rho": 48,
        "integration_points_theta": 144,
        "monte_carlo_enabled": True,
        "monte_carlo_samples": 5000,
        "monte_carlo_seed": 42,
    },
    "tca_refinement": {
        "fit_window_samples": 2,
        "search_window_multiplier": 1.0,
    },
    "maneuver": {
        "separation_constraint_km": 5.0,
        "burn_lead_orbits_min": 0.1,
        "burn_lead_orbits_max": 2.0,
        "burn_sweep_points": 24,
        "max_delta_v_km_s": 0.02,
        "fuel_remaining_default": 1.0,
        "fuel_penalty_weight": 0.25,
        "propellant_fraction_default": 0.3,
    },
    "performance": {"benchmark_counts": [100, 1000, 5000], "parallel_workers": 4},
    "synthetic_ml": {"training_samples": 2500, "random_seed": 42},
}

_CONFIG_CACHE: dict[str, Any] | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, force_reload: bool = False) -> dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload and path is None:
        return deepcopy(_CONFIG_CACHE)

    if path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    else:
        config_path = Path(path)

    config = deepcopy(DEFAULT_CONFIG)
    if config_path.exists() and yaml is not None:
        with config_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if isinstance(loaded, dict):
            config = _deep_merge(config, loaded)
    _CONFIG_CACHE = deepcopy(config)
    return config
