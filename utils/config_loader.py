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
        "hard_body_radius_m": 10.0,
        "integration_points_z": 81,
        "integration_points_rho": 36,
        "integration_points_theta": 72,
    },
    "tca_refinement": {"fit_window_samples": 2},
    "maneuver": {
        "separation_constraint_km": 2.0,
        "burn_lead_orbits_min": 0.1,
        "burn_lead_orbits_max": 2.0,
        "burn_sweep_points": 24,
        "max_delta_v_km_s": 0.02,
        "fuel_remaining_default": 1.0,
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
