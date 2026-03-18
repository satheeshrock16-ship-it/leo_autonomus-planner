"""Generate summary reports for synthetic validation outputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from config import PROCESSED_DATA_DIR, RESULTS_DIR


def generate_synthetic_report() -> dict[str, Any]:
    synthetic_root = PROCESSED_DATA_DIR / "synthetic"
    scenario_dirs = [p for p in synthetic_root.glob("*") if p.is_dir()]
    rows: list[dict[str, Any]] = []
    for scenario_dir in scenario_dirs:
        decision_path = scenario_dir / "decision.json"
        if not decision_path.exists():
            continue
        with decision_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        rows.append(payload)

    pcs = np.array([float(row.get("collision_probability", 0.0)) for row in rows], dtype=float)
    miss = np.array([float(row.get("closest_approach_km", 0.0)) for row in rows], dtype=float)
    maneuvers = [bool(row.get("maneuver_triggered", False)) for row in rows]

    summary = {
        "scenario_count": len(rows),
        "mean_collision_probability": float(np.mean(pcs)) if len(pcs) else 0.0,
        "max_collision_probability": float(np.max(pcs)) if len(pcs) else 0.0,
        "mean_closest_approach_km": float(np.mean(miss)) if len(miss) else 0.0,
        "maneuver_trigger_rate": float(np.mean(maneuvers)) if maneuvers else 0.0,
        "scenarios": [row.get("scenario", "UNKNOWN") for row in rows],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "synthetic_validation_report.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(generate_synthetic_report(), indent=2))
