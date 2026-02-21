"""Independent synthetic SAFE collision environment."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from physics.orbit_intersection import OrbitalElements
from pipeline.synthetic_case_engine import SyntheticScenario, run_synthetic_scenario


def run() -> dict:
    scenario = SyntheticScenario(
        name="SAFE",
        satellite=OrbitalElements(
            semi_major_axis_km=7000.0,
            eccentricity=0.0,
            inclination_deg=20.0,
            raan_deg=10.0,
            argp_deg=0.0,
        ),
        debris=OrbitalElements(
            semi_major_axis_km=7000.0,
            eccentricity=0.0,
            inclination_deg=22.0,
            raan_deg=30.0,
            argp_deg=0.0,
        ),
        target_miss_km=200.0,
        phase_offset_mode="target_over_sat_node_radius",
        expected_min_km=150.0,
        expected_max_km=250.0,
        expected_maneuver_triggered=False,
    )
    return run_synthetic_scenario(scenario)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
