"""Independent synthetic STRONG collision environment."""
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
        name="STRONG",
        satellite=OrbitalElements(
            semi_major_axis_km=7000.0,
            eccentricity=0.0,
            inclination_deg=55.0,
            raan_deg=40.0,
            argp_deg=0.0,
        ),
        debris=OrbitalElements(
            semi_major_axis_km=7000.0,
            eccentricity=0.0,
            inclination_deg=110.0,
            raan_deg=140.0,
            argp_deg=0.0,
        ),
        target_miss_km=3.0,
        phase_offset_mode="target_over_sat_node_radius",
        expected_min_km=2.5,
        expected_max_km=3.5,
        expected_maneuver_triggered=True,
    )
    return run_synthetic_scenario(scenario)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
