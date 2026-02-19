"""TLE to ECI conversion using SGP4."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from sgp4.api import Satrec, jday

from config import PROCESSED_DATA_DIR


def _to_eci_state(tle_1: str, tle_2: str, epoch: datetime | None = None) -> Dict[str, Any]:
    sat = Satrec.twoline2rv(tle_1, tle_2)
    epoch = epoch or datetime.utcnow()
    jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second)
    error_code, r_eci_km, v_eci_km_s = sat.sgp4(jd, fr)
    if error_code != 0:
        raise RuntimeError(f"SGP4 error code: {error_code}")

    return {
        "epoch_utc": epoch.isoformat(),
        "r_eci_km": r_eci_km,
        "v_eci_km_s": v_eci_km_s,
    }


def convert_tle_file_to_eci(input_file: Path, output_file: Path) -> Path:
    with input_file.open("r", encoding="utf-8") as f:
        records: List[Dict[str, Any]] = json.load(f)

    converted = []
    for rec in records:
        try:
            state = _to_eci_state(rec["TLE_LINE1"], rec["TLE_LINE2"])
            converted.append({"norad_cat_id": rec.get("NORAD_CAT_ID"), **state})
        except Exception:
            continue

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2)
    return output_file


def convert_default_files(satellite_tles: Path, debris_tles: Path) -> tuple[Path, Path]:
    sat_out = PROCESSED_DATA_DIR / "satellite_eci.json"
    deb_out = PROCESSED_DATA_DIR / "debris_eci.json"
    return (
        convert_tle_file_to_eci(satellite_tles, sat_out),
        convert_tle_file_to_eci(debris_tles, deb_out),
    )
