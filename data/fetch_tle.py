"""Space-Track TLE ingestion for protected satellites and debris."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import requests

from config import (
    SPACE_TRACK_BASE_URL,
    SPACE_TRACK_IDENTITY,
    SPACE_TRACK_PASSWORD,
    SATELLITE_DATA_DIR,
    DEBRIS_DATA_DIR,
)


class SpaceTrackClient:
    """Minimal Space-Track REST client with authenticated session management."""

    def __init__(self, identity: str = SPACE_TRACK_IDENTITY, password: str = SPACE_TRACK_PASSWORD):
        self.identity = identity
        self.password = password
        self._session = requests.Session()

    def login(self) -> None:
        if not self.identity or not self.password:
            raise ValueError("Space-Track credentials are missing. Set SPACE_TRACK_IDENTITY/PASSWORD.")

        response = self._session.post(
            f"{SPACE_TRACK_BASE_URL}/ajaxauth/login",
            data={"identity": self.identity, "password": self.password},
            timeout=20,
        )
        response.raise_for_status()

    def fetch_latest_tles(self, class_query: str, limit: int = 200) -> List[Dict[str, Any]]:
        endpoint = (
            f"{SPACE_TRACK_BASE_URL}/basicspacedata/query/class/gp/"
            f"{class_query}/orderby/EPOCH desc/limit/{limit}/format/json"
        )
        response = self._session.get(endpoint, timeout=30)
        response.raise_for_status()
        return response.json()


def _serialize_tles(objects: Iterable[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(list(objects), f, indent=2)


def fetch_and_store_tles(limit: int = 200) -> tuple[Path, Path]:
    """Fetch and store LEO satellite/debris TLE data into separate files."""
    client = SpaceTrackClient()
    client.login()

    # LEO approximation by mean motion threshold (>11 rev/day) and selectable object type.
    satellites = client.fetch_latest_tles("MEAN_MOTION/>11/OBJECT_TYPE/PAYLOAD", limit=limit)
    debris = client.fetch_latest_tles("MEAN_MOTION/>11/OBJECT_TYPE/DEBRIS", limit=limit)

    sat_path = SATELLITE_DATA_DIR / "satellite_tles.json"
    debris_path = DEBRIS_DATA_DIR / "debris_tles.json"
    _serialize_tles(satellites, sat_path)
    _serialize_tles(debris, debris_path)
    return sat_path, debris_path


if __name__ == "__main__":
    try:
        sat_file, debris_file = fetch_and_store_tles(limit=100)
        print(f"Saved satellite TLEs: {sat_file}")
        print(f"Saved debris TLEs: {debris_file}")
    except Exception as exc:  # Safe top-level error output for operators.
        print(f"TLE ingestion failed: {exc}")
