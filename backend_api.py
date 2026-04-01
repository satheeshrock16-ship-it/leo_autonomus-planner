"""Lightweight real-time API for Cesium object, alert, and maneuver views."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import random
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from sgp4.api import Satrec, jday

from config import DEBRIS_DATA_DIR, SATELLITE_DATA_DIR
from utils.eci_geodetic import eci_km_to_geodetic

ROOT = Path(__file__).resolve().parent
WEBAPP_DIR = ROOT / "webapp"
DEFAULT_OBJECT_TARGET = 800
DEFAULT_MIN_OBJECTS = 650
DEFAULT_REFRESH_SECONDS = 2.5
DEFAULT_ALERT_THRESHOLD_KM = 1800.0
CELESTRAK_ACTIVE_TLE_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
CELESTRAK_ACTIVE_JSON_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json"
DEBRIS_NAME_PATTERNS = ("DEB", "R/B", "FENGYUN", "COSMOS")


@dataclass(frozen=True)
class CatalogRecord:
    object_id: str
    name: str
    object_type: str
    line1: str
    line2: str
    phase_offset_seconds: int = 0


@dataclass
class LiveObject:
    object_id: str
    name: str
    object_type: str
    lat: float
    lon: float
    alt_m: float
    r_eci_km: tuple[float, float, float]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _jd_fr(epoch: datetime) -> tuple[float, float]:
    dt = epoch.astimezone(timezone.utc)
    seconds = dt.second + (dt.microsecond / 1_000_000.0)
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, seconds)


def _load_json_records(path: Path, object_type: str) -> list[CatalogRecord]:
    if not path.exists():
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    records: list[CatalogRecord] = []
    for raw in payload:
        line1 = str(raw.get("TLE_LINE1", "")).strip()
        line2 = str(raw.get("TLE_LINE2", "")).strip()
        if not line1 or not line2:
            continue
        base_id = str(raw.get("NORAD_CAT_ID") or raw.get("OBJECT_ID") or raw.get("OBJECT_NAME") or f"{object_type}-{len(records)+1}")
        name = str(raw.get("OBJECT_NAME") or raw.get("OBJECT_ID") or base_id)
        records.append(
            CatalogRecord(
                object_id=f"{object_type}:{base_id}",
                name=name,
                object_type=object_type,
                line1=line1,
                line2=line2,
            )
        )
    return records


def _parse_tle_triplets(text: str, object_type: str) -> list[CatalogRecord]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    records: list[CatalogRecord] = []
    index = 0

    while index < len(lines):
        if lines[index].startswith("1 ") and index + 1 < len(lines) and lines[index + 1].startswith("2 "):
            name = f"{object_type.title()} {len(records) + 1}"
            line1 = lines[index]
            line2 = lines[index + 1]
            index += 2
        elif (
            index + 2 < len(lines)
            and not lines[index].startswith("1 ")
            and lines[index + 1].startswith("1 ")
            and lines[index + 2].startswith("2 ")
        ):
            name = lines[index]
            line1 = lines[index + 1]
            line2 = lines[index + 2]
            index += 3
        else:
            index += 1
            continue

        norad = line1[2:7].strip() or f"{object_type}-{len(records) + 1}"
        records.append(
            CatalogRecord(
                object_id=f"{object_type}:{norad}",
                name=name,
                object_type=object_type,
                line1=line1,
                line2=line2,
            )
        )
    return records


def _fetch_celestrak_active_records(limit: int) -> list[CatalogRecord]:
    response = requests.get(CELESTRAK_ACTIVE_TLE_URL, timeout=4)
    response.raise_for_status()
    return _parse_tle_triplets(response.text, "satellite")[:limit]


def _dedupe_records(records: list[CatalogRecord]) -> list[CatalogRecord]:
    seen: set[str] = set()
    unique: list[CatalogRecord] = []
    for record in records:
        if record.object_id in seen:
            continue
        unique.append(record)
        seen.add(record.object_id)
    return unique


def _densify_records(records: list[CatalogRecord], minimum_objects: int) -> list[CatalogRecord]:
    if len(records) >= minimum_objects or not records:
        return records

    offsets = [0, 45, 90, 135, 180, 225, 270, 315, 360]
    dense: list[CatalogRecord] = []
    for offset in offsets:
        for record in records:
            if len(dense) >= minimum_objects:
                return dense
            if offset == 0:
                dense.append(record)
            else:
                dense.append(
                    replace(
                        record,
                        object_id=f"{record.object_id}@{offset}",
                        name=f"{record.name} [+{offset}s]",
                        phase_offset_seconds=offset,
                    )
                )
    return dense


def _propagate_record(record: CatalogRecord, epoch: datetime) -> LiveObject | None:
    sample_time = epoch + timedelta(seconds=record.phase_offset_seconds)
    satrec = Satrec.twoline2rv(record.line1, record.line2)
    jd, fr = _jd_fr(sample_time)
    error_code, r_eci_km, _ = satrec.sgp4(jd, fr)
    if error_code != 0:
        return None

    lat, lon, alt_m = eci_km_to_geodetic(r_eci_km, sample_time)
    return LiveObject(
        object_id=record.object_id,
        name=record.name,
        object_type=record.object_type,
        lat=float(lat),
        lon=float(lon),
        alt_m=float(alt_m),
        r_eci_km=(float(r_eci_km[0]), float(r_eci_km[1]), float(r_eci_km[2])),
    )


def _distance_km(a: LiveObject, b: LiveObject) -> float:
    dx = a.r_eci_km[0] - b.r_eci_km[0]
    dy = a.r_eci_km[1] - b.r_eci_km[1]
    dz = a.r_eci_km[2] - b.r_eci_km[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def generate_objects(n: int = 100) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []

    for i in range(n):
        obj_type = "satellite" if i % 2 == 0 else "debris"
        objects.append(
            {
                "id": f"{'SAT' if obj_type == 'satellite' else 'DEB'}-{i}",
                "type": obj_type,
                "lat": random.uniform(-90, 90),
                "lon": random.uniform(-180, 180),
                "alt": random.uniform(400, 1200),
                "velocity": random.uniform(7.0, 8.0),
            }
        )

    return objects


def _classify_object_type(name: str) -> str:
    normalized_name = name.upper()
    if any(pattern in normalized_name for pattern in DEBRIS_NAME_PATTERNS):
        return "debris"
    return "satellite"


def fetch_tle_data() -> list[dict[str, str]]:
    objects: list[dict[str, str]] = []

    try:
        response = requests.get(CELESTRAK_ACTIVE_TLE_URL, timeout=5)
        response.raise_for_status()

        for record in _parse_tle_triplets(response.text, "satellite"):
            objects.append(
                {
                    "name": record.name,
                    "line1": record.line1,
                    "line2": record.line2,
                    "type": _classify_object_type(record.name),
                }
            )
    except Exception as error:
        print("Active fetch failed:", error)

        try:
            for record in _load_json_records(
                SATELLITE_DATA_DIR / "satellite_tles.json",
                "satellite",
            ):
                objects.append(
                    {
                        "name": record.name,
                        "line1": record.line1,
                        "line2": record.line2,
                        "type": _classify_object_type(record.name),
                    }
                )
        except Exception as fallback_error:
            print("Active fallback failed:", fallback_error)

    if not objects:
        return []

    return objects[:400]


def generate_collisions(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    satellites = [obj for obj in objects if obj.get("type") == "satellite"]
    debris = [obj for obj in objects if obj.get("type") == "debris"]
    collisions: list[dict[str, Any]] = []

    if len(satellites) >= 2:
        collisions.append(
            {
                "sat1": satellites[0]["id"],
                "sat2": satellites[1]["id"],
                "probability": 0.8,
            }
        )

    if satellites and debris:
        collisions.append(
            {
                "sat1": satellites[0]["id"],
                "sat2": debris[0]["id"],
                "probability": 0.42,
            }
        )

    return collisions


def generate_analysis() -> dict[str, Any]:
    return {
        "collision_probability": 1.05e-307,
        "miss_distance_km": 579.28,
        "delta_v_km_s": 0.0,
        "runtime_seconds": random.uniform(40, 100),
        "tca_time": "2026-01-02 06:59:00+00:00",
    }


class RealtimeCatalogService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_refresh: datetime | None = None
        self._last_tle_refresh: datetime | None = None
        self._records: list[CatalogRecord] = []
        self._objects: list[LiveObject] = []
        self._alerts: list[dict[str, Any]] = []
        self._alert_index: dict[str, dict[str, Any]] = {}
        self.minimum_objects = DEFAULT_MIN_OBJECTS
        self.object_target = DEFAULT_OBJECT_TARGET
        self.refresh_seconds = DEFAULT_REFRESH_SECONDS
        self.alert_threshold_km = DEFAULT_ALERT_THRESHOLD_KM

    def _load_records(self) -> list[CatalogRecord]:
        now = _utc_now()
        if self._records and self._last_tle_refresh and (now - self._last_tle_refresh).total_seconds() < 1800:
            return self._records

        local_satellite = _load_json_records(SATELLITE_DATA_DIR / "satellite_tles.json", "satellite")
        local_debris = _load_json_records(DEBRIS_DATA_DIR / "debris_tles.json", "debris")
        records = _dedupe_records(local_satellite + local_debris)

        if len(records) < self.minimum_objects:
            try:
                extra_satellites = _fetch_celestrak_active_records(max(self.object_target, self.minimum_objects))
                records = _dedupe_records(records + extra_satellites)
            except Exception:
                pass

        if len(records) < self.minimum_objects:
            records = _densify_records(records, self.minimum_objects)

        self._records = records
        self._last_tle_refresh = now
        return records

    def _refresh(self) -> None:
        now = _utc_now()
        if self._last_refresh and (now - self._last_refresh).total_seconds() < self.refresh_seconds:
            return

        with self._lock:
            now = _utc_now()
            if self._last_refresh and (now - self._last_refresh).total_seconds() < self.refresh_seconds:
                return

            live_objects = []
            for record in self._load_records():
                live = _propagate_record(record, now)
                if live is not None:
                    live_objects.append(live)

            self._objects = live_objects
            self._alerts = self._build_alerts(live_objects, now)
            self._alert_index = {alert["id"]: alert for alert in self._alerts}
            self._last_refresh = now

    def _build_alerts(self, live_objects: list[LiveObject], epoch: datetime) -> list[dict[str, Any]]:
        satellites = [obj for obj in live_objects if obj.object_type == "satellite"]
        debris = [obj for obj in live_objects if obj.object_type == "debris"]

        pairs: list[dict[str, Any]] = []
        fallback_pairs: list[dict[str, Any]] = []

        for sat in satellites:
            for deb in debris:
                distance_km = _distance_km(sat, deb)
                entry = {
                    "id": f"{sat.object_id}__{deb.object_id}",
                    "sat1": sat.name,
                    "sat2": deb.name,
                    "sat1_id": sat.object_id,
                    "sat2_id": deb.object_id,
                    "distance": round(distance_km, 3),
                    "time": epoch.isoformat(),
                }
                fallback_pairs.append(entry)
                if distance_km <= self.alert_threshold_km:
                    pairs.append(entry)

        if not pairs:
            pairs = sorted(fallback_pairs, key=lambda item: item["distance"])[:5]
        else:
            pairs = sorted(pairs, key=lambda item: item["distance"])[:10]

        return pairs

    def objects_payload(self) -> list[dict[str, Any]]:
        self._refresh()
        return [
            {
                "id": obj.object_id,
                "name": obj.name,
                "lat": round(obj.lat, 6),
                "lon": round(obj.lon, 6),
                "alt": round(obj.alt_m, 2),
                "type": obj.object_type,
            }
            for obj in self._objects
        ]

    def alerts_payload(self) -> list[dict[str, Any]]:
        self._refresh()
        return self._alerts

    def active_catalog_payload(self) -> list[dict[str, Any]]:
        try:
            response = requests.get(CELESTRAK_ACTIVE_JSON_URL, timeout=6)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []
        return payload if isinstance(payload, list) else []

    def _scenario_payload(
        self,
        primary_name: str,
        secondary_name: str,
        distance_km: float,
        alert_id: str,
        event_time: str | None = None,
        relative_velocity_km_s: float | None = None,
    ) -> dict[str, Any] | None:
        if distance_km < 50.0:
            scenario_key = "critical"
        elif distance_km < 250.0:
            scenario_key = "strong"
        else:
            scenario_key = "safe"

        scenario_path = WEBAPP_DIR / "scenarios" / f"{scenario_key}.json"
        if not scenario_path.exists():
            return None

        payload = json.loads(scenario_path.read_text(encoding="utf-8"))
        payload["scenarioName"] = f"{primary_name} vs {secondary_name}"
        payload["description"] = (
            f"Backend maneuver visualization for {primary_name} and {secondary_name} "
            f"with miss distance {distance_km:.3f} km."
        )
        payload.setdefault("metadata", {})
        payload["metadata"]["scenarioName"] = payload["scenarioName"]
        payload["metadata"]["minimumDistanceKm"] = round(distance_km, 3)
        payload["metadata"]["collisionProbability"] = round(min(0.95, max(0.01, 120.0 / max(distance_km, 1.0) / 100.0)), 4)
        payload["metadata"]["maneuverTriggered"] = True
        payload["metadata"]["postManeuverDistanceKm"] = round(max(distance_km * 1.8, 25.0), 3)
        payload["metadata"]["selectedAlertId"] = alert_id
        payload["metadata"]["selectedAlertLabel"] = f"{primary_name} x {secondary_name}"
        payload["metadata"]["selectedAlertTime"] = event_time or _utc_now().isoformat()
        if relative_velocity_km_s is not None:
            payload["metadata"]["relativeVelocityKmS"] = round(float(relative_velocity_km_s), 4)
        return payload

    def maneuver_payload(self, alert_id: str) -> dict[str, Any] | None:
        self._refresh()
        alert = self._alert_index.get(alert_id)
        if alert is None:
            return None

        return self._scenario_payload(
            primary_name=str(alert["sat1"]),
            secondary_name=str(alert["sat2"]),
            distance_km=float(alert["distance"]),
            alert_id=alert_id,
            event_time=str(alert.get("time") or _utc_now().isoformat()),
        )

    def maneuver_payload_from_pair(self, request_payload: dict[str, Any]) -> dict[str, Any] | None:
        primary = request_payload.get("primary") or {}
        secondary = request_payload.get("secondary") or {}
        primary_name = str(
            request_payload.get("sat1")
            or primary.get("name")
            or primary.get("id")
            or "SAT-A"
        )
        secondary_name = str(
            request_payload.get("sat2")
            or secondary.get("name")
            or secondary.get("id")
            or "SAT-B"
        )
        raw_distance = (
            request_payload.get("distanceKm")
            or request_payload.get("distance")
            or request_payload.get("minimumDistanceKm")
            or 1000.0
        )
        try:
            distance_km = float(raw_distance)
        except (TypeError, ValueError):
            distance_km = 1000.0

        return self._scenario_payload(
            primary_name=primary_name,
            secondary_name=secondary_name,
            distance_km=distance_km,
            alert_id=str(request_payload.get("id") or f"{primary_name}__{secondary_name}"),
            event_time=str(request_payload.get("time") or _utc_now().isoformat()),
            relative_velocity_km_s=request_payload.get("relativeVelocityKmS") or request_payload.get("relative_velocity_km_s"),
        )


SERVICE = RealtimeCatalogService()


class ApiHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _write_json(self, status_code: int, payload: Any) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/catalog/active":
            self._write_json(200, SERVICE.active_catalog_payload())
            return

        if path == "/objects":
            analysis = generate_analysis()
            objects = fetch_tle_data()
            self._write_json(
                200,
                {
                    "analysis": analysis,
                    "objects": objects,
                },
            )
            return

        if path == "/alerts":
            self._write_json(200, SERVICE.alerts_payload())
            return

        if path.startswith("/maneuver/"):
            alert_id = path.split("/maneuver/", 1)[1]
            payload = SERVICE.maneuver_payload(alert_id)
            if payload is None:
                self._write_json(404, {"error": "maneuver not found"})
            else:
                self._write_json(200, payload)
            return

        if path == "/health":
            self._write_json(200, {"status": "ok"})
            return

        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path != "/maneuver":
            self._write_json(404, {"error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0") or 0)
        raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json(400, {"error": "invalid json"})
            return

        maneuver_payload = SERVICE.maneuver_payload_from_pair(payload if isinstance(payload, dict) else {})
        if maneuver_payload is None:
            self._write_json(404, {"error": "maneuver not found"})
            return

        self._write_json(200, maneuver_payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cesium real-time catalog and alerts.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    with ThreadingHTTPServer((args.host, args.port), ApiHandler) as server:
        print(f"Realtime API listening on http://{args.host}:{args.port}")
        server.serve_forever()


if __name__ == "__main__":
    main()
