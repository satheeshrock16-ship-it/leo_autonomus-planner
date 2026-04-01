"""ECI/TEME to geodetic conversion helpers for Cesium-ready visualization."""
from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Iterable

WGS84_A_KM = 6378.137
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def _utc_datetime(epoch: datetime) -> datetime:
    if epoch.tzinfo is None:
        return epoch.replace(tzinfo=timezone.utc)
    return epoch.astimezone(timezone.utc)


def julian_date(epoch: datetime) -> float:
    dt = _utc_datetime(epoch)
    year = dt.year
    month = dt.month
    day = dt.day

    if month <= 2:
        year -= 1
        month += 12

    a = year // 100
    b = 2 - a + (a // 4)

    day_fraction = (
        dt.hour / 24.0
        + dt.minute / 1440.0
        + (dt.second + dt.microsecond / 1_000_000.0) / 86400.0
    )

    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + day
        + day_fraction
        + b
        - 1524.5
    )


def gmst_radians(epoch: datetime) -> float:
    jd = julian_date(epoch)
    t = (jd - 2451545.0) / 36525.0
    gmst_deg = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * (t ** 2)
        - (t ** 3) / 38710000.0
    )
    return math.radians(gmst_deg % 360.0)


def eci_km_to_ecef_km(r_eci_km: Iterable[float], epoch: datetime) -> tuple[float, float, float]:
    x_eci, y_eci, z_eci = (float(v) for v in r_eci_km)
    theta = gmst_radians(epoch)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_ecef = cos_t * x_eci + sin_t * y_eci
    y_ecef = -sin_t * x_eci + cos_t * y_eci
    z_ecef = z_eci
    return x_ecef, y_ecef, z_ecef


def ecef_km_to_geodetic(x_km: float, y_km: float, z_km: float) -> tuple[float, float, float]:
    p = math.hypot(x_km, y_km)
    lon = math.atan2(y_km, x_km)

    if p < 1e-12:
      lat = math.copysign(math.pi / 2.0, z_km)
      alt_km = abs(z_km) - WGS84_A_KM * math.sqrt(1.0 - WGS84_E2)
      return math.degrees(lat), math.degrees(lon), alt_km * 1000.0

    lat = math.atan2(z_km, p * (1.0 - WGS84_E2))
    alt_km = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A_KM / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt_km = p / max(math.cos(lat), 1e-12) - n
        lat = math.atan2(z_km, p * (1.0 - (WGS84_E2 * n / max(n + alt_km, 1e-12))))

    lon_deg = ((math.degrees(lon) + 180.0) % 360.0) - 180.0
    lat_deg = math.degrees(lat)
    return lat_deg, lon_deg, alt_km * 1000.0


def eci_km_to_geodetic(r_eci_km: Iterable[float], epoch: datetime) -> tuple[float, float, float]:
    x_ecef, y_ecef, z_ecef = eci_km_to_ecef_km(r_eci_km, epoch)
    return ecef_km_to_geodetic(x_ecef, y_ecef, z_ecef)
