"""3D orbit plotting utilities."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR, PROCESSED_DATA_DIR


EARTH_RADIUS_KM = 6371.0


def _plot_earth(ax, radius_km: float = EARTH_RADIUS_KM) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 50)
    v = np.linspace(0.0, np.pi, 25)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightsteelblue", alpha=0.35, linewidth=0.0, label="Earth")


def _set_equal_axes(ax, points: np.ndarray) -> None:
    if points.size == 0:
        span = EARTH_RADIUS_KM * 1.1
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(-span, span)
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    half_span = max(float((maxs - mins).max() * 0.5), EARTH_RADIUS_KM * 1.1)

    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1, 1, 1))


def _load_all_debris_points(csv_path: Path) -> np.ndarray:
    if not csv_path.exists():
        return np.empty((0, 3), dtype=float)

    points: list[list[float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("object_role") != "debris":
                continue
            try:
                points.append([
                    float(row["x_km"]),
                    float(row["y_km"]),
                    float(row["z_km"]),
                ])
            except (KeyError, TypeError, ValueError):
                continue

    if not points:
        return np.empty((0, 3), dtype=float)
    return np.asarray(points, dtype=float)


def plot_orbits(
    protected_orbit: np.ndarray,
    debris_orbit: np.ndarray,
    title: str = "LEO Encounter",
    tca_satellite_km: np.ndarray | None = None,
    tca_debris_km: np.ndarray | None = None,
) -> str:
    protected_orbit = np.asarray(protected_orbit, dtype=float)
    debris_orbit = np.asarray(debris_orbit, dtype=float)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    _plot_earth(ax)
    earth_proxy = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightsteelblue", markersize=10, label="Earth")

    sat_line, = ax.plot(
        protected_orbit[:, 0],
        protected_orbit[:, 1],
        protected_orbit[:, 2],
        color="royalblue",
        linewidth=1.8,
        label="Protected Satellite",
    )

    debris_points = _load_all_debris_points(PROCESSED_DATA_DIR / "propagated_states.csv")
    debris_handle = None
    if debris_points.size > 0:
        downsample_step = max(1, int(len(debris_points) / 4000))
        sampled = debris_points[::downsample_step]
        debris_handle = ax.scatter(
            sampled[:, 0],
            sampled[:, 1],
            sampled[:, 2],
            color="red",
            s=6,
            alpha=0.55,
            label="Debris",
        )

    closest_line, = ax.plot(
        debris_orbit[:, 0],
        debris_orbit[:, 1],
        debris_orbit[:, 2],
        color="orange",
        linewidth=2.0,
        label="Closest Approach",
    )

    tca_handle = None
    if tca_satellite_km is not None and tca_debris_km is not None:
        tca_midpoint = (np.asarray(tca_satellite_km, dtype=float) + np.asarray(tca_debris_km, dtype=float)) * 0.5
        tca_handle = ax.scatter(
            tca_midpoint[0],
            tca_midpoint[1],
            tca_midpoint[2],
            color="yellow",
            edgecolors="black",
            s=90,
            marker="o",
            label="TCA",
        )

    ref_points = [protected_orbit, debris_orbit, np.array([[0.0, 0.0, 0.0]], dtype=float)]
    if debris_points.size > 0:
        ref_points.append(debris_points)
    if tca_satellite_km is not None:
        ref_points.append(np.asarray(tca_satellite_km, dtype=float).reshape(1, 3))
    if tca_debris_km is not None:
        ref_points.append(np.asarray(tca_debris_km, dtype=float).reshape(1, 3))
    all_points = np.vstack(ref_points)
    _set_equal_axes(ax, all_points)

    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")

    legend_handles = [earth_proxy, sat_line]
    if debris_handle is not None:
        legend_handles.append(debris_handle)
    legend_handles.append(closest_line)
    if tca_handle is not None:
        legend_handles.append(tca_handle)
    ax.legend(handles=legend_handles, loc="best")

    out = PLOTS_DIR / "orbit_before.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return str(out)
