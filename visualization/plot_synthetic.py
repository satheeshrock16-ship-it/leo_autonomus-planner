"""Synthetic scenario 3D plotting utilities."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


EARTH_RADIUS_KM = 6371.0


def _plot_earth(ax) -> None:
    u = np.linspace(0.0, 2.0 * np.pi, 50)
    v = np.linspace(0.0, np.pi, 25)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color="lightsteelblue", alpha=0.3, linewidth=0.0)


def _set_equal_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) * 0.5
    half_span = max(float((maxs - mins).max() * 0.5), EARTH_RADIUS_KM * 1.1)

    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1, 1, 1))


def plot_synthetic_scenario(
    scenario_name: str,
    satellite_orbit_km: np.ndarray,
    debris_orbit_km: np.ndarray,
    tca_point_km: np.ndarray,
    avoidance_orbit_km: np.ndarray,
    return_orbit_km: np.ndarray,
) -> str:
    satellite_orbit_km = np.asarray(satellite_orbit_km, dtype=float)
    debris_orbit_km = np.asarray(debris_orbit_km, dtype=float)
    tca_point_km = np.asarray(tca_point_km, dtype=float)
    avoidance_orbit_km = np.asarray(avoidance_orbit_km, dtype=float)
    return_orbit_km = np.asarray(return_orbit_km, dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    _plot_earth(ax)
    earth_proxy = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightsteelblue", markersize=10, label="Earth")

    sat_line, = ax.plot(
        satellite_orbit_km[:, 0],
        satellite_orbit_km[:, 1],
        satellite_orbit_km[:, 2],
        color="blue",
        linewidth=1.8,
        label="Original Satellite Orbit",
    )
    debris_line, = ax.plot(
        debris_orbit_km[:, 0],
        debris_orbit_km[:, 1],
        debris_orbit_km[:, 2],
        color="red",
        linewidth=1.6,
        label="Debris Trajectory",
    )
    tca_handle = ax.scatter(
        tca_point_km[0],
        tca_point_km[1],
        tca_point_km[2],
        color="yellow",
        edgecolors="black",
        s=90,
        marker="o",
        label="TCA Point",
    )
    avoid_line, = ax.plot(
        avoidance_orbit_km[:, 0],
        avoidance_orbit_km[:, 1],
        avoidance_orbit_km[:, 2],
        color="green",
        linewidth=1.8,
        label="Avoidance Trajectory",
    )
    return_line, = ax.plot(
        return_orbit_km[:, 0],
        return_orbit_km[:, 1],
        return_orbit_km[:, 2],
        color="purple",
        linewidth=1.8,
        label="Return-to-Orbit",
    )

    all_points = np.vstack(
        [
            satellite_orbit_km,
            debris_orbit_km,
            avoidance_orbit_km,
            return_orbit_km,
            tca_point_km.reshape(1, 3),
            np.array([[0.0, 0.0, 0.0]], dtype=float),
        ]
    )
    _set_equal_axes(ax, all_points)

    ax.set_title(f"Synthetic LEO Collision Scenario: {scenario_name}")
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend(handles=[earth_proxy, sat_line, debris_line, tca_handle, avoid_line, return_line], loc="best")

    output_path = PLOTS_DIR / f"synthetic_{scenario_name}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)
