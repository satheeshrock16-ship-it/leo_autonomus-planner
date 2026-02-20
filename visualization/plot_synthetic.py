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
    scenario_plot_tag: str,
    satellite_original_states: np.ndarray,
    debris_states: np.ndarray,
    tca_point_km: np.ndarray,
    avoidance_states: np.ndarray,
    return_states: np.ndarray,
) -> str:
    satellite_original_states = np.asarray(satellite_original_states, dtype=float)
    debris_states = np.asarray(debris_states, dtype=float)
    tca_point_km = np.asarray(tca_point_km, dtype=float)
    avoidance_states = np.asarray(avoidance_states, dtype=float)
    return_states = np.asarray(return_states, dtype=float)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    _plot_earth(ax)
    earth_proxy = plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="lightsteelblue", markersize=10, label="Earth")

    sat_line, = ax.plot(
        satellite_original_states[:, 0],
        satellite_original_states[:, 1],
        satellite_original_states[:, 2],
        color="blue",
        linewidth=2.4,
        label="Original Satellite Orbit",
        zorder=3,
    )
    debris_line, = ax.plot(
        debris_states[:, 0],
        debris_states[:, 1],
        debris_states[:, 2],
        color="red",
        linewidth=1.8,
        label="Debris Trajectory",
        zorder=4,
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
        avoidance_states[:, 0],
        avoidance_states[:, 1],
        avoidance_states[:, 2],
        color="green",
        linewidth=2.0,
        label="Avoidance Arc",
        zorder=5,
    )
    return_line, = ax.plot(
        return_states[:, 0],
        return_states[:, 1],
        return_states[:, 2],
        color="purple",
        linewidth=2.0,
        label="Return-to-Orbit",
        zorder=6,
    )

    all_points = np.vstack(
        [
            satellite_original_states,
            debris_states,
            avoidance_states,
            return_states,
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

    output_path = PLOTS_DIR / f"synthetic_{scenario_plot_tag}.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)
