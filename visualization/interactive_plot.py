"""Interactive 3D Plotly visualization for LEO encounters."""
from __future__ import annotations

from pathlib import Path

import numpy as np
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:  # pragma: no cover
    go = None


def _to_xyz(arr: np.ndarray | None) -> np.ndarray | None:
    if arr is None:
        return None
    xyz = np.asarray(arr, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.shape[0] == 0:
        return None
    return xyz


def _earth_surface(radius_km: float) -> go.Surface:
    u = np.linspace(0.0, 2.0 * np.pi, 64)
    v = np.linspace(0.0, np.pi, 32)
    x = radius_km * np.outer(np.cos(u), np.sin(v))
    y = radius_km * np.outer(np.sin(u), np.sin(v))
    z = radius_km * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x,
        y=y,
        z=z,
        opacity=0.35,
        colorscale=[[0.0, "#B0C4DE"], [1.0, "#B0C4DE"]],
        showscale=False,
        name="Earth",
    )


def plot_interactive_3d(
    earth_radius_km,
    satellite_xyz,
    debris_xyz,
    tca_point=None,
    avoidance_xyz=None,
    return_xyz=None,
    title="LEO Encounter",
    output_path="interactive.html",
):
    if go is None:
        raise RuntimeError("Plotly is not installed. Please install `plotly` to enable interactive visualization.")

    satellite_xyz = _to_xyz(satellite_xyz)
    debris_xyz = _to_xyz(debris_xyz)
    avoidance_xyz = _to_xyz(avoidance_xyz)
    return_xyz = _to_xyz(return_xyz)
    tca = None if tca_point is None else np.asarray(tca_point, dtype=float).reshape(-1)

    if satellite_xyz is None:
        raise ValueError("satellite_xyz must be a non-empty Nx3 array.")
    if debris_xyz is None:
        raise ValueError("debris_xyz must be a non-empty Nx3 array.")

    fig = go.Figure()
    fig.add_trace(_earth_surface(float(earth_radius_km)))

    fig.add_trace(
        go.Scatter3d(
            x=satellite_xyz[:, 0],
            y=satellite_xyz[:, 1],
            z=satellite_xyz[:, 2],
            mode="lines",
            line=dict(color="blue", width=6),
            name="Satellite Orbit",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=debris_xyz[:, 0],
            y=debris_xyz[:, 1],
            z=debris_xyz[:, 2],
            mode="lines",
            line=dict(color="red", width=5),
            name="Debris Trajectory",
        )
    )

    if tca is not None and tca.size == 3:
        fig.add_trace(
            go.Scatter3d(
                x=[float(tca[0])],
                y=[float(tca[1])],
                z=[float(tca[2])],
                mode="markers",
                marker=dict(color="yellow", size=8, line=dict(color="black", width=1)),
                name="TCA Point",
            )
        )

    if avoidance_xyz is not None:
        fig.add_trace(
            go.Scatter3d(
                x=avoidance_xyz[:, 0],
                y=avoidance_xyz[:, 1],
                z=avoidance_xyz[:, 2],
                mode="lines",
                line=dict(color="green", width=5),
                name="Avoidance Arc",
            )
        )

    if return_xyz is not None:
        fig.add_trace(
            go.Scatter3d(
                x=return_xyz[:, 0],
                y=return_xyz[:, 1],
                z=return_xyz[:, 2],
                mode="lines",
                line=dict(color="purple", width=5),
                name="Return-to-Orbit",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            aspectmode="data",
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(x=0.01, y=0.99),
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output))
    return str(output)
