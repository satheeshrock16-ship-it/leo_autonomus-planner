"""Plotly-based 3D visualization for synthetic encounter scenarios."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from config import PLOTS_DIR

try:
    import plotly.graph_objects as go
    import plotly.io as pio
except Exception:  # pragma: no cover - optional dependency
    go = None
    pio = None


EARTH_RADIUS_KM = 6378.137


def _earth_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(0.0, 2.0 * np.pi, 70)
    v = np.linspace(0.0, np.pi, 36)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def plot_3d_encounter(
    scenario_name: str,
    satellite_orbit_km: np.ndarray,
    debris_orbit_km: np.ndarray,
    tca_point_km: np.ndarray,
    delta_v_vector_km_s: np.ndarray,
    avoidance_arc_km: np.ndarray,
    return_arc_km: np.ndarray,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    sat = np.asarray(satellite_orbit_km, dtype=float)
    deb = np.asarray(debris_orbit_km, dtype=float)
    tca = np.asarray(tca_point_km, dtype=float)
    dv = np.asarray(delta_v_vector_km_s, dtype=float)
    avoid = np.asarray(avoidance_arc_km, dtype=float)
    ret = np.asarray(return_arc_km, dtype=float)

    out_dir = PLOTS_DIR if output_dir is None else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{scenario_name.upper()}_interactive.html"
    png_path = out_dir / f"{scenario_name.upper()}_snapshot.png"

    if go is None or pio is None:
        print("Plotly not available - skipping interactive visualization export.")
        return {"interactive_html": "", "snapshot_png": ""}

    earth_x, earth_y, earth_z = _earth_surface()
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=earth_x,
            y=earth_y,
            z=earth_z,
            opacity=0.35,
            colorscale="Blues",
            showscale=False,
            name="Earth",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=sat[:, 0],
            y=sat[:, 1],
            z=sat[:, 2],
            mode="lines",
            line=dict(color="royalblue", width=5),
            name="Satellite Orbit",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=deb[:, 0],
            y=deb[:, 1],
            z=deb[:, 2],
            mode="lines",
            line=dict(color="crimson", width=4),
            name="Debris Orbit",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[tca[0]],
            y=[tca[1]],
            z=[tca[2]],
            mode="markers",
            marker=dict(size=7, color="gold"),
            name="TCA",
        )
    )

    dv_mag = float(np.linalg.norm(dv))
    if dv_mag > 0.0:
        dv_hat = dv / dv_mag
    else:
        dv_hat = np.zeros(3, dtype=float)
    dv_tail = avoid[0]
    dv_len = max(float(np.linalg.norm(avoid[-1] - avoid[0])) * 0.15, 20.0)
    dv_head = dv_tail + dv_hat * dv_len
    fig.add_trace(
        go.Scatter3d(
            x=[dv_tail[0], dv_head[0]],
            y=[dv_tail[1], dv_head[1]],
            z=[dv_tail[2], dv_head[2]],
            mode="lines",
            line=dict(color="cyan", width=7),
            name="Delta-v Vector",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=avoid[:, 0],
            y=avoid[:, 1],
            z=avoid[:, 2],
            mode="lines",
            line=dict(color="limegreen", width=4),
            name="Avoidance Arc",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=ret[:, 0],
            y=ret[:, 1],
            z=ret[:, 2],
            mode="lines",
            line=dict(color="mediumpurple", width=4),
            name="Return-to-Orbit Arc",
        )
    )

    fig.update_layout(
        title=f"{scenario_name.upper()} Synthetic Encounter",
        scene=dict(
            xaxis_title="X (km)",
            yaxis_title="Y (km)",
            zaxis_title="Z (km)",
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    fig.write_html(str(html_path))
    try:
        pio.kaleido.scope.default_format = "png"
        fig.write_image(str(png_path))
    except Exception:
        print("Kaleido image export unavailable - HTML output generated only.")
        png_path = Path("")
    return {"interactive_html": str(html_path), "snapshot_png": str(png_path)}
