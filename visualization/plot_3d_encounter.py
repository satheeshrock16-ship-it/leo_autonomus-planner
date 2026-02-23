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


EARTH_RADIUS_KM = np.float64(6378.137)
VISUAL_OFFSET_KM = np.float64(120.0)


def _earth_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(np.float64(0.0), np.float64(2.0) * np.float64(np.pi), 70, dtype=np.float64)
    v = np.linspace(np.float64(0.0), np.float64(np.pi), 36, dtype=np.float64)
    x = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
    y = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
    z = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z


def _cone_vector(vec_km_s: np.ndarray, reference_scale_km: float) -> np.ndarray:
    vec = np.asarray(vec_km_s, dtype=np.float64)
    vec_norm = np.float64(np.linalg.norm(vec))
    if vec_norm <= np.float64(0.0):
        return np.zeros(3, dtype=np.float64)
    scale_km = np.float64(max(reference_scale_km, 120.0))
    return (vec / vec_norm * scale_km).astype(np.float64)


def _rotation_r1(angle_rad: float) -> np.ndarray:
    c = np.float64(np.cos(angle_rad))
    s = np.float64(np.sin(angle_rad))
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _rotation_r3(angle_rad: float) -> np.ndarray:
    c = np.float64(np.cos(angle_rad))
    s = np.float64(np.sin(angle_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _estimate_temporary_orbit_full(
    avoidance_arc_km: np.ndarray,
    burn1_point_km: np.ndarray,
) -> np.ndarray:
    arc = np.asarray(avoidance_arc_km, dtype=np.float64)
    burn1 = np.asarray(burn1_point_km, dtype=np.float64).reshape(3)
    if arc.ndim != 2 or arc.shape[1] != 3 or len(arc) < 8:
        return np.empty((0, 3), dtype=np.float64)

    # Fit the temporary orbital plane from avoidance samples.
    centered = arc - np.mean(arc, axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except Exception:
        return np.empty((0, 3), dtype=np.float64)

    u_hat = np.asarray(vh[0], dtype=np.float64)
    v_hat = np.asarray(vh[1], dtype=np.float64)
    h_hat = np.cross(u_hat, v_hat).astype(np.float64)
    h_norm = np.float64(np.linalg.norm(h_hat))
    if h_norm <= np.float64(1e-12):
        return np.empty((0, 3), dtype=np.float64)
    h_hat = h_hat / h_norm
    v_hat = np.cross(h_hat, u_hat).astype(np.float64)
    v_hat /= max(float(np.linalg.norm(v_hat)), 1e-12)

    # Polar coordinates in fitted plane.
    x_plane = arc @ u_hat
    y_plane = arc @ v_hat
    r_plane = np.sqrt((x_plane * x_plane) + (y_plane * y_plane))
    if np.any(r_plane <= np.float64(1e-9)):
        return np.empty((0, 3), dtype=np.float64)
    phi = np.arctan2(y_plane, x_plane)

    # Linearized conic fit with focus at origin:
    # 1/r = A + B cos(phi) + C sin(phi)
    design = np.column_stack([np.ones_like(phi), np.cos(phi), np.sin(phi)]).astype(np.float64)
    inv_r = (1.0 / r_plane).astype(np.float64)
    try:
        coeff, *_ = np.linalg.lstsq(design, inv_r, rcond=None)
    except Exception:
        return np.empty((0, 3), dtype=np.float64)

    a0, b0, c0 = float(coeff[0]), float(coeff[1]), float(coeff[2])
    if abs(a0) <= 1e-12:
        return np.empty((0, 3), dtype=np.float64)
    p = 1.0 / a0
    ecc = float(np.sqrt((b0 * b0) + (c0 * c0)) / max(abs(a0), 1e-12))
    ecc = float(np.clip(ecc, 0.0, 0.999999))
    if abs(1.0 - (ecc * ecc)) <= 1e-12:
        return np.empty((0, 3), dtype=np.float64)
    a = float(p / max(1.0 - (ecc * ecc), 1e-12))

    # Argument of periapsis inside fitted plane basis.
    argp_plane = float(np.arctan2(c0, b0))

    # Convert fitted basis to classical orientation (Omega, i, omega), then
    # use canonical perifocal->ECI rotation for full 0..2pi sweep.
    k_hat = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    node_vec = np.cross(k_hat, h_hat).astype(np.float64)
    node_norm = float(np.linalg.norm(node_vec))
    inc = float(np.arccos(np.clip(h_hat[2], -1.0, 1.0)))
    raan = float(np.arctan2(node_vec[1], node_vec[0])) if node_norm > 1e-12 else 0.0

    p_hat = (np.cos(argp_plane) * u_hat) + (np.sin(argp_plane) * v_hat)
    if node_norm > 1e-12:
        omega = float(np.arctan2(np.dot(np.cross(node_vec, p_hat), h_hat), np.dot(node_vec, p_hat)))
    else:
        omega = float(np.arctan2(p_hat[1], p_hat[0]))

    # M0_1 estimate at Burn 1 (computed and retained for geometric consistency).
    q_hat = np.cross(h_hat, p_hat).astype(np.float64)
    q_hat /= max(float(np.linalg.norm(q_hat)), 1e-12)
    x_b1 = float(np.dot(burn1, p_hat))
    y_b1 = float(np.dot(burn1, q_hat))
    f_b1 = float(np.arctan2(y_b1, x_b1))
    if ecc < 1e-10:
        m0_1 = f_b1
    else:
        e_anom = float(
            2.0
            * np.arctan2(
                np.sqrt(max(1.0 - ecc, 0.0)) * np.sin(0.5 * f_b1),
                np.sqrt(max(1.0 + ecc, 1e-12)) * np.cos(0.5 * f_b1),
            )
        )
        m0_1 = float(e_anom - (ecc * np.sin(e_anom)))
    _ = m0_1

    f_sweep = np.linspace(0.0, 2.0 * np.pi, 500, dtype=np.float64)
    denom = 1.0 + (ecc * np.cos(f_sweep))
    denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
    r_sweep = a * (1.0 - (ecc * ecc)) / denom
    r_pf = np.column_stack([r_sweep * np.cos(f_sweep), r_sweep * np.sin(f_sweep), np.zeros_like(f_sweep)])

    rot_pf_to_eci = _rotation_r3(raan) @ _rotation_r1(inc) @ _rotation_r3(omega)
    temp_orbit = (rot_pf_to_eci @ r_pf.T).T
    return np.asarray(temp_orbit, dtype=np.float64)


def plot_3d_encounter(
    scenario_name: str,
    satellite_orbit_km: np.ndarray,
    debris_orbit_km: np.ndarray,
    tca_point_km: np.ndarray,
    avoidance_arc_km: np.ndarray,
    restored_orbit_segment_km: np.ndarray,
    burn1_point_km: np.ndarray,
    burn2_point_km: np.ndarray,
    delta_v1_vector_km_s: np.ndarray,
    delta_v2_vector_km_s: np.ndarray,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    sat = np.asarray(satellite_orbit_km, dtype=np.float64)
    deb = np.asarray(debris_orbit_km, dtype=np.float64)
    tca = np.asarray(tca_point_km, dtype=np.float64).reshape(3)
    avoid = np.asarray(avoidance_arc_km, dtype=np.float64)
    restored = np.asarray(restored_orbit_segment_km, dtype=np.float64)
    burn1 = np.asarray(burn1_point_km, dtype=np.float64).reshape(3)
    burn2 = np.asarray(burn2_point_km, dtype=np.float64).reshape(3)
    dv1 = np.asarray(delta_v1_vector_km_s, dtype=np.float64).reshape(3)
    dv2 = np.asarray(delta_v2_vector_km_s, dtype=np.float64).reshape(3)
    temp_full = _estimate_temporary_orbit_full(avoid, burn1)
    # Rendering-only separation: keep physics arrays unchanged.
    sat_plot = sat.copy()
    sat_plot[:, 2] = sat_plot[:, 2] + VISUAL_OFFSET_KM

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
            x=sat_plot[:, 0],
            y=sat_plot[:, 1],
            z=sat_plot[:, 2],
            mode="lines",
            line=dict(color="lightskyblue", width=4),
            name="Original Orbit",
        )
    )
    if len(temp_full) > 1:
        fig.add_trace(
            go.Scatter3d(
                x=temp_full[:, 0],
                y=temp_full[:, 1],
                z=temp_full[:, 2],
                mode="lines",
                line=dict(color="red", width=2, dash="dash"),
                opacity=0.6,
                name="Temporary Orbit (Full)",
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=deb[:, 0],
            y=deb[:, 1],
            z=deb[:, 2],
            mode="lines",
            line=dict(color="crimson", width=3),
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

    fig.add_trace(
        go.Scatter3d(
            x=avoid[:, 0],
            y=avoid[:, 1],
            z=avoid[:, 2],
            mode="lines",
            line=dict(color="green", width=5),
            name="Avoidance Arc (Phase Shift Arc)",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=restored[:, 0],
            y=restored[:, 1],
            z=restored[:, 2],
            mode="lines",
            line=dict(color="blue", width=5),
            name="Restored Orbit Segment",
        )
    )

    fig.add_trace(
        go.Scatter3d(
            x=[burn1[0]],
            y=[burn1[1]],
            z=[burn1[2]],
            mode="markers+text",
            marker=dict(size=8, color="orange"),
            text=["Burn 1 \u2013 Avoidance"],
            textposition="top center",
            name="Burn 1 \u2013 Avoidance",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[burn2[0]],
            y=[burn2[1]],
            z=[burn2[2]],
            mode="markers+text",
            marker=dict(size=8, color="purple"),
            text=["Burn 2 \u2013 Restoration"],
            textposition="top center",
            name="Burn 2 \u2013 Restoration",
        )
    )

    reference_scale_km = float(np.linalg.norm(avoid[-1] - avoid[0])) * 0.08 if len(avoid) > 1 else 160.0
    dv1_plot_vec = _cone_vector(dv1, reference_scale_km)
    dv2_plot_vec = _cone_vector(dv2, reference_scale_km)

    if float(np.linalg.norm(dv1_plot_vec)) > 0.0:
        fig.add_trace(
            go.Cone(
                x=[burn1[0]],
                y=[burn1[1]],
                z=[burn1[2]],
                u=[dv1_plot_vec[0]],
                v=[dv1_plot_vec[1]],
                w=[dv1_plot_vec[2]],
                anchor="tail",
                sizemode="absolute",
                sizeref=max(reference_scale_km * 0.2, 30.0),
                colorscale=[[0.0, "orange"], [1.0, "orange"]],
                showscale=False,
                name="\u0394v1 Cone Arrow",
                showlegend=True,
            )
        )

    if float(np.linalg.norm(dv2_plot_vec)) > 0.0:
        fig.add_trace(
            go.Cone(
                x=[burn2[0]],
                y=[burn2[1]],
                z=[burn2[2]],
                u=[dv2_plot_vec[0]],
                v=[dv2_plot_vec[1]],
                w=[dv2_plot_vec[2]],
                anchor="tail",
                sizemode="absolute",
                sizeref=max(reference_scale_km * 0.2, 30.0),
                colorscale=[[0.0, "purple"], [1.0, "purple"]],
                showscale=False,
                name="\u0394v2 Cone Arrow",
                showlegend=True,
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
