"""3D orbit plotting utilities."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


def plot_orbits(
    protected_orbit: np.ndarray,
    debris_orbit: np.ndarray,
    title: str = "LEO Encounter",
    tca_satellite_km: np.ndarray | None = None,
    tca_debris_km: np.ndarray | None = None,
) -> str:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(protected_orbit[:, 0], protected_orbit[:, 1], protected_orbit[:, 2], label="Protected satellite")
    ax.plot(debris_orbit[:, 0], debris_orbit[:, 1], debris_orbit[:, 2], label="Highest-risk debris")
    if tca_satellite_km is not None and tca_debris_km is not None:
        tca_midpoint = (np.asarray(tca_satellite_km, dtype=float) + np.asarray(tca_debris_km, dtype=float)) * 0.5
        ax.scatter(
            tca_midpoint[0],
            tca_midpoint[1],
            tca_midpoint[2],
            c="red",
            s=80,
            marker="x",
            label="TCA",
        )
    ax.set_title(title)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_zlabel("Z (km)")
    ax.legend()
    out = PLOTS_DIR / "orbit_before.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)
