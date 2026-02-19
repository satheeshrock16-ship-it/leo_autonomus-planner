"""3D orbit plotting utilities."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


def plot_orbits(protected_orbit: np.ndarray, debris_orbit: np.ndarray, title: str = "LEO Encounter") -> str:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(protected_orbit[:, 0], protected_orbit[:, 1], protected_orbit[:, 2], label="Protected")
    ax.plot(debris_orbit[:, 0], debris_orbit[:, 1], debris_orbit[:, 2], label="Debris")
    ax.set_title(title)
    ax.set_xlabel("R")
    ax.set_ylabel("T")
    ax.set_zlabel("N")
    ax.legend()
    out = PLOTS_DIR / "orbit_before.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)
