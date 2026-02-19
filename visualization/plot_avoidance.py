"""Avoidance and return trajectory visualization."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import PLOTS_DIR


def plot_avoidance_3d(relative_traj: np.ndarray, thrust_vector: np.ndarray, return_delta_v: float) -> str:
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    mid = len(relative_traj) // 2
    before = relative_traj[:mid]
    after = relative_traj[mid:] + thrust_vector.reshape(1, 3) * 200
    return_leg = after + np.array([0.0, -return_delta_v * 200, 0.0])

    ax.plot(before[:, 0], before[:, 1], before[:, 2], label="Before maneuver")
    ax.scatter(before[-1, 0], before[-1, 1], before[-1, 2], c="red", label="Collision point")
    ax.plot(after[:, 0], after[:, 1], after[:, 2], label="Avoidance trajectory")
    ax.plot(return_leg[:, 0], return_leg[:, 1], return_leg[:, 2], label="Return trajectory")

    ax.set_title("Autonomous Collision Avoidance Trajectory")
    ax.legend()
    out = PLOTS_DIR / "avoidance_trajectory.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return str(out)
