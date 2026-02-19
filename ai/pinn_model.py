"""Physics-Informed NN scaffold for CW consistency validation."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PINNResult:
    mse_data: float
    mse_physics: float


class CWPhysicsInformedValidator:
    """Lightweight PINN-like residual checker (framework-agnostic)."""

    def evaluate(self, position: np.ndarray, velocity: np.ndarray, n: float, dt: float) -> PINNResult:
        acc = np.gradient(velocity, dt, axis=0)
        x, y, z = position[:, 0], position[:, 1], position[:, 2]
        xdot, ydot = velocity[:, 0], velocity[:, 1]

        r1 = acc[:, 0] - 2 * n * ydot - 3 * n**2 * x
        r2 = acc[:, 1] + 2 * n * xdot
        r3 = acc[:, 2] + n**2 * z

        mse_physics = float(np.mean(r1**2 + r2**2 + r3**2))
        mse_data = float(np.mean(position**2) * 1e-6)
        return PINNResult(mse_data=mse_data, mse_physics=mse_physics)
