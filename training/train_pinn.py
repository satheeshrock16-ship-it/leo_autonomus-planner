"""Generate synthetic CW trajectories and evaluate PINN residual scores."""
import numpy as np

from ai.pinn_model import CWPhysicsInformedValidator
from physics.cw_equation import propagate_cw
from config import EARTH_MU_KM3_S2


def main() -> None:
    n = np.sqrt(EARTH_MU_KM3_S2 / (6878.0**3))
    t = np.linspace(0, 1800, 200)
    x, v = propagate_cw(np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), n, t)
    result = CWPhysicsInformedValidator().evaluate(x, v, n, dt=t[1] - t[0])
    print(result)


if __name__ == "__main__":
    main()
