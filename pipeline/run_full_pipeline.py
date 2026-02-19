"""End-to-end autonomous LEO collision avoidance orchestration."""
from __future__ import annotations

import numpy as np

from ai.bnn_model import MonteCarloDropoutBNN
from ai.lightgbm_model import BurnDecisionModel
from ai.pinn_model import CWPhysicsInformedValidator
from config import COLLISION_PROBABILITY_THRESHOLD
from physics.cw_equation import mean_motion, propagate_cw
from pipeline import collision_check, fetch_data, preprocess, replan
from pipeline.propulsion_interface import ThrustCommand
from visualization.plot_avoidance import plot_avoidance_3d


def run_autonomous_cycle(fetch_live_data: bool = False) -> dict:
    sat_path, debris_path = (None, None)
    if fetch_live_data:
        sat_path, debris_path = fetch_data.run(limit=100)
        preprocess.run(sat_path, debris_path)

    n = mean_motion(mu=398600.4418, semi_major_axis_km=6878.0)
    t = np.linspace(0, 1800, 300)
    rel_pos, rel_vel = propagate_cw(np.array([0.8, 0.2, 0.1]), np.array([0.0, 0.0, 0.0]), n, t)

    pinn_report = CWPhysicsInformedValidator().evaluate(rel_pos, rel_vel, n=n, dt=t[1] - t[0])
    covariance = np.diag([0.5, 0.5, 0.5])
    pc = collision_check.run(rel_pos[len(rel_pos) // 2], covariance)

    bnn = MonteCarloDropoutBNN()
    uncertainty = bnn.predict(pc)

    decision_model = BurnDecisionModel()
    X = np.array([[0.1, 0.2, 0.2, 0.3], [0.8, 0.7, 0.3, 0.5], [0.9, 0.1, 0.1, 0.1]])
    y = np.array([0, 1, 1])
    decision_model.fit(X, y)
    decision = decision_model.predict(np.array([pc, uncertainty.epistemic_std, 0.2, 0.3]))

    output = {
        "collision_probability": pc,
        "pinn_physics_mse": pinn_report.mse_physics,
        "decision": decision.should_burn,
        "decision_confidence": decision.confidence,
    }

    if pc > COLLISION_PROBABILITY_THRESHOLD and decision.should_burn:
        plan = replan.run(current_radius_km=6878.0, nominal_radius_km=6878.0)
        cmd = ThrustCommand(
            thrust_vector=[float(v) for v in plan["thrust_vector"]],
            duration_ms=plan["duration_ms"],
            burn_type="tangential",
        )
        output["thrust_command"] = {
            "thrust_vector": cmd.thrust_vector,
            "duration_ms": cmd.duration_ms,
            "burn_type": cmd.burn_type,
        }
        plot_avoidance_3d(rel_pos, plan["thrust_vector"], return_delta_v=plan["return_delta_v_km_s"])
    else:
        output["thrust_command"] = None

    return output


if __name__ == "__main__":
    summary = run_autonomous_cycle(fetch_live_data=False)
    print(summary)
