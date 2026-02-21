"""Train a LightGBM regressor for required delta-v magnitude."""
from __future__ import annotations

import logging
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai.delta_v_regressor import DeltaVRegressor
from config import MODEL_DIR, RESULTS_DIR, SYNTHETIC_ML_CONFIG
from physics.maneuver_optimizer import analytical_required_delta_v_km_s


LOGGER = logging.getLogger(__name__)
MODEL_PATH = MODEL_DIR / "delta_v_model.txt"


def _build_dataset(sample_count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pc = rng.uniform(0.0, 0.25, size=sample_count)
    miss_distance = rng.uniform(0.05, 5.0, size=sample_count)
    rel_velocity = rng.uniform(0.1, 15.0, size=sample_count)
    time_to_tca = rng.uniform(600.0, 24.0 * 3600.0, size=sample_count)
    altitude = rng.uniform(450.0, 1200.0, size=sample_count)
    fuel_remaining = rng.uniform(0.2, 1.0, size=sample_count)

    X = np.column_stack([pc, miss_distance, rel_velocity, time_to_tca, altitude, fuel_remaining])
    targets = np.array(
        [
            analytical_required_delta_v_km_s(
                miss_distance_km=float(md),
                separation_constraint_km=2.0 + float(0.5 * risk),
                lead_time_s=float(ttc),
                max_delta_v_km_s=float(0.03 * fuel),
            )
            for risk, md, _, ttc, _, fuel in X
        ],
        dtype=float,
    )
    return X, targets


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    sample_count = int(SYNTHETIC_ML_CONFIG.get("training_samples", 2500))
    seed = int(SYNTHETIC_ML_CONFIG.get("random_seed", 42))
    X, y = _build_dataset(sample_count=sample_count, seed=seed)

    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = DeltaVRegressor()
    model.fit(X_train, y_train)
    model.save(MODEL_PATH)
    metrics = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = RESULTS_DIR / "delta_v_regression_comparison.png"
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=18, label="Samples")
    lim_min = float(min(np.min(y_test), np.min(y_pred)))
    lim_max = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "r--", label="Ideal")
    plt.xlabel("Analytical required delta-v (km/s)")
    plt.ylabel("ML predicted delta-v (km/s)")
    plt.title("Analytical vs ML delta-v")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()

    LOGGER.info("Saved model to %s", MODEL_PATH)
    LOGGER.info("MAE=%.6f RMSE=%.6f R2=%.6f", metrics.mae, metrics.rmse, metrics.r2)
    LOGGER.info("Saved comparison plot to %s", fig_path)


if __name__ == "__main__":
    main()
