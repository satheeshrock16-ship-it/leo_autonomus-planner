"""Train the burn/no-burn classifier on synthetic encounter features."""
import numpy as np

from ai.lightgbm_model import BurnDecisionModel


def main() -> None:
    rng = np.random.default_rng(42)
    X = rng.uniform(0, 1, size=(500, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.7).astype(int)

    model = BurnDecisionModel()
    model.fit(X, y)
    print(model.predict(np.array([0.8, 0.3, 0.2, 0.1])))


if __name__ == "__main__":
    main()
