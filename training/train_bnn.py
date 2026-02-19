"""Demonstration script for Bayesian uncertainty predictions."""
from ai.bnn_model import MonteCarloDropoutBNN


def main() -> None:
    model = MonteCarloDropoutBNN(samples=100)
    print(model.predict(0.2))


if __name__ == "__main__":
    main()
