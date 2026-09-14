"""Generate the Ant octopus-count experiment's simulated-reward results."""

import json
from pathlib import Path
import pickle
import sys

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "Cosmic Octopi"))
from simulated_reward_experiment import load_xi_matrix, run_scenario, validate_setup  # noqa: E402


def main():
    with (EXPERIMENT_DIR / "setup.json").open(encoding="utf-8") as file:
        setup = json.load(file)
    validate_setup(setup)
    xi_matrix = load_xi_matrix(
        EXPERIMENT_DIR / setup["xi_matrix_file"], setup["class_name"]
    )
    rng = np.random.default_rng(setup["seed"])
    scenarios = [
        run_scenario(
            setup, xi_matrix, n_octopi, setup["gamma_grav"],
            rng.integers(0, 2**63, size=setup["n_repetitions"]),
        )
        for n_octopi in setup["n_octopi_values"]
    ]
    result = {"setup": setup, "xi_matrix": xi_matrix, "scenarios": scenarios}
    output_path = EXPERIMENT_DIR / setup["output_file"]
    with output_path.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(scenarios)} octopus-count scenarios to {output_path}")


if __name__ == "__main__":
    main()
