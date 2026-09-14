"""Generate the two synchronization/request-heterogeneity plot datasets.

Run from the project root with:
    python "Synchronization and request rates/script.py"
The setup is read from setup.json beside this script. The result is
written there as result.pkl, regardless of the current working directory.
"""

import json
from pathlib import Path
import pickle
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "Cosmic Octopi"))

from utils import (  # noqa: E402
    generate_global_iterations_instants_holistic,
    generate_learning_periods,
    generate_triggering_events_instants,
)


CONFIG_PATH = Path(__file__).with_name("setup.json")


def generate_scenario(base_parameters, horizon, gamma_heter, gamma_sync):
    parameters = {
        **base_parameters,
        "gamma_heter": gamma_heter,
        "gamma_sync": gamma_sync,
    }
    global_iterations_instants, oracle_iterations, time_horizon = (
        generate_global_iterations_instants_holistic(parameters, horizon)
    )
    activation_periods, _ = generate_learning_periods(parameters, time_horizon)
    triggering_events_instants, number_of_episodes = (
        generate_triggering_events_instants(
            parameters, global_iterations_instants, activation_periods
        )
    )
    return {
        "parameters": parameters,
        "global_iterations_instants": global_iterations_instants,
        "activation_periods": activation_periods,
        "triggering_events_instants": triggering_events_instants,
        "oracle_iterations": oracle_iterations,
        "number_of_episodes": number_of_episodes,
        "time_horizon": time_horizon,
    }


def main():
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)
    np.random.seed(config["seed"])
    data = {
        "seed": config["seed"],
        "horizon": config["horizon"],
        "scenarios": [
            generate_scenario(config["base_parameters"], config["horizon"], **setting)
            for setting in config["scenarios"]
        ],
    }
    output_path = Path(__file__).with_name(config["output_file"])
    with output_path.open("wb") as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(data['scenarios'])} scenarios to {output_path}")


if __name__ == "__main__":
    main()
