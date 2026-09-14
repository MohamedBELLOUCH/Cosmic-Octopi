"""Run the fitness-margin experiment from Graphics.ipynb.

Reads setup.json beside this file and writes plot-ready histories to result.pkl.
Run from the project root with: python "Fitness margins/script.py"
"""

import gc
import json
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np
import torch


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "Cosmic Octopi"))

from utils import (  # noqa: E402
    generate_Xi_matrices,
    generate_global_iterations_instants_reductionist,
    generate_learning_periods,
    generate_triggering_events_instants,
)
from octopus import Octopus  # noqa: E402
from oracle import Oracle  # noqa: E402


def summarize_oracle(oracle, parameters, threshold, overheads, instabilities):
    """Keep the histories used by the figures, without live gym environments."""
    return {
        "fitness_margin": threshold,
        "fitness_history": [
            np.stack([octopus.fitness_history[c] for octopus in oracle.octopi])
            for c in range(parameters["n_classes"])
        ],
        "freshness_history": [
            np.stack([octopus.freshness_history[c] for octopus in oracle.octopi])
            for c in range(parameters["n_classes"])
        ],
        "active_octopi_history": [
            [[octopus.agent_id for octopus in group] for group in class_history]
            for class_history in oracle.active_octopi_history
        ],
        "candidate_octopi_history": [
            [[octopus.agent_id for octopus in group] for group in class_history]
            for class_history in oracle.candidate_octopi_history
        ],
        "contributing_octopi_history": [
            [[octopus.agent_id for octopus in group] for group in class_history]
            for class_history in oracle.contributing_octopi_history
        ],
        "overheads": overheads,
        "instabilities": instabilities,
        "n_octopi": parameters["n_octopi"],
    }


def run_scenario(setup, threshold, xi_matrices):
    parameters = setup["system_dynamics_parameters"]
    n_classes = parameters["n_classes"]
    horizon = setup["horizon"]
    global_iterations_instants, oracle_iterations, time_horizon = (
        generate_global_iterations_instants_reductionist(parameters, horizon)
    )
    activation_periods, _ = generate_learning_periods(parameters, time_horizon)
    triggering_events_instants, number_of_episodes = (
        generate_triggering_events_instants(
            parameters, global_iterations_instants, activation_periods
        )
    )
    gravities = np.random.normal(
        setup["gravity_mean"], parameters["gamma_grav"], parameters["n_octopi"]
    )
    octopi = []
    try:
        for i in range(parameters["n_octopi"]):
            octopi.append(
                Octopus(
                    i,
                    activation_periods[i],
                    triggering_events_instants[i],
                    number_of_episodes[i],
                    global_iterations_instants,
                    xi_matrices[i],
                    setup["hyper_parameters"],
                    gravities[i],
                    setup.get("environment_ids"),
                )
            )
        oracle = Oracle(octopi, n_classes, global_iterations_instants, setup["hyper_parameters"])
        overhead_increments = [[] for _ in range(n_classes)]
        instability_increments = [[] for _ in range(n_classes)]
        for iteration_number, (class_index, class_iteration) in enumerate(oracle_iterations, 1):
            response = oracle.step(
                class_index,
                class_iteration,
                threshold,
                simulated_rewards=setup["simulated_rewards"],
                n_parallel=setup["n_parallel"],
            )
            if response is not None:
                overhead_increments[class_index].append(response[0])
                instability_increments[class_index].append(response[1])
            if iteration_number % 25 == 0 or iteration_number == len(oracle_iterations):
                print(f"  fitness margin {threshold}: {iteration_number}/{len(oracle_iterations)} iterations", flush=True)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            overheads = [np.sum(values[:-1]) for values in overhead_increments]
            instabilities = [np.mean(values[1:]) for values in instability_increments]
        return summarize_oracle(oracle, parameters, threshold, overheads, instabilities)
    finally:
        for octopus in octopi:
            for name in ("cheetah_marionette", "ant_marionette", "leg_marionette", "humanoid_marionette"):
                environment = getattr(octopus, name, None)
                if environment is not None:
                    environment.close()
        del octopi
        gc.collect()


def main():
    with (EXPERIMENT_DIR / "setup.json").open(encoding="utf-8") as file:
        setup = json.load(file)
    np.random.seed(setup["seed"])
    torch.manual_seed(setup["seed"])
    xi_matrix_path = setup.get("xi_matrix_file")
    if xi_matrix_path is not None:
        xi_matrix_path = EXPERIMENT_DIR / xi_matrix_path
    xi_matrices = generate_Xi_matrices(
        setup["system_dynamics_parameters"]["n_octopi"], xi_matrix_path
    )
    results = [
        run_scenario(setup, threshold, xi_matrices)
        for threshold in setup["fitness_margins"]
    ]
    result = {
        "setup": setup,
        "xi_matrix": {name: matrix.copy() for name, matrix in xi_matrices[0].items()},
        "scenarios": results,
    }
    output_path = EXPERIMENT_DIR / setup["output_file"]
    with output_path.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(results)} scenarios to {output_path}")


if __name__ == "__main__":
    main()
