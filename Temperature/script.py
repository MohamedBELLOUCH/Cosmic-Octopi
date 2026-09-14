"""Run the Leg/Hopper fitness-margin and temperature sweep with simulated rewards.

This follows the Graphics notebook's request, candidate, upload, global-fitness,
overhead, and instability calculations with the paper's Leg class. Policy
weights are not trained when simulated_rewards is true, so the fixed Hopper
model size represents each uploaded update without constructing unused MuJoCo
environments or models.
"""

import json
from pathlib import Path
import pickle
import sys

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "Cosmic Octopi"))

from simulated_reward_experiment import simulate_ou_episodes  # noqa: E402
from utils import (  # noqa: E402
    CLASS_NAMES,
    Network,
    activations_candidacies,
    generate_global_iterations_instants_reductionist,
    generate_learning_periods,
    generate_triggering_events_instants,
    generate_Xi_matrices,
    get_model_size_in_kb,
)


def run_scenario(setup, xi_matrix, fitness_margin, temperature, seed, model_size_kb):
    parameters = setup["system_dynamics_parameters"]
    n_octopi = parameters["n_octopi"]
    class_index = setup["class_index"]
    np.random.seed(int(seed) % (2**32))
    reward_rng = np.random.default_rng(int(seed))

    global_instants, oracle_iterations, time_horizon = (
        generate_global_iterations_instants_reductionist(parameters, setup["horizon"])
    )
    learning_periods, _ = generate_learning_periods(parameters, time_horizon)
    triggering_instants, episode_counts = generate_triggering_events_instants(
        parameters, global_instants, learning_periods
    )
    class_iterations = [
        iteration for iteration in oracle_iterations if iteration[0] == class_index
    ][:setup["max_global_iterations"]]
    candidacies = [
        activations_candidacies(
            learning_periods[i], triggering_instants[i], global_instants
        )[1][class_index]
        for i in range(n_octopi)
    ]
    g = setup["gravity_mean"]
    gravities = np.random.gamma(
        shape=g**2 / parameters["gamma_grav"],
        scale=parameters["gamma_grav"] / g,
        size=n_octopi,
    )

    last_global_iteration = np.zeros(n_octopi, dtype=int)
    last_global_fitness = np.zeros(n_octopi)
    last_contributor_count = np.ones(n_octopi, dtype=int)
    lower_references = [[] for _ in range(n_octopi)]
    last_rewards = [None] * n_octopi
    global_fitness = -np.inf
    overhead_total = 0.0
    instability_total = 0.0
    responded_rounds = 0

    for _, k in class_iterations:
        instant = global_instants[class_index][k]
        candidates = [i for i in range(n_octopi) if candidacies[i][k] == 1]
        contributors = []
        response_maxima = []

        for i in candidates:
            event_index = np.searchsorted(
                triggering_instants[i][class_index], instant, side="right"
            ) - 1
            if event_index < 0:
                continue
            n_episodes = episode_counts[i][class_index][event_index]
            if n_episodes == 0:
                continue

            freshness = last_global_iteration[i] - k + 1
            last_global_iteration[i] = k
            initial, episode_rewards = simulate_ou_episodes(
                xi_matrix,
                last_global_fitness[i],
                last_contributor_count[i],
                [gravities[i]],
                n_episodes,
                reward_rng,
                setup["ou_substeps"],
            )
            rewards = np.concatenate((initial, episode_rewards[:, 0]))
            last_rewards[i] = rewards
            lower_references[i].append(float(rewards.min()))
            local_fitness = float(rewards.mean() * np.exp(temperature * freshness))
            response_maxima.append(float(rewards.max()))
            if local_fitness - global_fitness > fitness_margin or k == 0:
                contributors.append((i, local_fitness, freshness))

        if not response_maxima:
            continue
        responded_rounds += 1
        upper_reference = max(response_maxima)
        if contributors:
            global_fitness = sum(item[1] for item in contributors) / sum(
                np.exp(temperature * item[2]) for item in contributors
            )
            contributor_count = len(contributors)
            last_global_fitness[candidates] = global_fitness
            last_contributor_count[candidates] = contributor_count
            overhead_total += contributor_count * model_size_kb

        local_instabilities = []
        for i in candidates:
            if len(lower_references[i]) < 2:
                local_instabilities.append(0.0)
                continue
            lower_reference = min(lower_references[i][-2:])
            instability = 0.0
            if upper_reference > lower_reference:
                scaled_rewards = (
                    (last_rewards[i] - lower_reference)
                    / (upper_reference - lower_reference)
                )
                instability = float(np.mean(
                    np.exp(-setup["scaling_constant"] * scaled_rewards)
                ))
            local_instabilities.append(instability)
        instability_total += float(np.mean(local_instabilities))

    return overhead_total, instability_total, responded_rounds


def main():
    with (EXPERIMENT_DIR / "setup.json").open(encoding="utf-8") as file:
        setup = json.load(file)
    if not setup["simulated_rewards"]:
        raise ValueError("Temperature experiment requires simulated_rewards=true")
    if CLASS_NAMES[setup["class_index"]] != setup["class_name"]:
        raise ValueError("class_name and class_index disagree")
    parameters = setup["system_dynamics_parameters"]
    xi_matrix = generate_Xi_matrices(
        1, EXPERIMENT_DIR / setup["xi_matrix_file"]
    )[0][setup["class_name"]]
    model_size_kb = get_model_size_in_kb(Network(
        shape_in=setup["model_shape_in"],
        action_shape=setup["model_action_shape"],
    ))
    margins = setup["fitness_margins"]
    temperatures = setup["temperatures"]
    n_samples = setup["n_samples"]
    shape = (n_samples, len(temperatures), len(margins))
    overhead_samples = np.empty(shape)
    instability_samples = np.empty(shape)
    responded_rounds = np.empty(shape, dtype=int)
    scenario_seeds = np.empty(shape, dtype=np.uint64)
    seed_rng = np.random.default_rng(setup["seed"])

    for sample in range(n_samples):
        for margin_index, margin in enumerate(margins):
            for temperature_index, temperature in enumerate(temperatures):
                seed = int(seed_rng.integers(0, 2**63))
                scenario_seeds[sample, temperature_index, margin_index] = seed
                overhead, instability, rounds = run_scenario(
                    setup, xi_matrix, margin, temperature, seed, model_size_kb
                )
                overhead_samples[sample, temperature_index, margin_index] = overhead
                instability_samples[sample, temperature_index, margin_index] = instability
                responded_rounds[sample, temperature_index, margin_index] = rounds
        print(f"Completed {sample + 1}/{n_samples} repetitions", flush=True)

    result = {
        "setup": setup,
        "xi_matrix": xi_matrix,
        "model_size_kb": model_size_kb,
        "scenario_seeds": scenario_seeds,
        "responded_rounds": responded_rounds,
        "overhead_samples": overhead_samples,
        "instability_samples": instability_samples,
        "overhead_mean": overhead_samples.mean(axis=0),
        "overhead_std": overhead_samples.std(axis=0, ddof=1),
        "instability_mean": instability_samples.mean(axis=0),
        "instability_std": instability_samples.std(axis=0, ddof=1),
    }
    output_path = EXPERIMENT_DIR / setup["output_file"]
    with output_path.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved temperature experiment to {output_path}")


if __name__ == "__main__":
    main()
