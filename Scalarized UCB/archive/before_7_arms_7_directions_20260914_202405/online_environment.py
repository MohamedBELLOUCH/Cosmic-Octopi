"""Online holistic/reductionist simulation with one bandit decision per request.

The dynamics mirror Pareto fronts/script.py:evaluate_sequence. Only the margin
selection and observation callbacks are new. An observation is the actual
(overhead KiB, instability) increment, and the simulation state persists between
steps. Fixed-sequence parity is checked against the shared offline evaluator.
"""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Cosmic Octopi"))
from utils import (
    CLASS_NAMES, activations_candidacies,
    generate_global_iterations_instants_holistic, generate_global_iterations_instants_reductionist,
    generate_learning_periods,
    generate_triggering_events_instants,
)
from simulated_reward_experiment import simulate_ou_episodes


def run_online(setup, xi_matrices, model_sizes, seed, select_margin, observe, approach):
    """Select a normalized holistic arm or a physical reductionist margin.

    The reductionist path processes only the target class, exactly as the
    existing offline evaluator does. The setup horizon is the online horizon
    supplied by the caller, independent of the held-out evaluation horizon.
    """
    if approach not in ("holistic", "reductionist"):
        raise ValueError("Expected holistic or reductionist approach")
    parameters = setup["system_dynamics_parameters"]
    n_octopi = parameters["n_octopi"]
    n_classes = parameters["n_classes"]
    experiment = setup[approach]
    np.random.seed(int(seed) % (2**32))
    reward_rng = np.random.default_rng(int(seed))

    schedule = (generate_global_iterations_instants_holistic if approach == "holistic"
                else generate_global_iterations_instants_reductionist)
    global_instants, oracle_iterations, time_horizon = schedule(parameters, experiment["horizon"])
    if approach == "reductionist":
        class_index = CLASS_NAMES.index(experiment["class_name"])
        oracle_iterations = [item for item in oracle_iterations if item[0] == class_index][:experiment["horizon"]]
    learning_periods, _ = generate_learning_periods(parameters, time_horizon)
    triggering_instants, episode_counts = generate_triggering_events_instants(
        parameters, global_instants, learning_periods
    )
    candidacies = [
        activations_candidacies(
            learning_periods[i], triggering_instants[i], global_instants
        )[1]
        for i in range(n_octopi)
    ]
    g = setup["gravity_mean"]
    gravities = np.random.gamma(
        shape=g**2 / parameters["gamma_grav"],
        scale=parameters["gamma_grav"] / g,
        size=n_octopi,
    )

    last_global_iteration = np.zeros((n_octopi, n_classes), dtype=int)
    last_global_fitness = np.zeros((n_octopi, n_classes))
    last_contributor_count = np.ones((n_octopi, n_classes), dtype=int)
    lower_references = [[[] for _ in range(n_classes)] for _ in range(n_octopi)]
    last_rewards = [[None for _ in range(n_classes)] for _ in range(n_octopi)]
    global_fitness = np.full(n_classes, -np.inf)
    overhead_total = 0.0
    instability_total = 0.0
    temperature = setup["hyper_parameters"]["temperature"]
    scaling_constant = setup["hyper_parameters"]["scaling_constant"]

    for class_index, k in oracle_iterations:
        arm_value = float(select_margin(class_index, k))
        low, high = setup["fitness_margin_bounds"][CLASS_NAMES[class_index]]
        if approach == "holistic":
            if not np.isfinite(arm_value) or not 0 <= arm_value <= 1:
                raise ValueError("A holistic arm must be in [0,1]")
            margin = low + arm_value * (high - low)
        else:
            if not np.isfinite(arm_value) or not low <= arm_value <= high:
                raise ValueError("A reductionist arm must be a feasible class fitness margin")
            margin = arm_value
        previous_overhead, previous_instability = overhead_total, instability_total
        instant = global_instants[class_index][k]
        candidates = [
            i for i in range(n_octopi) if candidacies[i][class_index][k] == 1
        ]
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

            freshness = last_global_iteration[i, class_index] - k + 1
            last_global_iteration[i, class_index] = k
            initial, episode_rewards = simulate_ou_episodes(
                xi_matrices[CLASS_NAMES[class_index]],
                last_global_fitness[i, class_index],
                last_contributor_count[i, class_index],
                [gravities[i]], n_episodes, reward_rng, setup["ou_substeps"],
            )
            rewards = np.concatenate((initial, episode_rewards[:, 0]))
            last_rewards[i][class_index] = rewards
            lower_references[i][class_index].append(float(rewards.min()))
            local_fitness = float(rewards.mean() * np.exp(temperature * freshness))
            response_maxima.append(float(rewards.max()))
            if local_fitness - global_fitness[class_index] > margin or k == 0:
                contributors.append((local_fitness, freshness))

        if not response_maxima:
            observe(class_index, k, arm_value, np.zeros(2))
            continue
        upper_reference = max(response_maxima)
        if contributors:
            global_fitness[class_index] = sum(item[0] for item in contributors) / sum(
                np.exp(temperature * item[1]) for item in contributors
            )
            contributor_count = len(contributors)
            last_global_fitness[candidates, class_index] = global_fitness[class_index]
            last_contributor_count[candidates, class_index] = contributor_count
            overhead_total += contributor_count * model_sizes[class_index]

        local_instabilities = []
        for i in candidates:
            references = lower_references[i][class_index]
            if len(references) < 2:
                local_instabilities.append(0.0)
                continue
            lower_reference = min(references[-2:])
            instability = 0.0
            if upper_reference > lower_reference:
                scaled_rewards = (
                    (last_rewards[i][class_index] - lower_reference)
                    / (upper_reference - lower_reference)
                )
                instability = float(np.mean(np.exp(
                    -scaling_constant * scaled_rewards
                )))
            local_instabilities.append(instability)
        instability_total += float(np.mean(local_instabilities))
        observe(class_index, k, arm_value, np.array([
            overhead_total - previous_overhead, instability_total - previous_instability
        ]))

    return np.array([overhead_total, instability_total], dtype=float)
