"""Evaluate Sobol points for holistic and reductionist Pareto fronts.

The stochastic schedule, upload rule, global fitness, overhead, and instability
follow Graphics.ipynb and Oracle. Simulated rewards do not update policies, so
policy payload size is computed once per class instead of constructing MuJoCo
environments and copying identical model weights for every virtual octopus.
"""

import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.stats.qmc import Sobol


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "Cosmic Octopi"))

from front_utils import expected_pareto_front, RTS_pareto_front, STR_pareto_front  # noqa: E402
from simulated_reward_experiment import simulate_ou_episodes  # noqa: E402
from utils import (  # noqa: E402
    CLASS_NAMES,
    Network,
    activations_candidacies,
    generate_global_iterations_instants_holistic,
    generate_global_iterations_instants_reductionist,
    generate_learning_periods,
    generate_triggering_events_instants,
    generate_Xi_matrices,
    get_model_size_in_kb,
)


def evaluate_sequence(setup, xi_matrices, model_sizes, margins, mode, seed):
    """Return total (overhead KB, instability) for one stochastic realization."""
    parameters = setup["system_dynamics_parameters"]
    n_octopi = parameters["n_octopi"]
    n_classes = parameters["n_classes"]
    experiment = setup[mode]
    np.random.seed(int(seed) % (2**32))
    reward_rng = np.random.default_rng(int(seed))

    if mode == "holistic":
        global_instants, oracle_iterations, time_horizon = (
            generate_global_iterations_instants_holistic(
                parameters, experiment["horizon"]
            )
        )
    else:
        global_instants, oracle_iterations, time_horizon = (
            generate_global_iterations_instants_reductionist(
                parameters, experiment["horizon"]
            )
        )
        class_index = CLASS_NAMES.index(experiment["class_name"])
        oracle_iterations = [
            iteration for iteration in oracle_iterations if iteration[0] == class_index
        ][:experiment["horizon"]]

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

    margins = np.asarray(margins, dtype=float)
    normalized_sequence = (
        mode == "holistic"
        and experiment.get("margin_parameterization", "per_class") == "normalized_sequence"
    )
    if normalized_sequence:
        if margins.shape != (experiment["horizon"],) or not np.all(
            np.isfinite(margins) & (margins >= 0) & (margins <= 1)
        ):
            raise ValueError("Expected one normalized value in [0, 1] per global iteration")
    elif mode == "holistic" and margins.shape != (n_classes,):
        raise ValueError("Expected one fitness margin per class")
    for iteration_number, (class_index, k) in enumerate(oracle_iterations):
        if normalized_sequence:
            # The model-based formulation chooses x_p in [0, 1] at every
            # global iteration, then maps it to the active class's margin.
            low, high = setup["fitness_margin_bounds"][CLASS_NAMES[class_index]]
            unit_margin = margins[iteration_number]
            margin = low + unit_margin * (high - low)
        else:
            margin = margins[class_index] if mode == "holistic" else margins[k]
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

    return np.array([overhead_total, instability_total], dtype=float)


def sobol_points(dimension, n_points, seed):
    """Use the first n_points of the next power-of-two scrambled Sobol net."""
    engine = Sobol(d=dimension, scramble=True, seed=int(seed))
    power = int(np.ceil(np.log2(n_points)))
    return engine.random_base2(power)[:n_points]


def compute_fronts(objectives, setup):
    all_points = objectives.reshape(-1, 2)
    spans = np.ptp(all_points, axis=0)
    if np.any(spans <= 0):
        raise ValueError("Pareto fronts need variation on both objective axes")
    reference_point = all_points.max(axis=0) + (
        setup["front"]["reference_padding_fraction"] * spans
    )
    negative_samples = [-sample for sample in objectives]
    front_points = setup["front"]["n_points"]
    return {
        "reference_point": reference_point,
        "expected_front": -expected_pareto_front(
            negative_samples, -reference_point, n_points=front_points
        ),
        "rts_front": -RTS_pareto_front(
            negative_samples, -reference_point, n_points=front_points
        ),
        "str_front": -STR_pareto_front(
            negative_samples, -reference_point, n_points=front_points
        ),
    }


def run_approach(setup, xi_matrices, model_sizes, mode, seed_rng):
    experiment = setup[mode]
    n_points = experiment["n_sobol_points"]
    n_repetitions = experiment["n_repetitions_per_point"]
    if n_repetitions < 2:
        raise ValueError(
            f"{mode}: RTS and STR fronts require at least two realizations per Sobol point"
        )
    dimension = setup["system_dynamics_parameters"]["n_classes"] if mode == "holistic" else experiment["horizon"]
    sobol_seed = int(seed_rng.integers(0, 2**32))
    unit_points = sobol_points(dimension, n_points, sobol_seed)
    if mode == "holistic":
        bounds = np.array([
            setup["fitness_margin_bounds"][name] for name in CLASS_NAMES
        ])
        margin_points = bounds[:, 0] + unit_points * (bounds[:, 1] - bounds[:, 0])
    else:
        bounds = setup["fitness_margin_bounds"][experiment["class_name"]]
        margin_points = bounds[0] + unit_points * (bounds[1] - bounds[0])

    objectives = np.empty((n_points, n_repetitions, 2))
    scenario_seeds = np.empty((n_points, n_repetitions), dtype=np.uint64)
    for point_index, margins in enumerate(margin_points):
        for repetition in range(n_repetitions):
            seed = int(seed_rng.integers(0, 2**63))
            scenario_seeds[point_index, repetition] = seed
            objectives[point_index, repetition] = evaluate_sequence(
                setup, xi_matrices, model_sizes, margins, mode, seed
            )
        print(f"{mode}: {point_index + 1}/{n_points} Sobol points", flush=True)

    return {
        "sobol_seed": sobol_seed,
        "sobol_unit_points": unit_points,
        "fitness_margin_points": margin_points,
        "scenario_seeds": scenario_seeds,
        "objectives": objectives,
        **compute_fronts(objectives, setup),
    }


def main():
    with (EXPERIMENT_DIR / "setup.json").open(encoding="utf-8") as file:
        setup = json.load(file)
    if not setup["simulated_rewards"]:
        raise ValueError("Pareto fronts require simulated_rewards=true")
    if setup["system_dynamics_parameters"]["n_classes"] != len(CLASS_NAMES):
        raise ValueError("Expected four environment classes")
    xi_matrices = generate_Xi_matrices(
        1, EXPERIMENT_DIR / setup["xi_matrix_file"]
    )[0]
    model_sizes = np.array([
        get_model_size_in_kb(Network(*setup["policy_shapes"][name]))
        for name in CLASS_NAMES
    ])
    seed_rng = np.random.default_rng(setup["seed"])
    result = {
        "setup": setup,
        "xi_matrix": xi_matrices,
        "model_sizes_kb": dict(zip(CLASS_NAMES, model_sizes)),
        "holistic": run_approach(
            setup, xi_matrices, model_sizes, "holistic", seed_rng
        ),
        "reductionist": run_approach(
            setup, xi_matrices, model_sizes, "reductionist", seed_rng
        ),
    }
    output_path = EXPERIMENT_DIR / setup["output_file"]
    with output_path.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved both Pareto-front approaches to {output_path}")


if __name__ == "__main__":
    main()
