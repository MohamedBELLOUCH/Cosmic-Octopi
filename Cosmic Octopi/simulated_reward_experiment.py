"""Shared model-based Ant experiment steps for gravity and octopus-count sweeps.

Simulated rewards update scalar global fitness. They do not train policy weights,
so this models the reward side of Figure 9 rather than policy-parameter FedAvg.
"""

import json
from pathlib import Path

import numpy as np


def load_xi_matrix(path, class_name):
    with Path(path).open(encoding="utf-8") as file:
        matrices = json.load(file)
    matrix = np.asarray(matrices[class_name], dtype=float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{class_name} Xi matrix must be a finite 4x4 array")
    return matrix


def validate_setup(setup):
    if not setup["simulated_rewards"]:
        raise ValueError("This experiment requires simulated_rewards=true")
    if setup["episodes_per_round"] <= 0 or setup["n_episodes"] <= 0:
        raise ValueError("Episode counts must be positive")
    if setup["n_episodes"] % setup["episodes_per_round"]:
        raise ValueError("n_episodes must be divisible by episodes_per_round")
    if setup["n_repetitions"] <= 0 or setup["ou_substeps"] <= 0:
        raise ValueError("n_repetitions and ou_substeps must be positive")


def simulate_ou_episodes(xi_matrix, global_fitness, n_contributors, gravities,
                         n_episodes, rng, substeps):
    """Vectorized Milstein OU paths with Octopus's five-substep convention."""
    gravities = np.atleast_1d(np.asarray(gravities, dtype=float))
    features = np.column_stack((
        np.full(len(gravities), global_fitness),
        np.full(len(gravities), n_contributors),
        gravities,
        np.ones(len(gravities)),
    ))
    parameters = features @ xi_matrix
    values = parameters[:, 0].copy()
    reversion, long_term, sigma = np.exp(parameters[:, 1:]).T
    initial_values = values.copy()
    rewards = np.empty((n_episodes, len(gravities)))
    dt = 1.0 / substeps
    sqrt_dt = np.sqrt(dt)
    for step in range(n_episodes * substeps):
        values += reversion * (long_term - values) * dt
        values += sigma * (values > -10000) * sqrt_dt * rng.standard_normal(len(gravities))
        if (step + 1) % substeps == 0:
            rewards[step // substeps] = values
    return initial_values, rewards


def run_scenario(setup, xi_matrix, n_octopi, gravity_variance, seeds):
    if n_octopi <= 0 or gravity_variance <= 0:
        raise ValueError("n_octopi and gravity_variance must be positive")
    episodes_per_round = setup["episodes_per_round"]
    n_rounds = setup["n_episodes"] // episodes_per_round
    earth_gravity = setup["earth_gravity"]
    rewards = np.empty((len(seeds), setup["n_episodes"]))
    planet_gravities = np.empty((len(seeds), n_octopi))

    for repetition, seed in enumerate(seeds):
        rng = np.random.default_rng(int(seed))
        planets = rng.gamma(
            shape=earth_gravity ** 2 / gravity_variance,
            scale=gravity_variance / earth_gravity,
            size=n_octopi,
        )
        planet_gravities[repetition] = planets
        global_fitness = setup["initial_global_fitness"]
        last_contributor_count = setup["initial_contributor_count"]
        for round_index in range(n_rounds):
            local_initial, local_rewards = simulate_ou_episodes(
                xi_matrix, global_fitness, last_contributor_count,
                planets, episodes_per_round, rng, setup["ou_substeps"],
            )
            # All agents contribute the same number of episodes. Averaging
            # their local means matches Oracle's simulated-fitness update.
            global_fitness = float(
                (local_initial + local_rewards.sum(axis=0)).mean()
                / (episodes_per_round + 1)
            )
            last_contributor_count = n_octopi
            _, earth_rewards = simulate_ou_episodes(
                xi_matrix, global_fitness, n_octopi, [earth_gravity],
                episodes_per_round, rng, setup["ou_substeps"],
            )
            start = round_index * episodes_per_round
            rewards[repetition, start:start + episodes_per_round] = earth_rewards[:, 0]
        print(
            f"  gamma_grav={gravity_variance:g}, N={n_octopi}: "
            f"{repetition + 1}/{len(seeds)} repetitions",
            flush=True,
        )

    return {
        "gamma_grav": gravity_variance,
        "n_octopi": n_octopi,
        "repetition_seeds": np.asarray(seeds),
        "planet_gravities": planet_gravities,
        "rewards": rewards,
    }
