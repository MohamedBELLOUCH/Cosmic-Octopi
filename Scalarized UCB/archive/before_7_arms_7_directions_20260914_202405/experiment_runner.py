"""Shared RTS/STR scalarized-UCB runner for holistic and reductionist sweeps.

Public script_<approach>_<experiment>.py entrypoints supply an adjacent setup.
Online training iterations and held-out sequence horizon are separate budgets.
The optimization classes and the existing total-cost/HV evaluator are reused.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import pickle
import sys

import numpy as np

from RTS_Scalarized_UCB import RTS_ScalarizedMultiObjectiveUCB
from STR_Scalarized_UCB import STR_ScalarizedMultiObjectiveUCB
from online_environment import run_online

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(ROOT / "MORBO"))
from script_holistic_gravity import (
    CLASS_NAMES, Network, estimated_hypervolume, evaluate_sequence,
    generate_Xi_matrices, get_model_size_in_kb, save_result, unit_directions,
    source_fingerprints as shared_source_fingerprints,
)


def source_fingerprints(entrypoint):
    hashes = shared_source_fingerprints()
    for name in ("experiment_runner.py", Path(entrypoint).name, "RTS_Scalarized_UCB.py",
                 "STR_Scalarized_UCB.py", "online_environment.py", "bandits.ipynb"):
        path = EXPERIMENT_DIR / name
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def validate_setup(setup):
    if setup["simulated_rewards"] is not True or setup["n_independent_runs"] != 1:
        raise ValueError("This first experiment supports one run with simulated rewards")
    approach = setup["approach"]
    if approach not in ("holistic", "reductionist"):
        raise ValueError("Expected holistic or reductionist approach")
    expected = {"gravity": "gamma_grav", "synchronization": "gamma_sync", "heterogeneity": "gamma_heter"}
    if setup["experiment"] not in expected or setup["sweep"]["parameter"] != expected[setup["experiment"]]:
        raise ValueError("Experiment and sweep parameter must agree")
    bandit, evaluation = setup["bandit"], setup["evaluation"]
    for name, value in {
        "horizon": setup[approach]["horizon"], "online_iterations": bandit["n_online_iterations"],
        "n_arms": bandit["n_arms"],
        "n_scalarization_directions": bandit["n_scalarization_directions"],
        "n_random_sequences": evaluation["n_random_sequences"],
        "n_realizations_per_sequence": evaluation["n_realizations_per_sequence"],
        "hypervolume_directions": evaluation["hypervolume_directions"],
        "checkpoint_every": evaluation["checkpoint_every"],
        "workers": evaluation["workers"],
        "torch_num_threads": setup["torch_num_threads"],
    }.items():
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if bandit["n_arms"] < 2 or evaluation["n_realizations_per_sequence"] < 2:
        raise ValueError("Use at least two arms and two held-out realizations per sequence")
    if bandit["n_arms"] * bandit["n_scalarization_directions"] >= bandit["n_online_iterations"]:
        raise ValueError("Leave iterations after initialization for adaptive UCB choices")
    if approach == "holistic":
        expected_interval = [0, 1]
        if setup[approach]["margin_parameterization"] != "normalized_sequence":
            raise ValueError("Holistic margins must be a normalized sequence")
    else:
        if setup[approach]["class_name"] not in CLASS_NAMES:
            raise ValueError("Unknown reductionist class")
        expected_interval = setup["fitness_margin_bounds"][setup[approach]["class_name"]]
    if (bandit["arm_interval"] != expected_interval or bandit["arm_discretization"] != "linspace_inclusive"
            or bandit["initialization"] != "round_robin"
            or bandit["rts_mean_scope"] not in ("global", "scalarizer")
            or bandit["direction_rule"] != "uniform_angle_midpoint_quadrature"
            or bandit["reward_reference"] != [0, 0]):
        raise ValueError("Unsupported arm parameterization, initialization, direction rule, or reference")
    if bandit["observation_normalization"] != "one_minus_increment_divided_by_maximum_possible_increment":
        raise ValueError("Unsupported online reward normalization")
    if bandit["recommendation"] != "union_of_one_empirical_exploitation_maximizer_per_observed_scalarizer":
        raise ValueError("Unsupported recommendation rule")
    if evaluation["sampling_rule"] != "common_scrambled_sobol_uniform_draws_over_recommended_arms":
        raise ValueError("Unsupported evaluation sampling rule")
    if not np.isfinite(bandit["exploration_coefficient"]) or bandit["exploration_coefficient"] <= 0:
        raise ValueError("exploration_coefficient must be finite and positive")
    for epsilon in (bandit["direction_endpoint_epsilon"], evaluation["direction_endpoint_epsilon"]):
        if not 0 < epsilon < np.pi / 4:
            raise ValueError("Direction endpoint epsilon must be in (0, pi/4)")
    if setup["system_dynamics_parameters"]["n_classes"] != len(CLASS_NAMES):
        raise ValueError("Expected four environment classes")
    levels = np.asarray(setup["sweep"]["values"], dtype=float)
    if not len(levels) or not np.all(np.isfinite(levels) & (levels >= 0)) or len(set(levels)) != len(levels):
        raise ValueError("Sweep levels must be distinct, finite, and nonnegative")
    if setup["experiment"] == "gravity" and np.any(levels <= 0):
        raise ValueError("Gravity variances must be positive")
    if setup["experiment"] == "synchronization" and np.any((levels <= 0) | (levels > 1)):
        raise ValueError("Synchronization levels must be in (0,1]")
    reference = np.asarray(setup["reference_costs"], dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference) & (reference > 0)):
        raise ValueError("Two finite positive total-cost reference values are required")
    if not setup["formulations"] or not set(setup["formulations"]) <= {"RTS", "STR"}:
        raise ValueError("Expected RTS and/or STR")


def make_bandit(setup, formulation, weights, seed):
    settings = setup["bandit"]
    kwargs = dict(K=settings["n_arms"], D=2, weights_list=weights,
                  z=np.array(settings["reward_reference"]), seed=seed,
                  initialization=settings["initialization"],
                  exploration_coefficient=settings["exploration_coefficient"])
    if formulation == "RTS":
        return RTS_ScalarizedMultiObjectiveUCB(**kwargs, mean_scope=settings["rts_mean_scope"])
    return STR_ScalarizedMultiObjectiveUCB(**kwargs)


def train_online(setup, formulation, level, xi, model_sizes, arms, weights):
    """One live trajectory: select, observe the current increment, then update."""
    condition = deepcopy(setup)
    condition["system_dynamics_parameters"][setup["sweep"]["parameter"]] = level
    condition[setup["approach"]]["horizon"] = setup["bandit"]["n_online_iterations"]
    streams = np.random.SeedSequence(setup["seed"]).spawn(2)
    environment_seed, selection_seed = [int(s.generate_state(1, dtype=np.uint64)[0]) for s in streams]
    bandit = make_bandit(setup, formulation, weights, selection_seed)
    payload = (max(model_sizes) if setup["approach"] == "holistic" else
               model_sizes[CLASS_NAMES.index(setup["reductionist"]["class_name"])])
    bounds = np.array([setup["system_dynamics_parameters"]["n_octopi"] * payload, 1.0])
    history = {name: [] for name in ("arm_history", "scalarizer_history", "class_history",
               "class_iteration_history", "increment_costs", "reward_history", "recommended_arms")}

    def select_margin(class_index, k):
        arm, j = bandit.select_action()
        history["arm_history"].append(arm)
        history["scalarizer_history"].append(j)
        history["class_history"].append(class_index)
        history["class_iteration_history"].append(k)
        return arms[arm]

    def observe(class_index, k, unit_margin, costs):
        # Keep the affine transform unclipped: RTS must average before clipping.
        if not np.all(np.isfinite(costs) & (costs >= -1e-9) & (costs <= bounds + 1e-9)):
            raise ValueError("An online cost exceeded its documented normalization bounds")
        rewards = 1 - costs / bounds
        bandit.update(history["arm_history"][-1], history["scalarizer_history"][-1], rewards)
        history["increment_costs"].append(costs.copy())
        history["reward_history"].append(rewards)
        history["recommended_arms"].append(bandit.recommended_arms())

    totals = run_online(condition, xi, model_sizes, environment_seed, select_margin, observe, setup["approach"])
    for name in history:
        if name != "recommended_arms":
            history[name] = np.asarray(history[name])
    if len(history["arm_history"]) != setup["bandit"]["n_online_iterations"]:
        raise ValueError("Online simulator did not provide the requested number of global iterations")
    return {
        **history, "formulation": formulation, "sweep_parameter": setup["sweep"]["parameter"], "sweep_value": level,
        "environment_seed": environment_seed, "selection_seed": selection_seed,
        "increment_cost_upper_bounds": bounds, "training_total_costs": totals,
        "pull_counts": bandit.ni.copy(), "mean_reward_vectors": bandit.mean_vec.copy(),
        "scalar_ucbs": deepcopy(bandit.scalar_ucbs),
        "checkpoint_iterations": [], "hypervolume_history": [], "evaluation_keys": [],
        "training_complete": True, "complete": False,
    }


def evaluation_design(setup):
    import torch

    settings = setup["evaluation"]
    # Precompute fixed held-out draws; UCB never sees these observations.
    engine = torch.quasirandom.SobolEngine(setup[setup["approach"]]["horizon"], scramble=True,
                                         seed=settings["seed"])
    draws = engine.draw(settings["n_random_sequences"], dtype=torch.float64).numpy()
    seeds = np.random.default_rng(settings["seed"]).integers(
        0, 2**63, size=settings["n_realizations_per_sequence"], dtype=np.uint64)
    return draws, seeds


def evaluate_subset(setup, level, subset, arms, draws, seeds, xi, model_sizes, executor=None):
    """Sample the set subset**horizon; estimate RTS/STR from complete rollouts."""
    subset = np.asarray(subset, dtype=int)
    indices = subset[np.minimum((draws * len(subset)).astype(int), len(subset) - 1)]
    if setup["evaluation"]["include_constant_sequences"]:
        constants = np.repeat(subset[:, None], setup[setup["approach"]]["horizon"], axis=1)
        indices = np.vstack((constants, indices))
    indices = np.unique(indices, axis=0)
    sequences = arms[indices]
    condition = deepcopy(setup)
    condition["system_dynamics_parameters"][setup["sweep"]["parameter"]] = level
    costs = np.empty((len(sequences), len(seeds), 2))
    jobs = []
    for i, sequence in enumerate(sequences):
        for j, seed in enumerate(seeds):
            args = (condition, xi, model_sizes, sequence, setup["approach"], int(seed))
            if executor is None:
                costs[i, j] = evaluate_sequence(*args)
            else:
                jobs.append((i, j, executor.submit(evaluate_sequence, *args)))
    for i, j, future in jobs:
        costs[i, j] = future.result()
    if not np.all(np.isfinite(costs)):
        raise ValueError("Non-finite held-out cost")
    settings = setup["evaluation"]
    directions = unit_directions(settings["hypervolume_directions"], settings["direction_endpoint_epsilon"])
    return {
        "sweep_parameter": setup["sweep"]["parameter"], "sweep_value": level, "recommended_arms": subset.copy(),
        "arm_sequences": indices, "sequences": sequences,
        "X_unit": ((sequences - arms[0]) / (arms[-1] - arms[0])),
        "scenario_seeds": seeds.copy(),
        "cost_realizations": costs,
        "hypervolumes": {form: estimated_hypervolume(costs, form, directions, np.asarray(setup["reference_costs"]))
                         for form in ("RTS", "STR")},
    }


def run(setup, output_path, entrypoint, executor=None):
    import torch

    validate_setup(setup)
    torch.set_num_threads(setup["torch_num_threads"])
    arms = np.linspace(*setup["bandit"]["arm_interval"], setup["bandit"]["n_arms"])
    weights = unit_directions(setup["bandit"]["n_scalarization_directions"],
                              setup["bandit"]["direction_endpoint_epsilon"])
    xi = generate_Xi_matrices(1, EXPERIMENT_DIR / setup["xi_matrix_file"])[0]
    model_sizes = np.array([get_model_size_in_kb(Network(*setup["policy_shapes"][name]).float())
                            for name in CLASS_NAMES])
    fingerprints = source_fingerprints(entrypoint)
    if output_path.exists():
        with output_path.open("rb") as file:
            result = pickle.load(file)
        if (result.get("schema_version") != 2 or result.get("setup") != setup
                or result.get("source_sha256") != fingerprints
                or any(not np.array_equal(result["xi_matrices"][name], xi[name]) for name in CLASS_NAMES)):
            raise ValueError("Result settings/code differ; choose a new output_file before rerunning")
        if result["complete"]:
            print(f"Already complete: {output_path}", flush=True)
            return result
    else:
        result = {"schema_version": 2, "algorithm": "Scalarized UCB", "approach": setup["approach"],
                  "experiment": setup["experiment"], "setup": setup, "source_sha256": fingerprints,
                  "xi_matrices": xi, "model_sizes_kb": dict(zip(CLASS_NAMES, model_sizes)),
                  "arms": arms, "scalarization_directions": weights,
                  "runs": {}, "evaluation_cache": {}, "complete": False}
        save_result(output_path, result)
    draws, seeds = evaluation_design(setup)
    horizon = setup["bandit"]["n_online_iterations"]
    checkpoints = sorted(set(range(1, horizon + 1, setup["evaluation"]["checkpoint_every"])) | {horizon})
    for level in setup["sweep"]["values"]:
        for formulation in setup["formulations"]:
            key = f"{formulation}/{setup['sweep']['parameter']}={level:g}"
            if key not in result["runs"]:
                result["runs"][key] = train_online(setup, formulation, level, xi, model_sizes, arms, weights)
                save_result(output_path, result)
            trajectory = result["runs"][key]
            for iteration in checkpoints[len(trajectory["checkpoint_iterations"]):]:
                subset = trajectory["recommended_arms"][iteration - 1]
                cache_key = f"{setup['sweep']['parameter']}={level:g}/arms=" + ",".join(map(str, subset))
                if cache_key not in result["evaluation_cache"]:
                    result["evaluation_cache"][cache_key] = evaluate_subset(
                        setup, level, subset, arms, draws, seeds, xi, model_sizes, executor)
                    print(f"{key}: assessed subset at iteration {iteration}/{horizon}, "
                          f"arms={subset.tolist()}", flush=True)
                entry = result["evaluation_cache"][cache_key]
                trajectory["checkpoint_iterations"].append(iteration)
                trajectory["evaluation_keys"].append(cache_key)
                trajectory["hypervolume_history"].append(entry["hypervolumes"][formulation])
                save_result(output_path, result)
            trajectory["complete"] = True
            save_result(output_path, result)
            print(f"{key}: final HV={trajectory['hypervolume_history'][-1]:.6f}", flush=True)
    result["complete"] = True
    save_result(output_path, result)
    print(f"Saved {output_path}", flush=True)
    return result


def main(approach, experiment, entrypoint):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=EXPERIMENT_DIR / f"setup_{approach}_{experiment}.json")
    args = parser.parse_args()
    with args.setup.open(encoding="utf-8") as file:
        setup = json.load(file)
    os.environ["OMP_NUM_THREADS"] = str(setup["torch_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(setup["torch_num_threads"])
    validate_setup(setup)
    if setup["approach"] != approach or setup["experiment"] != experiment:
        raise ValueError("Setup does not match this script's approach and experiment")
    # Separate processes isolate the simulator's legacy np.random global state.
    # Threads would make concurrent scenario seeds interfere with one another.
    with ProcessPoolExecutor(max_workers=setup["evaluation"]["workers"],
                             mp_context=get_context("spawn")) as executor:
        run(setup, EXPERIMENT_DIR / setup["output_file"], entrypoint, executor)

