"""Shared runner for holistic and reductionist MORBO synchronization experiments.

Scalarization, acquisition and trust-region updates are reused from the gravity
experiments. Only gamma_sync is swept; gamma_grav and gamma_heter stay fixed in
the setup. Each public script supplies its own approach and setup path.
"""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle

import numpy as np
import torch

from script_holistic_gravity import (
    CLASS_NAMES, Network, acquisition_candidates, append_observation,
    estimated_hypervolume, evaluate_candidate as evaluate_holistic_candidate,
    finish_iteration, generate_Xi_matrices, get_model_size_in_kb, init_run,
    read_json, save_result, unit_directions,
)
from script_reductionist_gravity import (
    evaluate_candidate as evaluate_reductionist_candidate,
    source_fingerprints as gravity_source_fingerprints,
)

EXPERIMENT_DIR = Path(__file__).resolve().parent


def source_fingerprints(entrypoint):
    fingerprints = gravity_source_fingerprints()
    for path in (Path(__file__).resolve(), Path(entrypoint).resolve()):
        fingerprints[str(path.relative_to(EXPERIMENT_DIR.parent))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return fingerprints


def run_condition(result, key, formulation, synchronization_level, setup, hp,
                  xi_matrices, model_sizes, output_path, approach):
    # Common initial designs, scenario seeds and direction draws make conditions
    # comparable. Different objectives, rather than arbitrary seeds, drive changes.
    evaluate_candidate = (evaluate_holistic_candidate if approach == "holistic"
                          else evaluate_reductionist_candidate)
    seed = setup["seed"]
    run = result["runs"].setdefault(key, init_run(seed, hp))
    if approach == "reductionist":
        run["class_name"] = setup["reductionist"]["class_name"]
    run["formulation"] = formulation
    run["sweep_parameter"] = "gamma_sync"
    run["sweep_value"] = synchronization_level
    scenario_rng = np.random.default_rng()
    scenario_rng.bit_generator.state = run["rng_state"]
    if run["torch_rng_state"] is None:
        torch.manual_seed(seed)
    else:
        torch.set_rng_state(run["torch_rng_state"])
    condition_setup = deepcopy(setup)
    condition_setup["system_dynamics_parameters"]["gamma_sync"] = synchronization_level
    directions = unit_directions(
        hp["hypervolume_directions"], hp["direction_endpoint_epsilon"]
    )
    reference_costs = np.asarray(setup["reference_costs"], dtype=float)

    sobol = torch.quasirandom.SobolEngine(hp["dimension"], scramble=True, seed=seed)
    initial_x = sobol.draw(hp["n_initial_points"], dtype=torch.float64).numpy()
    while len(run["X_unit"]) < hp["n_initial_points"]:
        idx = len(run["X_unit"])
        costs, seeds = evaluate_candidate(
            initial_x[idx], scenario_rng, condition_setup, xi_matrices, model_sizes
        )
        append_observation(run, initial_x[idx], costs, seeds)
        run["rng_state"] = scenario_rng.bit_generator.state
        save_result(output_path, result)
        if (idx + 1) % 8 == 0 or idx + 1 == hp["n_initial_points"]:
            print(f"{key}: initial {idx + 1}/{hp['n_initial_points']}", flush=True)
    if not run["hypervolume_history"]:
        run["center"] = run["X_unit"].mean(axis=0)
        run["hypervolume_history"].append(estimated_hypervolume(
            run["cost_realizations"], formulation, directions, reference_costs
        ))
        run["evaluation_counts"].append(len(run["X_unit"]))
        save_result(output_path, result)

    while len(run["direction_angle_history"]) < hp["n_iterations"]:
        iteration = len(run["direction_angle_history"])
        if run["pending_angle"] is None:
            run["pending_angle"] = float(scenario_rng.uniform(
                hp["direction_endpoint_epsilon"],
                np.pi / 2 - hp["direction_endpoint_epsilon"]
            ))
            run["rng_state"] = scenario_rng.bit_generator.state
            save_result(output_path, result)
        angle = run["pending_angle"]
        if run["pending_X"] is None:
            run["pending_X"] = acquisition_candidates(
                run, formulation, angle, hp, reference_costs
            )
            run["torch_rng_state"] = torch.get_rng_state()
            save_result(output_path, result)
        batch_start = hp["n_initial_points"] + iteration * hp["batch_size"]
        while len(run["X_unit"]) < batch_start + hp["batch_size"]:
            j = len(run["X_unit"]) - batch_start
            candidate = run["pending_X"][j]
            costs, seeds = evaluate_candidate(
                candidate, scenario_rng, condition_setup, xi_matrices, model_sizes
            )
            append_observation(run, candidate, costs, seeds)
            run["rng_state"] = scenario_rng.bit_generator.state
            save_result(output_path, result)
        finish_iteration(
            run, formulation, angle, hp, directions, reference_costs
        )
        save_result(output_path, result)
        print(f"{key}: iteration {iteration + 1}/{hp['n_iterations']}, "
              f"{formulation} HV={run['hypervolume_history'][-1]:.5f}", flush=True)


def main(approach, setup_filename, entrypoint):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="8 initial points and 1 iteration per condition")
    args = parser.parse_args()
    if approach not in ("holistic", "reductionist"):
        raise ValueError("Expected holistic or reductionist approach")
    setup = read_json(EXPERIMENT_DIR / setup_filename)
    source_hp = read_json(EXPERIMENT_DIR / setup["hyperparameters_file"])
    hp = deepcopy(source_hp)
    hp["dimension"] = setup[approach]["horizon"]
    if args.pilot:
        hp["n_initial_points"] = 8
        hp["n_iterations"] = 1
    if not setup["simulated_rewards"]:
        raise ValueError("This experiment requires simulated_rewards=true")
    if approach == "reductionist" and setup[approach]["class_name"] not in CLASS_NAMES:
        raise ValueError("Unknown reductionist class")
    if setup["n_realizations_per_candidate"] < 2:
        raise ValueError("RTS and STR need at least two realizations per candidate")
    if setup["system_dynamics_parameters"]["n_classes"] != len(CLASS_NAMES):
        raise ValueError("Expected four environment classes")
    sweep = setup["sweep"]
    if (sweep["parameter"] != "gamma_sync" or not sweep["values"]
            or any(not 0 < value <= 1 for value in sweep["values"])
            or len(set(sweep["values"])) != len(sweep["values"])):
        raise ValueError("Synchronization levels must be distinct values in (0, 1]")
    if (not setup["formulations"]
            or any(value not in ("RTS", "STR") for value in setup["formulations"])):
        raise ValueError("Formulations must be RTS and/or STR")
    for name in ("dimension", "n_initial_points", "batch_size", "n_iterations",
                 "mc_samples", "hypervolume_directions", "acquisition_num_restarts",
                 "acquisition_raw_samples", "acquisition_batch_limit", "acquisition_maxiter"):
        if not isinstance(hp[name], int) or hp[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    reference = np.asarray(setup["reference_costs"], dtype=float)
    if reference.shape != (2,) or not np.all(np.isfinite(reference) & (reference > 0)):
        raise ValueError("Two finite, positive reference costs are required")
    if hp["dtype"] != "float64" or hp["device"] != "cpu":
        raise ValueError("This runner uses float64 GP calculations on CPU")
    xi_matrices = generate_Xi_matrices(
        1, EXPERIMENT_DIR / setup["xi_matrix_file"]
    )[0]
    model_sizes = np.array([
        get_model_size_in_kb(Network(*setup["policy_shapes"][name]).float())
        for name in CLASS_NAMES
    ])
    output_path = EXPERIMENT_DIR / setup["output_file"]
    profile = "pilot" if args.pilot else "full"
    fingerprints = source_fingerprints(entrypoint)
    result = None
    if output_path.exists():
        with output_path.open("rb") as file:
            previous = pickle.load(file)
        if profile == "full" and previous.get("profile") == "pilot":
            pilot_path = output_path.with_stem(output_path.stem + "_pilot")
            save_result(pilot_path, previous)
            print(f"Preserved the pilot at {pilot_path}", flush=True)
        else:
            if (previous.get("schema_version") != 2 or previous.get("profile") != profile
                    or previous.get("setup") != setup
                    or previous.get("hyperparameters") != hp
                    or previous.get("source_sha256") != fingerprints
                    or any(not np.array_equal(previous["xi_matrices"][name], xi_matrices[name])
                           for name in CLASS_NAMES)):
                raise ValueError("Existing result uses different settings/code; archive it before rerunning")
            result = previous
    if result is None:
        result = {
            "schema_version": 2,
            "profile": profile,
            "experiment": "synchronization",
            "approach": approach,
            "implementation": hp["implementation"],
            "source_sha256": fingerprints,
            "setup": setup,
            "shared_hyperparameters": source_hp,
            "hyperparameters": hp,
            "xi_matrices": xi_matrices,
            "model_sizes_kb": dict(zip(CLASS_NAMES, model_sizes)),
            "runtime": {"torch_num_threads": torch.get_num_threads()},
            "runs": {},
            "complete": False,
        }
        if approach == "reductionist":
            result["class_name"] = setup[approach]["class_name"]
        save_result(output_path, result)
    for level in sweep["values"]:
        for formulation in setup["formulations"]:
            key = f"{formulation}/gamma_sync={level:g}"
            run_condition(result, key, formulation, level, setup, hp,
                          xi_matrices, model_sizes, output_path, approach)
    result["complete"] = True
    save_result(output_path, result)
    print(f"Saved {profile} {approach} synchronization results to {output_path}", flush=True)
