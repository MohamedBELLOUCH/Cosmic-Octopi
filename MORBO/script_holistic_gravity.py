"""Holistic gravity optimization with separate RTS and STR trust-region runs.

This follows the single-trust-region BoTorch example in MORBO_success.ipynb.
RTS/STR require scalarizing simulated realizations in different orders, so the
acquisition here is scalar qLogNEI rather than the toy's mean-vector qLogNEHVI.
The implementation is not the published multi-region MORBO algorithm.

Run from any directory: python "MORBO/script_holistic_gravity.py"
Use --pilot for a short end-to-end validation with an explicitly marked result.
Results are checkpointed after every candidate and iteration; rerunning resumes.
"""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "Pareto fronts"))
from script import evaluate_sequence, generate_Xi_matrices, get_model_size_in_kb, Network, CLASS_NAMES  # noqa: E402


def read_json(path):
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def source_fingerprints():
    """Prevent a resumed result from mixing different simulator/optimizer code."""
    root = EXPERIMENT_DIR.parent
    paths = [Path(__file__).resolve(), root / "Pareto fronts" / "script.py",
             root / "Cosmic Octopi" / "simulated_reward_experiment.py",
             root / "Cosmic Octopi" / "utils.py"]
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def unit_directions(count, epsilon):
    """Midpoint quadrature directions, uniform in angle on the positive circle."""
    angles = np.clip((np.arange(count) + 0.5) * np.pi / (2 * count),
                     epsilon, np.pi / 2 - epsilon)
    return np.stack((np.cos(angles), np.sin(angles)), axis=-1)


def scalar_scores(costs, formulation, directions, reference_costs):
    """Empirical length scalarization of normalized cost realizations.

    costs: (candidate, realization, objective); directions: (direction, 2).
    Higher scores are better. The reference is the upper cost point eta, so
    utilities are max(1 - cost/eta, 0) in each objective.
    """
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 3 or costs.shape[-1] != 2:
        raise ValueError("Expected costs with shape (candidates, realizations, 2)")
    if formulation == "RTS":
        # Average before every nonlinear operation, including reference clipping.
        mean_utilities = np.maximum(1.0 - costs.mean(axis=1) / reference_costs, 0.0)
        return np.min(mean_utilities[:, None, :] / directions[None, :, :], axis=-1)
    if formulation == "STR":
        utilities = np.maximum(1.0 - costs / reference_costs, 0.0)
        lengths = np.min(
            utilities[:, :, None, :] / directions[None, None, :, :], axis=-1
        )
        return lengths.mean(axis=1)
    raise ValueError(f"Unknown formulation: {formulation}")


def estimated_hypervolume(costs, formulation, directions, reference_costs):
    """Quadrature estimate of the paper's polar RTS/STR hypervolume in 2D."""
    lengths = scalar_scores(costs, formulation, directions, reference_costs)
    best_lengths = lengths.max(axis=0)
    return float((np.pi / 4) * np.mean(best_lengths**2))


def save_result(path, result):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    # Windows indexing/antivirus can briefly hold the destination open.
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(0.05 * 2**attempt)


def evaluate_candidate(unit_sequence, scenario_rng, setup, xi_matrices, model_sizes):
    n_reps = setup["n_realizations_per_candidate"]
    values = np.empty((n_reps, 2), dtype=float)
    seeds = np.empty(n_reps, dtype=np.uint64)
    for rep in range(n_reps):
        seed = int(scenario_rng.integers(0, 2**63))
        seeds[rep] = seed
        values[rep] = evaluate_sequence(
            setup, xi_matrices, model_sizes, unit_sequence, "holistic", seed
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Simulator returned non-finite objective values")
    return values, seeds


def init_run(seed, hp):
    return {
        "X_unit": np.empty((0, hp["dimension"]), dtype=float),
        "cost_realizations": np.empty((0, 0, 2), dtype=float),
        "scenario_seeds": np.empty((0, 0), dtype=np.uint64),
        "rng_state": np.random.default_rng(seed).bit_generator.state,
        "torch_rng_state": None,
        "center": np.full(hp["dimension"], 0.5),
        "trust_region_length": hp["trust_region_length_initial"],
        "trust_region_length_history": [hp["trust_region_length_initial"]],
        "hypervolume_history": [],
        "evaluation_counts": [],
        "direction_angle_history": [],
        "pending_angle": None,
        "pending_X": None,
    }


def append_observation(run, x, costs, seeds):
    run["X_unit"] = np.vstack((run["X_unit"], np.asarray(x)[None, :]))
    if run["cost_realizations"].shape[1] == 0:
        run["cost_realizations"] = costs[None, :, :]
        run["scenario_seeds"] = seeds[None, :]
    else:
        run["cost_realizations"] = np.concatenate(
            (run["cost_realizations"], costs[None, :, :]), axis=0
        )
        run["scenario_seeds"] = np.vstack((run["scenario_seeds"], seeds))


def acquisition_candidates(run, formulation, angle, hp, reference_costs):
    direction = np.array([[np.cos(angle), np.sin(angle)]])
    scores = scalar_scores(
        run["cost_realizations"], formulation, direction, reference_costs
    )[:, 0]
    x = torch.as_tensor(run["X_unit"], dtype=torch.float64)
    y = torch.as_tensor(scores[:, None], dtype=torch.float64)
    center = torch.as_tensor(run["center"], dtype=torch.float64)
    length = run["trust_region_length"]
    tr_bounds = torch.stack(((center - length / 2).clamp(0, 1),
                             (center + length / 2).clamp(0, 1)))
    # Keep observations at their actual positions in the global unit cube.
    # Only the acquisition search is restricted to the current trust region.
    gp = SingleTaskGP(x, y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    acq = qLogNoisyExpectedImprovement(
        model=gp,
        X_baseline=x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([hp["mc_samples"]])),
        prune_baseline=hp["prune_baseline"],
    )
    x_new, _ = optimize_acqf(
        acq_function=acq,
        bounds=tr_bounds,
        q=hp["batch_size"],
        num_restarts=hp["acquisition_num_restarts"],
        raw_samples=hp["acquisition_raw_samples"],
        options={"batch_limit": hp["acquisition_batch_limit"],
                 "maxiter": hp["acquisition_maxiter"]},
    )
    return x_new.detach().cpu().numpy()


def finish_iteration(run, formulation, angle, hp, directions, reference_costs):
    batch_size = hp["batch_size"]
    scores = scalar_scores(
        run["cost_realizations"], formulation,
        np.array([[np.cos(angle), np.sin(angle)]]), reference_costs
    )[:, 0]
    improved = scores[-batch_size:].max() > scores[:-batch_size].max()
    if improved:
        run["trust_region_length"] = min(
            hp["trust_region_length_maximum"],
            run["trust_region_length"] * hp["trust_region_success_multiplier"]
        )
        run["center"] = (
            hp["trust_region_center_old_weight"] * run["center"]
            + hp["trust_region_center_new_weight"]
            * run["X_unit"][-batch_size:].mean(axis=0)
        )
    else:
        run["trust_region_length"] = max(
            hp["trust_region_length_minimum"],
            run["trust_region_length"] * hp["trust_region_failure_multiplier"]
        )
    run["trust_region_length_history"].append(run["trust_region_length"])
    run["direction_angle_history"].append(angle)
    run["hypervolume_history"].append(estimated_hypervolume(
        run["cost_realizations"], formulation, directions, reference_costs
    ))
    run["evaluation_counts"].append(len(run["X_unit"]))
    run["pending_X"] = None
    run["pending_angle"] = None


def run_condition(result, key, formulation, gravity_variance, setup, hp,
                  xi_matrices, model_sizes, output_path):
    # Common initial designs, scenario seeds and direction draws make conditions
    # comparable. Different objectives, rather than arbitrary seeds, drive changes.
    seed = setup["seed"]
    run = result["runs"].setdefault(key, init_run(seed, hp))
    run["formulation"] = formulation
    run["gravity_variance"] = gravity_variance
    scenario_rng = np.random.default_rng()
    scenario_rng.bit_generator.state = run["rng_state"]
    if run["torch_rng_state"] is None:
        torch.manual_seed(seed)
    else:
        torch.set_rng_state(run["torch_rng_state"])
    condition_setup = deepcopy(setup)
    condition_setup["system_dynamics_parameters"]["gamma_grav"] = gravity_variance
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="8 initial points and 1 iteration per condition")
    args = parser.parse_args()
    setup = read_json(EXPERIMENT_DIR / "setup_holistic_gravity.json")
    hp = read_json(EXPERIMENT_DIR / setup["hyperparameters_file"])
    if args.pilot:
        hp["n_initial_points"] = 8
        hp["n_iterations"] = 1
    if not setup["simulated_rewards"]:
        raise ValueError("This experiment requires simulated_rewards=true")
    if hp["dimension"] != setup["holistic"]["horizon"]:
        raise ValueError("MORBO dimension must equal the holistic horizon")
    if setup["n_realizations_per_candidate"] < 2:
        raise ValueError("RTS and STR need at least two realizations per candidate")
    if setup["system_dynamics_parameters"]["n_classes"] != len(CLASS_NAMES):
        raise ValueError("Expected four environment classes")
    if (not setup["gravity_variances"]
            or any(value <= 0 for value in setup["gravity_variances"])):
        raise ValueError("Gravity variances must be strictly positive")
    if (not setup["formulations"]
            or any(value not in ("RTS", "STR") for value in setup["formulations"])):
        raise ValueError("Formulations must be RTS and/or STR")
    for name in ("dimension", "n_initial_points", "batch_size", "n_iterations",
                 "mc_samples", "hypervolume_directions", "acquisition_num_restarts",
                 "acquisition_raw_samples", "acquisition_batch_limit", "acquisition_maxiter"):
        if not isinstance(hp[name], int) or hp[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if np.any(np.asarray(setup["reference_costs"]) <= 0):
        raise ValueError("Reference costs must be positive for normalization")
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
    fingerprints = source_fingerprints()
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
            "implementation": hp["implementation"],
            "source_sha256": fingerprints,
            "setup": setup,
            "hyperparameters": hp,
            "xi_matrices": xi_matrices,
            "model_sizes_kb": dict(zip(CLASS_NAMES, model_sizes)),
            "runs": {},
            "complete": False,
        }
        save_result(output_path, result)
    for gravity_variance in setup["gravity_variances"]:
        for formulation in setup["formulations"]:
            key = f"{formulation}/gamma_grav={gravity_variance}"
            run_condition(result, key, formulation, gravity_variance, setup, hp,
                          xi_matrices, model_sizes, output_path)
    result["complete"] = True
    save_result(output_path, result)
    print(f"Saved {profile} results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
