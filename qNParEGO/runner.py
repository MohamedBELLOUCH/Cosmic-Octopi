"""Global sequential-batch RTS/STR adaptation of qLogParEGO_success.ipynb.

The toy's global acquisition search and sequential batch construction are kept.
As in the MORBO experiment runner, fit a scalar GP to the empirical RTS or STR
length scores for the current direction, using qLogNEI. This is an adaptation
for the paper's stochastic scalarized objectives, not an unchanged invocation
of the toy's mean-vector qLogNParEGO. Hypervolume uses MORBO's fixed references.
"""

import argparse
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.optim.optimize import optimize_acqf_list
from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.mlls import ExactMarginalLogLikelihood

EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "MORBO"))
from script_holistic_gravity import (  # noqa: E402
    CLASS_NAMES, Network, append_observation, estimated_hypervolume,
    evaluate_candidate as evaluate_holistic_candidate,
    generate_Xi_matrices, get_model_size_in_kb,
    read_json, save_result, scalar_scores, unit_directions,
)
from script_reductionist_gravity import (  # noqa: E402
    evaluate_candidate as evaluate_reductionist_candidate,
    source_fingerprints as gravity_source_fingerprints,
)


def init_run(seed, hp):
    return {
        "X_unit": np.empty((0, hp["dimension"]), dtype=float),
        "cost_realizations": np.empty((0, 0, 2), dtype=float),
        "scenario_seeds": np.empty((0, 0), dtype=np.uint64),
        "rng_state": np.random.default_rng(seed).bit_generator.state,
        "torch_rng_state": None,
        "hypervolume_history": [],
        "evaluation_counts": [],
        "direction_angle_history": [],
        "pending_X": None,
        "pending_angle": None,
    }


def acquisition_candidates(run, formulation, angle, hp, reference_costs):
    direction = np.array([[np.cos(angle), np.sin(angle)]])
    scores = scalar_scores(run["cost_realizations"], formulation, direction, reference_costs)[:, 0]
    x = torch.as_tensor(run["X_unit"], dtype=torch.float64)
    y = torch.as_tensor(scores[:, None], dtype=torch.float64)
    gp = SingleTaskGP(x, y)
    fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
    acq = qLogNoisyExpectedImprovement(
        model=gp, X_baseline=x,
        sampler=SobolQMCNormalSampler(sample_shape=torch.Size([hp["mc_samples"]])),
        prune_baseline=hp["prune_baseline"],
    )
    # Like the toy, share one scalarization across the batch and condition each
    # successive candidate on previously proposed pending points. Search globally.
    bounds = torch.stack((torch.zeros(hp["dimension"], dtype=torch.float64),
                          torch.ones(hp["dimension"], dtype=torch.float64)))
    candidates, _ = optimize_acqf_list(
        acq_function_list=[acq] * hp["batch_size"], bounds=bounds,
        num_restarts=hp["acquisition_num_restarts"],
        raw_samples=hp["acquisition_raw_samples"],
        options={"batch_limit": hp["acquisition_batch_limit"],
                 "maxiter": hp["acquisition_maxiter"]},
    )
    return candidates.detach().cpu().numpy()


def finish_iteration(run, formulation, angle, hp, directions, reference_costs):
    run["direction_angle_history"].append(angle)
    completed_steps = len(run["direction_angle_history"])
    checkpoint_every = hp.get("hypervolume_checkpoint_every", 1)
    if (completed_steps % checkpoint_every == 0
            or completed_steps == hp["n_iterations"]):
        run["hypervolume_history"].append(estimated_hypervolume(
            run["cost_realizations"], formulation, directions, reference_costs
        ))
        run["evaluation_counts"].append(len(run["X_unit"]))
    run["pending_X"] = None
    run["pending_angle"] = None


def source_fingerprints(entrypoint):
    fingerprints = gravity_source_fingerprints()
    for path in (Path(__file__).resolve(), Path(entrypoint).resolve(),
                 EXPERIMENT_DIR / "qLogParEGO_success.ipynb"):
        fingerprints[str(path.relative_to(EXPERIMENT_DIR.parent))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    return fingerprints


def run_condition(result, key, formulation, parameter, level, setup, hp,
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
    run["sweep_parameter"] = parameter
    run["sweep_value"] = level
    if parameter == "gamma_grav":
        run["gravity_variance"] = level
    scenario_rng = np.random.default_rng()
    scenario_rng.bit_generator.state = run["rng_state"]
    if run["torch_rng_state"] is None:
        torch.manual_seed(seed)
    else:
        torch.set_rng_state(run["torch_rng_state"])
    condition_setup = deepcopy(setup)
    condition_setup["system_dynamics_parameters"][parameter] = level
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


def main(approach, experiment, entrypoint):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true",
                        help="8 initial points and 1 iteration per condition")
    args = parser.parse_args()
    if approach not in ("holistic", "reductionist"):
        raise ValueError("Expected holistic or reductionist approach")
    setup = read_json(EXPERIMENT_DIR / f"setup_{approach}_{experiment}.json")
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
    expected_parameter = {"gravity": "gamma_grav", "synchronization": "gamma_sync",
                          "heterogeneity": "gamma_heter"}[experiment]
    if (sweep["parameter"] != expected_parameter or not sweep["values"]
            or any(not np.isfinite(value) or value < 0 for value in sweep["values"])
            or len(set(sweep["values"])) != len(sweep["values"])):
        raise ValueError("Expected distinct, finite, nonnegative levels for this experiment")
    if expected_parameter == "gamma_sync" and any(not 0 < v <= 1 for v in sweep["values"]):
        raise ValueError("Synchronization values must be in (0, 1]")
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
            "experiment": experiment,
            "algorithm": "qNParEGO",
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
            key = f"{formulation}/{sweep['parameter']}={level:g}"
            run_condition(result, key, formulation, sweep["parameter"], level, setup, hp,
                          xi_matrices, model_sizes, output_path, approach)
    result["complete"] = True
    save_result(output_path, result)
    print(f"Saved {profile} qNParEGO {approach} {experiment} results to {output_path}", flush=True)
