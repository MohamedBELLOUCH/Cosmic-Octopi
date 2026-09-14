"""Shared online RTS/STR scalarized KG runner for both approaches and all sweeps.

Run with the project Python interpreter. All settings are in the adjacent JSON.
The shared UCB simulator and held-out evaluator are reused without modification;
only the optimizer and its observation convention differ. No BO scripts run.
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

from RTS_Scalarized_KG import RTS_ScalarizedMultiObjectiveKG
from STR_Scalarized_KG import STR_ScalarizedMultiObjectiveKG

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parent
sys.path.insert(0, str(ROOT / "Scalarized UCB"))
from experiment_runner import (
    CLASS_NAMES, Network, evaluate_subset, evaluation_design, generate_Xi_matrices,
    get_model_size_in_kb, save_result, unit_directions, load_xi_matrices,
    source_fingerprints as evaluation_source_fingerprints,
)
from online_environment import run_online


def source_fingerprints(entrypoint):
    hashes = evaluation_source_fingerprints(ROOT / "Scalarized UCB" / "script_holistic_gravity.py")
    for name in ("RTS_Scalarized_KG.py", "STR_Scalarized_KG.py", "kg_runner.py", Path(entrypoint).name):
        path = EXPERIMENT_DIR / name
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def make_bandit(setup, formulation, weights, seed):
    b = setup["bandit"]
    if formulation not in ("RTS", "STR"):
        raise ValueError("Expected RTS or STR")
    cls = RTS_ScalarizedMultiObjectiveKG if formulation == "RTS" else STR_ScalarizedMultiObjectiveKG
    return cls(K=b["n_arms"], D=2, weights_list=weights, horizon=b["n_online_iterations"],
               reference=b["cost_reference"], seed=seed,
               initial_pulls_per_pair=b["initial_pulls_per_pair"],
               exploration_coefficient=b["exploration_coefficient"])


def validate_setup(setup):
    if (setup["simulated_rewards"] is not True or setup["n_independent_runs"] != 1
            or setup["approach"] not in ("holistic", "reductionist")):
        raise ValueError("Each invocation supports one independent simulated-reward run")
    b, e = setup["bandit"], setup["evaluation"]
    approach = setup["approach"]
    parameters = {"gravity": "gamma_grav", "synchronization": "gamma_sync", "heterogeneity": "gamma_heter"}
    if setup["experiment"] not in parameters or setup["sweep"]["parameter"] != parameters[setup["experiment"]]:
        raise ValueError("Experiment and sweep parameter must agree")
    for name, value in {"horizon": setup[approach]["horizon"],
                        "workers": e["workers"], "torch_num_threads": setup["torch_num_threads"],
                        **{k: e[k] for k in ("n_random_sequences", "n_realizations_per_sequence",
                                            "hypervolume_directions", "checkpoint_every")}}.items():
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if e["n_realizations_per_sequence"] < 2:
        raise ValueError("RTS/STR evaluation requires at least two realizations")
    expected = {"initialization": "round_robin_full_passes", "statistics_scope": "arm_and_direction",
                "arm_discretization": "linspace_inclusive", "sample_variance_ddof": 1,
                "direction_rule": "uniform_angle_midpoint_quadrature",
                "observation_normalization": "increment_divided_by_maximum_possible_increment",
                "scalarization": "max_positive_cost_minus_reference_divided_by_direction",
                "standard_error": "sample_standard_deviation_divided_by_sqrt_pair_count",
                "exploration_scale": "remaining_global_iterations_times_n_arms_times_n_objectives",
                "zero_variance_rule": "zero_knowledge_gradient_without_artificial_variance_floor",
                "tie_breaking": "smallest_arm_index",
                "recommendation": "union_of_one_empirical_exploitation_maximizer_per_observed_scalarizer"}
    for key, value in expected.items():
        if b[key] != value:
            raise ValueError(f"Unsupported {key}")
    if approach == "holistic":
        interval = [0, 1]
        if setup[approach]["margin_parameterization"] != "normalized_sequence":
            raise ValueError("Holistic margins must be normalized")
    else:
        if setup[approach]["class_name"] not in CLASS_NAMES:
            raise ValueError("Unknown reductionist class")
        interval = setup["fitness_margin_bounds"][setup[approach]["class_name"]]
    if b["arm_interval"] != interval or np.any(np.asarray(b["cost_reference"]) >= 0):
        raise ValueError("Use the approach's arm interval and a strictly negative cost reference")
    for epsilon in (b["direction_endpoint_epsilon"], e["direction_endpoint_epsilon"]):
        if not 0 < epsilon < np.pi / 4:
            raise ValueError("Direction endpoint epsilon must be in (0, pi/4)")
    weights = unit_directions(b["n_scalarization_directions"], b["direction_endpoint_epsilon"])
    make_bandit(setup, "RTS", weights, setup["seed"])
    if e["sampling_rule"] != "common_scrambled_sobol_uniform_draws_over_recommended_arms":
        raise ValueError("Unsupported held-out sampling rule")
    levels = np.asarray(setup["sweep"]["values"], dtype=float)
    if (levels.ndim != 1 or not len(levels) or not np.all(np.isfinite(levels) & (levels >= 0))
            or len(set(levels)) != len(levels)):
        raise ValueError("Expected distinct finite nonnegative sweep levels")
    if setup["experiment"] == "gravity" and np.any(levels <= 0):
        raise ValueError("Gravity variances must be positive")
    if setup["experiment"] == "synchronization" and np.any((levels <= 0) | (levels > 1)):
        raise ValueError("Synchronization levels must be in (0,1]")
    ref = np.asarray(setup["reference_costs"], dtype=float)
    if ref.shape != (2,) or not np.all(np.isfinite(ref) & (ref > 0)):
        raise ValueError("Expected two positive total-cost hypervolume references")
    if setup["system_dynamics_parameters"]["n_classes"] != len(CLASS_NAMES):
        raise ValueError("Expected four classes")
    if not setup["formulations"] or not set(setup["formulations"]) <= {"RTS", "STR"}:
        raise ValueError("Expected RTS and/or STR")


def train_online(setup, formulation, level, xi, model_sizes, arms, weights):
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
               "class_iteration_history", "increment_costs", "normalized_cost_history", "recommended_arms")}
    diagnostics = []

    def select_margin(class_index, k):
        arm, j = bandit.select_action()
        history["arm_history"].append(arm)
        history["scalarizer_history"].append(j)
        history["class_history"].append(class_index)
        history["class_iteration_history"].append(k)
        if bandit.last_decision is not None:
            diagnostics.append({"iteration": bandit.n + 1, "direction": j,
                                **deepcopy(bandit.last_decision)})
        return arms[arm]

    def observe(class_index, k, margin, costs):
        if not np.all(np.isfinite(costs) & (costs >= -1e-9) & (costs <= bounds + 1e-9)):
            raise ValueError("An online cost exceeded its normalization bounds")
        normalized = costs / bounds
        bandit.update(history["arm_history"][-1], history["scalarizer_history"][-1], normalized)
        history["increment_costs"].append(costs.copy())
        history["normalized_cost_history"].append(normalized)
        history["recommended_arms"].append(bandit.recommended_arms())

    totals = run_online(condition, xi, model_sizes, environment_seed, select_margin, observe, setup["approach"])
    for name in history:
        if name != "recommended_arms":
            history[name] = np.asarray(history[name])
    if len(history["arm_history"]) != setup["bandit"]["n_online_iterations"]:
        raise ValueError("Online simulator returned an incomplete trajectory")
    return {**history, "formulation": formulation, "sweep_parameter": setup["sweep"]["parameter"], "sweep_value": level,
            "environment_seed": environment_seed, "selection_seed": selection_seed,
            "increment_cost_upper_bounds": bounds, "training_total_costs": totals,
            "bandit_state": bandit.state_dict(), "adaptive_decisions": diagnostics,
            "checkpoint_iterations": [], "hypervolume_history": [], "evaluation_keys": [],
            "training_complete": True, "complete": False}


def run(setup, output_path, entrypoint, executor=None):
    import torch

    validate_setup(setup)
    torch.set_num_threads(setup["torch_num_threads"])
    b = setup["bandit"]
    arms = np.linspace(*b["arm_interval"], b["n_arms"])
    weights = unit_directions(b["n_scalarization_directions"], b["direction_endpoint_epsilon"])
    xi = load_xi_matrices(setup, EXPERIMENT_DIR)
    sizes = np.array([get_model_size_in_kb(Network(*setup["policy_shapes"][name]).float()) for name in CLASS_NAMES])
    hashes = source_fingerprints(entrypoint)
    if output_path.exists():
        with output_path.open("rb") as file:
            result = pickle.load(file)
        if (result.get("schema_version") != 2 or result.get("algorithm") != "Scalarized KG"
                or result.get("setup") != setup or result.get("source_sha256") != hashes
                or any(not np.array_equal(result["xi_matrices"][name], xi[name]) for name in CLASS_NAMES)):
            raise ValueError("Result settings/code differ; use a new output_file or archive the old result")
        if result["complete"]:
            print(f"Already complete: {output_path}", flush=True)
            return result
    else:
        result = {"schema_version": 2, "algorithm": "Scalarized KG", "approach": setup["approach"],
                  "experiment": setup["experiment"], "setup": setup, "source_sha256": hashes,
                  "xi_matrices": xi, "model_sizes_kb": dict(zip(CLASS_NAMES, sizes)),
                  "arms": arms, "scalarization_directions": weights,
                  "runs": {}, "evaluation_cache": {}, "complete": False}
        save_result(output_path, result)
    draws, seeds = evaluation_design(setup)
    horizon = b["n_online_iterations"]
    checkpoints = sorted(set(range(1, horizon + 1, setup["evaluation"]["checkpoint_every"])) | {horizon})
    for level in setup["sweep"]["values"]:
        for formulation in setup["formulations"]:
            key = f"{formulation}/{setup['sweep']['parameter']}={level:g}"
            if key not in result["runs"]:
                result["runs"][key] = train_online(setup, formulation, level, xi, sizes, arms, weights)
                save_result(output_path, result)
            trajectory = result["runs"][key]
            for iteration in checkpoints[len(trajectory["checkpoint_iterations"]):]:
                subset = trajectory["recommended_arms"][iteration - 1]
                cache_key = f"{setup['sweep']['parameter']}={level:g}/arms=" + ",".join(map(str, subset))
                if cache_key not in result["evaluation_cache"]:
                    result["evaluation_cache"][cache_key] = evaluate_subset(
                        setup, level, subset, arms, draws, seeds, xi, sizes, executor)
                    print(f"{key}: assessed subset at iteration {iteration}/{horizon}, arms={subset.tolist()}", flush=True)
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
    setup = json.loads(args.setup.read_text(encoding="utf-8"))
    os.environ["OMP_NUM_THREADS"] = str(setup["torch_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(setup["torch_num_threads"])
    validate_setup(setup)
    if setup["approach"] != approach or setup["experiment"] != experiment:
        raise ValueError("Setup does not match this entrypoint")
    with ProcessPoolExecutor(max_workers=setup["evaluation"]["workers"], mp_context=get_context("spawn")) as executor:
        run(setup, EXPERIMENT_DIR / setup["output_file"], entrypoint, executor)

