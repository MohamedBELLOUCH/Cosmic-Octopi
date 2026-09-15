"""Final-arm-set comparison using the existing scalarized bandit implementations."""
import sys
sys.dont_write_bytecode = True
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import time
import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for folder in ("Scalarized UCB", "Scalarized Knowledge Gradient", "MORBO"):
    sys.path.insert(0, str(ROOT / folder))
from pareto_online import CostParetoUCB, DelayedCostFeedback
from online_environment import run_online
import experiment_runner as ucb_runner
import kg_runner
from script_holistic_gravity import CLASS_NAMES, Network, get_model_size_in_kb, unit_directions


def save(path, record):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(record, file, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    paths = [Path(__file__), HERE / "pareto_online.py", HERE / "setup_model_free_holistic.json",
             ROOT / "Pareto UCB" / "Pareto_UCB.py"]
    for folder, names in {
        "Scalarized UCB": ["experiment_runner.py", "online_environment.py", "RTS_Scalarized_UCB.py", "STR_Scalarized_UCB.py"],
        "Scalarized Knowledge Gradient": ["kg_runner.py", "RTS_Scalarized_KG.py", "STR_Scalarized_KG.py"],
        "Cosmic Octopi": ["utils.py", "simulated_reward_experiment.py", "front_utils.py"],
        "Pareto fronts": ["script.py"], "MORBO": ["script_holistic_gravity.py"],
    }.items():
        paths += [ROOT / folder / name for name in names]
    paths += [HERE / f"holistic {algorithm}.py" for algorithm in ("Pareto UCB", "Scalarized UCB", "Scalarized KG")]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def check_sources(record):
    for relative, expected in record["source_sha256"].items():
        if file_hash(ROOT / relative) != expected:
            raise ValueError(f"Source/settings changed: {relative}")


def load_setup():
    setup = json.loads((HERE / "setup_model_free_holistic.json").read_text(encoding="utf-8"))
    if setup["simulated_rewards"] is not True or setup["n_independent_runs"] != 1 or setup["approach"] != "holistic":
        raise ValueError("Expected one holistic simulated-reward run")
    if setup["feedback"] != {"Scalarized UCB": "current_overhead_current_instability",
                              "Scalarized KG": "current_overhead_current_instability",
                              "Pareto UCB": "current_overhead_next_global_iteration_instability"}:
        raise ValueError("Keep the explicitly selected feedback definitions")
    if setup["formulations"] != ["RTS", "STR"] or setup["holistic"]["margin_parameterization"] != "normalized_sequence":
        raise ValueError("Expected RTS/STR and normalized holistic sequences")
    for field in ("n_arms", "n_scalarization_directions", "n_online_iterations"):
        if type(setup[field]) is not int or setup[field] < 1:
            raise ValueError(f"Invalid {field}")
    for algorithm in ("Scalarized UCB", "Scalarized KG"):
        b = setup["algorithm_settings"][algorithm]
        if any(b[k] != setup[k] for k in ("n_arms", "n_scalarization_directions", "n_online_iterations")):
            raise ValueError("Match arms, directions and training budgets")
        cfg = scalarized_setup(setup, algorithm)
        (ucb_runner if algorithm == "Scalarized UCB" else kg_runner).make_bandit(
            cfg, "RTS", unit_directions(setup["n_scalarization_directions"], b["direction_endpoint_epsilon"]), setup["seed"])
    if setup["algorithm_settings"]["Pareto UCB"]["exploration_coefficient"] != 2.0:
        raise ValueError("Retain the extracted Pareto UCB coefficient")
    e = setup["evaluation"]
    if (e["sampling_rule"] != "common_iid_uniform_draws_over_recommended_arms"
            or e["include_constant_sequences"] or not e["deduplicate_sequences"]):
        raise ValueError("Use common random sequence draws, deduplicated, without added constants")
    for v in (setup["holistic"]["horizon"], e["workers"], e["n_random_sequences"], e["n_realizations_per_sequence"]):
        if type(v) is not int or v <= 0:
            raise ValueError("Positive integer budgets required")
    if e["n_realizations_per_sequence"] < 2 or len(set(e["scenario_seeds"])) != e["n_realizations_per_sequence"]:
        raise ValueError("Provide distinct common seeds and at least two realizations")
    if np.asarray(setup["reference_costs"]).shape != (2,) or np.any(np.asarray(setup["reference_costs"]) <= 0):
        raise ValueError("Positive two-objective reference required")
    paths = [setup["output_files"]["Pareto UCB"], setup["comparison_output_file"]]
    paths += [p for a in ("Scalarized UCB", "Scalarized KG") for p in setup["output_files"][a].values()]
    if len(set(paths)) != len(paths) or any(Path(p).name != p or not p.endswith('.pkl') for p in paths):
        raise ValueError("Use distinct local pickle output names")
    return setup


def scalarized_setup(setup, algorithm):
    cfg = deepcopy(setup)
    cfg["bandit"] = cfg["algorithm_settings"][algorithm]
    cfg["sweep"] = {"parameter": "gamma_grav", "values": [setup["system_dynamics_parameters"]["gamma_grav"]]}
    return cfg


def simulator_inputs(setup):
    import torch
    torch.set_num_threads(setup["torch_num_threads"])
    xi = {name: np.asarray(setup["Xi_matrices"][name], dtype=float) for name in CLASS_NAMES}
    if any(x.shape != (4, 4) or not np.isfinite(x).all() for x in xi.values()):
        raise ValueError("Invalid embedded Xi matrix")
    sizes = np.array([get_model_size_in_kb(Network(*setup["policy_shapes"][name]).float()) for name in CLASS_NAMES])
    return xi, sizes


def train_pareto(setup, xi, sizes, arms):
    condition = deepcopy(setup)
    condition["holistic"]["horizon"] = setup["n_online_iterations"]
    environment_seed, selection_seed = [int(s.generate_state(1, dtype=np.uint64)[0])
        for s in np.random.SeedSequence(setup["seed"]).spawn(2)]
    bandit = CostParetoUCB(len(arms), selection_seed)
    bounds = np.array([setup["system_dynamics_parameters"]["n_octopi"] * max(sizes), 1.0])
    feedback = DelayedCostFeedback(bandit, bounds)
    actions, classes, class_iterations, increments, recommendations = [], [], [], [], []

    def select(class_index, k):
        arm = bandit.select_action()
        actions.append(arm)
        classes.append(class_index)
        class_iterations.append(k)
        return arms[arm]

    def observe(class_index, k, margin, costs):
        increments.append(costs.copy())
        feedback.observe(len(actions) - 1, actions[-1], costs)
        recommendations.append(bandit.recommended_arms().copy())

    totals = run_online(condition, xi, sizes, environment_seed, select, observe, "holistic")
    assert len(actions) == setup["n_online_iterations"] and bandit.n == len(actions) - 1
    assert len(feedback.observations) == len(actions) - 1
    return {"arm_history": np.asarray(actions), "class_history": np.asarray(classes),
            "class_iteration_history": np.asarray(class_iterations), "increment_costs": np.asarray(increments),
            "recommended_arms": recommendations, "delayed_observations": feedback.observations,
            "pending_final_action": feedback.pending, "n_completed_observations": bandit.n,
            "pull_counts": bandit.ni.copy(), "mean_cost_vectors": -bandit.mean_vec.copy(),
            "increment_cost_upper_bounds": bounds, "training_total_costs": totals,
            "environment_seed": environment_seed, "selection_seed": selection_seed}


def train(algorithm, only_formulation=None):
    setup = load_setup()
    xi, sizes = simulator_inputs(setup)
    arms = np.linspace(0.0, 1.0, setup["n_arms"])
    weights = unit_directions(setup["n_scalarization_directions"],
                             setup["algorithm_settings"]["Scalarized UCB"]["direction_endpoint_epsilon"])
    formulations = [None] if algorithm == "Pareto UCB" else ([only_formulation] if only_formulation else setup["formulations"])
    fingerprints = source_hashes()
    for formulation in tqdm(formulations, desc=algorithm, unit="training run", ascii=True):
        filename = setup["output_files"][algorithm] if formulation is None else setup["output_files"][algorithm][formulation]
        path = HERE / filename
        if path.exists():
            with path.open("rb") as file:
                previous = pickle.load(file)
            if previous["setup"] != setup or previous["source_sha256"] != fingerprints or not previous["complete"]:
                raise ValueError(f"Existing result is not a matching completed run: {path}")
            print(f"Already complete: {path}", flush=True)
            continue
        start = time.perf_counter()
        if algorithm == "Pareto UCB":
            training = train_pareto(setup, xi, sizes, arms)
        else:
            runner = ucb_runner if algorithm == "Scalarized UCB" else kg_runner
            training = runner.train_online(scalarized_setup(setup, algorithm), formulation,
                       setup["system_dynamics_parameters"]["gamma_grav"], xi, sizes, arms, weights)
        subset = np.asarray(training["recommended_arms"][-1], dtype=int)
        if not len(subset) or np.any((subset < 0) | (subset >= len(arms))):
            raise ValueError("Invalid final recommendation")
        seed = int(training["environment_seed"]) % 2**32
        if seed in {int(s) % 2**32 for s in setup["evaluation"]["scenario_seeds"]}:
            raise ValueError("Training and held-out seeds collide")
        result = {"schema_version": 1, "kind": "holistic_model_free_recommended_arm_set",
                  "algorithm": algorithm, "formulation": formulation, "setup": setup,
                  "source_sha256": fingerprints, "complete": True, "training": training,
                  "model_sizes_kib": dict(zip(CLASS_NAMES, map(float, sizes))),
                  "arms_unit": arms, "recommended_arm_indices": subset,
                  "recommended_unit_margins": arms[subset],
                  "recommended_physical_margins_by_class": {
                      name: low + (high-low)*arms[subset]
                      for name,(low,high) in setup["fitness_margin_bounds"].items()},
                  "selection_note": "Scalarized algorithms return their exploitation union; Pareto UCB returns empirical nondominated arms. These are learned recommendations, not certified true Pareto sets.",
                  "elapsed_seconds": time.perf_counter()-start}
        check_sources(result)
        save(path, result)
        print(f"Saved {filename}; recommended arms: {subset.tolist()}", flush=True)


def main(algorithm):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--formulation", choices=("RTS", "STR"))
    args = parser.parse_args()
    if algorithm == "Pareto UCB" and args.formulation:
        parser.error("Pareto UCB learns one set; RTS/STR are applied during evaluation")
    if args.validate_only:
        load_setup()
        print(f"{algorithm}: setup validated; no simulation run")
    else:
        train(algorithm, args.formulation)
