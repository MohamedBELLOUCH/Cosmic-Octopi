"""Adapt existing holistic optimizers to the baseline comparison's shared setup.

The existing run_condition functions perform all optimization and simulation.
This module supplies inline settings, separate resumable output files, progress
reporting, and extraction of empirical RTS/STR solution sets. It never calls
the original CLI mains or reads their setup/hyperparameter/calibration JSONs.
"""

import sys

sys.dont_write_bytecode = True

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import importlib.util
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile
import time

import numpy as np
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
CLASS_NAMES = ("Cheetah", "Ant", "Leg", "Humanoid")


def validate_setup(setup):
    """Validate the settings used by both model-based comparison scripts."""
    def positive_int(value, name):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if setup["schema_version"] != 1 or setup["simulated_rewards"] is not True:
        raise ValueError("Expected schema_version=1 and simulated_rewards=true")
    if setup["n_independent_runs"] != 1:
        raise ValueError("These comparison scripts run one independent optimization")
    if type(setup["seed"]) is not int or not 0 <= setup["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    positive_int(setup["torch_num_threads"], "torch_num_threads")
    positive_int(setup["ou_substeps"], "ou_substeps")
    positive_int(setup["holistic"]["horizon"], "holistic.horizon")
    positive_int(setup["n_realizations_per_candidate"], "n_realizations_per_candidate")
    if setup["n_realizations_per_candidate"] < 2:
        raise ValueError("RTS/STR comparison requires at least two realizations per candidate")
    if setup["holistic"]["margin_parameterization"] != "normalized_sequence":
        raise ValueError("Expected normalized_sequence parameterization")
    if setup["formulations"] != ["RTS", "STR"]:
        raise ValueError("The shared comparison setup must include both RTS and STR")
    if setup["objective_names"] != ["total_overhead", "total_instability"] or setup["objective_senses"] != ["minimize", "minimize"]:
        raise ValueError("Minimize total overhead and total instability, in that order")
    if setup["objective_units"] != ["KiB", "dimensionless"] or setup["objective_normalization"] != "divide_each_cost_by_its_reference_cost":
        raise ValueError("Use the existing simulator's units and reference normalization")
    reference = np.asarray(setup["reference_costs"], dtype=float)
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Expected two finite positive reference costs")
    p = setup["system_dynamics_parameters"]
    if p["n_classes"] != len(CLASS_NAMES):
        raise ValueError("The shared simulator requires all four classes")
    positive_int(p["n_octopi"], "n_octopi")
    for name in ("gamma_learn", "gamma_break", "gamma_req", "gamma_grav", "gamma_epis"):
        if not np.isfinite(p[name]) or p[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(p["gamma_sync"]) or not 0 < p["gamma_sync"] <= 1:
        raise ValueError("gamma_sync must be in (0, 1]")
    if not np.isfinite(p["gamma_heter"]) or p["gamma_heter"] < 0:
        raise ValueError("gamma_heter must be finite and nonnegative")
    if not np.isfinite(setup["gravity_mean"]) or setup["gravity_mean"] <= 0:
        raise ValueError("gravity_mean must be finite and positive")
    for name in ("temperature", "scaling_constant"):
        if not np.isfinite(setup["hyper_parameters"][name]) or setup["hyper_parameters"][name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    for field in ("Xi_matrices", "policy_shapes", "fitness_margin_bounds"):
        if set(setup[field]) != set(CLASS_NAMES):
            raise ValueError(f"{field} must contain exactly the four simulator classes")
    bounds = np.asarray([setup["fitness_margin_bounds"][name] for name in CLASS_NAMES])
    if bounds.shape != (4, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]) or not np.all(bounds == bounds[0]):
        raise ValueError("The constant-margin comparison requires identical increasing class bounds")
    for name in CLASS_NAMES:
        matrix = np.asarray(setup["Xi_matrices"][name])
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{name}: expected a finite 4x4 Xi matrix")
        shape = setup["policy_shapes"][name]
        if len(shape) != 3:
            raise ValueError(f"{name}: expected input, action and hidden sizes")
        for size in shape:
            positive_int(size, f"{name} policy size")
    for algorithm in ("MORBO", "qNParEGO"):
        hp = setup["hyperparameters"][algorithm]
        for name in ("dimension", "n_initial_points", "n_iterations", "batch_size", "hypervolume_checkpoint_every",
                     "mc_samples", "acquisition_num_restarts", "acquisition_raw_samples",
                     "acquisition_batch_limit", "acquisition_maxiter", "hypervolume_directions"):
            positive_int(hp[name], f"{algorithm}.{name}")
        if hp["dimension"] != setup["holistic"]["horizon"]:
            raise ValueError(f"{algorithm}: dimension must equal the holistic horizon")
        expected = {"dtype": "float64", "device": "cpu", "initial_design": "scrambled_sobol",
                    "surrogate": "SingleTaskGP", "surrogate_fit": "ExactMarginalLogLikelihood",
                    "rts_str_acquisition": "qLogNoisyExpectedImprovement",
                    "hypervolume_direction_rule": "uniform_angle_midpoint_quadrature",
                    "reference_point_policy": "fixed_reference_costs_from_setup"}
        if any(hp[key] != value for key, value in expected.items()):
            raise ValueError(f"{algorithm}: settings must match the reused implementation")
        if not 0 < hp["direction_endpoint_epsilon"] < np.pi / 4:
            raise ValueError("direction_endpoint_epsilon must be in (0, pi/4)")
    morbo, qnparego = (setup["hyperparameters"][name] for name in ("MORBO", "qNParEGO"))
    for name in ("dimension", "n_initial_points", "n_iterations", "batch_size", "hypervolume_checkpoint_every",
                 "hypervolume_directions", "direction_endpoint_epsilon"):
        if morbo[name] != qnparego[name]:
            raise ValueError(f"Keep {name} matched across both algorithms")
    if not 0 < morbo["trust_region_length_minimum"] <= morbo["trust_region_length_initial"] <= morbo["trust_region_length_maximum"]:
        raise ValueError("Invalid trust-region lengths")
    if not morbo["trust_region_success_multiplier"] > 1 or not 0 < morbo["trust_region_failure_multiplier"] < 1:
        raise ValueError("Invalid trust-region update multipliers")
    weights = [morbo["trust_region_center_old_weight"], morbo["trust_region_center_new_weight"]]
    if not np.isfinite(weights).all() or np.any(np.asarray(weights) < 0) or not np.isclose(sum(weights), 1):
        raise ValueError("Trust-region center weights must be nonnegative and sum to one")
    filenames = [setup["output_file"], *(setup["comparison_output_files"][a][f]
                  for a in ("MORBO", "qNParEGO") for f in ("RTS", "STR"))]
    if len({name.casefold() for name in filenames}) != len(filenames):
        raise ValueError("Baseline and model-based outputs must have distinct filenames")
    if any(Path(name).name != name or Path(name).suffix != ".pkl" for name in filenames):
        raise ValueError("Outputs must be .pkl filenames beside the setup")


def load_backends(algorithm):
    sys.path.insert(0, str(PROJECT_ROOT / "MORBO"))
    core = importlib.import_module("script_holistic_gravity")
    if algorithm == "MORBO":
        return core, core
    if algorithm != "qNParEGO":
        raise ValueError(f"Unknown algorithm: {algorithm}")
    spec = importlib.util.spec_from_file_location("_baseline_qnparego", PROJECT_ROOT / "qNParEGO" / "runner.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return core, module


def source_fingerprints(algorithm, entrypoint):
    paths = [Path(__file__), Path(entrypoint), PROJECT_ROOT / "MORBO" / "script_holistic_gravity.py",
             PROJECT_ROOT / "Pareto fronts" / "script.py", PROJECT_ROOT / "Cosmic Octopi" / "utils.py",
             PROJECT_ROOT / "Cosmic Octopi" / "simulated_reward_experiment.py",
             PROJECT_ROOT / "Cosmic Octopi" / "front_utils.py"]
    if algorithm == "qNParEGO":
        paths += [PROJECT_ROOT / "qNParEGO" / "runner.py", PROJECT_ROOT / "MORBO" / "script_reductionist_gravity.py"]
    return {str(path.resolve().relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def nondominated_indices(costs):
    """Strict Pareto nondominance for minimization, preserving tied solutions."""
    costs = np.asarray(costs, dtype=float)
    if costs.ndim != 2 or not len(costs) or not np.isfinite(costs).all():
        raise ValueError("Expected a nonempty finite matrix of objective values")
    return np.asarray([i for i, point in enumerate(costs)
                       if not np.any(np.all(costs <= point, axis=1) & np.any(costs < point, axis=1))], dtype=int)


def extract_pareto_set(run, formulation, setup, hp, core):
    """Preserve empirical RTS/STR hypervolume using formulation-aware filtering.

    RTS minimizes mean costs. STR maximizes expected length scores across the
    configured direction grid; mean-cost filtering can discard an STR winner.
    STR is therefore a finite-direction approximation, not an ordinary 2D mean
    cost front. All original observations remain in the output's run record.
    """
    x = np.asarray(run["X_unit"], dtype=float)
    costs = np.asarray(run["cost_realizations"], dtype=float)
    if x.ndim != 2 or x.shape[1] != hp["dimension"] or costs.shape != (len(x), setup["n_realizations_per_candidate"], 2):
        raise ValueError("Inconsistent sequence or realization shapes")
    if not np.isfinite(costs).all() or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
        raise ValueError("Invalid optimization observations")
    directions = core.unit_directions(hp["hypervolume_directions"], hp["direction_endpoint_epsilon"])
    reference = np.asarray(setup["reference_costs"], dtype=float)
    scores = core.scalar_scores(costs, formulation, directions, reference)
    means = costs.mean(axis=1)
    indices = nondominated_indices(means if formulation == "RTS" else -scores)
    all_hv = core.estimated_hypervolume(costs, formulation, directions, reference)
    selected_hv = core.estimated_hypervolume(costs[indices], formulation, directions, reference)
    if not np.isclose(all_hv, selected_hv, rtol=1e-12, atol=1e-14):
        raise RuntimeError("Pareto filtering changed the empirical hypervolume")
    low, high = setup["fitness_margin_bounds"][CLASS_NAMES[0]]
    return {
        "pareto_indices": indices,
        "pareto_unit_sequences": x[indices].copy(),
        "pareto_margin_sequences": low + (high - low) * x[indices],
        "pareto_cost_realizations": costs[indices].copy(),
        "pareto_scenario_seeds": np.asarray(run["scenario_seeds"])[indices].copy(),
        "pareto_mean_costs": means[indices],
        "pareto_directional_scores": scores[indices],
        "pareto_selection": "nondominated_mean_costs" if formulation == "RTS" else "nondominated_empirical_directional_length_profiles",
        "hypervolume_directions": directions,
        "training_hypervolume": selected_hv,
        "held_out_evaluation": None,
        "solution_semantics": "Each row is a fitness-margin sequence with one entry per holistic global iteration.",
        "evaluation_note": "Selection uses training realizations. A later common held-out evaluation is required for the comparison barplot. STR mean costs are descriptive, not its Pareto-front representation.",
    }


def save_checkpoint(path, result):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".holistic_optimizer_", suffix=".tmp",
                                         dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(7):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.05 * 2**attempt)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_formulation(setup, algorithm, formulation, core, backend, xi_matrices,
                    model_sizes, output, fingerprints, show_progress=True):
    hp = setup["hyperparameters"][algorithm]
    metadata = {
        "schema_version": 1, "kind": "holistic_model_based_pareto_set", "profile": "full",
        "algorithm": algorithm, "formulation": formulation, "approach": "holistic",
        "setup": deepcopy(setup), "hyperparameters": deepcopy(hp), "source_sha256": fingerprints,
        "model_sizes_kib": dict(zip(CLASS_NAMES, map(float, model_sizes))),
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scipy", "torch", "botorch", "gpytorch")
        }},
    }
    if output.exists():
        with output.open("rb") as file:
            result = pickle.load(file)
        if any(result.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Settings, source, or versions differ from {output.name}; use a new --output-dir to keep the existing result")
        if result["complete"]:
            print(f"Already complete: {output}", flush=True)
            return result
    else:
        result = {**metadata, "complete": False, "runs": {}, "elapsed_seconds": 0.0}
        save_checkpoint(output, result)
    total = hp["n_initial_points"] + hp["n_iterations"] * hp["batch_size"]
    previous_elapsed = result["elapsed_seconds"]
    start = time.perf_counter()
    existing = result["runs"].get(formulation, {})
    with tqdm(total=total, initial=len(existing.get("X_unit", [])),
              desc=f"{algorithm} {formulation}", unit="candidate", ascii=True,
              dynamic_ncols=True, disable=not show_progress) as bar:
        def checkpoint(path, current):
            if Path(path) != output or current is not result:
                raise RuntimeError("Optimizer attempted to save an unexpected checkpoint")
            current["elapsed_seconds"] = previous_elapsed + time.perf_counter() - start
            save_checkpoint(output, current)
            run = current["runs"].get(formulation, {})
            count = len(run.get("X_unit", []))
            bar.update(max(0, count - bar.n))
            steps = len(run.get("direction_angle_history", []))
            phase = "initial design" if count < hp["n_initial_points"] else "optimization"
            if run.get("pending_angle") is not None and run.get("pending_X") is None:
                phase = "fitting GP/acquisition"
            bar.set_postfix(step=f"{steps}/{hp['n_iterations']}", phase=phase)

        # This is a process-local callback substitution. No shared source file
        # or other running Python process is modified.
        original_save = backend.save_result
        backend.save_result = checkpoint
        try:
            with redirect_stdout(io.StringIO()):
                tail = (setup, hp, xi_matrices, model_sizes, output)
                gravity = setup["system_dynamics_parameters"]["gamma_grav"]
                if algorithm == "MORBO":
                    backend.run_condition(result, formulation, formulation, gravity, *tail)
                else:
                    backend.run_condition(result, formulation, formulation, "gamma_grav", gravity, *tail, "holistic")
        finally:
            backend.save_result = original_save
    run = result["runs"][formulation]
    if len(run["X_unit"]) != total or len(run["direction_angle_history"]) != hp["n_iterations"]:
        raise RuntimeError("Optimizer stopped before completing the configured budget")
    if any(hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest() != digest for name, digest in fingerprints.items()):
        raise RuntimeError("Source files changed during optimization; checkpoint kept for inspection")
    result.update(extract_pareto_set(run, formulation, setup, hp, core))
    result["elapsed_seconds"] = previous_elapsed + time.perf_counter() - start
    result["complete"] = True
    save_checkpoint(output, result)
    print(f"Saved {len(result['pareto_indices'])} {algorithm} {formulation} sequences to {output}", flush=True)
    return result


def main(algorithm, entrypoint):
    parser = argparse.ArgumentParser(description=f"Holistic {algorithm} RTS/STR baseline comparison")
    parser.add_argument("--setup", type=Path, default=EXPERIMENT_DIR / "setup_model_free_holistic.json")
    parser.add_argument("--output-dir", type=Path, help="Optional separate directory for result files")
    parser.add_argument("--formulation", choices=("RTS", "STR"), help="Run only one formulation; default: both")
    parser.add_argument("--validate-only", action="store_true", help="Check settings and show the budget without running or writing results")
    args = parser.parse_args()
    setup_path = args.setup.resolve()
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    validate_setup(setup)
    output_dir = args.output_dir.resolve() if args.output_dir else setup_path.parent
    formulations = [args.formulation] if args.formulation else setup["formulations"]
    hp = setup["hyperparameters"][algorithm]
    total = hp["n_initial_points"] + hp["batch_size"] * hp["n_iterations"]
    print(f"{algorithm}: {hp['n_initial_points']} initial points + {hp['n_iterations']} steps x "
          f"{hp['batch_size']} candidates = {total} candidates per formulation; "
          f"{setup['n_realizations_per_candidate']} realizations each, horizon {hp['dimension']}.", flush=True)
    for formulation in formulations:
        print(f"  {formulation}: {output_dir / setup['comparison_output_files'][algorithm][formulation]}", flush=True)
    if args.validate_only:
        print("Setup valid. No optimization or output files created.", flush=True)
        return
    core, backend = load_backends(algorithm)
    import torch

    torch.set_num_threads(setup["torch_num_threads"])
    torch.manual_seed(setup["seed"])
    xi_matrices = {name: np.asarray(setup["Xi_matrices"][name], dtype=float) for name in CLASS_NAMES}
    model_sizes = np.asarray([core.get_model_size_in_kb(core.Network(*setup["policy_shapes"][name]).float()) for name in CLASS_NAMES])
    fingerprints = source_fingerprints(algorithm, entrypoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        for formulation in formulations:
            output = output_dir / setup["comparison_output_files"][algorithm][formulation]
            run_formulation(setup, algorithm, formulation, core, backend, xi_matrices,
                            model_sizes, output, fingerprints)
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume the saved candidate checkpoint.", flush=True)
        raise SystemExit(130)
