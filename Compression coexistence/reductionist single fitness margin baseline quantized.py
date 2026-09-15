"""Constant-margin Cheetah baseline using the same Gaussian/NSGA-II search."""

import sys

sys.dont_write_bytecode = True

import hashlib
import importlib.util
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import time

import argparse
from model_based_quantized import load_setup, simulator_inputs, payload_measurements, extra_source_paths

import numpy as np
from tqdm import tqdm

from reductionist_model_based_runner_quantized import validate_setup

HERE = Path(__file__).resolve().parent


def load_search_helpers():
    spec = importlib.util.spec_from_file_location(
        "_constant_margin_search_helpers", HERE / "holistic single fitness margin baseline quantized.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def physical_constant_sequence(margin, horizon):
    """The reductionist simulator consumes physical, not unit, margins."""
    return np.full(horizon, float(margin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    setup_path = HERE / "setup_model_based_reductionist_quantized.json"
    setup = load_setup(setup_path)
    validate_setup(setup)
    baseline = setup["baseline"]
    if baseline["n_realizations_per_margin"] != 1 or baseline["n_grid_points"] < 2:
        raise ValueError("Use at least two grid points and one realization per margin")
    if baseline["grid"] != "linspace_inclusive" or baseline["margin_parameterization"] != "one_physical_margin_for_all_selected_class_requests":
        raise ValueError("Expected fixed physical margins across selected-class requests")
    smoothing = baseline["smoothing"]
    if (smoothing["method"] != "nadaraya_watson_gaussian"
            or smoothing["bandwidth_units"] != "physical_fitness_margin"
            or not np.isfinite(smoothing["bandwidth"]) or smoothing["bandwidth"] <= 0):
        raise ValueError("Expected a positive Gaussian bandwidth in physical margin units")
    hp = baseline["nsga2"]
    if version("pymoo") != hp["pymoo_version"]:
        raise ValueError("Installed pymoo version differs from the setup")
    for name in ("pop_size", "n_offsprings", "n_generations"):
        if type(hp[name]) is not int or hp[name] <= 0:
            raise ValueError(f"Invalid NSGA-II {name}")
    if args.validate_only:
        payload_measurements(setup)
        print("Quantized reductionist baseline setup validated; no simulations or outputs.")
        return
    helpers = load_search_helpers()
    fingerprints = helpers.source_fingerprints()
    for path in (Path(__file__).resolve(), HERE / "reductionist_model_based_runner_quantized.py"):
        fingerprints[str(path.relative_to(HERE.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprints.update({str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in extra_source_paths(setup_path)})
    output = HERE / setup["output_file"]
    if output.exists():
        with output.open("rb") as file:
            previous = pickle.load(file)
        if previous["setup"] != setup or previous["source_sha256"] != fingerprints:
            raise ValueError("Existing baseline uses different settings/source; preserve it elsewhere before rerunning")
        if previous["complete"]:
            print(f"Already complete: {output}", flush=True)
            return
    start = time.perf_counter()
    simulator = helpers.load_simulator()
    import torch
    torch.set_num_threads(setup["torch_num_threads"])
    torch.manual_seed(setup["seed"])
    matrices, model_sizes = simulator_inputs(setup)
    experiment = setup["reductionist"]
    low, high = setup["fitness_margin_bounds"][experiment["class_name"]]
    grid = np.linspace(low, high, baseline["n_grid_points"])
    rng = np.random.default_rng(setup["seed"])
    seeds = rng.choice(2**32, size=len(grid), replace=False).astype(np.uint64)
    costs = np.empty((len(grid), 2))
    for i in tqdm(range(len(grid)), desc="Reductionist constant margins", unit="margin", ascii=True):
        costs[i] = simulator.evaluate_sequence(setup, matrices, model_sizes,
            physical_constant_sequence(grid[i], experiment["horizon"]), "reductionist", int(seeds[i]))
        if not np.isfinite(costs[i]).all():
            raise ValueError("Simulator returned non-finite costs")
    regression = helpers.GaussianCostRegression(grid, costs, smoothing["bandwidth"])
    optimized = helpers.optimize_surrogates(regression, np.asarray([low, high]), hp)
    if any(hashlib.sha256((HERE.parent / p).read_bytes()).hexdigest() != h for p, h in fingerprints.items()):
        raise ValueError("Calibration, setup or source changed during the baseline run")
    result = {
        "schema_version": 1, "kind": "reductionist_constant_fitness_margin_pareto_set",
        "complete": True, "approach": "reductionist", "class_name": experiment["class_name"],
        "setup": setup, "setup_sha256": hashlib.sha256(setup_path.read_bytes()).hexdigest(),
        "source_sha256": fingerprints, "elapsed_seconds": time.perf_counter() - start,
        "versions": {name: version(name) for name in ("numpy", "scipy", "torch", "pymoo")},
        "model_sizes_kib": dict(zip(simulator.CLASS_NAMES, model_sizes.tolist())),
        "payload_measurements": payload_measurements(setup), "compression": "int8",
        "training": {"margins": grid, "costs": costs, "scenario_seeds": seeds,
                     "n_simulator_evaluations": len(grid)},
        **optimized,
        "pareto_unit_margins": (optimized["pareto_margins"] - low) / (high - low),
        "solution_semantics": "Repeat a scalar physical margin for all requests of the selected class.",
        "prediction_note": "Surrogate costs are used for search. Held-out simulation provides the comparison metrics.",
        "held_out_evaluation": None,
    }
    if any(hashlib.sha256((HERE.parent / name).read_bytes()).hexdigest() != digest for name, digest in fingerprints.items()):
        raise RuntimeError("Source changed during the baseline run")
    helpers.save_result(output, result)
    print(f"Saved {len(result['pareto_margins'])} constant margins to {output}", flush=True)


if __name__ == "__main__":
    main()
