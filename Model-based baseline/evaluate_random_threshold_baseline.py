"""Evaluate only random Pareto-threshold sequences; never rerun any search.

The 20 sampled sequences are a finite comparison set. Each receives the same
10 held-out simulator seeds as the existing model-based solutions. RTS/STR
hypervolume is computed over this set, not over a single averaged random policy.
"""

import sys
sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from tqdm import tqdm

from holistic_model_based_runner import save_checkpoint

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONFIG = HERE / "setup_random_threshold_evaluation.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample_sequences(pareto_margins, n_sequences, horizon, seed, bounds):
    """Draw independently at every position from the distinct saved thresholds."""
    support = np.unique(np.asarray(pareto_margins, dtype=float))
    low, high = bounds
    if (support.ndim != 1 or not len(support) or not np.isfinite(support).all()
            or not np.isfinite([low, high]).all() or low >= high
            or np.any((support < low) | (support > high))):
        raise ValueError("Invalid Pareto thresholds or physical margin bounds")
    if type(n_sequences) is not int or n_sequences < 1 or type(horizon) is not int or horizon < 1:
        raise ValueError("Sequence count and horizon must be positive integers")
    indices = np.random.default_rng(seed).integers(len(support), size=(n_sequences, horizon))
    physical = support[indices]
    return support, indices, physical, (physical - low) / (high - low)


def check_sources(record):
    for relative, expected in record["source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Input or evaluation source changed: {relative}")


def prepare(case):
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    if (config["sampling"] != "independent_uniform_with_replacement_at_every_iteration"
            or config["interpretation"] != "hypervolume_of_the_sampled_sequence_set"
            or config["realizations"] != "reuse_the_10_shared_held_out_seeds_per_sequence"):
        raise ValueError("Unsupported sampling/evaluation convention")
    cfg = config["cases"][case]
    source = HERE / cfg["comparison_cache"]
    with source.open("rb") as file:
        previous = pickle.load(file)
    if not previous["complete"] or previous["kind"] != f"{case}_held_out_hypervolume_comparison":
        raise ValueError("Expected a completed comparison cache")
    for filename, expected in previous["input_sha256"].items():
        if digest(HERE / filename) != expected:
            raise ValueError(f"Original Pareto set changed: {filename}")
    for relative, expected in previous["evaluation_source_sha256"].items():
        if digest(ROOT / relative) != expected:
            raise ValueError(f"Original evaluation source changed: {relative}")
    setup = previous["setup"]
    selected_class = "Cheetah" if case == "holistic" else setup[case]["class_name"]
    bounds = setup["fitness_margin_bounds"][selected_class]
    if case == "holistic" and any(value != bounds for value in setup["fitness_margin_bounds"].values()):
        raise ValueError("Physical thresholds require matching class bounds")
    pareto_path = HERE / previous["groups"]["Constant margin"]["source_file"]
    with pareto_path.open("rb") as file:
        pareto = pickle.load(file)
    if not pareto["complete"] or pareto["setup"] != setup:
        raise ValueError("Baseline Pareto set has different settings")
    support, indices, physical, unit = sample_sequences(
        pareto["pareto_margins"], config["n_sequences"], setup[case]["horizon"],
        cfg["sampling_seed"], bounds,
    )
    seeds = previous["scenario_seeds"]
    if len(seeds) != 10 or len(set(seeds)) != 10:
        raise ValueError("Expected ten distinct shared held-out seeds")
    paths = {**previous["evaluation_source_sha256"],
             str(source.relative_to(ROOT)): digest(source),
             str(pareto_path.relative_to(ROOT)): digest(pareto_path),
             str(CONFIG.relative_to(ROOT)): digest(CONFIG),
             str(Path(__file__).resolve().relative_to(ROOT)): digest(__file__)}
    metadata = {
        "schema_version": 1, "kind": "random_pareto_threshold_sequence_evaluation",
        "case": case, "config": config, "setup": deepcopy(setup),
        "evaluation_settings": deepcopy(previous["evaluation_settings"]),
        "scenario_seeds": list(seeds), "sampling_seed": cfg["sampling_seed"],
        "source_cache": source.name, "source_pareto_set": pareto_path.name,
        "model_sizes_kib": deepcopy(previous["model_sizes_kib"]), "source_sha256": paths,
    }
    arrays = {"pareto_thresholds": support, "threshold_indices": indices,
              "physical_sequences": physical, "unit_sequences": unit}
    output = HERE / cfg["output_file"]
    if output.parent != HERE or output == source or output == pareto_path:
        raise ValueError("Use a separate output file beside this evaluator")
    return output, metadata, arrays


def worker_init(case, setup, sequences, seeds, sizes):
    global _worker_module
    _worker_module = importlib.import_module(f"evaluate_{case}_comparison")
    _worker_module.worker_init(setup, sequences, seeds, sizes)


def worker_evaluate(indices):
    return _worker_module.worker_evaluate(indices)


def baseline_statistics(result):
    if not result["complete"]:
        raise ValueError("Random-sequence evaluation is incomplete")
    costs = np.asarray(result["cost_realizations"], dtype=float)
    expected = (result["config"]["n_sequences"], len(result["scenario_seeds"]), 2)
    if costs.shape != expected or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError("Invalid sampled-sequence cost measurements")
    evaluator = importlib.import_module(f"evaluate_{result['case']}_comparison")
    sys.path.insert(0, str(ROOT / "MORBO"))
    core = importlib.import_module("script_holistic_gravity")
    cfg = result["evaluation_settings"]
    directions = core.unit_directions(cfg["hypervolume_directions"], cfg["direction_endpoint_epsilon"])
    reference = np.asarray(result["setup"]["reference_costs"])
    means = costs.mean(axis=1)
    exact = evaluator.standard_hypervolume(means, reference)
    rts = core.estimated_hypervolume(costs, "RTS", directions, reference)
    str_hv = core.estimated_hypervolume(costs, "STR", directions, reference)
    return [{"label": f"Random threshold {formulation}", "algorithm": "Random threshold",
             "formulation": formulation, "n_solutions": costs.shape[0],
             "n_realizations": costs.shape[1], "mean_costs": means,
             "hypervolume": exact if formulation == "RTS" else str_hv,
             "standard_hypervolume_of_means": exact, "rts_quadrature_hypervolume": rts,
             "str_hypervolume": str_hv,
             "metric": "Exact 2D hypervolume of mean costs" if formulation == "RTS"
                       else f"STR expected-length ({len(directions)} directions)"}
            for formulation in ("RTS", "STR")]


def evaluate(case, workers):
    path, metadata, arrays = prepare(case)
    shape = (len(arrays["unit_sequences"]), len(metadata["scenario_seeds"]), 2)
    if path.exists():
        with path.open("rb") as file:
            result = pickle.load(file)
        if any(result.get(k) != v for k, v in metadata.items()):
            raise ValueError("Existing randomized baseline cache has different inputs/settings")
        for key, value in arrays.items():
            np.testing.assert_array_equal(result[key], value)
        if result["cost_realizations"].shape != shape:
            raise ValueError("Cached cost dimensions differ")
        if result["complete"]:
            print(f"Already complete: {path}", flush=True)
            return result
    else:
        result = {**metadata, **arrays, "complete": False, "elapsed_seconds": 0.0,
                  "cost_realizations": np.full(shape, np.nan)}
        save_checkpoint(path, result)
    costs = result["cost_realizations"]
    finished = np.isfinite(costs).all(axis=-1)
    jobs = [tuple(map(int, pair)) for pair in np.argwhere(~finished)]
    start, previous_time = time.perf_counter(), result["elapsed_seconds"]
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(case, metadata["setup"], arrays["unit_sequences"],
                                       metadata["scenario_seeds"], metadata["model_sizes_kib"])) as pool:
        futures = [pool.submit(worker_evaluate, job) for job in jobs]
        try:
            with tqdm(total=finished.size, initial=int(finished.sum()), desc=f"{case} random baseline",
                      unit="simulation", ascii=True) as bar:
                for count, future in enumerate(as_completed(futures), 1):
                    i, j, values = future.result()
                    costs[i, j] = values
                    bar.update(1)
                    if count % 10 == 0:
                        result["elapsed_seconds"] = previous_time + time.perf_counter() - start
                        save_checkpoint(path, result)
        except BaseException:
            for future in futures:
                future.cancel()
            result["elapsed_seconds"] = previous_time + time.perf_counter() - start
            save_checkpoint(path, result)
            raise
    check_sources(result)
    if not np.isfinite(costs).all():
        raise RuntimeError("Some evaluations are missing")
    result["complete"] = True
    result["summary"] = baseline_statistics(result)
    result["elapsed_seconds"] = previous_time + time.perf_counter() - start
    save_checkpoint(path, result)
    for row in result["summary"]:
        print(f"{case} {row['label']}: {row['hypervolume']:.6f}", flush=True)
    print(f"Saved {path}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("holistic", "reductionist", "both"), default="both")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for case in (("holistic", "reductionist") if args.case == "both" else (args.case,)):
        if args.validate_only:
            _, metadata, arrays = prepare(case)
            print(f"{case}: {len(arrays['unit_sequences'])} sequences x {len(metadata['scenario_seeds'])} realizations; validation passed")
        else:
            evaluate(case, args.workers)
