"""Quantized reductionist comparison of MORBO, qNParEGO and the threshold baseline.

The baseline samples 20 sequences from its learned threshold set. Each algorithm
is scored under RTS/STR with the same seeds and references as the standard case.
Simulated costs and standard-versus-quantized differences are saved locally.
"""

import sys

sys.dont_write_bytecode = True

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
from tqdm import tqdm

from model_based_quantized import load_setup, extra_source_paths, sample_baseline_sequences, compare_standard, payload_measurements
from model_based_quantized import comparison_statistics as quantized_statistics
from reductionist_model_based_runner_quantized import save_checkpoint, validate_setup


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CACHE_NAME = "reductionist model-based hypervolume comparison quantized.pkl"
GROUPS = (
    ("MORBO RTS", "MORBO", "RTS"),
    ("qNParEGO RTS", "qNParEGO", "RTS"),
    ("MORBO STR", "MORBO", "STR"),
    ("qNParEGO STR", "qNParEGO", "STR"),
    ("Random threshold", "Random threshold", "standard"),
)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def standard_hypervolume(mean_costs, reference_costs):
    """Exact dominated area in normalized minimization space, reference (1, 1)."""
    costs = np.asarray(mean_costs, dtype=float)
    reference = np.asarray(reference_costs, dtype=float)
    if costs.ndim != 2 or costs.shape[1] != 2 or not np.isfinite(costs).all():
        raise ValueError("Expected finite costs of shape (solutions, 2)")
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("Expected two positive finite reference costs")
    points = costs / reference
    points = points[np.all(points < 1, axis=1)]
    area, best_y = 0.0, 1.0
    for x, y in points[np.argsort(points[:, 0], kind="stable")]:
        if y < best_y:
            area += (1 - x) * (best_y - y)
            best_y = y
    return float(area)


def prepare_inputs():
    setup_path = HERE / "setup_model_based_reductionist_quantized.json"
    setup = load_setup(setup_path)
    validate_setup(setup)
    evaluation = deepcopy(setup["held_out_evaluation"])
    evaluation["baseline_hypervolume_formulation"] = "both"
    evaluation["note"] = "Evaluate 20 random Pareto-threshold sequences under RTS and STR on the common held-out seeds."
    if (type(evaluation["n_realizations_per_solution"]) is not int
            or evaluation["n_realizations_per_solution"] < 2):
        raise ValueError("At least two held-out realizations are required")
    if (evaluation["scenario_seed_policy"] != "same_seed_list_for_every_solution_and_algorithm"
            or evaluation["hypervolume_direction_rule"] != "uniform_angle_midpoint_quadrature"):
        raise ValueError("Unsupported evaluation seed/direction policy")
    records, input_hashes, all_x, training_seeds = {}, {}, [], set()
    model_sizes = None
    for label, algorithm, formulation in GROUPS:
        filename = setup["output_file"] if formulation == "standard" else setup["comparison_output_files"][algorithm][formulation]
        path = HERE / filename
        input_hashes[filename] = file_hash(path)
        with path.open("rb") as file:
            record = pickle.load(file)
        if not record["complete"] or record["setup"] != setup:
            raise ValueError(f"Incomplete result or mismatched setup: {filename}")
        for relative, digest in record["source_sha256"].items():
            if file_hash(ROOT / relative) != digest:
                raise ValueError(f"Search source differs from its recorded version: {relative}")
        sizes = record["model_sizes_kib"]
        if model_sizes is None:
            model_sizes = sizes
        elif sizes != model_sizes:
            raise ValueError("Policy payload sizes differ between search results")
        if formulation == "standard":
            if record["kind"] != "reductionist_constant_fitness_margin_pareto_set":
                raise ValueError("Expected the constant-margin search result")
            margins = np.asarray(record["pareto_margins"], dtype=float)
            low, high = setup["fitness_margin_bounds"][setup["reductionist"]["class_name"]]
            unit = (margins - low) / (high - low)
            np.testing.assert_allclose(unit, record["pareto_unit_margins"])
            x = sample_baseline_sequences(margins, (low, high), setup, setup["reductionist"]["horizon"])
            seeds = record["training"]["scenario_seeds"]
        else:
            if record["algorithm"] != algorithm or record["formulation"] != formulation:
                raise ValueError(f"Unexpected algorithm/formulation in {filename}")
            run = record["runs"][formulation]
            hp = record["hyperparameters"]
            expected = hp["n_initial_points"] + hp["n_iterations"] * hp["batch_size"]
            if len(run["X_unit"]) != expected or len(run["direction_angle_history"]) != hp["n_iterations"]:
                raise ValueError(f"Optimization budget not completed: {filename}")
            x = np.asarray(record["pareto_unit_sequences"], dtype=float)
            np.testing.assert_array_equal(x, run["X_unit"][record["pareto_indices"]])
            seeds = run["scenario_seeds"]
        if x.ndim != 2 or not len(x) or x.shape[1] != setup["reductionist"]["horizon"] or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError(f"Invalid sequences in {filename}")
        training_seeds.update(int(seed) % 2**32 for seed in np.asarray(seeds).flat)
        records[label] = {"algorithm": algorithm, "formulation": formulation, "source_file": filename,
                          "n_solutions": len(x), "unit_sequences": x.copy()}
        all_x.append(x)
    unique_x, inverse = np.unique(np.vstack(all_x), axis=0, return_inverse=True)
    offset = 0
    for group in records.values():
        group["sequence_indices"] = inverse[offset:offset + group["n_solutions"]]
        offset += group["n_solutions"]
    held_out_seeds = setup["held_out_scenario_seeds"]
    if any(seed in training_seeds for seed in held_out_seeds):
        raise ValueError("A held-out seed collides with a training seed")
    sources = [Path(__file__), ROOT / "MORBO" / "script_holistic_gravity.py", HERE / "reductionist_model_based_runner_quantized.py",
               ROOT / "Pareto fronts" / "script.py", ROOT / "Cosmic Octopi" / "utils.py",
               ROOT / "Cosmic Octopi" / "simulated_reward_experiment.py",
               ROOT / "Cosmic Octopi" / "front_utils.py", ROOT / "MORBO" / "script_reductionist_gravity.py"]
    sources += extra_source_paths(setup_path)
    metadata = {"schema_version": 1, "kind": "reductionist_held_out_hypervolume_comparison",
                "setup": setup, "evaluation_settings": evaluation,
                "input_sha256": input_hashes, "setup_sha256": file_hash(setup_path),
                "evaluation_source_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
                "scenario_seeds": held_out_seeds, "model_sizes_kib": model_sizes}
    return metadata, records, unique_x


def worker_init(setup, sequences, seeds, sizes):
    global _worker
    spec = importlib.util.spec_from_file_location("_held_out_simulator", ROOT / "Pareto fronts" / "script.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    with redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    import torch
    torch.set_num_threads(1)
    matrices = {name: np.asarray(setup["Xi_matrices"][name], dtype=float) for name in module.CLASS_NAMES}
    payloads = np.asarray([sizes[name] for name in module.CLASS_NAMES])
    _worker = module, setup, matrices, payloads, sequences, seeds


def worker_evaluate(indices):
    sequence_index, repetition = indices
    module, setup, matrices, sizes, sequences, seeds = _worker
    low, high = setup["fitness_margin_bounds"][setup["reductionist"]["class_name"]]
    physical_sequence = low + (high - low) * sequences[sequence_index]
    costs = module.evaluate_sequence(setup, matrices, sizes, physical_sequence, "reductionist", seeds[repetition])
    if np.asarray(costs).shape != (2,) or not np.isfinite(costs).all() or np.any(costs < 0):
        raise ValueError(f"Invalid costs for sequence {sequence_index}, realization {repetition}")
    return sequence_index, repetition, costs


def comparison_statistics(result):
    sys.path.insert(0, str(ROOT / "MORBO"))
    core = importlib.import_module("script_holistic_gravity")
    return quantized_statistics(result, standard_hypervolume, core)

def evaluate(workers=4):
    metadata, groups, sequences = prepare_inputs()
    path = HERE / metadata["setup"]["comparison_cache_file"]
    shape = (len(sequences), len(metadata["scenario_seeds"]), 2)
    if path.exists():
        with path.open("rb") as file:
            result = pickle.load(file)
        if any(result.get(k) != v for k, v in metadata.items()):
            raise ValueError("Cached evaluation has different inputs/settings/source; preserve it elsewhere before changing the comparison")
        np.testing.assert_array_equal(result["unit_sequences"], sequences)
        if result["cost_realizations"].shape != shape:
            raise ValueError("Unexpected cached realization shape")
        if result["complete"]:
            print(f"Already complete: {path}", flush=True)
            return result
    else:
        result = {**metadata, "groups": groups, "unit_sequences": sequences,
                  "cost_realizations": np.full(shape, np.nan), "complete": False, "elapsed_seconds": 0.0}
        save_checkpoint(path, result)
    costs = result["cost_realizations"]
    finished = np.isfinite(costs).all(axis=-1)
    jobs = [tuple(map(int, pair)) for pair in np.argwhere(~finished)]
    start, previous = time.perf_counter(), result["elapsed_seconds"]
    print(f"Evaluating {len(sequences)} unique sequences x {shape[1]} shared seeds, {workers} workers.", flush=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    pool = ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                               initargs=(metadata["setup"], sequences, metadata["scenario_seeds"], metadata["model_sizes_kib"]))
    futures = []
    try:
        futures = [pool.submit(worker_evaluate, job) for job in jobs]
        with tqdm(total=finished.size, initial=int(finished.sum()), desc="Held-out comparison", unit="simulation", ascii=True) as bar:
            for index, future in enumerate(as_completed(futures), 1):
                sequence_index, repetition, values = future.result()
                costs[sequence_index, repetition] = values
                bar.update(1)
                if index % 10 == 0:
                    result["elapsed_seconds"] = previous + time.perf_counter() - start
                    save_checkpoint(path, result)
    except BaseException:
        for future in futures:
            future.cancel()
        result["elapsed_seconds"] = previous + time.perf_counter() - start
        save_checkpoint(path, result)
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for name, digest in metadata["input_sha256"].items():
        if file_hash(HERE / name) != digest:
            raise RuntimeError("An input Pareto set changed during evaluation")
    for relative, digest in metadata["evaluation_source_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise RuntimeError("Evaluation source changed during execution")
    if not np.isfinite(costs).all():
        raise RuntimeError("Some held-out simulations are missing")
    result["complete"] = True
    result["summary"] = comparison_statistics(result)
    result["standard_comparison"] = compare_standard(result)
    result["payload_measurements"] = payload_measurements(metadata["setup"])
    result["elapsed_seconds"] = previous + time.perf_counter() - start
    save_checkpoint(path, result)
    for row in result["summary"]:
        print(f"{row['label']}: {row['hypervolume']:.6f}", flush=True)
    for row in result["standard_comparison"]:
        print(f"{row['algorithm']} {row['formulation']}: standard={row['standard_hypervolume']:.6f}, quantized={row['quantized_hypervolume']:.6f}, difference={row['difference']:+.6f}", flush=True)
    print(f"Saved evaluation to {path}", flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    evaluate(args.workers)
