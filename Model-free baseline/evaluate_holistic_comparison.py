"""Evaluate final learned arm sets on shared held-out random sequences.

No optimization is performed. Pareto UCB's single set is scored under RTS and
STR using identical costs. Existing result files are preserved.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import importlib.util
import os
from pathlib import Path
import pickle
import time
import numpy as np
from tqdm import tqdm
from holistic_runner import HERE, ROOT, load_setup, save, file_hash, check_sources
from script_holistic_gravity import unit_directions, estimated_hypervolume, CLASS_NAMES


def exact_hypervolume(mean_costs, reference):
    points = np.asarray(mean_costs) / np.asarray(reference)
    points = points[np.all(points < 1, axis=1)]
    area, best_y = 0.0, 1.0
    for x, y in points[np.argsort(points[:, 0])]:
        if y < best_y:
            area += (1-x)*(best_y-y)
            best_y = y
    return float(area)


def prepare():
    setup = load_setup()
    cfg = setup["evaluation"]
    draws = np.random.default_rng(cfg["sampling_seed"]).random((cfg["n_random_sequences"], setup["holistic"]["horizon"]))
    groups, all_sequences, hashes = {}, [], {}
    sizes = None
    for algorithm, formulation in (("Scalarized UCB", "RTS"), ("Scalarized KG", "RTS"),
                                   ("Scalarized UCB", "STR"), ("Scalarized KG", "STR"), ("Pareto UCB", None)):
        filename = setup["output_files"][algorithm] if formulation is None else setup["output_files"][algorithm][formulation]
        path = HERE / filename
        with path.open("rb") as file:
            result = pickle.load(file)
        if (not result["complete"] or result["setup"] != setup or result["algorithm"] != algorithm
                or result["formulation"] != formulation):
            raise ValueError(f"Mismatched/incomplete training output: {filename}")
        check_sources(result)
        if sizes is None:
            sizes = result["model_sizes_kib"]
        elif sizes != result["model_sizes_kib"]:
            raise ValueError("Different model payload sizes")
        subset = result["recommended_arm_indices"]
        sampled_indices = subset[np.minimum((draws*len(subset)).astype(int), len(subset)-1)]
        unique_indices = np.unique(sampled_indices, axis=0)
        sequences = result["arms_unit"][unique_indices]
        label = algorithm if formulation is None else f"{algorithm} {formulation}"
        groups[label] = {"algorithm": algorithm, "formulation": formulation,
                         "source_file": filename, "recommended_arm_indices": subset,
                         "sampled_arm_sequences": sampled_indices, "unique_arm_sequences": unique_indices,
                         "n_solutions": len(sequences), "n_draws": len(draws)}
        all_sequences.append(sequences)
        hashes[str(path.relative_to(ROOT))] = file_hash(path)
        hashes.update(result["source_sha256"])
    hashes[str(Path(__file__).resolve().relative_to(ROOT))] = file_hash(__file__)
    unique, inverse = np.unique(np.vstack(all_sequences), axis=0, return_inverse=True)
    offset = 0
    for group in groups.values():
        group["sequence_indices"] = inverse[offset:offset+group["n_solutions"]]
        offset += group["n_solutions"]
    metadata = {"schema_version": 1, "kind": "holistic_model_free_baseline_comparison",
                "setup": setup, "source_sha256": hashes, "model_sizes_kib": sizes,
                "scenario_seeds": cfg["scenario_seeds"]}
    return metadata, groups, unique, draws


def worker_init(setup, sequences, seeds, sizes):
    global _worker
    spec = importlib.util.spec_from_file_location("_model_free_baseline_simulator", ROOT / "Pareto fronts" / "script.py")
    simulator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = simulator
    spec.loader.exec_module(simulator)
    import torch
    torch.set_num_threads(setup["torch_num_threads"])
    xi = {name: np.asarray(setup["Xi_matrices"][name]) for name in CLASS_NAMES}
    _worker = simulator, setup, xi, np.array([sizes[name] for name in CLASS_NAMES]), sequences, seeds


def worker_evaluate(job):
    i, j = job
    simulator, setup, xi, sizes, sequences, seeds = _worker
    cost = simulator.evaluate_sequence(setup, xi, sizes, sequences[i], "holistic", int(seeds[j]))
    if cost.shape != (2,) or not np.isfinite(cost).all() or np.any(cost < 0):
        raise ValueError("Invalid held-out cost vector")
    return i, j, cost


def statistics(result):
    if not result["complete"] or not np.isfinite(result["cost_realizations"]).all():
        raise ValueError("Evaluation incomplete")
    cfg = result["setup"]["evaluation"]
    directions = unit_directions(cfg["hypervolume_directions"], cfg["direction_endpoint_epsilon"])
    reference = np.asarray(result["setup"]["reference_costs"])
    rows = []
    for formulation in ("RTS", "STR"):
        for algorithm in ("Scalarized UCB", "Scalarized KG", "Pareto UCB"):
            label = algorithm if algorithm == "Pareto UCB" else f"{algorithm} {formulation}"
            group = result["groups"][label]
            costs = result["cost_realizations"][group["sequence_indices"]]
            means = costs.mean(axis=1)
            exact = exact_hypervolume(means, reference)
            polar = estimated_hypervolume(costs, formulation, directions, reference)
            rows.append({"algorithm": algorithm, "formulation": formulation,
                         "n_solutions": len(costs), "n_realizations": costs.shape[1],
                         "recommended_arm_indices": group["recommended_arm_indices"],
                         "mean_costs": means, "hypervolume": exact if formulation == "RTS" else polar,
                         "exact_mean_cost_hypervolume": exact, "quadrature_hypervolume": polar})
    return rows


def evaluate(workers=None):
    metadata, groups, sequences, draws = prepare()
    setup = metadata["setup"]
    workers = workers or setup["evaluation"]["workers"]
    path = HERE / setup["comparison_output_file"]
    shape = (len(sequences), len(metadata["scenario_seeds"]), 2)
    if path.exists():
        with path.open("rb") as file:
            result = pickle.load(file)
        if any(result.get(k) != v for k,v in metadata.items()):
            raise ValueError("Existing evaluation cache has different settings or inputs")
        np.testing.assert_array_equal(result["unit_sequences"], sequences)
        if result["complete"]:
            print(f"Already complete: {path}")
            return result
    else:
        result = {**metadata, "groups": groups, "unit_sequences": sequences, "sampling_draws": draws,
                  "cost_realizations": np.full(shape, np.nan), "complete": False, "elapsed_seconds": 0.0}
        save(path, result)
    costs = result["cost_realizations"]
    finished = np.isfinite(costs).all(axis=-1)
    jobs = [tuple(map(int, pair)) for pair in np.argwhere(~finished)]
    start, previous = time.perf_counter(), result["elapsed_seconds"]
    print(f"Held-out comparison: {len(sequences)} unique sequences x {shape[1]} shared seeds", flush=True)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init,
                             initargs=(setup, sequences, metadata["scenario_seeds"], metadata["model_sizes_kib"])) as pool:
        futures = [pool.submit(worker_evaluate, job) for job in jobs]
        try:
            with tqdm(total=finished.size, initial=int(finished.sum()), desc="Model-free comparison", unit="simulation", ascii=True) as bar:
                for count, future in enumerate(as_completed(futures), 1):
                    i,j,values = future.result()
                    costs[i,j] = values
                    bar.update(1)
                    if count % 10 == 0:
                        result["elapsed_seconds"] = previous + time.perf_counter()-start
                        save(path, result)
        except BaseException:
            for future in futures:
                future.cancel()
            result["elapsed_seconds"] = previous + time.perf_counter()-start
            save(path, result)
            raise
    check_sources(result)
    result["complete"] = True
    result["summary"] = statistics(result)
    result["elapsed_seconds"] = previous + time.perf_counter()-start
    save(path, result)
    for row in result["summary"]:
        print(f"{row['algorithm']} {row['formulation']}: {row['hypervolume']:.6f}")
    print(f"Saved {path}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    if args.workers is not None and args.workers < 1:
        parser.error("workers must be positive")
    evaluate(args.workers)
