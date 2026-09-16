"""Generate simulated Humanoid cost trajectories and assess stationarity.

Run with --stage generate or --stage analyze to separate simulation and analysis.
Completed trajectories and statistical tests are checkpointed in result.pkl.
No PPO training is performed. Existing experiment results are not modified.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
_CONTEXT = None
GENERATION_SOURCES = [
    ROOT / "Scalarized UCB/online_environment.py",
    ROOT / "Cosmic Octopi/utils.py",
    ROOT / "Cosmic Octopi/simulated_reward_experiment.py",
    Path(__file__),
]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def save_result(path, result):
    """Atomic checkpoint in the same directory/volume as the final pickle."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_setup(setup):
    if setup.get("simulated_rewards") is not True:
        raise ValueError("Only simulated_rewards=true is supported; training is disabled.")
    if setup["reductionist"]["class_name"] != "Humanoid":
        raise ValueError("This experiment targets Humanoid.")
    for key in ("n_independent_runs", "workers"):
        if type(setup[key]) is not int or setup[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    horizon = setup["reductionist"]["horizon"]
    arms = setup["arms"]
    if (type(arms["n_arms"]) is not int or arms["n_arms"] < 2
            or arms["randomization"] != "independent_uniform"):
        raise ValueError("Use at least two independently randomized arms.")
    if arms["interval"] != setup["fitness_margin_bounds"]["Humanoid"]:
        raise ValueError("Arm interval must match the Humanoid fitness-margin bounds.")
    if not np.isfinite(arms["interval"]).all() or not arms["interval"][0] < arms["interval"][1]:
        raise ValueError("Invalid arm interval")
    stats = setup["statistics"]
    horizons = stats["horizons"]
    lag = stats["ljung_box"]["max_lag"]
    if (type(lag) is not int or lag < 1 or not horizons
            or horizons != sorted(set(horizons))
            or any(type(p) is not int or not lag < p <= horizon for p in horizons)):
        raise ValueError("Horizons must be distinct increasing integers above the fixed lag.")
    if not 0 < stats["alpha"] < 1 or not 0 < stats["confidence_level"] < 1:
        raise ValueError("Invalid significance/confidence level")
    if stats["ljung_box"]["calibration"] != "joint_row_permutation":
        raise ValueError("Unsupported Ljung-Box calibration")
    if stats.get("second_test") == "quantile_coverage":
        q = stats["quantile_coverage"]
        if (not 0 < q["alpha"] < 1 or not 0 < q["minimum_coverage"] < 1
                or not 0 < q["calibration_fraction"] < 1
                or type(q["calibration_horizon"]) is not int
                or not 1 <= q["calibration_horizon"] <= horizon
                or q["sampling"] != "one_uniform_iteration_per_evaluation_trajectory"
                or q["correction"] != "holm_within_band_and_horizon"):
            raise ValueError("Invalid quantile coverage design")
        if not q["bands"] or any(not 0 <= b["lower"] < b["upper"] <= 1 for b in q["bands"]):
            raise ValueError("Invalid quantile bands")
    elif stats.get("second_test") == "manova":
        for method, count_key in (("ljung_box", "n_permutations"), ("manova", "n_bootstrap")):
            levels = stats[method]["significance_levels"]
            count = stats[method][count_key]
            if not levels or any(not 0 < level < 1 for level in levels):
                raise ValueError("Invalid significance levels")
            if type(count) is not int or count < 1:
                raise ValueError("Invalid resampling count")
            threshold = min(levels) / (arms["n_arms"] if method == "manova" else 1)
            if 1/(count+1) > threshold:
                raise ValueError(f"Too few resamples to resolve {method} significance levels")
        if (stats["manova"]["statistic"] != "pillai_trace"
                or stats["manova"]["calibration"] != "trajectory_wild_bootstrap_hc2"
                or stats["manova"]["correction"] != "holm_within_horizon"):
            raise ValueError("Unsupported MANOVA specification")
    elif stats["energy"]["statistic"] != "disco_between" or stats["energy"]["correction"] != "holm_within_horizon":
        raise ValueError("Unsupported energy test")
    for method in (() if stats.get("second_test") in ("manova", "quantile_coverage") else ("ljung_box", "energy")):
        b = stats[method]["n_permutations"]
        if type(b) is not int or b < 1:
            raise ValueError("Permutation counts must be positive integers")
    levels = stats.get("significance_levels", [stats["alpha"]])
    if not levels or any(not 0 < level < 1 for level in levels):
        raise ValueError("Invalid significance levels")
    if (stats.get("second_test") not in ("manova", "quantile_coverage") and
            1 / (stats["energy"]["n_permutations"] + 1) > min(stats["alpha"], *levels) / arms["n_arms"]):
        raise ValueError("Too few energy permutations to resolve the first Holm threshold")
    if stats["include_initial_iterations"] is not True:
        raise ValueError("The requested analysis includes initialization; no burn-in is removed.")
    for matrix in setup["Xi_matrices"].values():
        if np.asarray(matrix).shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Xi matrices must be finite 4x4 arrays")


def initialize_simulator(setup):
    global _CONTEXT
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / "Scalarized UCB"))
    from online_environment import run_online
    from utils import CLASS_NAMES, Network, get_model_size_in_kb
    import torch
    torch.set_num_threads(1)
    matrices = {name: np.asarray(value, float) for name, value in setup["Xi_matrices"].items()}
    sizes = np.array([get_model_size_in_kb(Network(*setup["policy_shapes"][name]).float())
                      for name in CLASS_NAMES])
    _CONTEXT = (setup, run_online, CLASS_NAMES, matrices, sizes)
    return CLASS_NAMES, sizes


def simulate_trajectory(task):
    run_index, seed, arm_indices = task
    setup, run_online, names, matrices, sizes = _CONTEXT
    target = names.index("Humanoid")
    values = np.linspace(*setup["arms"]["interval"], setup["arms"]["n_arms"])
    horizon = setup["reductionist"]["horizon"]
    costs = np.full((horizon, 2), np.nan)
    seen = np.zeros(horizon, dtype=bool)

    def select(c, k):
        if c != target or not 0 <= k < horizon:
            raise RuntimeError("Unexpected class/iteration in reductionist simulator")
        return values[arm_indices[k]]

    def observe(c, k, arm, vector):
        if c != target or seen[k] or arm != values[arm_indices[k]]:
            raise RuntimeError("Misaligned or duplicate cost observation")
        costs[k] = vector
        seen[k] = True

    start = time.perf_counter()
    total = run_online(setup, matrices, sizes, int(seed), select, observe, "reductionist")
    if not seen.all() or not np.isfinite(costs).all():
        raise RuntimeError("Incomplete/nonfinite simulated cost trajectory")
    np.testing.assert_allclose(costs.sum(axis=0), total, rtol=1e-10, atol=1e-9)
    bounds = np.array([setup["system_dynamics_parameters"]["n_octopi"] * sizes[target], 1.0])
    if np.any(costs < -1e-9) or np.any(costs > bounds + 1e-8):
        raise RuntimeError("A simulated cost exceeded its normalization bound")
    return run_index, costs, time.perf_counter() - start


def generation_signature(setup):
    settings = {k: v for k, v in setup.items() if k not in
                {"statistics", "workers", "output_file", "calibration_source"}}
    sources = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in GENERATION_SOURCES}
    return digest({"settings": settings, "sources": sources}), sources


def prepare_result(setup, path):
    signature, sources = generation_signature(setup)
    if path.exists():
        with path.open("rb") as stream:
            result = pickle.load(stream)
        if result["generation_signature"] != signature:
            raise ValueError("Simulation setup/source differs from this result. Use a new output_file.")
        return result
    names, sizes = initialize_simulator(setup)
    r, p = setup["n_independent_runs"], setup["reductionist"]["horizon"]
    streams = np.random.SeedSequence(setup["seed"]).spawn(2)
    arms = np.random.default_rng(streams[0]).integers(0, setup["arms"]["n_arms"], size=(r, p))
    seeds = np.array([int(s.generate_state(1)[0]) for s in streams[1].spawn(r)], dtype=np.uint32)
    if len(np.unique(seeds)) != r:
        raise ValueError("Environment seed collision; choose another master seed")
    scale = np.array([setup["system_dynamics_parameters"]["n_octopi"] * sizes[names.index("Humanoid")], 1.0])
    return {
        "schema_version": 1, "setup": deepcopy(setup), "generation_signature": signature,
        "source_sha256": sources, "class_names": names, "model_sizes_kib": sizes,
        "objective_names": ["overhead_increment", "instability_increment"],
        "objective_units": ["KiB", "dimensionless"], "objective_scale": scale,
        "arm_values": np.linspace(*setup["arms"]["interval"], setup["arms"]["n_arms"]),
        "arm_indices": arms, "environment_seeds": seeds,
        "costs": np.full((r, p, 2), np.nan), "completed_runs": np.zeros(r, dtype=bool),
        "trajectory_seconds": np.zeros(r), "analysis": None,
    }


def generate(setup, result, path):
    missing = np.flatnonzero(~result["completed_runs"])
    if not len(missing):
        print("All simulated trajectories are already saved.", flush=True)
        return
    tasks = [(int(i), int(result["environment_seeds"][i]), result["arm_indices"][i]) for i in missing]
    with ProcessPoolExecutor(max_workers=setup["workers"], mp_context=get_context("spawn"),
                             initializer=initialize_simulator, initargs=(setup,)) as pool:
        futures = [pool.submit(simulate_trajectory, task) for task in tasks]
        with tqdm(total=setup["n_independent_runs"], initial=int(result["completed_runs"].sum()),
                  desc="Simulated Humanoid trajectories", ascii=True) as progress:
            for future in as_completed(futures):
                i, costs, seconds = future.result()
                result["costs"][i] = costs
                result["trajectory_seconds"][i] = seconds
                result["completed_runs"][i] = True
                save_result(path, result)
                progress.update()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=HERE / "setup.json")
    parser.add_argument("--stage", choices=("all", "generate", "analyze"), default="all")
    parser.add_argument("--reanalyze", action="store_true", help="Replace statistical analysis, retaining trajectories")
    args = parser.parse_args()
    setup_path = args.setup.resolve()
    setup = json.loads(setup_path.read_text(encoding="utf-8"))
    validate_setup(setup)
    if setup["statistics"].get("second_test") == "quantile_coverage":
        if setup["statistics"]["quantile_coverage"].get("input_result_file") is None:
            from analysis_quantiles import analyze_generated
            output = setup_path.parent / setup["output_file"]
            result = prepare_result(setup, output)
            if args.stage in ("all", "generate"):
                generate(setup, result, output)
            if args.stage in ("all", "analyze"):
                analyze_generated(setup, result, output, save_result, reset=args.reanalyze)
            print(f"Saved {output}", flush=True)
            return
        if args.stage == "generate":
            raise ValueError("Quantile coverage reuses saved trajectories; use --stage analyze.")
        from analysis_quantiles import run_saved
        run_saved(setup, setup_path, reset=args.reanalyze)
        return
    if setup["statistics"].get("second_test") == "manova":
        if args.stage == "generate":
            raise ValueError("This configuration reuses saved trajectories; use --stage analyze.")
        from analysis_manova import run_saved
        run_saved(setup, setup_path, reset=args.reanalyze)
        return
    output = setup_path.parent / setup["output_file"]
    result = prepare_result(setup, output)
    if args.stage in ("all", "generate"):
        generate(setup, result, output)
    if args.stage in ("all", "analyze"):
        if not result["completed_runs"].all():
            raise ValueError("Generate all trajectories before statistical analysis")
        from analysis import analyze
        analyze(setup, result, output, save_result, reset=args.reanalyze)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
