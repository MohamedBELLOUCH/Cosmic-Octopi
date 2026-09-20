"""Saved-data independence diagnostics on iterations cutoff+1 through T.

Reuses the exact test functions from independence_comparison.py and analysis.py.
No simulation or training. The original prefix analysis is never overwritten.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from analysis import ljung_box_test
from independence_comparison import autohsic_test, bds_test, save_checkpoint


HERE = Path(__file__).resolve().parent
_CONTEXT = None
_THREAD_LIMIT = None


def suffix(series, cutoff):
    """1-based global iterations cutoff+1,...,T, with their original order."""
    if type(cutoff) is not int or not 0 <= cutoff < len(series):
        raise ValueError("Invalid cutoff")
    return series[cutoff:]


def validate_config(config, data):
    if (data["setup"].get("simulated_rewards") is not True
            or not np.asarray(data["completed_runs"]).all()
            or not np.isfinite(data["costs"]).all()):
        raise ValueError("Complete finite simulated trajectories are required")
    cutoffs = config["cutoff_horizons"]
    t = data["costs"].shape[1]
    if not cutoffs or cutoffs != sorted(set(cutoffs)) or any(type(p) is not int or not 0 <= p < t for p in cutoffs):
        raise ValueError("Invalid cutoff horizons")
    if not 0 < config["alpha"] < 1 or config["scaling"] != "saved_objective_scale":
        raise ValueError("Invalid significance or normalization")
    for name in ("ljung_box", "autohsic"):
        lag = config[name]["max_lag"]
        if type(lag) is not int or not 0 < lag <= t - max(cutoffs) - 4:
            raise ValueError("Every suffix must support the fixed maximum lag")
    for name, key in (("ljung_box", "n_permutations"), ("autohsic", "n_bootstrap"), ("bds", "n_permutations")):
        n = config[name][key]
        factor = data["costs"].shape[2] if name == "bds" else 1
        if type(n) is not int or n < 1 or factor/(n+1) > config["alpha"]:
            raise ValueError("Insufficient resamples for the significance level")
    if type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("Invalid worker count")
    required = {
        "ljung_box": {"calibration": "joint_row_permutation"},
        "autohsic": {"kernel": "gaussian", "bandwidth": "median_pairwise_distance_per_window",
                     "zero_median_fallback": "median_positive_distance_or_one",
                     "statistic": "sum_u_centered_hsic", "calibration": "independent_rademacher_weights_per_lag"},
        "bds": {"embedding_dimension": 2, "calibration": "joint_row_permutation_two_sided",
                "coordinate_correction": "bonferroni"}}
    for name, settings in required.items():
        if any(config[name].get(k) != v for k, v in settings.items()):
            raise ValueError(f"Unsupported {name} settings")
    if config["bds"]["distance_multiplier"] <= 0:
        raise ValueError("Invalid BDS distance multiplier")


def initialize_worker(config, normalized):
    global _CONTEXT, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _CONTEXT = config, normalized


def evaluate_task(task):
    hi, run = task
    config, normalized = _CONTEXT
    cutoff = config["cutoff_horizons"][hi]
    x = suffix(normalized[run], cutoff)
    lb = ljung_box_test(x, config["ljung_box"]["max_lag"], config["ljung_box"]["n_permutations"],
                       np.random.SeedSequence([config["ljung_box"]["seed"], 101, cutoff, run]))
    hsic = autohsic_test(x, config["autohsic"]["max_lag"], config["autohsic"]["n_bootstrap"],
                        np.random.SeedSequence([config["seed"], 110, cutoff, run]))
    bds = bds_test(x, config["bds"]["n_permutations"],
                   np.random.SeedSequence([config["seed"], 111, cutoff, run]), config["bds"]["distance_multiplier"])
    return hi, run, lb, hsic, bds


def analysis_signature(config, source_hash):
    settings = {k: v for k, v in config.items() if k not in ("workers", "notes", "references")}
    implementation = b"".join((HERE / name).read_bytes() for name in
                              ("independence_cutoff.py", "independence_comparison.py", "analysis.py"))
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode() + source_hash.encode() + implementation).hexdigest()


def main():
    config = json.loads((HERE / "setup_independence_cutoff.json").read_text(encoding="utf-8"))
    input_path = (HERE / config["input_result_file"]).resolve()
    output_path = (HERE / config["output_file"]).resolve()
    if input_path == output_path:
        raise ValueError("Analysis output must differ from saved trajectories")
    source_bytes = input_path.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source = pickle.loads(source_bytes)
    validate_config(config, source)
    signature = analysis_signature(config, source_hash)
    if output_path.exists():
        out = pickle.loads(output_path.read_bytes())
        if out["signature"] != signature:
            raise ValueError("Settings, input or implementation changed; use a new output filename")
        if out["complete"]:
            print(f"Already complete: {output_path}")
            return
    else:
        shape = (len(config["cutoff_horizons"]), len(source["costs"]))
        out = {"schema_version": 1, "configuration": config, "signature": signature,
               "source_sha256": source_hash, "complete": False, "seconds": 0.0,
               "cutoffs": np.array(config["cutoff_horizons"]),
               "window_lengths": source["costs"].shape[1] - np.array(config["cutoff_horizons"]),
               "final_iteration": source["costs"].shape[1], "n_trajectories": shape[1]}
        for name in ("lb_pvalue", "lb_statistic", "autohsic_pvalue", "autohsic_statistic", "autohsic_bandwidth", "bds_pvalue"):
            out[name] = np.full(shape, np.nan)
        for name in ("bds_statistic", "bds_coordinate_pvalue", "bds_epsilon"):
            out[name] = np.full((*shape, source["costs"].shape[2]), np.nan)
        save_checkpoint(output_path, out)
    normalized = source["costs"] / source["objective_scale"]
    done = np.isfinite(out["lb_pvalue"]) & np.isfinite(out["autohsic_pvalue"]) & np.isfinite(out["bds_pvalue"])
    tasks = [(int(hi), int(run)) for hi, run in zip(*np.where(~done))]
    print(f"Saved-data analysis only: {len(normalized)} trajectories, {len(out['cutoffs'])} cutoffs, "
          f"{config['workers']} workers; no training or simulation.", flush=True)
    last_save = time.perf_counter()
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=get_context("spawn"),
                             initializer=initialize_worker, initargs=(config, normalized)) as pool:
        with tqdm(total=done.size, initial=int(done.sum()), desc="Independence after cutoff",
                  ascii=True, mininterval=5) as progress:
            for hi, run, lb, hsic, bds in pool.map(evaluate_task, tasks, chunksize=5):
                for name, test, keys in (("lb", lb, ("statistic", "pvalue")),
                                        ("autohsic", hsic, ("statistic", "pvalue", "bandwidth")),
                                        ("bds", bds, ("statistic", "pvalue", "coordinate_pvalue", "epsilon"))):
                    for key in keys:
                        out[f"{name}_{key}"][hi, run] = test[key]
                progress.update()
                progress.set_postfix(cutoff=config["cutoff_horizons"][hi], refresh=False)
                now = time.perf_counter()
                if now-last_save >= 10 or run == len(normalized)-1:
                    out["seconds"] += now-last_save
                    save_checkpoint(output_path, out)
                    last_save = now
    for name in ("lb", "autohsic", "bds"):
        out[f"{name}_rejected"] = out[f"{name}_pvalue"] <= config["alpha"]
        out[f"{name}_rejection_rate"] = out[f"{name}_rejected"].mean(axis=1)
    if hashlib.sha256(input_path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Saved trajectory input changed during analysis")
    out["seconds"] += time.perf_counter()-last_save
    out["complete"] = True
    save_checkpoint(output_path, out)
    print("Cutoff | Remaining iterations | Ljung-Box | AutoHSIC | BDS", flush=True)
    for hi, cutoff in enumerate(out["cutoffs"]):
        print(f"{cutoff:6d} | {cutoff+1:3d}-{out['final_iteration']}              | "
              + " | ".join(f"{out[name+'_rejection_rate'][hi]:.2%}" for name in ("lb", "autohsic", "bds")), flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
