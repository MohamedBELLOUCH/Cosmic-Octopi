"""Held-out pooled-quantile coverage tests; no simulation or training.

One uniformly sampled time per evaluation trajectory/horizon produces independent
cost vectors. Exact binomial tests assess coverage >= minimum_coverage per arm.
Calibration uses separate complete trajectories. Horizons are pointwise.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import binomtest
from tqdm import tqdm

from analysis import holm_adjust, null_reference_band, binomial_interval


def partition_runs(n_runs, fraction, seed):
    if not 0 < fraction < 1 or n_runs < 4:
        raise ValueError("At least four trajectories and a nontrivial split are required")
    permutation = np.random.default_rng(seed).permutation(n_runs)
    split = int(n_runs*fraction)
    if split < 2 or n_runs-split < 2:
        raise ValueError("Both calibration and evaluation need at least two trajectories")
    return permutation[:split], permutation[split:]


def coverage_test(successes, n, minimum, confidence):
    if n == 0:
        return {"coverage": np.nan, "pvalue": 1.0, "interval": np.array([0., 1.])}
    return {"coverage": successes/n,
            "pvalue": float(binomtest(successes, n, minimum, alternative="less").pvalue),
            "interval": binomial_interval(successes, n, confidence)}


def compute_coverage(costs, arms, config):
    """Return inference conditional on the independently calibrated rectangles."""
    costs, arms = np.asarray(costs, float), np.asarray(arms, int)
    if costs.shape != (*arms.shape, 2) or not np.isfinite(costs).all():
        raise ValueError("Expected finite bivariate costs matching arm histories")
    q = config["quantile_coverage"]
    horizons = np.asarray(config["horizons"], int)
    if max(horizons) > arms.shape[1]:
        raise ValueError("Requested horizon exceeds the saved trajectory length")
    calibration, evaluation = partition_runs(len(costs), q["calibration_fraction"], q["split_seed"])
    n_arms = int(arms.max())+1
    calibration_points = costs[calibration, :q["calibration_horizon"]].reshape(-1, 2)
    rectangles = np.array([np.quantile(calibration_points, [b["lower"], b["upper"]],
                                       axis=0, method="linear") for b in q["bands"]])
    # A single U per evaluation trajectory couples horizons without compromising
    # uniform selection within any horizon. No selection uses outcomes or arms.
    u = np.random.default_rng(q["evaluation_seed"]).random(len(evaluation))
    positions = np.floor(horizons[:, None]*u).astype(int)
    shape = (len(rectangles), len(horizons), n_arms)
    successes = np.zeros(shape, int)
    counts = np.zeros((len(horizons), n_arms), int)
    coverage = np.full(shape, np.nan)
    intervals = np.zeros((*shape, 2))
    pvalues = np.ones(shape)
    pooled_coverage = np.zeros((len(rectangles), len(horizons)))
    for hi, h in enumerate(tqdm(horizons, desc="Held-out quantile coverage", ascii=True)):
        selected = costs[evaluation, positions[hi]]
        labels = arms[evaluation, positions[hi]]
        counts[hi] = np.bincount(labels, minlength=n_arms)
        for bi, (lower, upper) in enumerate(rectangles):
            # Inclusive boundaries explicitly accommodate atoms such as zero cost.
            inside = ((selected >= lower) & (selected <= upper)).all(axis=1)
            pooled_coverage[bi, hi] = inside.mean()
            for arm in range(n_arms):
                k = int(np.count_nonzero(inside & (labels == arm)))
                successes[bi, hi, arm] = k
                test = coverage_test(k, int(counts[hi, arm]), q["minimum_coverage"],
                                     config["confidence_level"])
                coverage[bi, hi, arm] = test["coverage"]
                intervals[bi, hi, arm] = test["interval"]
                pvalues[bi, hi, arm] = test["pvalue"]
    adjusted = np.array([[holm_adjust(row) for row in band] for band in pvalues])
    rejected = adjusted <= q["alpha"]
    return {"calibration_run_indices": calibration, "evaluation_run_indices": evaluation,
            "evaluation_time_indices": positions, "rectangles_normalized": rectangles,
            "counts": counts, "successes": successes, "coverage": coverage,
            "coverage_intervals": intervals, "pooled_coverage": pooled_coverage,
            "pvalue": pvalues, "pvalue_holm": adjusted, "rejected": rejected,
            "rejection_count": rejected.sum(axis=2), "rejection_rate": rejected.mean(axis=2),
            "no_data": counts == 0}


def analyze_generated(setup, result, path, save, reset=False):
    """Checkpoint Ljung-Box tests, then evaluate held-out quantile coverage."""
    from analysis import ljung_box_test
    if not result["completed_runs"].all() or not np.isfinite(result["costs"]).all():
        raise ValueError("Generate all trajectories before statistical analysis")
    config = setup["statistics"]
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()
        + Path(__file__).read_bytes()
        + Path(__file__).with_name('analysis.py').read_bytes()).hexdigest()
    a = result.get("analysis")
    if a is not None and a["signature"] != signature and not reset:
        raise ValueError("Statistical settings/source changed. Use --reanalyze; trajectories are retained.")
    horizons = np.asarray(config["horizons"], int)
    n_runs = len(result["costs"])
    if a is None or reset:
        a = {"signature": signature, "configuration": config, "complete": False,
             "horizons": horizons,
             "lb_statistic": np.full((len(horizons), n_runs), np.nan),
             "lb_pvalue": np.full((len(horizons), n_runs), np.nan),
             "arm_counts_by_iteration": np.array([
                 (result["arm_indices"] == arm).sum(axis=0)
                 for arm in range(len(result["arm_values"]))]).T}
        result["analysis"] = a
        save(path, result)
    if a["complete"]:
        print("Statistical analysis is already complete.", flush=True)
        return
    normalized = result["costs"] / result["objective_scale"]
    with tqdm(total=len(horizons)*n_runs,
              initial=int(np.isfinite(a["lb_pvalue"]).sum()),
              desc="Ljung-Box tests", ascii=True) as progress:
        for hi, p in enumerate(horizons):
            for r in range(n_runs):
                if np.isfinite(a["lb_pvalue"][hi, r]):
                    continue
                test = ljung_box_test(normalized[r, :p], config["ljung_box"]["max_lag"],
                    config["ljung_box"]["n_permutations"],
                    np.random.SeedSequence([config["seed"], 1, int(p), r]))
                a["lb_pvalue"][hi, r] = test["pvalue"]
                a["lb_statistic"][hi, r] = test["statistic"]
                progress.update()
                if (r+1) % 20 == 0:
                    save(path, result)
            save(path, result)
    a["quantile_coverage"] = compute_coverage(normalized, result["arm_indices"], config)
    a["quantile_coverage"]["rectangles_original_units"] = (
        a["quantile_coverage"]["rectangles_normalized"] * result["objective_scale"])
    a["lb_by_significance_level"] = [{"alpha": alpha,
        "lb_rejection_rate": (a["lb_pvalue"] <= alpha).mean(axis=1),
        "lb_rejection_count": (a["lb_pvalue"] <= alpha).sum(axis=1),
        "lb_null_reference_band": null_reference_band(n_runs, alpha, config["confidence_level"])}
        for alpha in config["ljung_box"]["significance_levels"]]
    a["complete"] = True
    save(path, result)


def run_saved(setup, setup_path, reset=False):
    from script import save_result
    folder = Path(setup_path).parent
    config = setup["statistics"]
    q = config["quantile_coverage"]
    source = folder / q["input_result_file"]
    target = folder / setup["output_file"]
    if source.resolve() == target.resolve():
        raise ValueError("Original results must not be overwritten")
    raw = pickle.loads(source.read_bytes())
    if (raw["setup"].get("simulated_rewards") is not True or not raw["completed_runs"].all()
            or len(raw["costs"]) != setup["n_independent_runs"]
            or not np.array_equal(raw["arm_values"], np.linspace(*setup["arms"]["interval"], setup["arms"]["n_arms"]))):
        raise ValueError("Saved simulation does not match the completed simulated design")
    for key in ("system_dynamics_parameters", "Xi_matrices", "hyper_parameters", "seed",
                "gravity_mean", "ou_substeps", "policy_shapes", "reductionist"):
        if raw["setup"][key] != setup[key]:
            raise ValueError(f"Saved simulation setting differs: {key}")
    previous = raw["analysis"]
    if not previous["complete"] or not np.array_equal(previous["horizons"], config["horizons"]):
        raise ValueError("A completed Ljung-Box result with the same horizons is required")
    for key in ("max_lag", "n_permutations", "calibration"):
        if previous["configuration"]["ljung_box"][key] != config["ljung_box"][key]:
            raise ValueError("Saved Ljung-Box specification differs; cannot reuse its p-values")
    data_hash = hashlib.sha256(raw["costs"].tobytes() + raw["arm_indices"].tobytes()
                               + raw["objective_scale"].tobytes()
                               + previous["lb_pvalue"].tobytes()).hexdigest()
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + data_hash.encode()
                               + Path(__file__).read_bytes() + (folder/'analysis.py').read_bytes()).hexdigest()
    if target.exists() and not reset:
        existing = pickle.loads(target.read_bytes())
        if existing["analysis"]["signature"] == signature and existing["analysis"]["complete"]:
            print(f"Already complete: {target}")
            return
        raise ValueError("Settings/source changed; use --reanalyze to refresh coverage analysis only")
    result = {key: value for key, value in raw.items() if key != "analysis"}
    a = {"signature": signature, "configuration": config, "complete": False,
         "input_result_file": str(source), "input_data_sha256": data_hash,
         "horizons": previous["horizons"], "lb_pvalue": previous["lb_pvalue"].copy(),
         "lb_statistic": previous["lb_statistic"].copy(),
         "arm_counts_by_iteration": previous["arm_counts_by_iteration"].copy()}
    a["quantile_coverage"] = compute_coverage(raw["costs"]/raw["objective_scale"], raw["arm_indices"], config)
    a["quantile_coverage"]["rectangles_original_units"] = a["quantile_coverage"]["rectangles_normalized"] * raw["objective_scale"]
    a["lb_by_significance_level"] = [{"alpha": alpha,
        "lb_rejection_rate": (a["lb_pvalue"] <= alpha).mean(axis=1),
        "lb_rejection_count": (a["lb_pvalue"] <= alpha).sum(axis=1),
        "lb_null_reference_band": null_reference_band(len(raw["costs"]), alpha, config["confidence_level"])}
        for alpha in config["ljung_box"]["significance_levels"]]
    a["complete"] = True
    result["analysis"] = a
    save_result(target, result)
    print(f"Saved {target}. No simulated trajectories or Ljung-Box tests were rerun.")
