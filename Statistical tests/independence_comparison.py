"""Compare Ljung-Box, AutoHSIC and coordinate-wise BDS on SAVED trajectories.

Analysis only: no simulator imports or training. Checkpoints resume automatically.
AutoHSIC: Li, Xu & Zhou (2026), eq. (5), portmanteau sum and Algorithm 1.
BDS: m=2, two-sided permutation calibration of sqrt(n-1)*(C2-C1**2).
Its standard-error factor depends only on unordered observations, so omitting
that factor does not change permutation p-values when the factor is positive.
The two coordinate p-values are combined by Bonferroni, not averaged.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from multiprocessing import get_context
import os
from pathlib import Path
import pickle
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from analysis import ljung_box_test, null_reference_band

# Optional acceleration; correctness and reproducibility do not depend on numba.
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda function: function

HERE = Path(__file__).resolve().parent
_CONTEXT = None
_THREAD_LIMIT = None


def u_center(kernel):
    """Unbiased U-centering, with diagonals excluded from every sum."""
    k = np.array(kernel, dtype=float, copy=True)
    n = len(k)
    if k.shape != (n, n) or n < 4:
        raise ValueError("U-centering needs a square matrix of size >= 4")
    np.fill_diagonal(k, 0)
    rows = k.sum(axis=1)
    centered = k - (rows[:, None] + rows[None, :]) / (n - 2)
    centered += rows.sum() / ((n - 1) * (n - 2))
    np.fill_diagonal(centered, 0)
    return centered


def autohsic_test(x, lag, n_bootstrap, seed):
    """Gaussian AutoHSIC; independent Rademacher weights at each lag.

    The kernel bandwidth rule uses all unordered distances within the prefix.
    The same kernel is used for both sides of each lagged pair. No detrending,
    whitening, thinning or removal of initial observations is performed.
    """
    x = np.asarray(x, dtype=float)
    if (x.ndim != 2 or not np.isfinite(x).all() or not 0 < lag <= len(x) - 4
            or n_bootstrap < 1):
        raise ValueError("Invalid AutoHSIC series, lag or bootstrap count")
    distances = pdist(x)
    bandwidth = float(np.median(distances))
    if bandwidth <= 0:
        positive = distances[distances > 0]
        bandwidth = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-cdist(x, x, metric="sqeuclidean") / (2 * bandwidth**2))
    rng = np.random.default_rng(seed)
    observed = 0.0
    bootstrapped = np.zeros(n_bootstrap)
    for h in range(1, lag + 1):
        n = len(x) - h
        product = u_center(kernel[h:, h:]) * u_center(kernel[:-h, :-h])
        product /= n * (n - 3)
        observed += product.sum()
        for start in range(0, n_bootstrap, 256):
            batch = min(256, n_bootstrap - start)
            # Fresh independent draws for every lag, as in Algorithm 1.
            weights = 2.0 * rng.integers(0, 2, size=(batch, n)) - 1.0
            bootstrapped[start:start + batch] += np.sum((weights @ product) * weights, axis=1)
    tolerance = 1e-12 * max(1.0, abs(observed))
    pvalue = (1 + np.count_nonzero(bootstrapped >= observed - tolerance)) / (n_bootstrap + 1)
    return {"statistic": float(observed), "pvalue": float(pvalue), "bandwidth": bandwidth}


@njit(cache=False)
def _bds_effects(indicators, permutations):
    """m=2 correlation integrals for each coordinate and ordering.

    C1 uses the final n-1 entries, matching the standard BDS implementation.
    C2 counts pairs close at both times of the two-dimensional embedding.
    """
    d, n, _ = indicators.shape
    answer = np.zeros((len(permutations), d))
    pairs = (n - 1) * (n - 2) / 2
    for b in range(len(permutations)):
        order = permutations[b]
        for c in range(d):
            c1 = 0
            c2 = 0
            for i in range(1, n):
                for j in range(i + 1, n):
                    close = indicators[c, order[i], order[j]]
                    c1 += close
                    c2 += close and indicators[c, order[i - 1], order[j - 1]]
            answer[b, c] = np.sqrt(n - 1) * (c2 / pairs - (c1 / pairs)**2)
    return answer


def bds_test(x, n_permutations, seed, distance_multiplier=1.5):
    """One trajectory decision from two coordinate-wise BDS permutation tests.

    Whole vectors are shuffled jointly; epsilon stays fixed under permutation.
    Constant coordinates receive p=1 (they contain no serial variation).
    """
    x = np.asarray(x, float)
    if (x.ndim != 2 or len(x) < 4 or not np.isfinite(x).all()
            or n_permutations < 1 or distance_multiplier <= 0):
        raise ValueError("Invalid BDS input")
    epsilon = distance_multiplier * x.std(axis=0, ddof=1)
    indicators = np.ascontiguousarray(np.moveaxis(
        np.abs(x[:, None, :] - x[None, :, :]) < epsilon, -1, 0))
    identity = np.arange(len(x))[None, :]
    observed = _bds_effects(indicators, identity)[0]
    rng = np.random.default_rng(seed)
    exceedances = np.zeros(x.shape[1], int)
    for start in range(0, n_permutations, 128):
        batch = min(128, n_permutations - start)
        indices = rng.permuted(np.tile(identity, (batch, 1)), axis=1)
        stats = _bds_effects(indicators, indices)
        exceedances += np.count_nonzero(np.abs(stats) >= np.abs(observed) - 1e-12, axis=0)
    pvalues = (1 + exceedances) / (1 + n_permutations)
    return {"statistic": observed, "coordinate_pvalue": pvalues,
            "pvalue": min(1.0, len(pvalues) * float(pvalues.min())), "epsilon": epsilon}


def save_checkpoint(path, result):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(config, result):
    if not np.asarray(result["completed_runs"]).all() or not np.isfinite(result["costs"]).all():
        raise ValueError("All saved trajectories must be complete and finite")
    horizons = config["horizons"]
    if (not horizons or horizons != sorted(set(horizons))
            or any(type(p) is not int or not 4 <= p <= result["costs"].shape[1] for p in horizons)):
        raise ValueError("Invalid horizons")
    if not 0 < config["alpha"] < 1 or not 0 < config["confidence_level"] < 1:
        raise ValueError("Invalid significance/confidence")
    if config["include_initial_iterations"] is not True or config["scaling"] != "saved_objective_scale":
        raise ValueError("Retain all initial observations and the saved normalization")
    for name, key in (("ljung_box", "n_permutations"), ("autohsic", "n_bootstrap"), ("bds", "n_permutations")):
        count = config[name][key]
        factor = result["costs"].shape[2] if name == "bds" else 1
        if type(count) is not int or count < 1 or factor / (count + 1) > config["alpha"]:
            raise ValueError(f"Insufficient resamples for {name} at this significance level")
    for name in ("ljung_box", "autohsic"):
        lag = config[name]["max_lag"]
        if type(lag) is not int or not 0 < lag <= min(horizons) - 4:
            raise ValueError("Invalid fixed maximum lag")
    required = {
        "ljung_box": {"calibration": "joint_row_permutation"},
        "autohsic": {"kernel": "gaussian", "bandwidth": "median_pairwise_distance_per_prefix",
                     "zero_median_fallback": "median_positive_distance_or_one",
                     "statistic": "sum_u_centered_hsic", "calibration": "independent_rademacher_weights_per_lag"},
        "bds": {"embedding_dimension": 2, "calibration": "joint_row_permutation_two_sided",
                "coordinate_correction": "bonferroni"},
    }
    for name, settings in required.items():
        if any(config[name].get(key) != value for key, value in settings.items()):
            raise ValueError(f"Unsupported {name} specification")
    if config["bds"]["distance_multiplier"] <= 0 or type(config["workers"]) is not int or config["workers"] < 1:
        raise ValueError("Invalid distance multiplier or worker count")


def initialize_worker(config, normalized):
    global _CONTEXT, _THREAD_LIMIT
    _THREAD_LIMIT = threadpool_limits(limits=1)
    _CONTEXT = config, normalized


def evaluate_task(task):
    hi, r, need_lb = task
    config, normalized = _CONTEXT
    p = config["horizons"][hi]
    x = normalized[r, :p]
    hsic = autohsic_test(x, config["autohsic"]["max_lag"], config["autohsic"]["n_bootstrap"],
                        np.random.SeedSequence([config["seed"], 10, p, r]))
    bds = bds_test(x, config["bds"]["n_permutations"],
                   np.random.SeedSequence([config["seed"], 11, p, r]), config["bds"]["distance_multiplier"])
    lb = None
    if need_lb:
        lb = ljung_box_test(x, config["ljung_box"]["max_lag"], config["ljung_box"]["n_permutations"],
                            np.random.SeedSequence([config["ljung_box"]["seed"], 1, p, r]))
    return hi, r, hsic, bds, lb


def prepare_output(config, source, signature):
    shape = (len(config["horizons"]), len(source["costs"]))
    out = {"schema_version": 1, "signature": signature, "configuration": config,
           "complete": False, "horizons": np.array(config["horizons"]),
           "n_trajectories": shape[1], "source_generation_signature": source["generation_signature"],
           "source_setup": source["setup"], "seconds": 0.0, "lb_reused": False}
    for name in ("lb_pvalue", "lb_statistic", "autohsic_pvalue", "autohsic_statistic", "autohsic_bandwidth", "bds_pvalue"):
        out[name] = np.full(shape, np.nan)
    for name in ("bds_statistic", "bds_coordinate_pvalue", "bds_epsilon"):
        out[name] = np.full((*shape, source["costs"].shape[2]), np.nan)
    saved = source.get("analysis") or {}
    old = saved.get("configuration", {})
    wanted = config["ljung_box"]
    if (saved.get("complete") and old.get("seed") == wanted["seed"]
            and all(old.get("ljung_box", {}).get(k) == wanted[k]
                    for k in ("max_lag", "n_permutations", "calibration"))
            and old.get("include_initial_iterations") is True
            and np.array_equal(saved["horizons"], out["horizons"])):
        out["lb_pvalue"] = np.array(saved["lb_pvalue"], copy=True)
        out["lb_statistic"] = np.array(saved["lb_statistic"], copy=True)
        out["lb_reused"] = True
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=HERE / "setup_independence.json")
    args = parser.parse_args()
    setup_path = args.setup.resolve()
    config = json.loads(setup_path.read_text(encoding="utf-8"))
    input_path = (setup_path.parent / config["input_result_file"]).resolve()
    output_path = (setup_path.parent / config["output_file"]).resolve()
    if input_path == output_path:
        raise ValueError("Output must differ from the saved simulation input")
    source_hash = file_sha256(input_path)
    with input_path.open("rb") as stream:
        source = pickle.load(stream)
    validate_config(config, source)
    signing_config = {k: v for k, v in config.items() if k not in ("workers", "notes", "references")}
    signature = hashlib.sha256(json.dumps(signing_config, sort_keys=True).encode()
                               + source_hash.encode() + Path(__file__).read_bytes()
                               + (HERE / "analysis.py").read_bytes()).hexdigest()
    if output_path.exists():
        with output_path.open("rb") as stream:
            out = pickle.load(stream)
        if out["signature"] != signature:
            raise ValueError("Analysis settings, source or input changed; choose a new output_file")
    else:
        out = prepare_output(config, source, signature)
        out["source_sha256"] = source_hash
        save_checkpoint(output_path, out)
    if out["complete"]:
        print(f"Comparison already complete: {output_path}", flush=True)
        return
    normalized = source["costs"] / source["objective_scale"]
    done = np.isfinite(out["autohsic_pvalue"]) & np.isfinite(out["bds_pvalue"]) & np.isfinite(out["lb_pvalue"])
    tasks = [(int(hi), int(r), not np.isfinite(out["lb_pvalue"][hi, r])) for hi, r in zip(*np.where(~done))]
    print(f"Analysis only: {len(normalized)} saved trajectories; Ljung-Box reused: {out['lb_reused']}", flush=True)
    start = time.perf_counter()
    last_save = start
    with ProcessPoolExecutor(max_workers=config["workers"], mp_context=get_context("spawn"),
                             initializer=initialize_worker, initargs=(config, normalized)) as pool:
        with tqdm(total=done.size, initial=int(done.sum()), desc="Independence comparison", ascii=True, mininterval=5) as progress:
            for hi, r, hsic, bds, lb in pool.map(evaluate_task, tasks, chunksize=5):
                for key in ("pvalue", "statistic", "bandwidth"):
                    out[f"autohsic_{key}"][hi, r] = hsic[key]
                for key in ("pvalue", "statistic", "coordinate_pvalue", "epsilon"):
                    out[f"bds_{key}"][hi, r] = bds[key]
                if lb is not None:
                    for key in ("pvalue", "statistic"):
                        out[f"lb_{key}"][hi, r] = lb[key]
                progress.update()
                progress.set_postfix(horizon=config["horizons"][hi], refresh=False)
                now = time.perf_counter()
                if now - last_save >= 10 or r == len(normalized) - 1:
                    out["seconds"] += now - last_save
                    save_checkpoint(output_path, out)
                    last_save = now
    for name in ("lb", "autohsic", "bds"):
        out[f"{name}_rejected"] = out[f"{name}_pvalue"] <= config["alpha"]
        out[f"{name}_rejection_rate"] = out[f"{name}_rejected"].mean(axis=1)
    out["nominal_binomial_reference_band"] = null_reference_band(len(normalized), config["alpha"], config["confidence_level"])
    out["complete"] = True
    out["seconds"] += time.perf_counter() - last_save
    save_checkpoint(output_path, out)
    print("Horizon | Ljung-Box | AutoHSIC | BDS", flush=True)
    for i, p in enumerate(config["horizons"]):
        print(f"{p:7d} | " + " | ".join(f"{out[name + '_rejection_rate'][i]:8.2%}" for name in ("lb", "autohsic", "bds")), flush=True)
    print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
