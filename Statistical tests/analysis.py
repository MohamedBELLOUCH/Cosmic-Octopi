"""Multivariate portmanteau and K-sample energy diagnostics.

Ljung-Box: n(n+2) sum_h tr(C_h C_0^-1 C_h.T C_0^-1)/(n-h).
Energy: DISCO between-sample component, with Euclidean distances (exponent 1).
Both use joint-vector permutations, never separate permutations of coordinates.
Energy permutations require exchangeability within each arm under the joint iid
null. They are not a dependence-robust standalone test of marginal stationarity.
"""

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import beta, binom, chi2
from tqdm import tqdm


def _lb_statistics(whitened, lag):
    """Vectorized statistic for arrays (permutations, time, dimensions)."""
    n = whitened.shape[1]
    statistic = np.zeros(len(whitened))
    for h in range(1, lag + 1):
        covariance = np.einsum("bti,btj->bij", whitened[:, h:], whitened[:, :-h]) / n
        statistic += np.sum(covariance ** 2, axis=(1, 2)) / (n - h)
    return n * (n + 2) * statistic


def ljung_box_test(x, lag, n_permutations, seed):
    """Fixed-lag multivariate Ljung-Box with iid-null permutation calibration.

Covariance whitening makes the statistic invariant to nonsingular linear
rescaling. All cross-lag coordinate relationships are included. No AR model is
fitted and no arm means, time trend, or initialization samples are removed.
"""
    x = np.asarray(x, float)
    if x.ndim != 2 or not np.isfinite(x).all() or not 0 < lag < len(x):
        raise ValueError("Expected a finite multivariate series longer than the lag")
    if n_permutations < 1:
        raise ValueError("At least one permutation is required")
    centered = x - x.mean(axis=0)
    covariance = centered.T @ centered / len(x)
    values, vectors = np.linalg.eigh(covariance)
    if values[-1] <= 0 or values[0] <= values[-1] * 1e-12:
        raise ValueError("Singular covariance: multivariate Ljung-Box is not identifiable")
    whitened = centered @ (vectors / np.sqrt(values))
    observed = float(_lb_statistics(whitened[None], lag)[0])
    rng = np.random.default_rng(seed)
    exceedances = 0
    tolerance = 1e-12 * max(1.0, abs(observed))
    for start in range(0, n_permutations, 64):
        batch = min(64, n_permutations - start)
        indices = rng.permuted(np.tile(np.arange(len(x)), (batch, 1)), axis=1)
        permuted = _lb_statistics(whitened[indices], lag)
        exceedances += int(np.count_nonzero(permuted >= observed - tolerance))
    return {
        "statistic": observed,
        "pvalue": (1 + exceedances) / (1 + n_permutations),
        "chi2_pvalue": float(chi2.sf(observed, lag * x.shape[1] ** 2)),
        "degrees_of_freedom": lag * x.shape[1] ** 2,
    }


def holm_adjust(pvalues):
    p = np.asarray(pvalues, float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Holm correction requires finite p-values in [0,1]")
    order = np.argsort(p)
    adjusted = np.minimum(1.0, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def binomial_interval(rejections, n, confidence=0.95):
    """Clopper-Pearson CI for the observed rejection probability."""
    tail = (1 - confidence) / 2
    lower = 0.0 if rejections == 0 else beta.ppf(tail, rejections, n - rejections + 1)
    upper = 1.0 if rejections == n else beta.ppf(1 - tail, rejections + 1, n - rejections)
    return np.array([lower, upper], float)


def null_reference_band(n, alpha=0.05, confidence=0.95):
    """Pointwise central binomial range under the nominal null, not an observed CI."""
    tail = (1 - confidence) / 2
    return binom.ppf([tail, 1 - tail], n, alpha) / n


def _energy_layout(sizes):
    sizes = np.asarray(sizes, int)
    if len(sizes) < 2 or np.any(sizes < 1):
        raise ValueError("Energy test needs at least two nonempty groups")
    offsets = np.r_[0, np.cumsum(sizes)]
    positions = np.arange(sizes.max())[None, :]
    mask = positions < sizes[:, None]
    slots = np.minimum(offsets[:-1, None] + positions, sizes.sum() - 1)
    weights = (mask[:, :, None] & mask[:, None, :]) / sizes[:, None, None]
    return slots, weights


def _within_dispersion(distance, indices, slots, weights):
    groups = indices[slots]
    return float(np.sum(distance[groups[:, :, None], groups[:, None, :]] * weights))


def energy_statistic(distance, sizes):
    """DISCO B = 1/2 [sum_ij D_ij/N - sum_g sum_(i,j in g) D_ij/n_g]."""
    distance = np.asarray(distance, float)
    slots, weights = _energy_layout(sizes)
    n = sum(sizes)
    if distance.shape != (n, n) or not np.isfinite(distance).all():
        raise ValueError("Distance matrix and group sizes do not match")
    within = _within_dispersion(distance, np.arange(n), slots, weights)
    return max(0.0, 0.5 * (distance.sum() / n - within))


def energy_test(distance, sizes, n_permutations, seed):
    """K-sample energy permutation test, preserving each iteration's sample size."""
    sizes = np.asarray(sizes, int)
    n = int(sizes.sum())
    if distance.shape != (n, n) or n_permutations < 1:
        raise ValueError("Invalid energy test inputs")
    slots, weights = _energy_layout(sizes)
    indices = np.arange(n)
    within = _within_dispersion(distance, indices, slots, weights)
    total = float(distance.sum()) / n
    observed = max(0.0, 0.5 * (total - within))
    tolerance = 1e-12 * max(1.0, abs(total), abs(within))
    rng = np.random.default_rng(seed)
    exceedances = 0
    for _ in range(n_permutations):
        # The pooled total distance is invariant under permutations. Larger
        # between-group energy is equivalent to smaller within-group dispersion.
        permuted_within = _within_dispersion(distance, rng.permutation(indices), slots, weights)
        exceedances += int(permuted_within <= within + tolerance)
    return {"statistic": observed, "pvalue": (1 + exceedances) / (1 + n_permutations)}


def summarize_levels(analysis, levels, confidence=0.95):
    """Apply each significance threshold to the same tests, with Holm per horizon."""
    levels = np.asarray(levels, float)
    if (levels.ndim != 1 or not len(levels) or not np.isfinite(levels).all()
            or np.any((levels <= 0) | (levels >= 1))
            or len(np.unique(levels)) != len(levels)):
        raise ValueError("Significance levels must be distinct numbers in (0,1)")
    adjusted = np.array([holm_adjust(p) for p in analysis["energy_pvalue"]])
    n_runs = analysis["lb_pvalue"].shape[1]
    summaries = []
    for alpha in levels:
        lb_rejected = analysis["lb_pvalue"] <= alpha
        counts = lb_rejected.sum(axis=1)
        energy_rejected = adjusted <= alpha
        summaries.append({
            "alpha": float(alpha),
            "lb_rejected": lb_rejected,
            "lb_rejection_count": counts,
            "lb_rejection_rate": lb_rejected.mean(axis=1),
            "lb_rate_confidence_interval": np.array([
                binomial_interval(int(n), n_runs, confidence) for n in counts]),
            "lb_null_reference_band": null_reference_band(n_runs, alpha, confidence),
            "energy_rejected": energy_rejected,
            "energy_rejection_count": energy_rejected.sum(axis=1),
            "energy_rejection_rate": energy_rejected.mean(axis=1),
        })
    return summaries


def analyze(setup, result, path, save, reset=False):
    config = setup["statistics"]
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + Path(__file__).read_bytes()).hexdigest()
    if result.get("analysis") is not None and result["analysis"]["signature"] != signature and not reset:
        raise ValueError("Statistical settings/source changed. Use --reanalyze; trajectories are retained.")
    horizons = np.asarray(config["horizons"], int)
    n_runs, p_max = result["arm_indices"].shape
    n_arms = len(result["arm_values"])
    if result.get("analysis") is None or reset:
        result["analysis"] = {
            "signature": signature, "configuration": config, "complete": False,
            "horizons": horizons,
            "lb_statistic": np.full((len(horizons), n_runs), np.nan),
            "lb_pvalue": np.full((len(horizons), n_runs), np.nan),
            "lb_chi2_pvalue": np.full((len(horizons), n_runs), np.nan),
            "energy_statistic": np.full((len(horizons), n_arms), np.nan),
            "energy_pvalue": np.full((len(horizons), n_arms), np.nan),
            "arm_counts_by_iteration": np.array([(result["arm_indices"] == a).sum(axis=0)
                                                   for a in range(n_arms)]).T,
            "seconds": 0.0,
        }
        save(path, result)
    analysis = result["analysis"]
    if analysis["complete"]:
        print("Statistical analysis is already complete.", flush=True)
        return
    normalized = result["costs"] / result["objective_scale"]
    if not np.isfinite(normalized).all():
        raise ValueError("Analysis requires all finite cost trajectories")
    min_size = int(analysis["arm_counts_by_iteration"].min())
    if min_size < config["energy"]["min_group_size"]:
        raise ValueError(f"Smallest arm/iteration group has {min_size} observations; increase the number of runs")
    start = time.perf_counter()
    initial = int(np.isfinite(analysis["lb_pvalue"]).sum())
    with tqdm(total=len(horizons) * n_runs, initial=initial, desc="Ljung-Box tests", ascii=True) as progress:
        for hi, p in enumerate(horizons):
            for r in range(n_runs):
                if np.isfinite(analysis["lb_pvalue"][hi, r]):
                    continue
                seed = np.random.SeedSequence([config["seed"], 1, int(p), r])
                test = ljung_box_test(normalized[r, :p], config["ljung_box"]["max_lag"],
                                     config["ljung_box"]["n_permutations"], seed)
                analysis["lb_statistic"][hi, r] = test["statistic"]
                analysis["lb_pvalue"][hi, r] = test["pvalue"]
                analysis["lb_chi2_pvalue"][hi, r] = test["chi2_pvalue"]
                progress.update()
            save(path, result)
    initial = int(np.isfinite(analysis["energy_pvalue"]).sum())
    with tqdm(total=len(horizons) * n_arms, initial=initial, desc="K-sample energy tests", ascii=True) as progress:
        for arm in range(n_arms):
            if np.isfinite(analysis["energy_pvalue"][:, arm]).all():
                continue
            # Time-major order makes each group contiguous and all horizons
            # prefixes of one distance matrix. Every observation is retained.
            time_indices, run_indices = np.nonzero(result["arm_indices"].T == arm)
            points = normalized[run_indices, time_indices]
            distance = cdist(points, points, metric="euclidean")
            for hi, p in enumerate(horizons):
                if np.isfinite(analysis["energy_pvalue"][hi, arm]):
                    continue
                sizes = analysis["arm_counts_by_iteration"][:p, arm]
                n = int(sizes.sum())
                seed = np.random.SeedSequence([config["seed"], 2, int(p), arm])
                test = energy_test(distance[:n, :n], sizes, config["energy"]["n_permutations"], seed)
                analysis["energy_statistic"][hi, arm] = test["statistic"]
                analysis["energy_pvalue"][hi, arm] = test["pvalue"]
                save(path, result)
                progress.set_postfix(arm=f"{result['arm_values'][arm]:g}", horizon=p, refresh=False)
                progress.update()
            del distance
    alpha = config["alpha"]
    analysis["lb_rejected"] = analysis["lb_pvalue"] <= alpha
    analysis["lb_rejection_count"] = analysis["lb_rejected"].sum(axis=1)
    analysis["lb_rejection_rate"] = analysis["lb_rejected"].mean(axis=1)
    analysis["lb_rate_confidence_interval"] = np.array([
        binomial_interval(int(n), n_runs, config["confidence_level"])
        for n in analysis["lb_rejection_count"]])
    analysis["lb_null_reference_band"] = null_reference_band(n_runs, alpha, config["confidence_level"])
    analysis["energy_pvalue_holm"] = np.array([holm_adjust(p) for p in analysis["energy_pvalue"]])
    analysis["energy_rejected"] = analysis["energy_pvalue_holm"] <= alpha
    analysis["energy_rejection_count"] = analysis["energy_rejected"].sum(axis=1)
    analysis["energy_rejection_rate"] = analysis["energy_rejected"].mean(axis=1)
    analysis["by_significance_level"] = summarize_levels(
        analysis, config.get("significance_levels", [alpha]), config["confidence_level"])
    analysis["seconds"] += time.perf_counter() - start
    analysis["complete"] = True
    save(path, result)
    print("Horizon | Ljung-Box rejections | Energy rejections (Holm)", flush=True)
    for i, p in enumerate(horizons):
        print(f"{p:7d} | {analysis['lb_rejection_count'][i]:3d}/{n_runs}"
              f" | {analysis['energy_rejection_count'][i]}/{n_arms}", flush=True)
