"""Three mean-equality diagnostics on saved suffixes; never runs simulations.

Pillai MANOVA, Welch-James (Johansen F approximation), and parametric-bootstrap
MATS. These are independent-group reference calibrations, not cluster-robust
tests. Welch-James uses the estimable contrast space when sample covariance is
singular; that rank-adjusted extension is reported explicitly in the results.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.linalg import block_diag, helmert
from scipy.stats import f
from threadpoolctl import threadpool_limits
from tqdm import tqdm


TEST_NAMES = ["MANOVA", "Welch-James", "MATS"]


def group_summary(values, mask):
    values, mask = np.asarray(values, float), np.asarray(mask, bool)
    if values.shape[:2] != mask.shape or not np.isfinite(values).all():
        raise ValueError("Invalid saved observations")
    varying = np.ptp(values[mask], axis=0) > 0
    if not varying.any():
        raise ValueError("All coordinates are constant; no variance-based test is available")
    values = values[..., varying]
    counts = mask.sum(axis=0)
    if len(counts) < 2 or np.any(counts < 2):
        raise ValueError("At least two observations per iteration are required")
    means = (values * mask[..., None]).sum(axis=0) / counts[:, None]
    residuals = (values - means) * mask[..., None]
    cov = np.einsum("rti,rtj->tij", residuals, residuals) / (counts - 1)[:, None, None]
    return means, cov, counts, varying


def pillai_test(means, cov, counts):
    n, g, d = counts.sum(), len(counts), means.shape[1]
    overall = np.average(means, axis=0, weights=counts)
    delta = means - overall
    h = (delta * counts[:, None]).T @ delta
    e = (cov * (counts - 1)[:, None, None]).sum(axis=0)
    if np.linalg.matrix_rank(e) < d:
        raise ValueError("Within-group covariance is singular after constant-coordinate removal")
    v = float(np.trace(np.linalg.solve(e + h, h)))
    q, error_df = g - 1, n - g
    s = min(d, q)
    m, nu = (abs(d - q) - 1) / 2, (error_df - d - 1) / 2
    df1, df2 = s * (2*m + s + 1), s * (2*nu + s + 1)
    stat = np.inf if v >= s else (df2 / df1) * max(v, 0) / (s - v)
    return v, float(f.sf(stat, df1, df2))


def welch_james_test(means, cov, counts, rank_tolerance=1e-12):
    """Johansen's Welch-James F approximation for a vector mean contrast.

For singular empirical contrast covariance, project onto its positive-eigenvalue
space and use its rank as numerator df. This is an exploratory rank-adjusted
extension, not the ordinary full-rank reference theorem. A nonzero observed
contrast in a zero-variance direction raises rather than being silently ignored.
"""
    g, d = means.shape
    contrast = np.kron(helmert(g, full=False), np.eye(d))
    blocks = cov / counts[:, None, None]
    contrast_cov = contrast @ block_diag(*blocks) @ contrast.T
    eig, basis = np.linalg.eigh(contrast_cov)
    keep = eig > max(float(eig[-1]), 0) * rank_tolerance
    q = int(keep.sum())
    if not q:
        raise ValueError("No estimable mean contrasts")
    delta = contrast @ means.ravel()
    coordinates = basis.T @ delta
    if np.linalg.norm(coordinates[~keep]) > 1e-8 * max(np.linalg.norm(delta), 1):
        raise ValueError("Nonzero mean contrast has zero estimated variance")
    inverse = (basis[:, keep] / eig[keep]) @ basis[:, keep].T
    statistic = float(np.sum(coordinates[keep]**2 / eig[keep]))
    a = 0.0
    for j in range(g):
        cj = contrast[:, j*d:(j+1)*d]
        bj = (cj.T @ inverse @ cj) @ blocks[j]
        a += (np.trace(bj @ bj) + np.trace(bj)**2) / (2 * (counts[j] - 1))
    c = q + 2*a - 6*a/(q + 2)
    df2 = q * (q + 2) / (3*a)
    return statistic, float(f.sf(statistic / c, q, df2)), q, contrast.shape[0]


def mats_statistic(means, mean_variances):
    """One-way MATS, equivalent to mu'T(T diag(s^2/n) T)^+T mu.

Supports leading bootstrap dimensions. In each response dimension this is a
weighted residual sum of squares. Zero-variance groups fix the common mean;
their means must agree, otherwise a zero-variance contrast is untestable.
"""
    means, mean_variances = np.broadcast_arrays(means, mean_variances)
    positive = mean_variances > 0
    weights = np.divide(1, mean_variances, out=np.zeros_like(mean_variances), where=positive)
    zeros = ~positive
    nzero = zeros.sum(axis=-2)
    weighted_mean = np.divide((weights * means).sum(axis=-2), weights.sum(axis=-2),
                              out=np.zeros_like(nzero, dtype=float), where=weights.sum(axis=-2) > 0)
    zero_mean = (means * zeros).sum(axis=-2) / np.maximum(nzero, 1)
    center = np.where(nzero > 0, zero_mean, weighted_mean)
    if np.any(np.abs((means - zero_mean[..., None, :]) * zeros) > 1e-9):
        raise ValueError("Different constant group means cannot be calibrated by MATS")
    return (weights * (means - center[..., None, :])**2).sum(axis=(-2, -1))


def mats_test(means, cov, counts, draws, seed, batch_size=256):
    """Gaussian parametric bootstrap with re-estimated variances in every draw.

Gaussian sample means and covariances are independent. Drawing means normally
and covariances via the Bartlett Wishart construction exactly reproduces their
joint sufficient-statistic distribution, including singular covariance roots.
Both cost coordinates are generated jointly, preserving within-vector covariance.
"""
    g, d = means.shape
    observed = float(mats_statistic(means, np.diagonal(cov, axis1=-2, axis2=-1) / counts[:, None]))
    eig, vectors = np.linalg.eigh(cov)
    roots = vectors * np.sqrt(np.maximum(eig, 0))[:, None, :]
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, draws, batch_size):
        size = min(batch_size, draws - start)
        z = rng.standard_normal((size, g, d))
        boot_means = np.einsum("gij,bgj->bgi", roots, z) / np.sqrt(counts)[None, :, None]
        bartlett = np.zeros((size, g, d, d))
        for j in range(d):
            bartlett[..., j, j] = np.sqrt(rng.chisquare(counts - 1 - j, size=(size, g)))
            for k in range(j):
                bartlett[..., j, k] = rng.standard_normal((size, g))
        factor = np.einsum("gij,bgjk->bgik", roots, bartlett)
        variances = (factor**2).sum(axis=-1) / (counts - 1)[None, :, None]
        statistics = mats_statistic(boot_means, variances / counts[None, :, None])
        exceedances += np.count_nonzero(statistics >= observed - 1e-12)
    return observed, (exceedances + 1) / (draws + 1)


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    return adjusted


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def main():
    folder = Path(__file__).resolve().parent
    config = json.loads((folder / "setup_mean_comparison.json").read_text(encoding="utf-8"))
    source = folder / config["input_result_file"]
    target = folder / config["output_file"]
    if source.resolve() == target.resolve():
        raise ValueError("Analysis output must be separate from saved trajectories")
    source_bytes = source.read_bytes()
    data = pickle.loads(source_bytes)
    if not data["completed_runs"].all() or data["setup"]["simulated_rewards"] is not True:
        raise ValueError("Complete simulated trajectories are required")
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode() + source_hash.encode() + Path(__file__).read_bytes()).hexdigest()
    x = data["costs"] / data["objective_scale"]
    cutoffs = np.asarray(config["cutoff_horizons"], dtype=int)
    n_arms = len(data["arm_values"])
    if np.any(cutoffs < 0) or np.any(cutoffs >= x.shape[1]-1):
        raise ValueError("Cutoffs must leave at least two iteration groups")
    if target.exists():
        result = pickle.loads(target.read_bytes())
        if result["signature"] != signature:
            raise ValueError("Analysis configuration or code changed; choose a new output filename")
        if result["complete"]:
            print(f"Already complete: {target}")
            return
    else:
        shape = (3, len(cutoffs), n_arms)
        result = {"configuration": config, "signature": signature, "source_sha256": source_hash,
                  "complete": False, "test_names": TEST_NAMES, "cutoffs": cutoffs,
                  "final_iteration": x.shape[1], "n_trajectories": len(x), "arm_values": data["arm_values"],
                  "statistics": np.full(shape, np.nan), "pvalues": np.full(shape, np.nan),
                  "response_dimensions": np.zeros(shape[1:], int),
                  "welch_james_rank": np.zeros(shape[1:], int),
                  "welch_james_full_rank": np.zeros(shape[1:], int)}
    with threadpool_limits(limits=2), tqdm(total=len(cutoffs)*n_arms,
            initial=int(np.isfinite(result["pvalues"]).all(axis=0).sum()), desc="Mean tests after cutoff", ascii=True) as progress:
        for hi, cutoff in enumerate(cutoffs):
            for arm in range(n_arms):
                if np.isfinite(result["pvalues"][:, hi, arm]).all():
                    continue
                means, cov, counts, varying = group_summary(x[:, cutoff:], data["arm_indices"][:, cutoff:] == arm)
                pillai = pillai_test(means, cov, counts)
                welch = welch_james_test(means, cov, counts, config["covariance_rank_tolerance"])
                mats = mats_test(means, cov, counts, config["mats_bootstrap_draws"],
                    np.random.SeedSequence([config["seed"], int(cutoff), arm]), config["bootstrap_batch_size"])
                for i, test in enumerate([pillai, welch, mats]):
                    result["statistics"][i, hi, arm], result["pvalues"][i, hi, arm] = test[:2]
                result["response_dimensions"][hi, arm] = varying.sum()
                result["welch_james_rank"][hi, arm], result["welch_james_full_rank"][hi, arm] = welch[2:]
                save(target, result)
                progress.update()
    result["adjusted_pvalues"] = np.array([[holm(p) for p in test] for test in result["pvalues"]])
    result["rejection_percentage"] = 100 * (result["adjusted_pvalues"] <= config["alpha"]).mean(axis=-1)
    result["complete"] = True
    save(target, result)
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Source trajectories changed during analysis")
    print("Cutoff | MANOVA | Welch-James | MATS")
    for hi, cutoff in enumerate(cutoffs):
        print(f"{cutoff:6d} | " + " | ".join(f"{v:.1f}%" for v in result["rejection_percentage"][:, hi]))
    print(f"Saved {target}")


if __name__ == "__main__":
    main()
