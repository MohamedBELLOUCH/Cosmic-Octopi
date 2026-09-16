"""Ljung-Box and MANOVA on saved simulated trajectories, without resimulation.

MANOVA uses Pillai's trace, calibrated by a trajectory-level wild residual
bootstrap. Conventional independent-observation F p-values are diagnostic only.
"""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.stats import f
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from analysis import ljung_box_test, holm_adjust, null_reference_band


def pillai(hypothesis, total):
    return np.trace(np.linalg.solve(total, hypothesis), axis1=-2, axis2=-1)


def manova_test(values, mask, n_bootstrap, seed):
    """Test equal bivariate means across time groups for one arm.

    Full-model residuals are centered within iteration, HC2 adjusted, then
    multiplied by one Rademacher draw per independent trajectory. The same
    multiplier applies to every retained observation and both coordinates.
    This is approximate wild-bootstrap MANOVA, not the classical exact F test.
    """
    values, mask = np.asarray(values, float), np.asarray(mask, bool)
    if values.shape != (*mask.shape, 2) or not np.isfinite(values).all():
        raise ValueError("Invalid MANOVA observations")
    counts = mask.sum(axis=0)
    if len(counts) < 2 or np.any(counts < 2) or n_bootstrap < 1:
        raise ValueError("MANOVA requires at least two observations per time group")
    means = (values * mask[..., None]).sum(axis=0) / counts[:, None]
    n = int(counts.sum())
    overall = (means * counts[:, None]).sum(axis=0) / n
    delta = means - overall
    hypothesis = (delta * counts[:, None]).T @ delta
    residual = (values - means) * mask[..., None]
    error = np.einsum("rti,rtj->ij", residual, residual)
    total = hypothesis + error
    eigenvalues = np.linalg.eigvalsh(total)
    if eigenvalues[-1] <= 0 or eigenvalues[0] <= eigenvalues[-1] * 1e-12:
        raise ValueError("Singular MANOVA response covariance")
    observed = float(pillai(hypothesis, total))
    # Conventional Pillai F approximation assumes independent observations.
    # Retain it only for comparison, never for plotted rejection decisions.
    d, q, v = 2, len(counts)-1, n-len(counts)
    s = min(d, q)
    m, nu = (abs(d-q)-1)/2, (v-d-1)/2
    df1, df2 = s*(2*m+s+1), s*(2*nu+s+1)
    f_stat = np.inf if observed >= s else df2/df1 * observed/(s-observed)
    classical_p = float(f.sf(f_stat, df1, df2))

    adjusted = residual / np.sqrt(1-1/counts)[None, :, None]
    gram = np.einsum("rti,rtj->ij", adjusted, adjusted)
    flattened = adjusted.reshape(len(values), -1)
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, n_bootstrap, 256):
        size = min(256, n_bootstrap-start)
        signs = rng.choice([-1.0, 1.0], size=(size, len(values)))
        sums = (signs @ flattened).reshape(size, len(counts), 2)
        total_sum = sums.sum(axis=1)
        centering = np.einsum("bi,bj->bij", total_sum, total_sum) / n
        h = np.einsum("bti,btj,t->bij", sums, sums, 1/counts) - centering
        t = gram[None] - centering
        bootstrap = pillai(h, t)
        exceedances += np.count_nonzero(bootstrap >= observed - 1e-12)
    return {"statistic": observed, "pvalue": (1+exceedances)/(1+n_bootstrap),
            "classical_f_pvalue": classical_p, "classical_f": f_stat,
            "df1": df1, "df2": df2, "group_means": means}


def run_saved(setup, setup_path, reset=False):
    from script import save_result
    config = setup["statistics"]
    folder = Path(setup_path).parent
    source = folder / config["input_result_file"]
    target = folder / setup["output_file"]
    if source.resolve() == target.resolve():
        raise ValueError("Saved input and new analysis output must be separate")
    raw = pickle.loads(source.read_bytes())
    if raw["setup"].get("simulated_rewards") is not True or not raw["completed_runs"].all():
        raise ValueError("Input must contain complete simulated trajectories")
    expected = np.linspace(*setup["arms"]["interval"], setup["arms"]["n_arms"])
    if (not np.array_equal(raw["arm_values"], expected)
            or len(raw["costs"]) != setup["n_independent_runs"]
            or raw["costs"].shape[1] < max(config["horizons"])):
        raise ValueError("Saved trajectory design does not match the setup")
    for key in ("system_dynamics_parameters", "Xi_matrices", "hyper_parameters",
                "gravity_mean", "ou_substeps", "policy_shapes", "seed", "reductionist"):
        if raw["setup"][key] != setup[key]:
            raise ValueError(f"Saved simulation setting differs: {key}")
    data_hash = hashlib.sha256(raw["costs"].tobytes() + raw["arm_indices"].tobytes()
                               + raw["objective_scale"].tobytes()).hexdigest()
    source_code = Path(__file__).read_bytes() + (folder / "analysis.py").read_bytes()
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()
                               + data_hash.encode() + source_code).hexdigest()
    if target.exists() and not reset:
        result = pickle.loads(target.read_bytes())
        if result["analysis"]["signature"] != signature:
            raise ValueError("Analysis settings changed; use --reanalyze. Raw data are retained.")
    else:
        result = {key: value for key, value in raw.items() if key != "analysis"}
        horizons = np.asarray(config["horizons"])
        r, j = len(raw["costs"]), len(expected)
        result["analysis"] = {
            "signature": signature, "configuration": config, "complete": False,
            "input_file": str(source), "input_data_sha256": data_hash,
            "horizons": horizons,
            "lb_statistic": np.full((len(horizons), r), np.nan),
            "lb_pvalue": np.full((len(horizons), r), np.nan),
            "manova_statistic": np.full((len(horizons), j), np.nan),
            "manova_pvalue": np.full((len(horizons), j), np.nan),
            "manova_classical_f_pvalue": np.full((len(horizons), j), np.nan),
            "arm_counts_by_iteration": np.array([(raw["arm_indices"] == a).sum(axis=0)
                                                   for a in range(j)]).T,
        }
        save_result(target, result)
    a = result["analysis"]
    if a["complete"]:
        print(f"Already complete: {target}")
        return
    x = raw["costs"] / raw["objective_scale"]
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite simulated costs")
    if a["arm_counts_by_iteration"].min() < config["manova"]["min_group_size"]:
        raise ValueError("Insufficient observations per arm/iteration")
    r, j = len(x), len(expected)
    with tqdm(total=a["lb_pvalue"].size, initial=int(np.isfinite(a["lb_pvalue"]).sum()),
              desc="Ljung-Box tests", ascii=True) as progress:
        for hi, p in enumerate(a["horizons"]):
            for run in range(r):
                if np.isfinite(a["lb_pvalue"][hi, run]):
                    continue
                t = ljung_box_test(x[run, :p], config["ljung_box"]["max_lag"],
                        config["ljung_box"]["n_permutations"],
                        np.random.SeedSequence([config["seed"], 1, int(p), run]))
                a["lb_statistic"][hi, run], a["lb_pvalue"][hi, run] = t["statistic"], t["pvalue"]
                progress.update()
            save_result(target, result)
    with threadpool_limits(limits=2), tqdm(total=a["manova_pvalue"].size,
            initial=int(np.isfinite(a["manova_pvalue"]).sum()), desc="MANOVA tests", ascii=True) as progress:
        for hi, p in enumerate(a["horizons"]):
            for arm in range(j):
                if np.isfinite(a["manova_pvalue"][hi, arm]):
                    continue
                t = manova_test(x[:, :p], raw["arm_indices"][:, :p] == arm,
                        config["manova"]["n_bootstrap"],
                        np.random.SeedSequence([config["seed"], 3, int(p), arm]))
                a["manova_statistic"][hi, arm] = t["statistic"]
                a["manova_pvalue"][hi, arm] = t["pvalue"]
                a["manova_classical_f_pvalue"][hi, arm] = t["classical_f_pvalue"]
                save_result(target, result)
                progress.update()
    a["manova_pvalue_holm"] = np.array([holm_adjust(p) for p in a["manova_pvalue"]])
    a["lb_by_significance_level"] = [{"alpha": level,
        "lb_rejection_rate": (a["lb_pvalue"] <= level).mean(axis=1),
        "lb_rejection_count": (a["lb_pvalue"] <= level).sum(axis=1),
        "lb_null_reference_band": null_reference_band(r, level, config["confidence_level"])}
        for level in config["ljung_box"]["significance_levels"]]
    a["manova_by_significance_level"] = [{"alpha": level,
        "manova_rejection_rate": (a["manova_pvalue_holm"] <= level).mean(axis=1),
        "manova_rejection_count": (a["manova_pvalue_holm"] <= level).sum(axis=1)}
        for level in config["manova"]["significance_levels"]]
    a["complete"] = True
    save_result(target, result)
    print(f"Saved {target}; original simulation and energy-test results preserved.")
