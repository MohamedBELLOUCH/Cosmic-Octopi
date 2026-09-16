"""Practical-equivalence sensitivity analysis of completed simulated trajectories.

Reads existing data only. Does not train, simulate, or overwrite original results.
Confidence bounds use a whole-trajectory bootstrap and are approximate.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ENERGY_SQUARED_BOUND = 2 * np.sqrt(2.0)  # costs lie in [0,1]^2


def save(path, data):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def validate(config):
    if config["simulated_rewards_only"] is not True:
        raise ValueError("Only simulated-reward input is allowed")
    horizons = config["horizons"]
    lag = config["max_lag"]
    if (type(lag) is not int or lag < 1 or not horizons
            or horizons != sorted(set(horizons))
            or any(type(p) is not int or p <= lag for p in horizons)):
        raise ValueError("Use increasing horizons greater than the positive integer lag")
    for name, maximum in (("correlation_tolerances", 1),
                          ("energy_distance_tolerances", np.sqrt(ENERGY_SQUARED_BOUND))):
        values = np.asarray(config[name], float)
        if (values.ndim != 1 or not len(values) or not np.isfinite(values).all()
                or np.any(values <= 0) or np.any(values >= maximum)):
            raise ValueError(f"Invalid {name}")
    if not 0 < config["confidence_level"] < 1:
        raise ValueError("Invalid confidence level")
    for key in ("bootstrap_replicates", "min_group_size", "blas_threads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid {key}")


def bootstrap_weights(n_runs, n_bootstrap, seed):
    rng = np.random.default_rng(seed)
    # Row zero is the original empirical sample. Other rows are counts of
    # independent trajectories, not resampled time points.
    return np.vstack((np.ones(n_runs),
                      rng.multinomial(n_runs, np.full(n_runs, 1 / n_runs), n_bootstrap)))


def correlation_vectors(costs, weights, horizon, lag):
    """Pearson correlations of pooled lag pairs, retaining between-time drift."""
    vectors = []
    for h in range(1, lag + 1):
        x, y = costs[:, :horizon-h], costs[:, h:horizon]
        count = weights.sum(axis=1) * (horizon - h)
        mx = weights @ x.sum(axis=1) / count[:, None]
        my = weights @ y.sum(axis=1) / count[:, None]
        vx = weights @ (x*x).sum(axis=1) / count[:, None] - mx*mx
        vy = weights @ (y*y).sum(axis=1) / count[:, None] - my*my
        products = np.einsum("rti,rtj->rij", x, y).reshape(len(costs), -1)
        cross = (weights @ products / count[:, None]).reshape(-1, 2, 2)
        cross -= mx[:, :, None] * my[:, None, :]
        denominator = np.sqrt(np.maximum(vx[:, :, None] * vy[:, None, :], 0))
        valid = (vx[:, :, None] > 1e-14) & (vy[:, None, :] > 1e-14)
        rho = np.full_like(cross, np.nan)
        np.divide(cross, denominator, out=rho, where=valid)
        vectors.append(np.clip(rho, -1, 1).reshape(len(weights), -1))
    return np.concatenate(vectors, axis=1)


def classify(lower, upper, tolerances):
    """Never turn failure to establish equivalence into a demonstrated violation."""
    lower, upper = np.asarray(lower), np.asarray(upper)
    output = []
    for tolerance in tolerances:
        supported = np.isfinite(upper) & (upper < tolerance)
        beyond = np.isfinite(lower) & (lower > tolerance)
        output.append({"tolerance": tolerance, "supported": supported,
                       "beyond_tolerance": beyond,
                       "inconclusive": ~(supported | beyond)})
    return output


def correlation_summary(costs, weights, config):
    effects, lower, upper, errors = [], [], [], []
    for horizon in tqdm(config["horizons"], desc="Correlation bounds", ascii=True):
        rho = correlation_vectors(costs, weights, horizon, config["max_lag"])
        if not np.isfinite(rho[0]).all():
            # Undefined correlations must not be treated as evidence of equivalence.
            effects.append(np.nan); lower.append(0.0); upper.append(1.0)
            errors.append(1.0)
            continue
        error = np.max(np.abs(rho[1:] - rho[0]), axis=1)
        error[~np.isfinite(error)] = 2.0
        radius = np.quantile(error, config["confidence_level"], method="higher")
        effect = np.max(np.abs(rho[0]))
        effects.append(effect); lower.append(max(0, effect-radius))
        upper.append(min(1, effect+radius)); errors.append(radius)
    return {"effect": np.asarray(effects), "lower": np.asarray(lower),
            "upper": np.asarray(upper), "radius": np.asarray(errors),
            "decisions": classify(lower, upper, config["correlation_tolerances"])}


def arm_energy(costs, arms, arm, weights, horizons, min_group_size, progress=None):
    """Exact Euclidean V-statistic, all time pairs; cluster bootstrap weights.

    Compute squared energy first. Bounds are transformed by sqrt afterwards.
    This avoids bootstrapping an extra square-root singularity at zero.
    No time grouping, subsampling, projection, or selection of favorable pairs.
    """
    groups, probabilities, within, valid_groups = [], [], [], []
    n_replicates = len(weights)
    for t in range(max(horizons)):
        indices = np.flatnonzero(arms[:, t] == arm)
        if len(indices) < min_group_size:
            raise ValueError(f"Arm {arm}, iteration {t+1}: too few observations")
        x = costs[indices, t]
        w = weights[:, indices]
        counts = w.sum(axis=1)
        valid_groups.append(counts > 0)
        w = w / np.maximum(counts[:, None], 1)
        groups.append(x); probabilities.append(w)
        within.append(np.einsum("bi,bi->b", w @ cdist(x, x), w))
    effect = 0.0
    errors = np.zeros(n_replicates - 1)
    effects, prefix_errors = [], []
    for t, (x, wx) in enumerate(zip(groups, probabilities)):
        for u in range(t):
            cross = np.einsum("bi,bi->b", wx @ cdist(x, groups[u]), probabilities[u])
            energy = np.maximum(0, 2 * cross - within[t] - within[u])
            effect = max(effect, energy[0])
            difference = np.abs(energy[1:] - energy[0])
            valid = valid_groups[t][1:] & valid_groups[u][1:]
            # Empty bootstrap groups receive the full possible uncertainty.
            difference[~valid] = ENERGY_SQUARED_BOUND
            errors = np.maximum(errors, difference)
        if t+1 in horizons:
            effects.append(effect)
            prefix_errors.append(errors.copy())
        if progress is not None:
            progress.update()
    return np.asarray(effects), np.asarray(prefix_errors).T


def distribution_summary(effect_squared, bootstrap_errors, config):
    # A shared resampling scheme makes this max simultaneous across arms and
    # all time pairs within each horizon. Horizons remain pointwise.
    error = np.max(bootstrap_errors, axis=0)  # bootstrap x horizon
    radius = np.quantile(error, config["confidence_level"], axis=0, method="higher")
    lower = np.sqrt(np.maximum(0, effect_squared-radius))
    upper = np.sqrt(np.minimum(ENERGY_SQUARED_BOUND, effect_squared+radius))
    return {"effect_by_arm": np.sqrt(effect_squared), "lower_by_arm": lower,
            "upper_by_arm": upper, "squared_energy_radius": radius,
            "effect": np.sqrt(effect_squared.max(axis=0)),
            "lower": lower.max(axis=0), "upper": upper.max(axis=0),
            "decisions_by_arm": classify(lower, upper, config["energy_distance_tolerances"]),
            "decisions": classify(lower.max(axis=0), upper.max(axis=0),
                                  config["energy_distance_tolerances"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=HERE / "setup_practical_equivalence.json")
    args = parser.parse_args()
    setup_path = args.setup.resolve()
    config = json.loads(setup_path.read_text(encoding="utf-8"))
    validate(config)
    source = setup_path.parent / config["input_file"]
    target = setup_path.parent / config["output_file"]
    if source.resolve() == target.resolve():
        raise ValueError("Input and output must differ")
    raw = pickle.loads(source.read_bytes())
    if raw["setup"].get("simulated_rewards") is not True:
        raise ValueError("Input must contain simulated rewards only")
    if not raw["completed_runs"].all():
        raise ValueError(f"Finish trajectory generation first: {raw['completed_runs'].sum()}"
                         f"/{len(raw['completed_runs'])} complete. Use script.py --stage generate.")
    costs = np.asarray(raw["costs"], float) / raw["objective_scale"]
    arms = np.asarray(raw["arm_indices"], int)
    if (costs.shape != (*arms.shape, 2) or not np.isfinite(costs).all()
            or np.any(costs < -1e-9) or np.any(costs > 1+1e-9)
            or max(config["horizons"]) > arms.shape[1]):
        raise ValueError("Invalid normalized cost trajectories or horizons")
    input_hash = hashlib.sha256(raw["costs"].tobytes() + arms.tobytes()
                                + np.asarray(raw["objective_scale"]).tobytes()
                                + np.asarray(raw["arm_values"]).tobytes()).hexdigest()
    signature = hashlib.sha256((json.dumps(config, sort_keys=True)+input_hash).encode()
                               + Path(__file__).read_bytes()).hexdigest()
    n_arms, n_runs = len(raw["arm_values"]), len(costs)
    if target.exists():
        output = pickle.loads(target.read_bytes())
        if output["signature"] != signature:
            raise ValueError("Settings, input, or source changed; choose a new output_file")
    else:
        output = {"signature": signature, "configuration": config,
                  "input_data_sha256": input_hash, "simulation_setup": raw["setup"],
                  "n_runs": n_runs, "arm_values": raw["arm_values"],
                  "correlation": None, "distribution": None, "complete": False,
                  "arm_complete": np.zeros(n_arms, bool),
                  "energy_squared": np.full((n_arms, len(config["horizons"])), np.nan),
                  "energy_bootstrap_errors": np.full((n_arms, config["bootstrap_replicates"],
                                                       len(config["horizons"])), np.nan)}
    if output["complete"]:
        print(f"Already complete: {target}")
        return
    weights = bootstrap_weights(n_runs, config["bootstrap_replicates"], config["bootstrap_seed"])
    with threadpool_limits(limits=config["blas_threads"]):
        if output["correlation"] is None:
            output["correlation"] = correlation_summary(costs, weights, config)
            save(target, output)
        horizon = max(config["horizons"])
        with tqdm(total=n_arms*horizon, initial=int(output["arm_complete"].sum())*horizon,
                  desc="Energy-distance bounds", ascii=True) as progress:
            for arm in range(n_arms):
                if output["arm_complete"][arm]:
                    continue
                progress.set_postfix(arm=arm+1, refresh=False)
                values, errors = arm_energy(costs, arms, arm, weights, config["horizons"],
                                           config["min_group_size"], progress)
                output["energy_squared"][arm] = values
                output["energy_bootstrap_errors"][arm] = errors
                output["arm_complete"][arm] = True
                save(target, output)
    output["distribution"] = distribution_summary(output["energy_squared"],
                                                  output["energy_bootstrap_errors"], config)
    output["complete"] = True
    save(target, output)
    print(f"Saved {target}. Original simulations and hypothesis tests were not changed.")


if __name__ == "__main__":
    main()
