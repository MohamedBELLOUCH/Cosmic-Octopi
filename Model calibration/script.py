"""Synthetic Humanoid Xi calibration and downstream metric comparison.

The paper's wet PPO training is replaced here by reward traces generated from
the existing Xi matrix. Fitted matrices are compared with that source matrix
on paired reductionist overhead and instability evaluations. This is a check
of the calibration pipeline, not an independent empirical calibration.
"""

from contextlib import redirect_stdout
import importlib.util
from itertools import combinations
import io
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from pymle.core.TransitionDensity import ExactDensity
from pymle.fit.AnalyticalMLE import AnalyticalMLE
from pymle.fit.Minimizer import ScipyMinimizer
from pymle.models import OrnsteinUhlenbeck


EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXPERIMENT_DIR.parent / "Cosmic Octopi"))

from simulated_reward_experiment import load_xi_matrix, simulate_ou_episodes  # noqa: E402
from utils import CLASS_NAMES, Network, get_model_size_in_kb  # noqa: E402


class RecordingMinimizer(ScipyMinimizer):
    """Expose the convergence flag that PyMLE's EstimatedResult omits."""

    def minimize(self, function, bounds=None, guess=None):
        self.last_result = super().minimize(function, bounds, guess)
        return self.last_result


def validate_setup(setup):
    if not setup["simulated_rewards"]:
        raise ValueError("This calibration requires simulated_rewards=true")
    for key in ("n_episodes", "ou_substeps"):
        if not isinstance(setup[key], int) or setup[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if setup["n_episodes"] < 3:
        raise ValueError("At least three episodes are needed to fit an OU process")
    if setup["gravity_mean"] <= 0 or setup["gravity_variance"] <= 0:
        raise ValueError("Gravity mean and variance must be positive")
    if setup["pre_aggregation_fitness_half_width"] <= 0:
        raise ValueError("Fitness half-width must be positive")
    for name, bounds in setup["ou_fit_bounds"].items():
        if len(bounds) != 2 or not 0 < bounds[0] < bounds[1]:
            raise ValueError(f"{name} bounds must be ordered and positive")
    metric_setup = setup["metric_error"]
    agents = metric_setup["agent_counts"]
    experiments = metric_setup["wet_experiment_counts"]
    if not agents or any(not isinstance(n, int) or not 1 < n <= 12 for n in agents):
        raise ValueError("Metric-error agent counts must be integers from 2 to 12")
    if not experiments or any(not isinstance(n, int) or n < 2 for n in experiments):
        raise ValueError("At least two wet experiments are needed for the metric-error sweep")
    if metric_setup["n_scenarios"] < 2 or metric_setup["horizon"] < 1:
        raise ValueError("Metric-error scenarios and horizon are invalid")
    if (metric_setup["ridge_alpha"] < 0
            or metric_setup["ridge_alpha_per_sample"] < 0
            or metric_setup["bootstrap_replicates"] < 1):
        raise ValueError("Metric-error ridge alpha and bootstrap count are invalid")
    if metric_setup["system_dynamics_parameters"]["n_classes"] != len(CLASS_NAMES):
        raise ValueError("The metric evaluator expects four environment classes")


def simulate_path(xi_matrix, fitness, count, gravity, episodes, rng, substeps):
    initial, rewards = simulate_ou_episodes(
        xi_matrix, fitness, count, [gravity], episodes, rng, substeps
    )
    return np.concatenate(([initial[0]], rewards[:, 0]))


def generate_dataset(setup, xi_matrix, offset, rng):
    """Replace wet local training by synthetic warmup traces, then use all subsets."""
    n_agents = setup["n_agents"]
    episodes = setup["n_episodes"]
    gravity_mean = setup["gravity_mean"]
    gravity_variance = setup["gravity_variance"]
    subsets = [
        subset
        for size in range(1, n_agents + 1)
        for subset in combinations(range(n_agents), size)
    ]
    n_rows = setup["n_wet_experiments"] * len(subsets)
    features = np.empty((n_rows, 4))
    paths = np.empty((n_rows, episodes + 1))
    wet_experiment_ids = np.empty(n_rows, dtype=int)
    subset_masks = np.empty(n_rows, dtype=np.uint64)
    target_agent_ids = np.empty(n_rows, dtype=int)
    warmup_episode_counts = np.empty((setup["n_wet_experiments"], n_agents), dtype=int)
    agent_gravities = np.empty_like(warmup_episode_counts, dtype=float)
    local_fitnesses = np.empty_like(agent_gravities)
    row = 0

    for experiment in range(setup["n_wet_experiments"]):
        max_warmup = int(rng.integers(1, episodes + 1))
        gravities = rng.gamma(
            shape=gravity_mean**2 / gravity_variance,
            scale=gravity_variance / gravity_mean,
            size=n_agents,
        )
        starting_fitnesses = rng.uniform(
            offset - setup["pre_aggregation_fitness_half_width"],
            offset + setup["pre_aggregation_fitness_half_width"],
            size=n_agents,
        )
        fitnesses = np.empty(n_agents)
        for agent in range(n_agents):
            warmup = int(rng.integers(1, max_warmup + 1))
            warmup_episode_counts[experiment, agent] = warmup
            local_path = simulate_path(
                xi_matrix, starting_fitnesses[agent], 1, gravities[agent],
                warmup, rng, setup["ou_substeps"],
            )
            fitnesses[agent] = local_path.mean()

        agent_gravities[experiment] = gravities
        local_fitnesses[experiment] = fitnesses
        for subset in subsets:
            target = int(rng.choice(subset))
            fitness = float(fitnesses[list(subset)].mean())
            count = len(subset)
            gravity = float(gravities[target])
            features[row] = (fitness, count, gravity, 1.0)
            paths[row] = simulate_path(
                xi_matrix, fitness, count, gravity, episodes, rng,
                setup["ou_substeps"],
            )
            wet_experiment_ids[row] = experiment
            subset_masks[row] = sum(1 << agent for agent in subset)
            target_agent_ids[row] = target
            row += 1
        print(
            f"Generated {experiment + 1}/{setup['n_wet_experiments']} "
            "synthetic wet experiments",
            flush=True,
        )

    return {
        "features": features,
        "reward_paths": paths,
        "wet_experiment_ids": wet_experiment_ids,
        "subset_masks": subset_masks,
        "target_agent_ids": target_agent_ids,
        "warmup_episode_counts": warmup_episode_counts,
        "agent_gravities": agent_gravities,
        "local_fitnesses": local_fitnesses,
    }


def fit_ou_path(path, bounds):
    model = OrnsteinUhlenbeck()
    minimizer = RecordingMinimizer(
        method="L-BFGS-B", tol=1e-8, options={"maxiter": 300}
    )
    estimator = AnalyticalMLE(
        sample=path,
        param_bounds=bounds,
        dt=1.0,
        density=ExactDensity(model),
        minimizer=minimizer,
    )
    guess = np.array([
        0.01,
        max(float(path.mean()), bounds[1][0] * 2),
        max(float(np.std(np.diff(path))), bounds[2][0] * 2),
    ])
    guess = np.clip(guess, [pair[0] for pair in bounds], [pair[1] for pair in bounds])
    with redirect_stdout(io.StringIO()):
        result = estimator.estimate_params(guess)
    return result.params, result.log_like, minimizer.last_result.success


def regress_matrix_regularized(features, targets, alpha):
    """Fit noisy log-OU targets with standardized ridge; keep the intercept free.

    Initial reward is observed exactly in each synthetic path, so its column
    remains ordinary least squares. The other three columns are estimated from
    short traces and need shrinkage to avoid divergent surrogate OU paths at
    the smallest calibration sample counts.
    """
    means = features[:, :3].mean(axis=0)
    scales = features[:, :3].std(axis=0)
    if np.any(scales <= 0):
        raise ValueError("Calibration features have no variation")
    standardized = (features[:, :3] - means) / scales
    centered_targets = targets[:, 1:] - targets[:, 1:].mean(axis=0)
    weights = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(3),
        standardized.T @ centered_targets,
    )
    matrix = np.empty((4, 4))
    matrix[:3, 1:] = weights / scales[:, None]
    matrix[3, 1:] = targets[:, 1:].mean(axis=0) - means @ matrix[:3, 1:]
    matrix[:, 0], *_ = np.linalg.lstsq(features, targets[:, 0], rcond=None)
    return matrix


def fit_paths(paths, bounds, label):
    fitted_parameters = np.empty((len(paths), 3))
    log_likelihoods = np.empty(len(paths))
    converged = np.empty(len(paths), dtype=bool)
    for index, path in enumerate(paths):
        params, log_likelihood, success = fit_ou_path(path, bounds)
        fitted_parameters[index] = params
        log_likelihoods[index] = log_likelihood
        converged[index] = success
        if (index + 1) % 100 == 0 or index + 1 == len(paths):
            print(f"{label}: fitted {index + 1}/{len(paths)} OU paths", flush=True)
    valid = np.all(np.isfinite(fitted_parameters), axis=1) & np.all(
        fitted_parameters > 0, axis=1
    ) & np.isfinite(log_likelihoods)
    transformed = np.column_stack((paths[:, 0], np.log(fitted_parameters)))
    return {
        "fitted_parameters": fitted_parameters,
        "log_likelihoods": log_likelihoods,
        "converged": converged,
        "valid": valid,
        "transformed": transformed,
    }


def load_metric_evaluator():
    path = EXPERIMENT_DIR.parent / "Pareto fronts" / "script.py"
    spec = importlib.util.spec_from_file_location("pareto_front_experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.evaluate_sequence


def run_metric_error_sweep(setup, xi_matrix, offset, bounds):
    """Compare paired source-Xi and fitted-Xi objective values (Figure 13 style)."""
    metric_setup = setup["metric_error"]
    agent_counts = metric_setup["agent_counts"]
    experiment_counts = metric_setup["wet_experiment_counts"]
    horizon = metric_setup["horizon"]
    n_scenarios = metric_setup["n_scenarios"]
    evaluate_sequence = load_metric_evaluator()
    evaluation_setup = {
        "system_dynamics_parameters": metric_setup["system_dynamics_parameters"],
        "reductionist": {"class_name": setup["class_name"], "horizon": horizon},
        "gravity_mean": setup["gravity_mean"],
        "ou_substeps": setup["ou_substeps"],
        "hyper_parameters": metric_setup["hyper_parameters"],
    }
    model_sizes = np.zeros(len(CLASS_NAMES))
    model_sizes[CLASS_NAMES.index(setup["class_name"])] = get_model_size_in_kb(
        Network(*metric_setup["policy_shape"])
    )
    rng = np.random.default_rng(setup["seed"] + 99_999)
    margin_bounds = metric_setup["fitness_margin_bounds"]
    margin_sequences = rng.uniform(
        margin_bounds[0], margin_bounds[1], size=(n_scenarios, horizon)
    )
    scenario_seeds = rng.integers(0, 2**63, size=n_scenarios, dtype=np.int64)
    source_metrics = np.empty((n_scenarios, 2))
    source_matrices = {setup["class_name"]: xi_matrix}
    for index, (margins, seed) in enumerate(zip(margin_sequences, scenario_seeds)):
        source_metrics[index] = evaluate_sequence(
            evaluation_setup, source_matrices, model_sizes, margins,
            "reductionist", int(seed),
        )
    print(f"Evaluated {n_scenarios} paired source-Xi scenarios", flush=True)

    shape = (len(agent_counts), len(experiment_counts))
    matrices = np.empty((*shape, 4, 4))
    ranks = np.empty(shape, dtype=int)
    sample_counts = np.empty(shape, dtype=int)
    effective_ridge_alphas = np.empty(shape)
    surrogate_metrics = np.empty((*shape, n_scenarios, 2))
    datasets = {}
    for agent_index, n_agents in enumerate(agent_counts):
        group_setup = dict(setup, n_agents=n_agents, n_wet_experiments=max(experiment_counts))
        group_dataset = generate_dataset(
            group_setup, xi_matrix, offset,
            np.random.default_rng(setup["seed"] + 1_000 * n_agents),
        )
        group_fits = fit_paths(group_dataset["reward_paths"], bounds, f"{n_agents} agents")
        datasets[n_agents] = {**group_dataset, **group_fits}
        for experiment_index, n_experiments in enumerate(experiment_counts):
            selected = (
                (group_dataset["wet_experiment_ids"] < n_experiments)
                & group_fits["valid"]
            )
            design = group_dataset["features"][selected]
            ranks[agent_index, experiment_index] = np.linalg.matrix_rank(design)
            sample_counts[agent_index, experiment_index] = len(design)
            effective_alpha = max(
                metric_setup["ridge_alpha"],
                metric_setup["ridge_alpha_per_sample"] * len(design),
            )
            effective_ridge_alphas[agent_index, experiment_index] = effective_alpha
            try:
                if ranks[agent_index, experiment_index] < 4:
                    raise ValueError("The calibration feature matrix is not full rank")
                fitted_matrix = regress_matrix_regularized(
                    design, group_fits["transformed"][selected],
                    effective_alpha,
                )
            except ValueError as error:
                raise ValueError(
                    f"Cannot calibrate {n_agents} agents with "
                    f"{n_experiments} wet experiments"
                ) from error
            matrices[agent_index, experiment_index] = fitted_matrix
            fitted_matrices = {setup["class_name"]: fitted_matrix}
            for scenario, (margins, seed) in enumerate(zip(margin_sequences, scenario_seeds)):
                surrogate_metrics[agent_index, experiment_index, scenario] = (
                    evaluate_sequence(
                        evaluation_setup, fitted_matrices, model_sizes,
                        margins, "reductionist", int(seed),
                    )
                )
            print(
                f"Metric comparison: {n_agents} agents, "
                f"{n_experiments} wet experiments",
                flush=True,
            )

    absolute_errors = np.abs(surrogate_metrics - source_metrics[None, None, :, :])
    if not np.all(np.isfinite(absolute_errors)):
        raise ValueError("Metric-error evaluation produced nonfinite objectives")
    if np.any(np.abs(source_metrics) < 1e-12):
        raise ValueError("Percentage error is undefined for a zero source objective")
    percentage_errors = 100 * absolute_errors / np.abs(source_metrics[None, None, :, :])
    rmse = np.sqrt(np.mean(absolute_errors**2, axis=2))
    rmspe = np.sqrt(np.mean(percentage_errors**2, axis=2))
    bootstrap_rng = np.random.default_rng(setup["seed"] + 199_999)
    bootstrap_indices = bootstrap_rng.integers(
        0, n_scenarios,
        size=(metric_setup["bootstrap_replicates"], n_scenarios),
    )
    bootstrap_rmse = np.sqrt(np.mean(absolute_errors[:, :, bootstrap_indices, :]**2, axis=3))
    band_lower, band_upper = np.quantile(bootstrap_rmse, [0.05, 0.95], axis=2)
    bootstrap_rmspe = np.sqrt(
        np.mean(percentage_errors[:, :, bootstrap_indices, :]**2, axis=3)
    )
    percentage_band_lower, percentage_band_upper = np.quantile(
        bootstrap_rmspe, [0.05, 0.95], axis=2
    )
    return {
        "agent_counts": np.asarray(agent_counts),
        "wet_experiment_counts": np.asarray(experiment_counts),
        "scenario_seeds": scenario_seeds,
        "fitness_margin_sequences": margin_sequences,
        "evaluation_setup": evaluation_setup,
        "model_sizes_kb": model_sizes,
        "source_metrics": source_metrics,
        "surrogate_metrics": surrogate_metrics,
        "absolute_errors": absolute_errors,
        "absolute_percentage_errors": percentage_errors,
        "mean_absolute_error": absolute_errors.mean(axis=2),
        "mean_absolute_percentage_error": percentage_errors.mean(axis=2),
        "root_mean_square_error": rmse,
        "root_mean_square_percentage_error": rmspe,
        "rmse_band_lower": band_lower,
        "rmse_band_upper": band_upper,
        "rmspe_band_lower": percentage_band_lower,
        "rmspe_band_upper": percentage_band_upper,
        "ridge_alpha": metric_setup["ridge_alpha"],
        "ridge_alpha_per_sample": metric_setup["ridge_alpha_per_sample"],
        "effective_ridge_alphas": effective_ridge_alphas,
        "absolute_error_std": absolute_errors.std(axis=2, ddof=1),
        "estimated_matrices": matrices,
        "feature_ranks": ranks,
        "calibration_sample_counts": sample_counts,
        "calibration_datasets": datasets,
    }


def main():
    with (EXPERIMENT_DIR / "setup.json").open(encoding="utf-8") as file:
        setup = json.load(file)
    validate_setup(setup)
    xi_matrix = load_xi_matrix(EXPERIMENT_DIR / setup["xi_matrix_file"], setup["class_name"])
    with (EXPERIMENT_DIR / setup["offsets_file"]).open(encoding="utf-8") as file:
        offset = float(json.load(file)[setup["class_name"]])
    bounds = [
        setup["ou_fit_bounds"][name]
        for name in ("mean_reversion_rate", "long_term_cumulative_reward", "sigma")
    ]
    result = {
        "setup": setup,
        "source_xi_matrix": xi_matrix,
        "offset": offset,
        "metric_error": run_metric_error_sweep(setup, xi_matrix, offset, bounds),
    }
    output_path = EXPERIMENT_DIR / setup["output_file"]
    with output_path.open("wb") as file:
        pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved Humanoid Xi calibration to {output_path}")
    metric_result = result["metric_error"]
    for index, n_agents in enumerate(metric_result["agent_counts"]):
        first, last = metric_result["root_mean_square_percentage_error"][index, [0, -1]]
        print(
            f"{n_agents} agents: overhead RMSPE {first[0]:.3g}→{last[0]:.3g}%, "
            f"instability RMSPE {first[1]:.3g}→{last[1]:.3g}% "
            f"({metric_result['wet_experiment_counts'][0]}→"
            f"{metric_result['wet_experiment_counts'][-1]} wet experiments)"
        )


if __name__ == "__main__":
    main()
