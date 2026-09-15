r"""Find an approximate Pareto set of constant holistic fitness margins.

Read setup_model_based_holistic_quantized.json beside this file. Evaluate one realization
at each of C physical margins, fit Gaussian Nadaraya-Watson regressions, and
minimize the two predicted costs with NSGA-II. Only the C training evaluations
call the simulator. The saved objectives are surrogate predictions; evaluation
of random sequences sampled from the returned thresholds is a later step.

CMD, from the project root:
    ".venv\Scripts\python.exe" -B "Compression coexistence\holistic single fitness margin baseline quantized.py"

The output is one pickle beside the setup; no other experiment files are edited.
NSGA-II implementation: https://pymoo.org/algorithms/moo/nsga2.html
"""

import sys

sys.dont_write_bytecode = True

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import tempfile
import time

from model_based_quantized import load_setup, simulator_inputs, payload_measurements, extra_source_paths

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.callback import Callback
from pymoo.core.problem import Problem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.survival.rank_and_crowding import RankAndCrowding
from pymoo.optimize import minimize
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent


def positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_setup(setup):
    """Reject settings that would silently change the constant-margin experiment."""
    if setup["schema_version"] != 1 or setup["simulated_rewards"] is not True:
        raise ValueError("Expected schema_version=1 and simulated_rewards=true")
    if setup["n_independent_runs"] != 1:
        raise ValueError("This single-experiment script supports one independent run")
    if type(setup["seed"]) is not int or not 0 <= setup["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    for name in ("torch_num_threads", "ou_substeps"):
        positive_integer(setup[name], name)
    positive_integer(setup["holistic"]["horizon"], "holistic.horizon")
    if setup["holistic"]["margin_parameterization"] != "normalized_sequence":
        raise ValueError("The shared holistic simulator expects normalized_sequence")

    classes = ("Cheetah", "Ant", "Leg", "Humanoid")
    parameters = setup["system_dynamics_parameters"]
    if parameters["n_classes"] != len(classes):
        raise ValueError("The shared simulator requires all four classes")
    positive_integer(parameters["n_octopi"], "n_octopi")
    for name in ("gamma_learn", "gamma_break", "gamma_req", "gamma_grav", "gamma_epis"):
        if not np.isfinite(parameters[name]) or parameters[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(parameters["gamma_sync"]) or not 0 < parameters["gamma_sync"] <= 1:
        raise ValueError("gamma_sync must be in (0, 1]")
    if not np.isfinite(parameters["gamma_heter"]) or parameters["gamma_heter"] < 0:
        raise ValueError("gamma_heter must be finite and nonnegative")
    if not np.isfinite(setup["gravity_mean"]) or setup["gravity_mean"] <= 0:
        raise ValueError("gravity_mean must be finite and positive")
    for name in ("temperature", "scaling_constant"):
        value = setup["hyper_parameters"][name]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")

    for name in ("fitness_margin_bounds", "policy_shapes", "Xi_matrices"):
        if set(setup[name]) != set(classes):
            raise ValueError(f"{name} must contain exactly {classes}")
    bounds = np.asarray([setup["fitness_margin_bounds"][name] for name in classes], dtype=float)
    if bounds.shape != (4, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
        raise ValueError("Each class must have finite, increasing margin bounds")
    if not np.all(bounds == bounds[0]):
        raise ValueError("A common constant physical margin requires identical class bounds")
    for name in classes:
        matrix = np.asarray(setup["Xi_matrices"][name], dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{name}: Xi must be a finite 4x4 matrix")
        if len(setup["policy_shapes"][name]) != 3:
            raise ValueError(f"{name}: expected input, action and hidden policy sizes")
        for size in setup["policy_shapes"][name]:
            positive_integer(size, f"{name} policy size")

    baseline = setup["baseline"]
    positive_integer(baseline["n_grid_points"], "baseline.n_grid_points")
    if baseline["n_grid_points"] < 2 or baseline["n_realizations_per_margin"] != 1:
        raise ValueError("Use at least two grid points and exactly one realization per margin")
    expected = {
        "grid": "linspace_inclusive",
        "margin_parameterization": "one_physical_margin_shared_by_all_classes_and_iterations",
        "scenario_seed_policy": "distinct_seed_per_grid_point_from_setup_seed",
    }
    if any(baseline[key] != value for key, value in expected.items()):
        raise ValueError("Unsupported baseline grid, parameterization or seed policy")
    smoothing = baseline["smoothing"]
    if smoothing["method"] != "nadaraya_watson_gaussian" or smoothing["bandwidth_units"] != "physical_fitness_margin":
        raise ValueError("Use Gaussian kernel regression in physical margin units")
    if not np.isfinite(smoothing["bandwidth"]) or smoothing["bandwidth"] <= 0:
        raise ValueError("Gaussian bandwidth must be finite and positive")

    hp = baseline["nsga2"]
    for name in ("pop_size", "n_offsprings", "n_generations"):
        positive_integer(hp[name], f"nsga2.{name}")
    if hp["pop_size"] < 2 or hp["initial_sampling"] != "linspace_inclusive":
        raise ValueError("NSGA-II requires a population of at least two, initialized on the interval")
    if hp["tournament_type"] != "comp_by_rank_and_crowding" or hp["crowding_function"] != "cd":
        raise ValueError("Use rank-and-crowding tournaments and ordinary crowding distance")
    if hp["implementation"] != "pymoo.algorithms.moo.nsga2.NSGA2":
        raise ValueError("Unsupported optimizer implementation")
    if version("pymoo") != hp["pymoo_version"]:
        raise ValueError(f"Install pymoo=={hp['pymoo_version']} as specified in the setup")
    if type(hp["seed"]) is not int or not 0 <= hp["seed"] < 2**32:
        raise ValueError("nsga2.seed must be an integer in [0, 2**32)")
    if type(hp["eliminate_duplicates"]) is not bool:
        raise ValueError("eliminate_duplicates must be a boolean")
    for name in ("crossover_probability", "crossover_probability_per_variable",
                 "mutation_probability", "mutation_probability_per_variable"):
        if not np.isfinite(hp[name]) or not 0 < hp[name] <= 1:
            raise ValueError(f"nsga2.{name} must be in (0, 1]")
    for name in ("crossover_eta", "mutation_eta"):
        if not np.isfinite(hp[name]) or hp[name] <= 0:
            raise ValueError(f"nsga2.{name} must be finite and positive")
    if setup["objective_senses"] != ["minimize", "minimize"]:
        raise ValueError("Both overhead and instability must be minimized")
    reference = np.asarray(setup["reference_costs"], dtype=float)
    if reference.shape != (2,) or not np.isfinite(reference).all() or np.any(reference <= 0):
        raise ValueError("reference_costs must contain two finite positive costs")
    output_name = Path(setup["output_file"])
    if output_name.name != str(output_name) or output_name.suffix != ".pkl":
        raise ValueError("output_file must be a .pkl filename beside the setup")
    return bounds[0]


def load_simulator():
    """Reuse the current sequence evaluator without calling any experiment main."""
    path = PROJECT_ROOT / "Pareto fronts" / "script.py"
    spec = importlib.util.spec_from_file_location("_constant_margin_simulator", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GaussianCostRegression:
    """Two Nadaraya-Watson regressions with a shared physical-margin bandwidth."""

    def __init__(self, margins, costs, bandwidth):
        self.margins = np.asarray(margins, dtype=float)
        self.costs = np.asarray(costs, dtype=float)
        self.bandwidth = float(bandwidth)
        if self.margins.ndim != 1 or self.costs.shape != (len(self.margins), 2):
            raise ValueError("Expected training margins (C,) and costs (C, 2)")
        if not np.isfinite(self.margins).all() or not np.isfinite(self.costs).all():
            raise ValueError("Training observations must be finite")

    def predict(self, margins):
        x = np.asarray(margins, dtype=float).reshape(-1)
        log_weights = -0.5 * ((x[:, None] - self.margins[None, :]) / self.bandwidth)**2
        # Subtracting the row maximum leaves normalized weights unchanged and
        # prevents every weight from underflowing when the bandwidth is small.
        weights = np.exp(log_weights - log_weights.max(axis=1, keepdims=True))
        weights /= weights.sum(axis=1, keepdims=True)
        return weights @ self.costs


class ConstantMarginProblem(Problem):
    def __init__(self, regression, bounds):
        super().__init__(n_var=1, n_obj=2, xl=bounds[0], xu=bounds[1])
        self.regression = regression

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.regression.predict(x[:, 0])


class GenerationProgress(Callback):
    def __init__(self, bar):
        super().__init__()
        self.bar = bar

    def notify(self, algorithm):
        self.bar.update(max(0, int(algorithm.n_gen) - self.bar.n))
        self.bar.set_postfix(surrogate_evaluations=algorithm.evaluator.n_eval, refresh=False)


def optimize_surrogates(regression, bounds, hp):
    """Optimize only the cheap smoothed objectives; retain the returned front."""
    problem = ConstantMarginProblem(regression, bounds)
    algorithm = NSGA2(
        pop_size=hp["pop_size"],
        n_offsprings=hp["n_offsprings"],
        sampling=np.linspace(*bounds, hp["pop_size"])[:, None],
        crossover=SBX(prob=hp["crossover_probability"],
                      prob_var=hp["crossover_probability_per_variable"], eta=hp["crossover_eta"]),
        mutation=PM(prob=hp["mutation_probability"],
                    prob_var=hp["mutation_probability_per_variable"], eta=hp["mutation_eta"]),
        survival=RankAndCrowding(crowding_func=hp["crowding_function"]),
        eliminate_duplicates=hp["eliminate_duplicates"],
    )
    algorithm.tournament_type = hp["tournament_type"]
    with tqdm(total=hp["n_generations"], desc="NSGA-II", unit="generation", ascii=True) as bar:
        result = minimize(
            problem, algorithm, ("n_gen", hp["n_generations"]), seed=hp["seed"],
            callback=GenerationProgress(bar), verbose=False, save_history=False,
        )
    if result.X is None:
        raise RuntimeError("NSGA-II returned no Pareto solutions")
    margins = np.unique(np.asarray(result.X, dtype=float).reshape(-1))
    if not np.isfinite(margins).all() or np.any(margins < bounds[0]) or np.any(margins > bounds[1]):
        raise RuntimeError("NSGA-II returned invalid margin values")
    predictions = regression.predict(margins)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Kernel regression returned non-finite predictions")
    indices = NonDominatedSorting().do(predictions, only_non_dominated_front=True)
    order = indices[np.argsort(margins[indices])]
    return {
        "pareto_margins": margins[order],
        "pareto_predicted_costs": predictions[order],
        "final_population_margins": result.pop.get("X").reshape(-1),
        "final_population_predicted_costs": result.pop.get("F"),
        "n_surrogate_evaluations": int(result.algorithm.evaluator.n_eval),
    }


def source_fingerprints():
    paths = [Path(__file__).resolve(), PROJECT_ROOT / "Pareto fronts" / "script.py",
             PROJECT_ROOT / "Cosmic Octopi" / "simulated_reward_experiment.py",
             PROJECT_ROOT / "Cosmic Octopi" / "utils.py",
             PROJECT_ROOT / "Cosmic Octopi" / "front_utils.py"]
    return {str(path.relative_to(PROJECT_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def save_result(path, result):
    """Replace the output atomically after a successful run; clean up staging."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", prefix=".constant_margin_",
                                         suffix=".tmp", dir=path.parent, delete=False) as file:
            temporary_path = Path(file.name)
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(7):
            try:
                temporary_path.replace(path)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.05 * 2**attempt)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path,
                        default=EXPERIMENT_DIR / "setup_model_based_holistic_quantized.json",
                        help="Self-contained setup JSON; output is saved beside it")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    setup_path = args.setup.resolve()
    setup_bytes = setup_path.read_bytes()
    setup = load_setup(setup_path)
    bounds = validate_setup(setup)
    if args.validate_only:
        payload_measurements(setup)
        print("Quantized single-margin search setup valid; no simulations or outputs.")
        return
    source_sha256 = source_fingerprints()
    source_sha256.update({str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in extra_source_paths(setup_path)})
    existing_path = setup_path.parent / setup["output_file"]
    if existing_path.exists():
        with existing_path.open("rb") as stream:
            previous = pickle.load(stream)
        if previous["complete"] and previous["setup"] == setup and previous["source_sha256"] == source_sha256:
            print(f"Already complete: {existing_path}")
            return
        raise ValueError("Existing baseline output does not match current inputs")
    start = time.perf_counter()
    simulator = load_simulator()
    import torch

    torch.set_num_threads(setup["torch_num_threads"])
    torch.manual_seed(setup["seed"])
    # Use complete measured int8 payload sizes, identical to optimizer inputs.
    xi_matrices, model_sizes = simulator_inputs(setup)
    baseline = setup["baseline"]
    grid = np.linspace(*bounds, baseline["n_grid_points"])
    scenario_rng = np.random.default_rng(setup["seed"])
    seeds = scenario_rng.choice(2**32, size=len(grid), replace=False).astype(np.uint64)
    costs = np.empty((len(grid), 2))
    horizon = setup["holistic"]["horizon"]
    print(f"Constant-margin baseline: {len(grid)} margins, {horizon} global iterations, "
          f"one realization per margin; gamma_sync={setup['system_dynamics_parameters']['gamma_sync']}, "
          f"gamma_grav={setup['system_dynamics_parameters']['gamma_grav']}, "
          f"gamma_heter={setup['system_dynamics_parameters']['gamma_heter']}", flush=True)
    with tqdm(total=len(grid), desc="Simulating constant margins", unit="margin", ascii=True) as bar:
        for index, (margin, seed) in enumerate(zip(grid, seeds)):
            bar.set_postfix(margin=f"{margin:.3f}")
            unit_margin = (margin - bounds[0]) / (bounds[1] - bounds[0])
            # Equal class bounds make this the same PHYSICAL threshold at
            # every global iteration, whichever class receives the request.
            sequence = np.full(horizon, unit_margin)
            costs[index] = simulator.evaluate_sequence(
                setup, xi_matrices, model_sizes, sequence, "holistic", int(seed)
            )
            if not np.isfinite(costs[index]).all():
                raise RuntimeError(f"Non-finite simulated costs at margin {margin}")
            bar.update(1)

    regression = GaussianCostRegression(grid, costs, baseline["smoothing"]["bandwidth"])
    optimized = optimize_surrogates(regression, bounds, baseline["nsga2"])
    if any(hashlib.sha256((PROJECT_ROOT / p).read_bytes()).hexdigest() != h for p, h in source_sha256.items()) or setup_path.read_bytes() != setup_bytes:
        raise RuntimeError("Setup or simulator source changed during the run; rerun with stable files")
    result = {
        "schema_version": 1,
        "kind": "holistic_constant_fitness_margin_pareto_set",
        "complete": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - start,
        "setup": setup,
        "setup_sha256": hashlib.sha256(setup_bytes).hexdigest(),
        "source_sha256": source_sha256,
        "versions": {"python": platform.python_version(), **{
            name: version(name) for name in ("numpy", "scipy", "torch", "pymoo", "tqdm")
        }},
        "model_sizes_kib": dict(zip(simulator.CLASS_NAMES, model_sizes.tolist())),
        "payload_measurements": payload_measurements(setup), "compression": "int8",
        "training": {"margins": grid, "costs": costs, "scenario_seeds": seeds,
                     "n_simulator_evaluations": len(grid)},
        **optimized,
        "pareto_unit_margins": (optimized["pareto_margins"] - bounds[0]) / (bounds[1] - bounds[0]),
        "solution_semantics": "Repeat one pareto_margins value for all holistic iterations and classes.",
        "prediction_note": "pareto_predicted_costs are Gaussian surrogate predictions in the setup objective units, not held-out simulated costs.",
        "held_out_evaluation": None,
    }
    output_path = setup_path.parent / setup["output_file"]
    save_result(output_path, result)
    print(f"Saved {len(result['pareto_margins'])} Pareto margins to {output_path}", flush=True)


if __name__ == "__main__":
    main()
