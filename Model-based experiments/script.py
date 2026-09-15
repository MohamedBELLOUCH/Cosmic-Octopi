"""Run all model-based experiments from the adjacent, self-contained setup.json.

Launch later from any directory:
    python -B "<project>/Model-based experiments/script.py"
An alternative configuration can be selected with --setup PATH.

This orchestrator calls the existing MORBO and qNParEGO run_condition functions;
it does not implement simulation, acquisition, scalarization, or hypervolume.
Each independent run starts a fresh optimization with a distinct seed. Seeds
are paired across methods and conditions within that run for comparison.

The only persistent project output is model_based_results.pkl beside this script.
Per-experiment checkpoints use cleaned-up system temporary files. The console
progress bar shows completed experiments and the current optimization step.
Rerunning resumes compatible checkpoints in the combined pickle.
The original single-run result files and configuration files are never changed.
The notebook continues to show those original results until updated separately.
"""

import argparse
from contextlib import contextmanager, nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
SWEEP_PARAMETERS = {
    "gravity": "gamma_grav",
    "synchronization": "gamma_sync",
    "heterogeneity": "gamma_heter",
}
CLASS_NAMES = ("Cheetah", "Ant", "Leg", "Humanoid")


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def finite_positive(values):
    return all(math.isfinite(value) and value > 0 for value in values)


def validate_setup(config):
    """Validate the complete suite before importing optimizers or writing files."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported suite schema_version")
    for name in ("n_independent_runs", "seed_increment", "torch_num_threads"):
        positive_integer(config[name], name)
    final_seed = (config["base_seed"]
                  + (config["n_independent_runs"] - 1) * config["seed_increment"])
    if (type(config["base_seed"]) is not int
            or not 0 <= config["base_seed"] <= final_seed < 2**32):
        raise ValueError("Run seeds must be distinct integers in [0, 2**32)")
    filename = Path(config["output_file"])
    if filename.name != str(filename) or filename.suffix != ".pkl":
        raise ValueError("output_file must be a pickle filename beside this script")
    if (EXPERIMENT_DIR / filename).resolve().parent != EXPERIMENT_DIR.resolve():
        raise ValueError("output_file must stay inside Model-based experiments")
    if set(config["hyperparameters"]) != {"MORBO", "qNParEGO"}:
        raise ValueError("Supply hyperparameters for MORBO and qNParEGO")
    if set(config["approaches"]) != {"holistic", "reductionist"}:
        raise ValueError("Supply holistic and reductionist configurations")
    if set(config["experiments"]) != set(SWEEP_PARAMETERS):
        raise ValueError("Supply gravity, synchronization, and heterogeneity experiments")

    common = config["common_setup"]
    if common["simulated_rewards"] is not True:
        raise ValueError("These experiments require simulated_rewards=true")
    positive_integer(common["n_realizations_per_candidate"], "n_realizations_per_candidate")
    positive_integer(common["ou_substeps"], "ou_substeps")
    if common["n_realizations_per_candidate"] < 2:
        raise ValueError("RTS/STR comparisons require at least two realizations per candidate")
    if (not common["formulations"]
            or len(set(common["formulations"])) != len(common["formulations"])
            or not set(common["formulations"]) <= {"RTS", "STR"}):
        raise ValueError("Use distinct RTS and/or STR formulations")
    if not finite_positive([common["gravity_mean"], *common["hyper_parameters"].values()]):
        raise ValueError("Gravity mean, temperature, and scaling constant must be positive")
    if (common["objective_names"] != ["total_overhead", "total_instability"]
            or common["objective_units"] != ["KiB", "dimensionless"]
            or common["objective_normalization"] != "divide_each_cost_by_its_reference_cost"):
        raise ValueError("The existing runners use reference-normalized overhead (KiB) and instability")
    for mapping in (config["Xi_matrices"], common["fitness_margin_bounds"], common["policy_shapes"]):
        if set(mapping) != set(CLASS_NAMES):
            raise ValueError(f"Supply values for all classes: {CLASS_NAMES}")
    for name in CLASS_NAMES:
        matrix = config["Xi_matrices"][name]
        if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                or not all(math.isfinite(value) for row in matrix for value in row)):
            raise ValueError(f"{name}: Xi must be a finite 4x4 matrix")
        bounds = common["fitness_margin_bounds"][name]
        if len(bounds) != 2 or not all(map(math.isfinite, bounds)) or bounds[0] >= bounds[1]:
            raise ValueError(f"{name}: expected increasing finite fitness margin bounds")
        shape = common["policy_shapes"][name]
        if len(shape) != 3:
            raise ValueError(f"{name}: expected input, output, and hidden policy dimensions")
        for value in shape:
            positive_integer(value, f"{name} policy dimension")
    for approach, settings in config["approaches"].items():
        positive_integer(settings[approach]["horizon"], f"{approach} horizon")
        if len(settings["reference_costs"]) != 2 or not finite_positive(settings["reference_costs"]):
            raise ValueError("Each approach needs two finite positive reference costs")
    if config["approaches"]["holistic"]["holistic"]["margin_parameterization"] != "normalized_sequence":
        raise ValueError("The holistic runner requires normalized_sequence margins")
    if config["approaches"]["reductionist"]["reductionist"]["class_name"] not in CLASS_NAMES:
        raise ValueError("Unknown reductionist class")
    for experiment, settings in config["experiments"].items():
        parameters = settings["system_dynamics_parameters"]
        for name in ("n_octopi", "n_classes"):
            positive_integer(parameters[name], name)
        if parameters["n_classes"] != len(CLASS_NAMES):
            raise ValueError("Expected four classes")
        if not finite_positive([parameters[k] for k in
                                ("gamma_learn", "gamma_break", "gamma_req", "gamma_grav", "gamma_epis")]):
            raise ValueError("Learning, break, request, gravity, and episode parameters must be positive")
        if not 0 < parameters["gamma_sync"] <= 1 or not math.isfinite(parameters["gamma_heter"]) or parameters["gamma_heter"] < 0:
            raise ValueError("Invalid synchronization or request heterogeneity parameter")
        sweep = settings["sweep"]
        levels = sweep["values"]
        if (sweep["parameter"] != SWEEP_PARAMETERS[experiment] or not levels
                or len(set(levels)) != len(levels)
                or not all(math.isfinite(value) and value >= 0 for value in levels)):
            raise ValueError(f"Invalid {experiment} sweep")
        if experiment == "gravity" and not finite_positive(levels):
            raise ValueError("Gravity variances must be strictly positive")
        if experiment == "synchronization" and not all(0 < value <= 1 for value in levels):
            raise ValueError("Synchronization values must be in (0, 1]")
    for algorithm, hp in config["hyperparameters"].items():
        for name in ("n_initial_points", "batch_size", "n_iterations", "hypervolume_checkpoint_every", "mc_samples",
                     "hypervolume_directions", "acquisition_num_restarts",
                     "acquisition_raw_samples", "acquisition_batch_limit", "acquisition_maxiter"):
            positive_integer(hp[name], f"{algorithm}.{name}")
        if hp["device"] != "cpu" or hp["dtype"] != "float64":
            raise ValueError("The existing optimizers use float64 on CPU")
        if not 0 < hp["direction_endpoint_epsilon"] < math.pi / 4:
            raise ValueError("direction_endpoint_epsilon must be in (0, pi/4)")
        if (hp["initial_design"] != "scrambled_sobol"
                or hp["surrogate"] != "SingleTaskGP"
                or hp["surrogate_fit"] != "ExactMarginalLogLikelihood"
                or hp["rts_str_acquisition"] != "qLogNoisyExpectedImprovement"
                or hp["hypervolume_direction_rule"] != "uniform_angle_midpoint_quadrature"
                or hp["reference_point_policy"] != "fixed_reference_costs_from_setup"):
            raise ValueError("Algorithm choices must match the referenced Python implementations")
    morbo, qnparego = (config["hyperparameters"][name] for name in ("MORBO", "qNParEGO"))
    for name in ("n_initial_points", "batch_size", "n_iterations", "hypervolume_checkpoint_every", "hypervolume_directions",
                 "direction_endpoint_epsilon"):
        if morbo[name] != qnparego[name]:
            raise ValueError(f"Keep {name} matched across algorithms for comparison")
    if not 0 < morbo["trust_region_length_minimum"] <= morbo["trust_region_length_initial"] <= morbo["trust_region_length_maximum"]:
        raise ValueError("Invalid trust-region length bounds")
    if (not morbo["trust_region_success_multiplier"] > 1
            or not 0 < morbo["trust_region_failure_multiplier"] < 1
            or not math.isclose(morbo["trust_region_center_old_weight"] + morbo["trust_region_center_new_weight"], 1)
            or not all(0 <= morbo[k] <= 1 for k in ("trust_region_center_old_weight", "trust_region_center_new_weight"))):
        raise ValueError("Invalid trust-region update parameters")


def load_backends():
    """Import implementations without invoking their JSON-reading CLI mains."""
    sys.path.insert(0, str(PROJECT_ROOT / "MORBO"))
    backends = {name: importlib.import_module(module) for name, module in {
        "holistic": "script_holistic_gravity",
        "reductionist": "script_reductionist_gravity",
        "synchronization": "synchronization_runner",
        "heterogeneity": "heterogeneity_runner",
    }.items()}
    # An explicit filename avoids collisions with similarly named MORBO modules.
    spec = importlib.util.spec_from_file_location(
        "_model_based_qnparego", PROJECT_ROOT / "qNParEGO" / "runner.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    backends["qNParEGO"] = module
    return backends


def source_fingerprints(backends):
    fingerprints = backends["reductionist"].source_fingerprints()
    paths = [Path(__file__), *(Path(module.__file__) for module in backends.values()),
             PROJECT_ROOT / "qNParEGO" / "qLogParEGO_success.ipynb",
             PROJECT_ROOT / "Cosmic Octopi" / "front_utils.py"]
    for path in paths:
        fingerprints[str(path.resolve().relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return fingerprints


def build_setup(config, approach, experiment, seed):
    setup = deepcopy(config["common_setup"])
    setup.update(deepcopy(config["approaches"][approach]))
    setup.update(deepcopy(config["experiments"][experiment]))
    setup["seed"] = seed
    if experiment == "gravity":
        setup["gravity_variances"] = list(setup["sweep"]["values"])
    return setup


def run_experiment(config, backends, fingerprints, run_number, algorithm, approach,
                   experiment, xi_matrices, model_sizes, output):
    """Provide inline settings and isolated checkpoints to the existing runners."""
    import numpy as np

    seed = config["base_seed"] + (run_number - 1) * config["seed_increment"]
    setup = build_setup(config, approach, experiment, seed)
    hp = deepcopy(config["hyperparameters"][algorithm])
    hp["dimension"] = setup[approach]["horizon"]
    metadata = {
        "schema_version": 2, "profile": "full", "suite_schema_version": 1,
        "algorithm": algorithm, "approach": approach, "experiment": experiment,
        "independent_run": run_number, "seed": seed,
        "implementation": hp["implementation"], "suite_setup": config,
        "setup": setup, "hyperparameters": hp,
        "shared_hyperparameters": config["hyperparameters"][algorithm],
        "source_sha256": fingerprints,
        "model_sizes_kb": dict(zip(CLASS_NAMES, map(float, model_sizes))),
        "runtime": {"torch_num_threads": config["torch_num_threads"]},
    }
    if approach == "reductionist":
        metadata["class_name"] = setup[approach]["class_name"]
    save_result = backends["holistic"].save_result
    if output.exists():
        with output.open("rb") as file:
            result = pickle.load(file)
        if (any(result.get(key) != value for key, value in metadata.items())
                or any(not np.array_equal(result.get("xi_matrices", {}).get(name), xi_matrices[name])
                       for name in CLASS_NAMES)):
            raise ValueError(f"Checkpoint settings/code differ: {output}. Choose a new output_file.")
        if result["complete"]:
            print(f"Already complete: {output}", flush=True)
            return result
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        result = {**metadata, "xi_matrices": xi_matrices, "runs": {}, "complete": False}
        save_result(output, result)
    print(f"Run {run_number}/{config['n_independent_runs']}, seed {seed}: "
          f"{algorithm} / {approach} / {experiment}", flush=True)
    parameter = setup["sweep"]["parameter"]
    for level in setup["sweep"]["values"]:
        for formulation in setup["formulations"]:
            key = f"{formulation}/{parameter}={level:g}"
            arguments = (result, key, formulation)
            tail = (setup, hp, xi_matrices, model_sizes, output)
            if algorithm == "qNParEGO":
                backends[algorithm].run_condition(*arguments, parameter, level, *tail, approach)
            elif experiment == "gravity":
                backends[approach].run_condition(*arguments, level, *tail)
            else:
                backends[experiment].run_condition(*arguments, level, *tail, approach)
    result["complete"] = True
    save_result(output, result)
    print(f"Saved {output}", flush=True)
    return result


def save_bundle(path, result):
    """Atomically replace the sole output, staging outside the project."""
    with tempfile.TemporaryDirectory(prefix="cosmic_model_based_save_") as directory:
        temporary = Path(directory) / "bundle.pkl"
        with temporary.open("wb") as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        for attempt in range(7):
            try:
                temporary.replace(path)
                return
            except PermissionError:
                if attempt == 6:
                    raise
                time.sleep(0.05 * 2**attempt)


class BatchProgress:
    """ASCII CMD bar with experiment totals and current BO step."""

    def __init__(self, config, bundle):
        from tqdm import tqdm

        self.config = config
        total = (config["n_independent_runs"] * len(config["hyperparameters"])
                 * len(config["approaches"]) * len(config["experiments"]))
        completed = sum(record["complete"] for record in bundle["experiments"].values())
        self.bar = tqdm(total=total, initial=completed, desc="Experiments",
                        unit="exp", ascii=True, dynamic_ncols=True,
                        mininterval=0.2, file=sys.stderr,
                        bar_format="{desc} {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} "
                                   "[{elapsed}<{remaining}]{postfix}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.bar.close()

    def start(self, number, algorithm, approach, experiment, saved):
        self.bar.set_description_str(f"Run {number}/{self.config['n_independent_runs']} "
                                     f"{algorithm} {approach[0].upper()}-{experiment[:4]}")
        if saved is None:
            self.bar.set_postfix_str("initializing")
        else:
            self.observe(saved)

    def observe(self, result):
        runs = result.get("runs", {})
        if not runs:
            return
        key = next(reversed(runs))
        trajectory = runs[key]
        form, condition = key.split("/", 1)
        label = f"{form}/{condition.rsplit('=', 1)[-1]}"
        hp = result["hyperparameters"]
        candidates = len(trajectory.get("X_unit", []))
        if candidates < hp["n_initial_points"]:
            status = f"init {candidates}/{hp['n_initial_points']}"
        else:
            steps = len(trajectory.get("direction_angle_history", []))
            status = f"step {steps}/{hp['n_iterations']}"
        self.bar.set_postfix_str(f"{label} {status}")

    @contextmanager
    def reporting(self, backends):
        originals = {name: backend.save_result for name, backend in backends.items()}
        try:
            for name, backend in backends.items():
                def save_and_report(path, result, original=originals[name]):
                    original(path, result)
                    self.observe(result)
                backend.save_result = save_and_report
            with redirect_stdout(io.StringIO()):
                yield
        finally:
            for name, backend in backends.items():
                backend.save_result = originals[name]

    def finish(self):
        self.bar.set_postfix_str("saved", refresh=False)
        self.bar.update(1)
        self.bar.refresh()


def run_suite(config, backends, xi_matrices, model_sizes, show_progress=False):
    """Store all runs in one pickle, without touching the individual results."""
    output = (EXPERIMENT_DIR / config["output_file"]).resolve()
    fingerprints = source_fingerprints(backends)
    metadata = {"schema_version": 1, "kind": "model_based_suite", "setup": config,
                "source_sha256": fingerprints}
    if output.exists():
        with output.open("rb") as file:
            bundle = pickle.load(file)
        if any(bundle.get(key) != value for key, value in metadata.items()):
            raise ValueError(f"Settings/code differ from {output}; choose a new output_file.")
        if bundle["complete"]:
            print(f"Already complete: {output}", flush=True)
            return bundle
    else:
        bundle = {**metadata, "experiments": {}, "complete": False}
        save_bundle(output, bundle)

    with BatchProgress(config, bundle) if show_progress else nullcontext() as progress:
        for number in range(1, config["n_independent_runs"] + 1):
            seed = config["base_seed"] + (number - 1) * config["seed_increment"]
            for algorithm in config["hyperparameters"]:
                for approach in config["approaches"]:
                    for experiment in config["experiments"]:
                        key = f"run_{number:02d}_seed_{seed}/{algorithm}/{approach}/{experiment}"
                        saved = bundle["experiments"].get(key)
                        if saved is not None and saved["complete"]:
                            continue
                        if progress:
                            progress.start(number, algorithm, approach, experiment, saved)
                        with tempfile.TemporaryDirectory(prefix="cosmic_model_based_run_") as directory:
                            checkpoint = Path(directory) / "checkpoint.pkl"
                            if saved is not None:
                                with checkpoint.open("wb") as file:
                                    pickle.dump(saved, file, protocol=pickle.HIGHEST_PROTOCOL)
                            try:
                                with progress.reporting(backends) if progress else nullcontext():
                                    result = run_experiment(config, backends, fingerprints, number,
                                                            algorithm, approach, experiment,
                                                            xi_matrices, model_sizes, checkpoint)
                            except (Exception, KeyboardInterrupt):
                                if checkpoint.exists():
                                    with checkpoint.open("rb") as file:
                                        bundle["experiments"][key] = pickle.load(file)
                                    save_bundle(output, bundle)
                                raise
                            if not result["complete"]:
                                raise RuntimeError(f"Incomplete experiment returned: {key}")
                            bundle["experiments"][key] = result
                            save_bundle(output, bundle)
                            if progress:
                                progress.finish()
    bundle["complete"] = True
    save_bundle(output, bundle)
    print(f"All model-based independent runs saved in {output}", flush=True)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=EXPERIMENT_DIR / "setup.json")
    args = parser.parse_args()
    with args.setup.open(encoding="utf-8") as file:
        config = json.load(file)
    validate_setup(config)
    # Set BLAS thread limits before importing numpy, torch, or the optimizers.
    os.environ["OMP_NUM_THREADS"] = str(config["torch_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["torch_num_threads"])
    backends = load_backends()
    import numpy as np
    import torch

    torch.set_num_threads(config["torch_num_threads"])
    base = backends["holistic"]
    if tuple(base.CLASS_NAMES) != CLASS_NAMES:
        raise ValueError("Simulator class order differs from this suite")
    xi_matrices = {name: np.asarray(config["Xi_matrices"][name], dtype=float)
                   for name in CLASS_NAMES}
    model_sizes = np.array([
        base.get_model_size_in_kb(base.Network(*config["common_setup"]["policy_shapes"][name]).float())
        for name in CLASS_NAMES
    ])
    run_suite(config, backends, xi_matrices, model_sizes, show_progress=True)


if __name__ == "__main__":
    main()
