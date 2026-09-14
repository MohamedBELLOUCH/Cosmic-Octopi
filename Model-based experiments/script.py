"""Run all model-based experiments from the adjacent, self-contained setup.json.

Launch later from any directory:
    python "<project>/Model-based experiments/script.py"
An alternative configuration can be selected with --setup PATH.

This orchestrator calls the existing MORBO and qNParEGO run_condition functions;
it does not implement simulation, acquisition, scalarization, or hypervolume.
Each independent run starts a fresh optimization with a distinct seed. Seeds
are paired across methods and conditions within that run for comparison.

Results are checkpointed by the existing runners, separately for every run,
algorithm, approach, and experiment. Rerunning resumes compatible checkpoints.
The original single-run result files and configuration files are never changed.
The notebook continues to show those original results until updated separately.
"""

import argparse
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import sys


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
    output = (EXPERIMENT_DIR / config["output_directory"]).resolve()
    if output == EXPERIMENT_DIR or not output.is_relative_to(EXPERIMENT_DIR):
        raise ValueError("output_directory must be a subdirectory of Model-based experiments")
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
        for name in ("n_initial_points", "batch_size", "n_iterations", "mc_samples",
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
    for name in ("n_initial_points", "batch_size", "n_iterations", "hypervolume_directions",
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
                   experiment, xi_matrices, model_sizes):
    """Provide inline settings and isolated checkpoints to the existing runners."""
    import numpy as np

    seed = config["base_seed"] + (run_number - 1) * config["seed_increment"]
    setup = build_setup(config, approach, experiment, seed)
    hp = deepcopy(config["hyperparameters"][algorithm])
    hp["dimension"] = setup[approach]["horizon"]
    output = (EXPERIMENT_DIR / config["output_directory"]
              / f"run_{run_number:02d}_seed_{seed}" / algorithm
              / f"result_{approach}_{experiment}.pkl")
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
            raise ValueError(f"Checkpoint settings/code differ: {output}. Choose a new output_directory.")
        if result["complete"]:
            print(f"Already complete: {output}", flush=True)
            return
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
    fingerprints = source_fingerprints(backends)
    for run_number in range(1, config["n_independent_runs"] + 1):
        for algorithm in config["hyperparameters"]:
            for approach in config["approaches"]:
                for experiment in config["experiments"]:
                    run_experiment(config, backends, fingerprints, run_number, algorithm,
                                   approach, experiment, xi_matrices, model_sizes)
    print("All model-based experiments are complete.", flush=True)


if __name__ == "__main__":
    main()
