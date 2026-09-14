"""Run the complete model-free suite later, using the adjacent setup.json.

Calls the existing Scalarized UCB and Scalarized KG scripts in fresh processes.
No bandit algorithm, simulator, or hypervolume calculation is implemented here.
All settings and calibration matrices come from this suite's JSON. Current
single-run results in the algorithm folders are never read or overwritten.

Run when ready: python "Model-free experiments/script.py"
Use --setup PATH for another self-contained configuration.
"""

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DIRECTORIES = {"UCB": "Scalarized UCB", "KG": "Scalarized Knowledge Gradient"}
SWEEPS = {"gravity": "gamma_grav", "synchronization": "gamma_sync", "heterogeneity": "gamma_heter"}
CLASSES = ("Cheetah", "Ant", "Leg", "Humanoid")


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def validate_setup(config):
    """Validate the suite and all referenced entrypoints before writing anything."""
    if config["schema_version"] != 1:
        raise ValueError("Unsupported suite schema")
    for name in ("n_independent_runs", "seed_increment", "evaluation_seed_increment", "torch_num_threads"):
        positive_integer(config[name], name)
    for base, increment in (("base_seed", "seed_increment"), ("evaluation_base_seed", "evaluation_seed_increment")):
        last = config[base] + (config["n_independent_runs"] - 1) * config[increment]
        if type(config[base]) is not int or not 0 <= config[base] <= last < 2**32:
            raise ValueError("Distinct run seeds must be in [0, 2**32)")
    destination = (HERE / config["output_directory"]).resolve()
    if destination == HERE or not destination.is_relative_to(HERE):
        raise ValueError("output_directory must be a subdirectory of Model-free experiments")
    if set(config["algorithms"]) != set(DIRECTORIES):
        raise ValueError("Supply UCB and KG configurations")
    if set(config["approaches"]) != {"holistic", "reductionist"} or set(config["experiments"]) != set(SWEEPS):
        raise ValueError("Supply both approaches and all three sweeps")
    b = config["common_bandit"]
    for key in ("n_arms", "n_scalarization_directions", "n_online_iterations"):
        positive_integer(b[key], key)
    if b["n_arms"] < 2:
        raise ValueError("At least two arms are required")
    if (b["cost_reference"] != [-0.01, -0.01]
            or b["observation_normalization"] != "increment_divided_by_maximum_possible_increment"):
        raise ValueError("Both algorithms must use the shared normalized-cost convention")
    common = config["common_setup"]
    if common["simulated_rewards"] is not True or common["formulations"] != ["RTS", "STR"]:
        raise ValueError("This suite uses simulated rewards and both RTS and STR")
    positive_integer(common["ou_substeps"], "ou_substeps")
    if set(config["Xi_matrices"]) != set(CLASSES):
        raise ValueError("Embed Xi matrices for all four classes")
    for name, matrix in config["Xi_matrices"].items():
        if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                or not all(math.isfinite(x) for row in matrix for x in row)):
            raise ValueError(f"{name}: Xi must be a finite 4x4 matrix")
    for algorithm, settings in config["algorithms"].items():
        if settings["script_directory"] != DIRECTORIES[algorithm]:
            raise ValueError("Unexpected algorithm script directory")
        if set(b) & set(settings["bandit"]):
            raise ValueError("Algorithm settings must not override shared bandit settings")
        initial_pulls = settings["bandit"].get("initial_pulls_per_pair", 1)
        positive_integer(initial_pulls, "initial_pulls_per_pair")
        if b["n_online_iterations"] <= initial_pulls * b["n_arms"] * b["n_scalarization_directions"]:
            raise ValueError("Leave adaptive iterations after each algorithm's initialization")
        for approach in config["approaches"]:
            for experiment in config["experiments"]:
                entrypoint = ROOT / settings["script_directory"] / f"script_{approach}_{experiment}.py"
                if not entrypoint.is_file():
                    raise FileNotFoundError(entrypoint)
    for approach, settings in config["approaches"].items():
        positive_integer(settings[approach]["horizon"], f"{approach} evaluation horizon")
        if (len(settings["reference_costs"]) != 2
                or not all(math.isfinite(x) and x > 0 for x in settings["reference_costs"])):
            raise ValueError("Supply fixed positive total-cost references")
        interval = settings["arm_interval"]
        if len(interval) != 2 or not all(map(math.isfinite, interval)) or interval[0] >= interval[1]:
            raise ValueError("Supply increasing finite arm intervals")
        if approach == "holistic" and interval != [0, 1]:
            raise ValueError("Holistic arms must span [0,1]")
        if approach == "reductionist":
            name = settings[approach]["class_name"]
            if name != "Cheetah" or interval != common["fitness_margin_bounds"][name]:
                raise ValueError("Reductionist arms must match the Cheetah margin interval")
    for name, settings in config["experiments"].items():
        sweep = settings["sweep"]
        if sweep["parameter"] != SWEEPS[name]:
            raise ValueError("Sweep parameter does not match the experiment")
        values = sweep["values"]
        if not values or len(set(values)) != len(values) or not all(math.isfinite(x) and x >= 0 for x in values):
            raise ValueError("Sweep levels must be distinct, finite, and nonnegative")
        if name == "gravity" and min(values) <= 0:
            raise ValueError("Gravity variances must be positive")
        if name == "synchronization" and not all(0 < x <= 1 for x in values):
            raise ValueError("Synchronization levels must be in (0,1]")
    e = config["evaluation"]
    for key in ("workers", "n_random_sequences", "n_realizations_per_sequence", "hypervolume_directions", "checkpoint_every"):
        positive_integer(e[key], key)
    if e["n_realizations_per_sequence"] < 2:
        raise ValueError("RTS/STR evaluation requires at least two held-out realizations")


def build_setup(config, run_number, algorithm, approach, experiment):
    """Materialize a complete child configuration, with run-specific paired seeds."""
    seed = config["base_seed"] + (run_number - 1) * config["seed_increment"]
    evaluation_seed = config["evaluation_base_seed"] + (run_number - 1) * config["evaluation_seed_increment"]
    method = config["algorithms"][algorithm]
    approach_settings = config["approaches"][approach]
    destination = (HERE / config["output_directory"] / f"run_{run_number:02d}_seed_{seed}" / algorithm).resolve()
    output = destination / f"result_{approach}_{experiment}.pkl"
    setup = deepcopy(config["common_setup"])
    setup.update(deepcopy(config["experiments"][experiment]))
    setup.update({approach: deepcopy(approach_settings[approach]),
                  "reference_costs": deepcopy(approach_settings["reference_costs"]),
                  "algorithm": f"Scalarized {algorithm}", "approach": approach, "experiment": experiment,
                  "implementation": method["implementation"], "seed": seed, "n_independent_runs": 1,
                  "torch_num_threads": config["torch_num_threads"], "output_file": str(output),
                  "Xi_matrices": deepcopy(config["Xi_matrices"]),
                  "comparison_note": config["comparison_note"], "evaluation": deepcopy(config["evaluation"])})
    setup["evaluation"]["seed"] = evaluation_seed
    setup["bandit"] = {**deepcopy(config["common_bandit"]), **deepcopy(method["bandit"]),
                       "arm_interval": list(approach_settings["arm_interval"])}
    if experiment == "gravity":
        setup["gravity_variances"] = list(setup["sweep"]["values"])
    if "sources" in method:
        setup["sources"] = deepcopy(method["sources"])
    setup["suite"] = {
        "independent_run": run_number,
        "n_independent_runs": config["n_independent_runs"],
        "setup_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    return setup, destination / f"setup_{approach}_{experiment}.json"


def save_compatible_setup(path, setup):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != setup:
            raise ValueError(f"Batch settings changed: {path}. Choose a new output_directory.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(setup, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=HERE / "setup.json")
    args = parser.parse_args()
    config = json.loads(args.setup.read_text(encoding="utf-8"))
    validate_setup(config)
    env = dict(os.environ, OMP_NUM_THREADS=str(config["torch_num_threads"]),
               MKL_NUM_THREADS=str(config["torch_num_threads"]))
    for run_number in range(1, config["n_independent_runs"] + 1):
        for algorithm, method in config["algorithms"].items():
            for approach in config["approaches"]:
                for experiment in config["experiments"]:
                    setup, path = build_setup(config, run_number, algorithm, approach, experiment)
                    save_compatible_setup(path, setup)
                    entrypoint = ROOT / method["script_directory"] / f"script_{approach}_{experiment}.py"
                    print(f"Run {run_number}/{config['n_independent_runs']}: "
                          f"{algorithm} / {approach} / {experiment}, seed={setup['seed']}", flush=True)
                    subprocess.run([sys.executable, str(entrypoint), "--setup", str(path)],
                                   cwd=ROOT, env=env, check=True)
    print("All model-free independent runs are complete.", flush=True)


if __name__ == "__main__":
    main()
