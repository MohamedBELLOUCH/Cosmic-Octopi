"""Run the complete model-free suite later, using the adjacent setup.json.

Calls the existing Scalarized UCB and Scalarized KG runner functions.
No bandit algorithm, simulator, or hypervolume calculation is implemented here.
All settings and calibration matrices come from this suite's JSON. Current
single-run results in the algorithm folders are never read or overwritten.

The only persistent output is model_free_results.pkl beside this script.
Intermediate checkpoints stay in the system temporary directory and are removed.
Run when ready: python -B "Model-free experiments/script.py"
Use --setup PATH for another self-contained configuration.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import hashlib
import importlib
import io
import json
import math
from multiprocessing import get_context
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

# Also applies to spawned evaluation workers, even when -B is omitted.
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


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
    filename = Path(config["output_file"])
    if filename.name != str(filename) or filename.suffix != ".pkl":
        raise ValueError("output_file must be a pickle filename beside this script")
    destination = (HERE / filename).resolve()
    if destination.parent != HERE.resolve():
        raise ValueError("output_file must stay inside Model-free experiments")
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
    output = (HERE / config["output_file"]).resolve()
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
    return setup


def load_backends():
    """Import implementations without invoking their file-writing CLI mains."""
    for directory in DIRECTORIES.values():
        sys.path.insert(0, str(ROOT / directory))
    return {"UCB": importlib.import_module("experiment_runner"),
            "KG": importlib.import_module("kg_runner")}


def source_fingerprints(config, backends):
    hashes = {str(Path(__file__).resolve().relative_to(ROOT)):
              hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for algorithm, method in config["algorithms"].items():
        for approach in config["approaches"]:
            for experiment in config["experiments"]:
                entrypoint = ROOT / method["script_directory"] / f"script_{approach}_{experiment}.py"
                hashes.update(backends[algorithm].source_fingerprints(entrypoint))
    return hashes


def save_bundle(path, result):
    """Stage outside the project, then atomically replace the sole output file."""
    # Atomic replacement requires the staging directory to be on the same volume.
    # Refuse a cross-volume write instead of risking an incomplete existing file.
    with tempfile.TemporaryDirectory(prefix="cosmic_model_free_save_") as directory:
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
    """One ASCII console bar, with per-checkpoint status from the active runner."""

    def __init__(self, config, bundle):
        from tqdm import tqdm

        self.config = config
        self.job_label = ""
        total = (config["n_independent_runs"] * len(config["algorithms"])
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

    def start(self, run_number, algorithm, approach, experiment, saved):
        self.job_label = f"Run {run_number}/{self.config['n_independent_runs']} " \
                         f"{algorithm} {approach[0].upper()}-{experiment[:4]}"
        self.bar.set_description_str(self.job_label)
        self.observe(saved or {"runs": {}})

    def observe(self, result):
        runs = result.get("runs", {})
        if runs:
            key = next(reversed(runs))
            trajectory = runs[key]
            checkpoints = trajectory.get("checkpoint_iterations", [])
            iteration = checkpoints[-1] if checkpoints else 0
            form, condition = key.split("/", 1)
            level = condition.rsplit("=", 1)[-1]
            horizon = self.config["common_bandit"]["n_online_iterations"]
            self.bar.set_postfix_str(f"{form}/{level}: {iteration}/{horizon}")
        else:
            self.bar.set_postfix_str("training")

    def run(self, backend, setup, checkpoint, entrypoint, executor):
        original_save = backend.save_result

        def save_and_report(path, result):
            original_save(path, result)
            self.observe(result)

        # Changes only the in-memory callback; the source file and numerical
        # implementation stay untouched. Always restore it, including on Ctrl+C.
        backend.save_result = save_and_report
        try:
            # Routine runner prints would overwrite the console bar. Exceptions
            # and warnings still go to stderr; no log file is created.
            with redirect_stdout(io.StringIO()):
                return backend.run(setup, checkpoint, entrypoint, executor)
        finally:
            backend.save_result = original_save

    def finish(self):
        self.bar.set_postfix_str("saved", refresh=False)
        self.bar.update(1)
        self.bar.refresh()


def run_suite(config, backends, executor=None, show_progress=False):
    """Collect 120 experiment records into one resumable pickle dictionary."""
    output = (HERE / config["output_file"]).resolve()
    metadata = {"schema_version": 1, "kind": "model_free_suite", "setup": config,
                "source_sha256": source_fingerprints(config, backends)}
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
        for run_number in range(1, config["n_independent_runs"] + 1):
            for algorithm, method in config["algorithms"].items():
                for approach in config["approaches"]:
                    for experiment in config["experiments"]:
                        setup = build_setup(config, run_number, algorithm, approach, experiment)
                        key = f"run_{run_number:02d}_seed_{setup['seed']}/{algorithm}/{approach}/{experiment}"
                        saved = bundle["experiments"].get(key)
                        if saved is not None and saved["complete"]:
                            continue
                        entrypoint = ROOT / method["script_directory"] / f"script_{approach}_{experiment}.py"
                        if progress:
                            progress.start(run_number, algorithm, approach, experiment, saved)
                        else:
                            print(f"Run {run_number}/{config['n_independent_runs']}: "
                                  f"{algorithm} / {approach} / {experiment}, seed={setup['seed']}", flush=True)
                        # Existing runners keep their ordinary checkpoint/resume logic,
                        # but only temporary files receive their per-experiment outputs.
                        with tempfile.TemporaryDirectory(prefix="cosmic_model_free_run_") as directory:
                            checkpoint = Path(directory) / "checkpoint.pkl"
                            if saved is not None:
                                with checkpoint.open("wb") as file:
                                    pickle.dump(saved, file, protocol=pickle.HIGHEST_PROTOCOL)
                            try:
                                if progress:
                                    result = progress.run(backends[algorithm], setup, checkpoint, entrypoint, executor)
                                else:
                                    result = backends[algorithm].run(setup, checkpoint, entrypoint, executor)
                            except (Exception, KeyboardInterrupt):
                                if checkpoint.exists():
                                    with checkpoint.open("rb") as file:
                                        bundle["experiments"][key] = pickle.load(file)
                                    save_bundle(output, bundle)
                                raise
                            if not result["complete"]:
                                raise RuntimeError(f"Runner returned an incomplete experiment: {key}")
                            bundle["experiments"][key] = result
                            save_bundle(output, bundle)
                            if progress:
                                progress.finish()
    bundle["complete"] = True
    save_bundle(output, bundle)
    print(f"All model-free independent runs saved in {output}", flush=True)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setup", type=Path, default=HERE / "setup.json")
    args = parser.parse_args()
    config = json.loads(args.setup.read_text(encoding="utf-8"))
    validate_setup(config)
    os.environ["OMP_NUM_THREADS"] = str(config["torch_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(config["torch_num_threads"])
    backends = load_backends()
    with ProcessPoolExecutor(max_workers=config["evaluation"]["workers"],
                             mp_context=get_context("spawn")) as executor:
        run_suite(config, backends, executor, show_progress=True)


if __name__ == "__main__":
    main()
