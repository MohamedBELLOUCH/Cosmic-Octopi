"""Inputs, provenance and paired reporting for quantized model-based runs."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from model_free_quantized import payload_measurements, simulator_inputs

HERE = Path(__file__).resolve().parent
CLASSES = ("Cheetah", "Ant", "Leg", "Humanoid")


def read_standard(setup, random_baseline=False):
    key = "standard_random_baseline_file" if random_baseline else "standard_comparison_file"
    path = (HERE / setup[key]).resolve()
    if path.parent != HERE.parent / "Model-based baseline":
        raise ValueError("Expected a saved standard Model-based baseline result")
    with path.open("rb") as stream:
        result = pickle.load(stream)
    if not result["complete"]:
        raise ValueError(f"Standard result is incomplete: {path.name}")
    return result


def check_standard_settings(setup):
    standard = read_standard(setup)
    baseline = read_standard(setup, True)
    ignore = {"Xi_matrices", "output_file", "comparison_output_files", "description"}
    for key, value in standard["setup"].items():
        if key not in ignore and setup[key] != value:
            raise ValueError(f"Quantized/standard model-based settings differ in {key}")
    if standard["setup"] != baseline["setup"]:
        raise ValueError("Standard optimizers and random baseline use different settings")
    if setup["held_out_scenario_seeds"] != standard["scenario_seeds"] or standard["scenario_seeds"] != baseline["scenario_seeds"]:
        raise ValueError("Preserve the actual standard held-out seeds")
    if (setup["baseline_sampling"]["n_sequences"] != baseline["config"]["n_sequences"]
            or setup["baseline_sampling"]["seed"] != baseline["sampling_seed"]
            or setup["baseline_sampling"]["rule"] != "independent_uniform_with_replacement_at_every_iteration"):
        raise ValueError("Preserve the standard random-threshold sequence sampling")


def load_setup(path):
    setup = json.loads(Path(path).read_text(encoding="utf-8"))
    if "Xi_matrices" in setup:
        raise ValueError("Read Xi from the quantized file instead of embedding coefficients")
    xi_path = (HERE / setup["xi_matrix_file"]).resolve()
    if xi_path.parent != HERE:
        raise ValueError("Use the Xi file inside Compression coexistence")
    raw = xi_path.read_bytes()
    matrices = json.loads(raw)
    if set(matrices) != set(CLASSES):
        raise ValueError("Xi must contain all four classes")
    if any(np.asarray(m).shape != (4, 4) or not np.isfinite(m).all() for m in matrices.values()):
        raise ValueError("Xi coefficients must be finite 4x4 matrices")
    setup["Xi_matrices"] = matrices
    setup["xi_matrix_sha256"] = hashlib.sha256(raw).hexdigest()
    q = setup["quantization"]
    if (q["bits"] != 8 or q["scheme"] != "symmetric_per_tensor"
            or q["codec_file"] != "utils quantized.py"
            or q["payload_size_rule"] != "fixed_serialized_representative_model_per_class"):
        raise ValueError("Unsupported quantized transport convention")
    check_standard_settings(setup)
    files = [setup["output_file"], setup["comparison_cache_file"],
             *(name for group in setup["comparison_output_files"].values() for name in group.values())]
    if len(set(files)) != len(files) or any(Path(p).name != p or not p.endswith("quantized.pkl") for p in files):
        raise ValueError("Use distinct local quantized.pkl output filenames")
    return setup


def extra_source_paths(setup_path):
    raw = json.loads(Path(setup_path).read_text(encoding="utf-8"))
    return [Path(__file__), HERE / "model_free_quantized.py", HERE / "utils quantized.py",
            Path(setup_path).resolve(), HERE / raw["xi_matrix_file"],
            (HERE / raw["standard_comparison_file"]).resolve(),
            (HERE / raw["standard_random_baseline_file"]).resolve()]


def sample_baseline_sequences(margins, bounds, setup, horizon):
    support = np.unique(np.asarray(margins, dtype=float))
    low, high = bounds
    if not len(support) or not np.isfinite(support).all() or np.any((support < low) | (support > high)):
        raise ValueError("Invalid learned threshold support")
    cfg = setup["baseline_sampling"]
    draws = np.random.default_rng(cfg["seed"]).integers(len(support), size=(cfg["n_sequences"], horizon))
    return (support[draws] - low) / (high - low)


def comparison_statistics(result, exact_hypervolume, core):
    if not result["complete"] or not np.isfinite(result["cost_realizations"]).all():
        raise ValueError("Held-out evaluation is incomplete")
    cfg = result["evaluation_settings"]
    directions = core.unit_directions(cfg["hypervolume_directions"], cfg["direction_endpoint_epsilon"])
    reference = np.asarray(result["setup"]["reference_costs"])
    rows = []
    for label, group in result["groups"].items():
        costs = result["cost_realizations"][group["sequence_indices"]]
        means = costs.mean(axis=1)
        exact = exact_hypervolume(means, reference)
        forms = ("RTS", "STR") if group["algorithm"] == "Random threshold" else (group["formulation"],)
        for form in forms:
            polar = core.estimated_hypervolume(costs, form, directions, reference)
            rows.append({"algorithm": group["algorithm"], "formulation": form,
                         "label": f"{group['algorithm']} {form}", "n_solutions": len(costs),
                         "n_realizations": costs.shape[1], "mean_costs": means,
                         "hypervolume": exact if form == "RTS" else polar,
                         "standard_hypervolume_of_means": exact, "quadrature_hypervolume": polar})
    return rows


def compare_standard(result):
    setup = result["setup"]
    check_standard_settings(setup)
    old_rows = read_standard(setup)["summary"] + read_standard(setup, True)["summary"]
    lookup = {(r["algorithm"], r["formulation"]): float(r["hypervolume"]) for r in old_rows}
    comparisons = []
    for row in result["summary"]:
        old = lookup[row["algorithm"], row["formulation"]]
        new = float(row["hypervolume"])
        comparisons.append({"algorithm": row["algorithm"], "formulation": row["formulation"],
                            "standard_hypervolume": old, "quantized_hypervolume": new,
                            "difference": new - old,
                            "relative_change_percent": 100 * (new / old - 1) if old else None})
    return comparisons
