"""Quantized inputs and standard-case comparison for simulated bandit runs."""
from functools import lru_cache
import hashlib
import importlib.util
import json
from pathlib import Path
import pickle
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
CLASSES = ("Cheetah", "Ant", "Leg", "Humanoid")


def load_quantized_setup(path):
    setup = json.loads(Path(path).read_text(encoding="utf-8"))
    if "Xi_matrices" in setup:
        raise ValueError("Read Xi from the quantized file; do not embed stale coefficients")
    xi_path = HERE / setup["xi_matrix_file"]
    if xi_path.resolve().parent != HERE or not xi_path.is_file():
        raise ValueError("Xi must be an existing file in Compression coexistence")
    source = xi_path.read_bytes()
    matrices = json.loads(source)
    if set(matrices) != set(CLASSES):
        raise ValueError("The quantized Xi file must map all four class names")
    for value in matrices.values():
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError("Each quantized Xi must be a finite 4x4 matrix")
    setup["Xi_matrices"] = matrices  # Freeze the actual input in result/worker setup.
    setup["xi_matrix_sha256"] = hashlib.sha256(source).hexdigest()
    q = setup["quantization"]
    if (q["bits"] != 8 or q["scheme"] != "symmetric_per_tensor"
            or q["payload_size_rule"] != "fixed_serialized_representative_model_per_class"
            or q["codec_file"] != "utils quantized.py"):
        raise ValueError("Unsupported quantized model-size convention")
    check_standard_settings(setup)
    return setup


def read_standard(setup):
    path = (HERE / setup["standard_comparison_file"]).resolve()
    if path.parent != HERE.parent / "Model-free baseline":
        raise ValueError("Expected the saved standard Model-free baseline comparison")
    with path.open("rb") as stream:
        standard = pickle.load(stream)
    if not standard["complete"]:
        raise ValueError("The standard comparison is incomplete")
    return standard


def check_standard_settings(setup):
    standard = read_standard(setup)["setup"]
    keys = ["seed", "simulated_rewards", "n_independent_runs", "gravity_mean", "ou_substeps",
            "system_dynamics_parameters", "hyper_parameters", "fitness_margin_bounds",
            "policy_shapes", "approach", setup["approach"], "reference_costs",
            "objective_names", "objective_units", "objective_normalization",
            "n_online_iterations", "n_arms", "n_scalarization_directions",
            "formulations", "algorithm_settings", "feedback", "evaluation"]
    for key in keys:
        if setup[key] != standard[key]:
            raise ValueError(f"Quantized and saved standard comparison differ in {key}")


@lru_cache(maxsize=8)
def _measure_payloads(configuration):
    import torch
    config = json.loads(configuration)
    torch.set_num_threads(config["torch_num_threads"])
    spec = importlib.util.spec_from_file_location("_model_free_int8_codec", HERE / "utils quantized.py")
    codec = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(codec)
    measurements = {}
    # The simulator has no trained tensor state. Serialize one seeded Network per
    # class and use its complete payload size for each simulated contribution.
    # Preserve RNG state so measuring communication does not change reward draws.
    with torch.random.fork_rng(devices=[]):
        for index, name in enumerate(CLASSES):
            torch.random.default_generator.manual_seed(config["measurement_seed"] + index)
            model = codec.Network(*config["policy_shapes"][name]).float()
            payload = codec.quantize_model(model)
            decoded = codec.dequantize_model(payload)
            raw_bytes = sum(t.numel() * t.element_size() for t in model.state_dict().values())
            if len(payload) >= raw_bytes:
                raise ValueError(f"{name}: quantized payload is not smaller")
            if list(decoded.state_dict()) != list(model.state_dict()):
                raise ValueError(f"{name}: invalid codec round trip")
            measurements[name] = {"quantized_bytes": len(payload), "float32_bytes": raw_bytes,
                                  "quantized_kib": len(payload) / 1024., "float32_kib": raw_bytes / 1024.,
                                  "reduction_fraction": 1 - len(payload) / raw_bytes,
                                  "payload_sha256": hashlib.sha256(payload).hexdigest()}
    return measurements


def payload_measurements(setup):
    config = {"policy_shapes": setup["policy_shapes"], "torch_num_threads": setup["torch_num_threads"],
              "measurement_seed": setup["quantization"]["measurement_seed"]}
    return _measure_payloads(json.dumps(config, sort_keys=True))


def simulator_inputs(setup):
    sizes = payload_measurements(setup)
    return ({name: np.asarray(setup["Xi_matrices"][name], dtype=float) for name in CLASSES},
            np.asarray([sizes[name]["quantized_kib"] for name in CLASSES]))


def extra_source_paths(setup_path):
    setup = json.loads(Path(setup_path).read_text(encoding="utf-8"))
    return [Path(__file__), HERE / "utils quantized.py", HERE / setup["xi_matrix_file"],
            (HERE / setup["standard_comparison_file"]).resolve()]


def compare_standard(result):
    setup = result["setup"]
    check_standard_settings(setup)
    standard = read_standard(setup)
    lookup = {(row["algorithm"], row["formulation"]): row for row in standard["summary"]}
    rows = []
    for row in result["summary"]:
        previous = lookup[row["algorithm"], row["formulation"]]
        before, after = float(previous["hypervolume"]), float(row["hypervolume"])
        rows.append({"algorithm": row["algorithm"], "formulation": row["formulation"],
                     "standard_hypervolume": before, "quantized_hypervolume": after,
                     "difference": after - before,
                     "relative_change_percent": 100 * (after / before - 1) if before else None})
    return rows
