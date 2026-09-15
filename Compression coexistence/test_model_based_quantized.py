"""Check quantized routing and held-out preparation using stand-in simulations."""
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import model_based_quantized as common
import holistic_model_based_runner_quantized as holistic
import reductionist_model_based_runner_quantized as reductionist
import evaluate_holistic_model_based_comparison_quantized as holistic_eval
import evaluate_reductionist_model_based_comparison_quantized as reductionist_eval


def worker_snapshot():
    _, setup, matrices, sizes, _, _ = reductionist_eval._worker
    return setup["reference_costs"], matrices["Cheetah"], sizes


def load_baseline(mode):
    path = common.HERE / f"{mode} single fitness margin baseline quantized.py"
    spec = importlib.util.spec_from_file_location(f"test_{mode}_quantized_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelBasedQuantizedTests(unittest.TestCase):
    def setup_for(self, mode):
        return common.load_setup(common.HERE / f"setup_model_based_{mode}_quantized.json")

    def test_optimizer_dispatch_uses_quantized_payloads_for_all_four_variants(self):
        for mode, runner in (("holistic", holistic), ("reductionist", reductionist)):
            setup = self.setup_for(mode)
            xi, sizes = common.simulator_inputs(setup)
            for algorithm in ("MORBO", "qNParEGO"):
                hp = setup["hyperparameters"][algorithm]
                calls = []

                def fake_condition(*args):
                    result, _, formulation = args[:3]
                    position = 6 if algorithm == "MORBO" else 7
                    self.assertIs(args[position], xi)
                    np.testing.assert_array_equal(args[position + 1], sizes)
                    if algorithm == "qNParEGO":
                        self.assertEqual(args[-1], mode)
                    calls.append(formulation)
                    total = hp["n_initial_points"] + hp["n_iterations"] * hp["batch_size"]
                    result["runs"][formulation] = {"X_unit": np.zeros((total, hp["dimension"])),
                                                   "direction_angle_history": np.zeros(hp["n_iterations"])}

                backend = SimpleNamespace(run_condition=fake_condition, save_result=object())
                original_save = backend.save_result
                for form in ("RTS", "STR"):
                    output = common.HERE / setup["comparison_output_files"][algorithm][form]
                    with patch.object(runner, "save_checkpoint"), \
                         patch.object(runner, "extract_pareto_set", return_value={"pareto_indices": [0]}):
                        record = runner.run_formulation(setup, algorithm, form, None, backend,
                                                        xi, sizes, output, {}, show_progress=False)
                    self.assertEqual(record["compression"], "int8")
                    self.assertIs(backend.save_result, original_save)
                self.assertEqual(calls, ["RTS", "STR"])

    def test_both_baseline_searches_receive_quantized_inputs(self):
        for mode in ("holistic", "reductionist"):
            setup = self.setup_for(mode)
            xi, sizes = common.simulator_inputs(setup)
            baseline = load_baseline(mode)
            helpers = baseline if mode == "holistic" else baseline.load_search_helpers()
            calls = []

            def fake_evaluate(cfg, matrices, payloads, sequence, approach, seed):
                np.testing.assert_array_equal(payloads, sizes)
                np.testing.assert_array_equal(matrices["Cheetah"], xi["Cheetah"])
                self.assertEqual(approach, mode)
                self.assertTrue(np.all(sequence == sequence[0]))
                calls.append(sequence[0])
                return np.array([sizes[0] * 10, .5])

            simulator = SimpleNamespace(CLASS_NAMES=common.CLASSES, evaluate_sequence=fake_evaluate)
            low, high = setup["fitness_margin_bounds"]["Cheetah"]
            with patch.object(helpers, "load_simulator", return_value=simulator), \
                 patch.object(helpers, "optimize_surrogates", return_value={"pareto_margins": np.array([low, high])}), \
                 patch.object(helpers, "save_result"), patch.object(sys, "argv", ["test"]):
                if mode == "reductionist":
                    with patch.object(baseline, "load_search_helpers", return_value=helpers):
                        baseline.main()
                else:
                    baseline.main()
            np.testing.assert_allclose(calls, np.linspace(0, 1, 10) if mode == "holistic" else np.linspace(low, high, 10))

    def test_preparation_samples_random_baseline_and_reports_six_comparable_values(self):
        for mode, evaluator in (("holistic", holistic_eval), ("reductionist", reductionist_eval)):
            setup = self.setup_for(mode)
            _, sizes = common.simulator_inputs(setup)
            fake_files = {}
            for label, algorithm, form in evaluator.GROUPS:
                baseline = form == "standard"
                filename = setup["output_file"] if baseline else setup["comparison_output_files"][algorithm][form]
                record = {"complete": True, "setup": setup, "source_sha256": {},
                          "model_sizes_kib": dict(zip(common.CLASSES, sizes))}
                if baseline:
                    low, high = setup["fitness_margin_bounds"]["Cheetah"]
                    record.update(kind=f"{mode}_constant_fitness_margin_pareto_set",
                                  pareto_margins=np.array([low, high]), pareto_unit_margins=np.array([0., 1.]),
                                  training={"scenario_seeds": [1, 2]})
                else:
                    hp = setup["hyperparameters"][algorithm]
                    count = hp["n_initial_points"] + hp["n_iterations"] * hp["batch_size"]
                    x = np.zeros((count, setup[mode]["horizon"]))
                    record.update(algorithm=algorithm, formulation=form, hyperparameters=hp,
                                  pareto_unit_sequences=x[:1], pareto_indices=[0],
                                  runs={form: {"X_unit": x, "direction_angle_history": np.zeros(hp["n_iterations"]),
                                               "scenario_seeds": [3, 4]}})
                fake_files[common.HERE / filename] = pickle.dumps(record)
            original_open = Path.open

            def fake_open(path, *args, **kwargs):
                return io.BytesIO(fake_files[path]) if path in fake_files else original_open(path, *args, **kwargs)

            with patch.object(Path, "open", fake_open):
                metadata, groups, sequences = evaluator.prepare_inputs()
            baseline = groups["Random threshold"]["unit_sequences"]
            self.assertEqual(baseline.shape, (20, setup[mode]["horizon"]))
            self.assertTrue(np.all(np.ptp(baseline, axis=1) > 0))
            self.assertEqual(metadata["scenario_seeds"], common.read_standard(setup)["scenario_seeds"])
            result = {**metadata, "groups": groups, "complete": True,
                      "cost_realizations": np.full((len(sequences), 10, 2), .25) * np.asarray(setup["reference_costs"])}
            result["summary"] = evaluator.comparison_statistics(result)
            rows = common.compare_standard(result)
            self.assertEqual(len(rows), 6)
            self.assertEqual({r["formulation"] for r in rows if r["algorithm"] == "Random threshold"}, {"RTS", "STR"})
            changed = deepcopy(setup)
            changed["reference_costs"][0] /= 4
            with self.assertRaisesRegex(ValueError, "reference_costs"):
                common.check_standard_settings(changed)

    def test_windows_worker_initialization_preserves_quantized_sizes(self):
        setup = self.setup_for("reductionist")
        xi, sizes = common.simulator_inputs(setup)
        with ProcessPoolExecutor(max_workers=1, initializer=reductionist_eval.worker_init,
                initargs=(setup, np.array([[0., 1.]]), [123], dict(zip(common.CLASSES, sizes)))) as pool:
            reference, matrix, received_sizes = pool.submit(worker_snapshot).result(timeout=60)
        self.assertEqual(reference, setup["reference_costs"])
        np.testing.assert_array_equal(matrix, xi["Cheetah"])
        np.testing.assert_array_equal(received_sizes, sizes)


if __name__ == "__main__":
    unittest.main(verbosity=2)
