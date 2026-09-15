"""Check dispatch, stochastic Pareto selection and resume without paper runs.

The real MORBO/qNParEGO run loops are exercised with cheap deterministic
stand-ins for simulation and GP acquisition. All results go to system Temp.
"""

import sys

sys.dont_write_bytecode = True

from copy import deepcopy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holistic_model_based_runner as runner


class HolisticComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.setup = json.loads((runner.EXPERIMENT_DIR / "setup_model_free_holistic.json").read_text())
        cls.core, cls.morbo = runner.load_backends("MORBO")
        _, cls.qnparego = runner.load_backends("qNParEGO")
        torch.set_num_threads(1)

    def small_setup(self):
        setup = deepcopy(self.setup)
        setup["holistic"]["horizon"] = 3
        for hp in setup["hyperparameters"].values():
            hp.update(dimension=3, n_initial_points=2, n_iterations=2, batch_size=1)
        return setup

    def test_shared_setup_and_output_collisions(self):
        runner.validate_setup(self.setup)
        bad = deepcopy(self.setup)
        bad["comparison_output_files"]["MORBO"]["RTS"] = bad["output_file"].upper().replace(".PKL", ".pkl")
        with self.assertRaisesRegex(ValueError, "distinct"):
            runner.validate_setup(bad)
        bad = deepcopy(self.setup)
        bad["hyperparameters"]["qNParEGO"]["n_iterations"] += 1
        with self.assertRaisesRegex(ValueError, "matched"):
            runner.validate_setup(bad)

    def test_str_keeps_winner_that_mean_cost_filter_would_remove(self):
        setup = self.small_setup()
        setup["reference_costs"] = [10, 10]
        hp = setup["hyperparameters"]["MORBO"]
        run = {
            "X_unit": np.array([[0, 0.5, 1], [0.25, 0.5, 0.75]]),
            "cost_realizations": np.array([[[0, 10], [10, 0]], [[6, 6], [6, 6]]], dtype=float),
            "scenario_seeds": np.array([[1, 2], [3, 4]], dtype=np.uint64),
        }
        rts = runner.extract_pareto_set(run, "RTS", setup, hp, self.core)
        str_result = runner.extract_pareto_set(run, "STR", setup, hp, self.core)
        np.testing.assert_array_equal(rts["pareto_indices"], [0])
        np.testing.assert_array_equal(str_result["pareto_indices"], [1])
        np.testing.assert_array_equal(rts["pareto_margin_sequences"], [[-10, 5, 20]])
        self.assertGreater(str_result["training_hypervolume"], 0)
        self.assertIsNone(str_result["held_out_evaluation"])

    def test_pareto_filter_preserves_hypervolume_with_ties(self):
        costs = np.array([[1, 2], [2, 1], [2, 2], [1, 2]])
        np.testing.assert_array_equal(runner.nondominated_indices(costs), [0, 1, 3])
        rng = np.random.default_rng(10)
        setup = self.small_setup()
        hp = setup["hyperparameters"]["MORBO"]
        observations = rng.random((15, 2, 2)) * np.asarray(setup["reference_costs"])
        run = {"X_unit": rng.random((15, 3)), "cost_realizations": observations,
               "scenario_seeds": np.arange(30).reshape(15, 2)}
        for formulation in ("RTS", "STR"):
            result = runner.extract_pareto_set(run, formulation, setup, hp, self.core)
            full_hv = self.core.estimated_hypervolume(observations, formulation,
                       result["hypervolume_directions"], np.asarray(setup["reference_costs"]))
            self.assertAlmostEqual(result["training_hypervolume"], full_hv, places=13)

    def test_real_run_loops_resume_and_do_not_touch_existing_outputs(self):
        setup = self.small_setup()
        runner.validate_setup(setup)
        xi = {name: np.asarray(setup["Xi_matrices"][name]) for name in runner.CLASS_NAMES}
        model_sizes = np.ones(4)

        def fake_acquisition(run, formulation, angle, hp, reference):
            return torch.rand((hp["batch_size"], hp["dimension"]), dtype=torch.float64).numpy()

        for algorithm, backend in (("MORBO", self.morbo), ("qNParEGO", self.qnparego)):
            evaluator_name = "evaluate_candidate" if algorithm == "MORBO" else "evaluate_holistic_candidate"
            fingerprints = runner.source_fingerprints(algorithm, runner.EXPERIMENT_DIR / f"holistic {algorithm}.py")
            for formulation in ("RTS", "STR"):
                with self.subTest(algorithm=algorithm, formulation=formulation), tempfile.TemporaryDirectory(prefix="cosmic_baseline_test_") as directory:
                    folder = Path(directory)
                    sentinel = folder / "existing.pkl"
                    sentinel.write_bytes(b"existing result must not change")
                    baseline_output = folder / "uninterrupted.pkl"
                    resumed_output = folder / "resumed.pkl"
                    original_save = backend.save_result
                    calls = {"count": 0, "interrupt_at": None}

                    def fake_evaluate(x, scenario_rng, config, matrices, sizes):
                        calls["count"] += 1
                        if calls["count"] == calls["interrupt_at"]:
                            raise KeyboardInterrupt()
                        self.assertEqual(config["system_dynamics_parameters"], setup["system_dynamics_parameters"])
                        self.assertEqual(config["holistic"], setup["holistic"])
                        seeds = scenario_rng.integers(0, 2**63, size=2, dtype=np.uint64)
                        noise = np.random.default_rng(int(seeds[0])).uniform(0, 0.1, (2, 2))
                        costs = (np.array([x.mean(), 1 - x.mean()]) + noise) * np.array([1e6, 100])
                        return costs, seeds

                    def execute(output):
                        return runner.run_formulation(setup, algorithm, formulation, self.core,
                            backend, xi, model_sizes, output, fingerprints, show_progress=False)

                    with patch.object(backend, "acquisition_candidates", side_effect=fake_acquisition), patch.object(backend, evaluator_name, side_effect=fake_evaluate):
                        uninterrupted = execute(baseline_output)
                        self.assertEqual(calls["count"], 4)
                        calls.update(count=0, interrupt_at=3)
                        with self.assertRaises(KeyboardInterrupt):
                            execute(resumed_output)
                        self.assertIs(backend.save_result, original_save)
                        with resumed_output.open("rb") as file:
                            partial = pickle.load(file)
                        self.assertFalse(partial["complete"])
                        self.assertEqual(len(partial["runs"][formulation]["X_unit"]), 2)
                        self.assertIsNotNone(partial["runs"][formulation]["pending_X"])
                        calls.update(count=0, interrupt_at=None)
                        resumed = execute(resumed_output)
                        self.assertEqual(calls["count"], 2)
                        self.assertTrue(resumed["complete"])
                        before = resumed_output.read_bytes()
                        execute(resumed_output)
                        self.assertEqual(calls["count"], 2)
                        self.assertEqual(resumed_output.read_bytes(), before)
                    for key in ("X_unit", "cost_realizations", "scenario_seeds", "direction_angle_history", "hypervolume_history", "evaluation_counts"):
                        np.testing.assert_array_equal(uninterrupted["runs"][formulation][key], resumed["runs"][formulation][key])
                    for key in ("pareto_indices", "pareto_unit_sequences", "pareto_margin_sequences", "pareto_directional_scores"):
                        np.testing.assert_array_equal(uninterrupted[key], resumed[key])
                    self.assertEqual(resumed["runs"][formulation]["evaluation_counts"], [2, 4])
                    self.assertEqual(sentinel.read_bytes(), b"existing result must not change")
                    self.assertEqual({p.name for p in folder.iterdir()}, {"existing.pkl", "uninterrupted.pkl", "resumed.pkl"})
                    self.assertIs(backend.save_result, original_save)
                    changed = deepcopy(setup)
                    changed["system_dynamics_parameters"]["gamma_heter"] = 0.8
                    with self.assertRaisesRegex(ValueError, "Settings, source, or versions"):
                        runner.run_formulation(changed, algorithm, formulation, self.core,
                            backend, xi, model_sizes, resumed_output, fingerprints, show_progress=False)


if __name__ == "__main__":
    unittest.main()
