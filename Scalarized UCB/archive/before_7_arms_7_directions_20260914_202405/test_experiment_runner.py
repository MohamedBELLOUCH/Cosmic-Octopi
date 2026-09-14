"""Check reductionist feedback and shared sweep/evaluation semantics."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
import unittest
from unittest.mock import patch

import numpy as np

import experiment_runner as runner
from online_environment import run_online

HERE = Path(__file__).resolve().parent


def read_setup(approach, experiment):
    return json.loads((HERE / f"setup_{approach}_{experiment}.json").read_text())


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(1)
        cls.setup = read_setup("reductionist", "gravity")
        cls.xi = runner.generate_Xi_matrices(1, HERE / cls.setup["xi_matrix_file"])[0]
        cls.sizes = np.array([runner.get_model_size_in_kb(runner.Network(*cls.setup["policy_shapes"][n]).float())
                             for n in runner.CLASS_NAMES])

    def test_all_new_setups_match_model_based_conditions_and_reference(self):
        for approach in ("holistic", "reductionist"):
            for experiment in ("gravity", "synchronization", "heterogeneity"):
                if (approach, experiment) == ("holistic", "gravity"):
                    continue
                setup = read_setup(approach, experiment)
                runner.validate_setup(setup)
                for algorithm in ("MORBO", "qNParEGO"):
                    existing = json.loads((HERE.parent / algorithm / f"setup_{approach}_{experiment}.json").read_text())
                    for name in (approach, "reference_costs", "system_dynamics_parameters",
                                 "fitness_margin_bounds", "policy_shapes", "hyper_parameters",
                                 "gravity_mean", "ou_substeps", "seed", "xi_matrix_file"):
                        self.assertEqual(setup[name], existing[name], (approach, experiment, name))
                    if experiment == "gravity":
                        self.assertEqual(setup["sweep"]["values"], existing["gravity_variances"])
                    else:
                        self.assertEqual(setup["sweep"], existing["sweep"])
                self.assertEqual(setup["bandit"]["n_arms"], 10)
                self.assertEqual(setup["bandit"]["n_scalarization_directions"], 5)
                self.assertEqual(setup["bandit"]["n_online_iterations"], 100)

    def test_reductionist_online_equals_offline_for_both_horizons_and_all_sweeps(self):
        for experiment in ("gravity", "synchronization", "heterogeneity"):
            for horizon in (50, 100):
                setup = read_setup("reductionist", experiment)
                setup["reductionist"]["horizon"] = horizon
                setup["system_dynamics_parameters"][setup["sweep"]["parameter"]] = setup["sweep"]["values"][-1]
                sequence = np.random.default_rng(11).uniform(-10, 20, horizon)
                increments, classes, chosen = [], [], []

                def select(c, k):
                    self.assertEqual(len(chosen), len(increments))
                    self.assertEqual(k, len(chosen))
                    x = sequence[len(chosen)]
                    chosen.append(x)
                    return x

                def observe(c, k, x, cost):
                    self.assertEqual(x, chosen[-1])
                    classes.append(c)
                    increments.append(cost)

                online = run_online(setup, self.xi, self.sizes, 71, select, observe, "reductionist")
                offline = runner.evaluate_sequence(setup, self.xi, self.sizes, sequence, "reductionist", 71)
                np.testing.assert_array_equal(online, offline)
                np.testing.assert_allclose(np.sum(increments, axis=0), offline, rtol=1e-13)
                self.assertEqual(len(classes), horizon)
                self.assertEqual(set(classes), {runner.CLASS_NAMES.index("Cheetah")})

    def test_generalized_holistic_runner_preserves_completed_gravity_training(self):
        setup = read_setup("holistic", "gravity")
        setup.update(approach="holistic", experiment="gravity", sweep={"parameter": "gamma_grav", "values": [1, 5]})
        setup["bandit"]["n_online_iterations"] = 100
        previous = pickle.loads((HERE / "result_holistic_gravity.pkl").read_bytes())
        for formulation in ("RTS", "STR"):
            actual = runner.train_online(setup, formulation, 1, self.xi, self.sizes,
                                         previous["arms"], previous["scalarization_directions"])
            expected = previous["runs"][f"{formulation}/gamma_grav=1"]
            for name in ("arm_history", "scalarizer_history", "class_history", "increment_costs", "training_total_costs"):
                np.testing.assert_array_equal(actual[name], expected[name])

    def test_evaluation_uses_physical_margins_and_separate_50_request_horizon(self):
        for experiment in ("gravity", "synchronization", "heterogeneity"):
            setup = read_setup("reductionist", experiment)
            original = deepcopy(setup)
            arms = np.linspace(-10, 20, 10)
            draws, seeds = runner.evaluation_design(setup)
            calls = []

            def fake_eval(condition, xi, sizes, sequence, approach, seed):
                self.assertEqual(approach, "reductionist")
                self.assertEqual(condition["reductionist"]["horizon"], 50)
                self.assertEqual(len(sequence), 50)
                self.assertTrue(set(sequence) <= {arms[0], arms[-1]})
                self.assertEqual(condition["system_dynamics_parameters"][setup["sweep"]["parameter"]], setup["sweep"]["values"][-1])
                for name, value in setup["system_dynamics_parameters"].items():
                    if name != setup["sweep"]["parameter"]:
                        self.assertEqual(condition["system_dynamics_parameters"][name], value)
                calls.append(seed)
                return np.array([2000, 20])

            with patch.object(runner, "evaluate_sequence", side_effect=fake_eval):
                entry = runner.evaluate_subset(setup, setup["sweep"]["values"][-1], [0, 9], arms,
                                                draws, seeds, self.xi, self.sizes)
            self.assertEqual(setup, original)
            self.assertEqual(set(calls), set(map(int, seeds)))
            self.assertEqual(entry["sequences"].shape[1], 50)
            np.testing.assert_allclose(entry["X_unit"], (entry["sequences"] + 10) / 30)

    def test_reductionist_online_normalization_uses_cheetah_payload(self):
        setup = self.setup
        run = runner.train_online(setup, "RTS", 1, self.xi, self.sizes,
                                   np.linspace(-10, 20, 10), runner.unit_directions(5, 0.001))
        bound = setup["system_dynamics_parameters"]["n_octopi"] * self.sizes[runner.CLASS_NAMES.index("Cheetah")]
        np.testing.assert_array_equal(run["increment_cost_upper_bounds"], [bound, 1])
        np.testing.assert_allclose(run["reward_history"], 1 - run["increment_costs"] / [bound, 1])
        self.assertEqual(len(run["arm_history"]), 100)
        self.assertEqual(setup["reductionist"]["horizon"], 50)


if __name__ == "__main__":
    unittest.main()
