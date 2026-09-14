"""Checks for RTS/STR ordering and parity of online/offline simulator dynamics."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from RTS_Scalarized_UCB import RTS_ScalarizedMultiObjectiveUCB
from STR_Scalarized_UCB import STR_ScalarizedMultiObjectiveUCB
from holistic_environment import run_holistic_online

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_ucb_experiment", HERE / "script_holistic_gravity.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


class BanditTests(unittest.TestCase):
    def test_rts_str_choose_different_arms_for_crossing_observations(self):
        weights = [[2**-0.5, 2**-0.5]]
        for scope in ("global", "scalarizer"):
            rts = RTS_ScalarizedMultiObjectiveUCB(2, 2, weights, z=[0, 0], mean_scope=scope)
            str_bandit = STR_ScalarizedMultiObjectiveUCB(2, 2, weights, z=[0, 0])
            for arm, rewards in ((0, ([0.9, 0.1], [0.1, 0.9])), (1, ([0.4, 0.4], [0.4, 0.4]))):
                for reward in rewards:
                    rts.update(arm, 0, reward)
                    str_bandit.update(arm, 0, reward)
            np.testing.assert_array_equal(rts.recommended_arms(), [0])
            np.testing.assert_array_equal(str_bandit.recommended_arms(), [1])
            self.assertEqual(rts.select_action(), (0, 0))
            self.assertEqual(str_bandit.select_action(), (1, 0))

    def test_rts_includes_latest_reward_and_refreshes_pooled_scores(self):
        b = RTS_ScalarizedMultiObjectiveUCB(1, 2, [[1, 1], [1, 2]], z=[0, 0])
        b.update(0, 0, [0.6, 0.8])
        self.assertAlmostEqual(b.scalar_ucbs[0]["mean_scalar"][0], 0.6)
        self.assertAlmostEqual(b.scalar_ucbs[1]["mean_scalar"][0], 0.4)
        b.update(0, 1, [0.2, 0.4])
        self.assertAlmostEqual(b.scalar_ucbs[0]["mean_scalar"][0], 0.4)
        self.assertAlmostEqual(b.scalar_ucbs[1]["mean_scalar"][0], 0.3)

    def test_paper_rts_statistics_are_scalarizer_specific(self):
        b = RTS_ScalarizedMultiObjectiveUCB(1, 2, [[1, 1], [1, 2]], z=[0, 0], mean_scope="scalarizer")
        b.update(0, 0, [0.8, 0.8])
        b.update(0, 1, [0.2, 0.2])
        self.assertAlmostEqual(b.scalar_ucbs[0]["mean_scalar"][0], 0.8)
        self.assertAlmostEqual(b.scalar_ucbs[1]["mean_scalar"][0], 0.1)

    def test_round_robin_initializes_every_pair_before_ucb(self):
        for cls in (RTS_ScalarizedMultiObjectiveUCB, STR_ScalarizedMultiObjectiveUCB):
            b = cls(10, 2, experiment.unit_directions(5, 0.001), seed=42, initialization="round_robin")
            self.assertEqual(len(b.recommended_arms()), 0)
            for t in range(50):
                arm, j = b.select_action()
                self.assertEqual((arm, j), (t // 5, t % 5))
                b.update(arm, j, [0.2, 0.3])
            for sub in b.scalar_ucbs:
                np.testing.assert_array_equal(sub["nji"], np.ones(10))
            self.assertEqual(b.n, 50)
            self.assertEqual(len(b.select_action()), 2)

    def test_bandit_rng_is_independent_of_simulator_global_rng(self):
        for cls in (RTS_ScalarizedMultiObjectiveUCB, STR_ScalarizedMultiObjectiveUCB):
            a = cls(2, 2, [[1, 1], [2, 1]], seed=123)
            b = cls(2, 2, [[1, 1], [2, 1]], seed=123)
            for t in range(20):
                first = a.select_action()
                np.random.seed(t)
                np.random.random(100)
                second = b.select_action()
                self.assertEqual(first, second)
                a.update(*first, [0.2, 0.3])
                b.update(*second, [0.2, 0.3])


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        torch.set_num_threads(3)
        cls.setup = json.loads((HERE / "setup_holistic_gravity.json").read_text())
        experiment.validate_setup(cls.setup)
        cls.xi = experiment.generate_Xi_matrices(1, HERE / cls.setup["xi_matrix_file"])[0]
        cls.sizes = np.array([experiment.get_model_size_in_kb(experiment.Network(*cls.setup["policy_shapes"][n]).float())
                              for n in experiment.CLASS_NAMES])

    def test_online_fixed_sequence_equals_existing_offline_evaluator(self):
        for gravity in (1, 5):
            setup = json.loads(json.dumps(self.setup))
            setup["system_dynamics_parameters"]["gamma_grav"] = gravity
            for sequence in (np.zeros(100), np.ones(100), np.random.default_rng(7).random(100)):
                observations = []
                decisions = []

                def choose(c, k):
                    self.assertEqual(len(decisions), len(observations))
                    x = sequence[len(decisions)]
                    decisions.append(x)
                    return x

                def observe(c, k, x, costs):
                    self.assertEqual(x, decisions[-1])
                    observations.append(costs)

                online = run_holistic_online(setup, self.xi, self.sizes, 99, choose, observe)
                offline = experiment.evaluate_sequence(setup, self.xi, self.sizes, sequence, "holistic", 99)
                np.testing.assert_array_equal(online, offline)
                np.testing.assert_allclose(np.sum(observations, axis=0), offline, rtol=1e-13)
                self.assertEqual(len(observations), 100)

    def test_sampler_only_uses_recommended_arms_and_scalarizes_realizations(self):
        setup = json.loads(json.dumps(self.setup))
        setup["evaluation"]["n_random_sequences"] = 2
        draws, seeds = experiment.evaluation_design(setup)
        arms = np.linspace(0, 1, 10)
        entry = experiment.evaluate_subset(setup, 1, [2, 7], arms, draws, seeds, self.xi, self.sizes)
        self.assertEqual(set(np.unique(entry["arm_sequences"])), {2, 7})
        self.assertEqual(entry["cost_realizations"].shape[1:], (2, 2))
        self.assertTrue(np.any(np.all(entry["arm_sequences"] == 2, axis=1)))
        self.assertTrue(np.any(np.all(entry["arm_sequences"] == 7, axis=1)))
        self.assertTrue(all(np.isfinite(list(entry["hypervolumes"].values()))))


if __name__ == "__main__":
    unittest.main()
