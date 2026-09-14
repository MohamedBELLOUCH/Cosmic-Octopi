"""Checks of the paper's KG formula, ordering, statistics, and experiment wiring."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from RTS_Scalarized_KG import RTS_ScalarizedMultiObjectiveKG, gaussian_kg_index, length_cost
from STR_Scalarized_KG import STR_ScalarizedMultiObjectiveKG

HERE = Path(__file__).resolve().parent


class KGTests(unittest.TestCase):
    def make(self, cls=RTS_ScalarizedMultiObjectiveKG, K=2, D=2, horizon=20):
        return cls(K, D, [np.ones(D) / np.sqrt(D)], horizon, seed=3)

    def test_gaussian_index_matches_integrated_expected_improvement(self):
        means = np.array([0.2, 0.6, 0.4])
        se = np.array([0.15, 0.3, 0.2])
        expected = []
        for i in range(3):
            competitor = np.delete(means, i).min()
            # E[min(current means) - min(mean_i + se_i Z, competitor)].
            threshold = (competitor - means[i]) / se[i]
            value = quad(lambda z: (min(means[i], competitor) - min(means[i] + se[i]*z, competitor))
                         * norm.pdf(z), -12, threshold, epsabs=1e-12)[0]
            value += quad(lambda z: (min(means[i], competitor) - competitor) * norm.pdf(z),
                          threshold, 12, epsabs=1e-12)[0]
            expected.append(value)
        np.testing.assert_allclose(gaussian_kg_index(means, se), expected, rtol=1e-10, atol=1e-12)
        # Excluding the arm itself matters for the unique best arm.
        self.assertLess(expected[0], se[0] / np.sqrt(2*np.pi))

    def test_ties_zero_variance_and_extreme_gaps(self):
        np.testing.assert_array_equal(gaussian_kg_index([1, 1], [0, 0]), [0, 0])
        np.testing.assert_allclose(gaussian_kg_index([1, 1], [0.2, 0.5]), np.array([0.2, 0.5])/np.sqrt(2*np.pi))
        values = gaussian_kg_index([0, 1], [1e-300, 1e-300])
        self.assertTrue(np.isfinite(values).all())
        np.testing.assert_array_equal(values, [0, 0])

    def test_sample_statistics_and_standard_errors(self):
        samples = np.array([[0.1, 0.9], [0.4, 0.3], [0.8, 0.2]])
        for cls in (RTS_ScalarizedMultiObjectiveKG, STR_ScalarizedMultiObjectiveKG):
            bandit = self.make(cls)
            for cost in samples:
                bandit.update(0, 0, cost)
                bandit.update(1, 0, cost + 0.1)
            np.testing.assert_allclose(bandit.mean_costs[0, 0], samples.mean(axis=0))
            np.testing.assert_allclose(bandit.m2_costs[0, 0]/2, samples.var(axis=0, ddof=1))
            values = (samples if cls is RTS_ScalarizedMultiObjectiveKG else
                      length_cost(samples, bandit.weights[0], bandit.reference))
            np.testing.assert_allclose(bandit.decision_statistics(0)['standard_errors'][0],
                                       values.std(axis=0, ddof=1)/np.sqrt(3))

    def test_rts_str_ordering_and_bonus_placement(self):
        rts, st = self.make(), self.make(STR_ScalarizedMultiObjectiveKG)
        for bandit in (rts, st):
            for cost in ([0, 1], [1, 0]): bandit.update(0, 0, cost)
            for _ in range(2): bandit.update(1, 0, [0.6, 0.6])
        self.assertEqual(rts.recommended_arms().tolist(), [0])
        self.assertEqual(st.recommended_arms().tolist(), [1])
        # STR scalar observations are identical despite vector variability.
        np.testing.assert_array_equal(st.decision_statistics(0)['kg_indices'], [0, 0])
        a = rts.decision_statistics(0)
        self.assertTrue((a['kg_indices'][0] > 0).all())
        np.testing.assert_allclose(a['scores'], -length_cost(rts.mean_costs[0]-a['exploration_bonus'], rts.weights[0], rts.reference))
        self.assertFalse(np.allclose(a['scores'], a['exploitation_scores']+a['exploration_bonus'].max(axis=1)))

    def test_initialization_pair_isolation_horizon_and_random_stream(self):
        directions = np.array([[0.6, 0.8], [0.8, 0.6]])
        for cls in (RTS_ScalarizedMultiObjectiveKG, STR_ScalarizedMultiObjectiveKG):
            bandit = cls(2, 2, directions, 12, seed=10)
            for t in range(8):
                self.assertEqual(bandit.select_action(), ((t % 4)//2, t % 2))
                arm, j = (t % 4)//2, t % 2
                bandit.update(arm, j, [0.1+j*0.2, 0.2+arm*0.1])
            np.testing.assert_array_equal(bandit.counts, np.full((2, 2), 2))
            self.assertNotEqual(bandit.mean_costs[0, 0, 0], bandit.mean_costs[1, 0, 0])
            rng = np.random.default_rng(10)
            for t in range(8, 12):
                np.random.seed(t)
                np.random.random(40)
                arm, j = bandit.select_action()
                self.assertEqual(j, int(rng.integers(2)))
                self.assertEqual(bandit.last_decision['remaining_iterations'], 11-t)
                if t == 11:
                    np.testing.assert_array_equal(bandit.last_decision['exploration_bonus'], 0)
                    np.testing.assert_allclose(bandit.last_decision['scores'], bandit.exploitation_scores(j))
                bandit.update(arm, j, [0.1, 0.2])
            with self.assertRaises(RuntimeError): bandit.select_action()
            with self.assertRaises(ValueError): cls(2, 2, directions, 8)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('kg_gravity_runner', HERE/'kg_runner.py')
        cls.runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.runner)
        cls.setup = json.loads((HERE/'setup_holistic_gravity.json').read_text())
        import torch
        torch.set_num_threads(1)

    def test_setup_and_evaluation_match_previous_holistic_gravity(self):
        self.runner.validate_setup(self.setup)
        for folder in ('Scalarized UCB', 'MORBO', 'qNParEGO'):
            previous = json.loads((HERE.parent/folder/'setup_holistic_gravity.json').read_text())
            for key in ('reference_costs', 'holistic', 'system_dynamics_parameters', 'fitness_margin_bounds',
                        'gravity_variances', 'hyper_parameters', 'seed', 'ou_substeps', 'policy_shapes'):
                self.assertEqual(self.setup[key], previous[key], key)
        b = self.setup['bandit']
        self.assertEqual((b['n_arms'], b['n_scalarization_directions'], b['n_online_iterations']), (7, 7, 150))
        wrong = deepcopy(self.setup)
        wrong['bandit']['n_online_iterations'] = 98
        with self.assertRaises(ValueError): self.runner.validate_setup(wrong)
        draws, _ = self.runner.evaluation_design(self.setup)
        self.assertEqual(draws.shape, (16, 100))

    def test_training_budget_cost_conversion_and_150_step_simulator_parity(self):
        runner = self.runner
        xi = runner.generate_Xi_matrices(1, HERE/self.setup['xi_matrix_file'])[0]
        sizes = np.array([runner.get_model_size_in_kb(runner.Network(*self.setup['policy_shapes'][n]).float()) for n in runner.CLASS_NAMES])
        arms = np.linspace(0, 1, 7)
        weights = runner.unit_directions(7, 0.001)
        for form in ('RTS', 'STR'):
            record = runner.train_online(self.setup, form, 5, xi, sizes, arms, weights)
            self.assertEqual(len(record['arm_history']), 150)
            self.assertEqual(len(record['adaptive_decisions']), 52)
            np.testing.assert_allclose(record['normalized_cost_history'], record['increment_costs']/record['increment_cost_upper_bounds'])
            condition = deepcopy(self.setup)
            condition['holistic']['horizon'] = 150
            condition['system_dynamics_parameters']['gamma_grav'] = 5
            from experiment_runner import evaluate_sequence
            expected = evaluate_sequence(condition, xi, sizes, arms[record['arm_history']], 'holistic', record['environment_seed'])
            np.testing.assert_array_equal(record['training_total_costs'], expected)
        self.assertEqual(self.setup['holistic']['horizon'], 100)

    def test_all_sweeps_route_online_budget_margins_and_cost_bounds(self):
        runner = self.runner
        sizes = np.array([10., 20., 30., 40.])
        for approach in ('holistic', 'reductionist'):
            for experiment in ('gravity', 'synchronization', 'heterogeneity'):
                setup = json.loads((HERE/f'setup_{approach}_{experiment}.json').read_text())
                original = deepcopy(setup)
                runner.validate_setup(setup)
                arms = np.linspace(*setup['bandit']['arm_interval'], 7)
                weights = runner.unit_directions(7, 0.001)
                level = setup['sweep']['values'][-1]

                def fake_online(condition, xi, payloads, seed, select, observe, mode):
                    self.assertEqual(mode, approach)
                    self.assertEqual(condition[approach]['horizon'], 150)
                    self.assertEqual(condition['system_dynamics_parameters'][setup['sweep']['parameter']], level)
                    totals = np.zeros(2)
                    for t in range(150):
                        c = t % 4 if mode == 'holistic' else 0
                        margin = select(c, t)
                        self.assertIn(margin, arms)
                        cost = np.array([10.+t%7, (t%3)/10])
                        observe(c, t, margin, cost)
                        totals += cost
                    return totals

                for form in ('RTS', 'STR'):
                    with patch.object(runner, 'run_online', side_effect=fake_online):
                        record = runner.train_online(setup, form, level, {}, sizes, arms, weights)
                    bound = 40 * (40 if approach == 'holistic' else 10)
                    np.testing.assert_array_equal(record['increment_cost_upper_bounds'], [bound, 1])
                    self.assertEqual(record['sweep_parameter'], setup['sweep']['parameter'])
                    self.assertEqual(len(record['adaptive_decisions']), 52)
                self.assertEqual(setup, original)

    def test_inline_xi_does_not_read_calibration_files(self):
        import experiment_runner as shared
        setup = deepcopy(self.setup)
        setup['Xi_matrices'] = {name: np.eye(4).tolist() for name in shared.CLASS_NAMES}
        setup.pop('xi_matrix_file')
        with patch.object(shared, 'generate_Xi_matrices', side_effect=AssertionError('external file read')):
            actual = shared.load_xi_matrices(setup, HERE)
        for name in shared.CLASS_NAMES:
            np.testing.assert_array_equal(actual[name], np.eye(4))


if __name__ == '__main__':
    unittest.main(verbosity=2)
