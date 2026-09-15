"""No simulator runs: check physical margins, normalization and delegation."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import reductionist_runner as runner
import evaluate_reductionist_comparison as evaluation


class ReductionistTests(unittest.TestCase):
    def setUp(self):
        self.setup = runner.load_setup()
        self.sizes = np.array([44.80078125, 100.0, 30.0, 1000.0])

    def test_physical_conversion_and_held_out_dispatch(self):
        expected = [-10.0, 5.0, 20.0]
        np.testing.assert_allclose(runner.unit_to_physical(self.setup, [0, .5, 1]), expected)
        with self.assertRaises(ValueError):
            runner.unit_to_physical(self.setup, [-.1])
        calls = []

        class FakeSimulator:
            def evaluate_sequence(self, setup, xi, sizes, sequence, mode, seed):
                calls.append((sequence.copy(), mode, seed))
                return np.array([3.0, .2])

        with patch.object(evaluation, '_worker', (FakeSimulator(), self.setup, {}, self.sizes,
                                                np.array([[0, .5, 1]]), [123]), create=True):
            evaluation.worker_evaluate((0, 0))
        np.testing.assert_allclose(calls[0][0], expected)
        self.assertEqual(calls[0][1:], ('reductionist', 123))

    def test_pareto_delayed_pairs_and_cheetah_payload(self):
        captured = []

        def fake_online(setup, xi, sizes, seed, select, observe, mode):
            self.assertEqual(mode, 'reductionist')
            self.assertEqual(setup['reductionist']['horizon'], 150)
            for k in range(150):
                margin = select(0, k)
                captured.append(margin)
                observe(0, k, margin, np.array([10., k/200.]))
            return np.array([1500., sum(k/200. for k in range(150))])

        arms = np.linspace(-10, 20, 7)
        with patch.object(runner, 'run_online', fake_online):
            result = runner.train_pareto(self.setup, {}, self.sizes, arms)
        np.testing.assert_allclose(captured[:7], arms)
        self.assertTrue(np.isin(captured, arms).all())
        self.assertEqual(result['increment_cost_upper_bounds'][0], 40*self.sizes[0])
        self.assertEqual(result['n_completed_observations'], 149)
        pairs = np.asarray([r['costs'] for r in result['delayed_observations']])
        np.testing.assert_allclose(pairs[:, 1], result['increment_costs'][1:, 1])
        self.assertEqual(result['pending_final_action'][0], 149)

    def test_scalarized_training_receives_physical_arms_without_writing_results(self):
        for algorithm, backend in (('Scalarized UCB', runner.ucb_runner), ('Scalarized KG', runner.kg_runner)):
            calls, saved = [], []

            def fake_train(setup, formulation, level, xi, sizes, arms, weights):
                self.assertEqual(setup['approach'], 'reductionist')
                self.assertEqual(setup['reductionist']['class_name'], 'Cheetah')
                self.assertEqual(setup['bandit']['n_online_iterations'], 150)
                np.testing.assert_allclose(arms, np.linspace(-10, 20, 7))
                calls.append(formulation)
                return {'recommended_arms': [np.array([0, 6])], 'environment_seed': 123}

            with patch.object(runner, 'simulator_inputs', return_value=({}, self.sizes)), \
                 patch.object(backend, 'train_online', fake_train), \
                 patch.object(runner, 'save', side_effect=lambda path, record: saved.append(record)):
                runner.train(algorithm)
            self.assertEqual(calls, ['RTS', 'STR'])
            for record in saved:
                np.testing.assert_allclose(record['recommended_unit_margins'], [0, 1])
                np.testing.assert_allclose(record['recommended_physical_margins'], [-10, 20])


if __name__ == '__main__':
    unittest.main()
