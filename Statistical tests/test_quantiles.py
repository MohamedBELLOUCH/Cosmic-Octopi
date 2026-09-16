import unittest
from unittest.mock import patch
from copy import deepcopy
import numpy as np
from scipy.stats import binom
from analysis_quantiles import partition_runs, coverage_test, compute_coverage, analyze_generated


class QuantileCoverageTests(unittest.TestCase):
    def test_generated_analysis_resumes_and_skips_finished_tests(self):
        config = {'horizons': [4, 8], 'seed': 12, 'confidence_level': .8,
            'ljung_box': {'max_lag': 1, 'n_permutations': 9, 'significance_levels': [.01]},
            'quantile_coverage': {'calibration_fraction': .5, 'split_seed': 11,
                'evaluation_seed': 12, 'calibration_horizon': 8,
                'minimum_coverage': .8, 'alpha': .01,
                'bands': [{'lower': .05, 'upper': .95}]}}
        result = {'completed_runs': np.ones(40, bool),
            'costs': np.random.default_rng(3).uniform(size=(40, 8, 2)),
            'arm_indices': np.random.default_rng(4).integers(0, 2, (40, 8)),
            'arm_values': np.array([0, 1]), 'objective_scale': np.ones(2), 'analysis': None}
        saves = []
        with patch('analysis.ljung_box_test', return_value={'pvalue': .5, 'statistic': 1}) as test:
            analyze_generated({'statistics': config}, result, None,
                              lambda p, r: saves.append(deepcopy(r)))
            self.assertEqual(test.call_count, 80)
            self.assertTrue(result['analysis']['complete'])
            checkpoint = next(r for r in saves
                if np.isfinite(r['analysis']['lb_pvalue']).sum() == 20)
            test.reset_mock()
            analyze_generated({'statistics': config}, checkpoint, None, lambda p, r: None)
            self.assertEqual(test.call_count, 60)
            np.testing.assert_equal(checkpoint['analysis']['quantile_coverage']['pvalue'],
                                    result['analysis']['quantile_coverage']['pvalue'])
            test.reset_mock()
            analyze_generated({'statistics': config}, checkpoint, None, lambda p, r: None)
            test.assert_not_called()

    def test_exact_test_direction_and_zero_observations(self):
        self.assertAlmostEqual(coverage_test(5, 20, .8, .8)["pvalue"], binom.cdf(5, 20, .8))
        self.assertEqual(coverage_test(20, 20, .8, .8)["pvalue"], 1.)
        self.assertEqual(coverage_test(0, 0, .8, .8)["pvalue"], 1.)
        np.testing.assert_equal(coverage_test(0, 0, .8, .8)["interval"], [0, 1])

    def test_confidence_changes_intervals_not_pvalues(self):
        a, b = coverage_test(15, 20, .8, .8), coverage_test(15, 20, .8, .95)
        self.assertEqual(a['pvalue'], b['pvalue'])
        self.assertGreater(a['interval'][0], b['interval'][0])
        self.assertLess(a['interval'][1], b['interval'][1])

    def test_disjoint_split_and_no_calibration_leakage(self):
        config = {'horizons': [4, 8], 'confidence_level': .8, 'quantile_coverage': {
            'calibration_fraction': .5, 'split_seed': 11, 'evaluation_seed': 12,
            'calibration_horizon': 8, 'minimum_coverage': .8, 'alpha': .01,
            'bands': [{'lower': .025, 'upper': .975}, {'lower': .25, 'upper': .75}]}}
        x = np.random.default_rng(3).uniform(size=(40, 8, 2))
        arms = np.random.default_rng(4).integers(0, 2, size=(40, 8))
        a = compute_coverage(x, arms, config)
        cal, evaluation = partition_runs(40, .5, 11)
        self.assertEqual(len(set(cal) & set(evaluation)), 0)
        self.assertEqual(len(set(cal) | set(evaluation)), 40)
        x[evaluation] += 10
        b = compute_coverage(x, arms, config)
        np.testing.assert_equal(a['rectangles_normalized'], b['rectangles_normalized'])
        np.testing.assert_equal(a['counts'].sum(axis=1), [20, 20])
        np.testing.assert_equal(a['evaluation_time_indices'], b['evaluation_time_indices'])
        self.assertTrue((b['successes'] == 0).all())
        self.assertTrue((a['successes'][0] >= a['successes'][1]).all())


if __name__ == '__main__':
    unittest.main()
