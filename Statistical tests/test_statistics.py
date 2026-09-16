"""Numerical/statistical validation; no PPO training or production simulation."""

import itertools
import json
from pathlib import Path
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from analysis import (binomial_interval, energy_statistic, energy_test, holm_adjust,
                      ljung_box_test, null_reference_band, summarize_levels)


class StatisticalTests(unittest.TestCase):
    def test_real_training_is_rejected(self):
        from script import validate_setup
        setup = json.loads((Path(__file__).parent / "setup.json").read_text())
        validate_setup(setup)
        setup["simulated_rewards"] = False
        with self.assertRaisesRegex(ValueError, "training is disabled"):
            validate_setup(setup)

    def test_ljung_box_matches_univariate_definition(self):
        x = np.random.default_rng(12).normal(size=80)
        centered = x - x.mean()
        correlations = np.correlate(centered, centered, mode="full")[len(x)-1:]
        correlations /= correlations[0]
        expected = len(x) * (len(x) + 2) * sum(correlations[h] ** 2 / (len(x) - h)
                                               for h in range(1, 6))
        result = ljung_box_test(x[:, None], 5, 19, 3)
        self.assertAlmostEqual(result["statistic"], expected, places=10)

    def test_multivariate_test_detects_cross_lag_dependence(self):
        noise = np.random.default_rng(17).normal(size=201)
        # Each coordinate alone is white noise; dependence exists across
        # coordinates at lag one, so separate univariate tests would miss it.
        x = np.column_stack((noise[1:], noise[:-1]))
        result = ljung_box_test(x, 5, 199, 8)
        self.assertLessEqual(result["pvalue"], 0.01)

    def test_ljung_box_invariant_to_coordinate_mixing_and_scaling(self):
        x = np.random.default_rng(27).normal(size=(80, 2))
        a = ljung_box_test(x, 5, 99, 35)
        b = ljung_box_test(x @ np.array([[3., 2.], [0.2, 1.]]), 5, 99, 35)
        self.assertAlmostEqual(a["statistic"], b["statistic"], places=9)
        self.assertEqual(a["pvalue"], b["pvalue"])

    def test_short_prefix_iid_calibration(self):
        rng = np.random.default_rng(611)
        pvalues = [ljung_box_test(rng.normal(size=(20, 2)), 5, 199, i)["pvalue"]
                   for i in range(100)]
        rejection_count = np.count_nonzero(np.asarray(pvalues) <= .05)
        # Broad deterministic sanity check, not a claim of demonstrated size.
        self.assertLessEqual(rejection_count, 13)
        self.assertGreaterEqual(rejection_count, 1)

    def test_singular_covariance_is_not_silently_accepted(self):
        with self.assertRaisesRegex(ValueError, "Singular"):
            ljung_box_test(np.ones((20, 2)), 5, 19, 1)

    def test_energy_matches_pairwise_definition_for_unequal_groups(self):
        sizes = [3, 7, 5]
        groups = [np.random.default_rng(i).normal(size=(n, 2)) for i, n in enumerate(sizes)]
        expected = 0.0
        for i, j in itertools.combinations(range(3), 2):
            energy_distance = (2 * cdist(groups[i], groups[j]).mean()
                               - cdist(groups[i], groups[i]).mean()
                               - cdist(groups[j], groups[j]).mean())
            expected += sizes[i] * sizes[j] / (2 * sum(sizes)) * energy_distance
        pooled = np.concatenate(groups)
        self.assertAlmostEqual(energy_statistic(cdist(pooled, pooled), sizes), expected, places=11)

    def test_energy_constant_samples_and_known_separation(self):
        self.assertEqual(energy_test(np.zeros((8, 8)), [3, 5], 19, 7)["pvalue"], 1)
        x = np.r_[np.zeros(3), np.full(4, 3.)][:, None]
        self.assertAlmostEqual(energy_statistic(cdist(x, x), [3, 4]), 36/7, places=10)

    def test_energy_permutation_pvalue_against_exhaustive_small_case(self):
        x = np.array([0., 0., 3., 3.])[:, None]
        d = cdist(x, x)
        observed = energy_statistic(d, [2, 2])
        exact = np.mean([energy_statistic(d[np.ix_(ix, ix)], [2, 2]) >= observed - 1e-10
                         for ix in itertools.permutations(range(4))])
        result = energy_test(d, [2, 2], 999, 715)
        self.assertAlmostEqual(exact, 1/3)
        self.assertLess(abs(result["pvalue"] - exact), .045)

    def test_energy_detects_spread_change_without_mean_shift(self):
        rng = np.random.default_rng(42)
        a = rng.normal(size=(60, 2)); a -= a.mean(axis=0)
        b = rng.normal(scale=8, size=(60, 2)); b -= b.mean(axis=0)
        x = np.concatenate((a, b))
        self.assertLessEqual(energy_test(cdist(x, x), [60, 60], 199, 93)["pvalue"], .01)

    def test_holm_and_binomial_summaries(self):
        np.testing.assert_allclose(holm_adjust([.04, .01, .03]), [.06, .03, .06])
        self.assertEqual(binomial_interval(0, 350)[0], 0)
        self.assertEqual(binomial_interval(350, 350)[1], 1)
        band = null_reference_band(350)
        self.assertLess(band[0], .05)
        self.assertGreater(band[1], .05)

    def test_multiple_levels_share_tests_and_apply_holm(self):
        analysis = {"lb_pvalue": np.array([[.05, .15, .25, .35]]),
                    "energy_pvalue": np.array([[.01, .08, .20]])}
        summaries = summarize_levels(analysis, [.1, .2, .3])
        self.assertEqual([s["lb_rejection_count"][0] for s in summaries], [1, 2, 3])
        self.assertEqual([s["energy_rejection_count"][0] for s in summaries], [1, 3, 3])
        for s in summaries:
            np.testing.assert_allclose(s["lb_null_reference_band"],
                                       null_reference_band(4, s["alpha"], .95))
        with self.assertRaises(ValueError):
            summarize_levels(analysis, [.1, .1])


if __name__ == "__main__":
    unittest.main()
