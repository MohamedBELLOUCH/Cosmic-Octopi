"""Small numerical checks only; no experiment or training is run."""
import unittest

import numpy as np
from scipy.spatial.distance import cdist

from practical_equivalence import (arm_energy, bootstrap_weights, classify,
                                   correlation_summary, correlation_vectors,
                                   distribution_summary)


class PracticalEquivalenceTests(unittest.TestCase):
    def test_whole_trajectory_resampling(self):
        weights = bootstrap_weights(25, 19, 17)
        self.assertEqual(weights.shape, (20, 25))
        np.testing.assert_array_equal(weights[0], np.ones(25))
        np.testing.assert_array_equal(weights.sum(axis=1), np.full(20, 25))
        np.testing.assert_array_equal(weights, bootstrap_weights(25, 19, 17))

    def test_correlations_match_explicit_resampled_data(self):
        x = np.random.default_rng(43).normal(size=(8, 12, 2))
        w = bootstrap_weights(8, 3, 47)
        actual = correlation_vectors(x, w, 10, 3)
        for b in range(len(w)):
            sample = x[np.repeat(np.arange(8), w[b].astype(int))]
            expected = []
            for h in range(1, 4):
                left = sample[:, :10-h].reshape(-1, 2)
                right = sample[:, h:10].reshape(-1, 2)
                expected.extend(np.corrcoef(left.T, right.T)[:2, 2:].ravel())
            np.testing.assert_allclose(actual[b], expected, atol=1e-12)

    def test_energy_matches_all_time_pair_definition(self):
        x = np.random.default_rng(7).uniform(size=(12, 5, 2))
        arms = np.zeros((12, 5), int)
        w = bootstrap_weights(12, 5, 81)
        effects, errors = arm_energy(x, arms, 0, w, [3, 5], 2)
        maxima = np.zeros((6, 2))
        differences = np.zeros((5, 2))
        for hi, horizon in enumerate([3, 5]):
            for t in range(horizon):
                for u in range(t):
                    values = []
                    for b in range(len(w)):
                        indices = np.repeat(np.arange(12), w[b].astype(int))
                        a, c = x[indices, t], x[indices, u]
                        values.append(max(0, 2*cdist(a,c).mean()
                                          - cdist(a,a).mean() - cdist(c,c).mean()))
                    maxima[:, hi] = np.maximum(maxima[:, hi], values)
                    differences[:, hi] = np.maximum(differences[:, hi],
                                                     np.abs(np.array(values[1:])-values[0]))
        np.testing.assert_allclose(effects, maxima[0], atol=1e-12)
        np.testing.assert_allclose(errors, differences, atol=1e-12)

    def test_empty_bootstrap_groups_are_not_evidence_of_equivalence(self):
        x = np.zeros((4, 2, 2))
        arms = np.array([[0, 1], [0, 1], [1, 0], [1, 0]])
        weights = np.array([[1, 1, 1, 1], [4, 0, 0, 0]])
        effect, errors = arm_energy(x, arms, 0, weights, [2], 2)
        output = distribution_summary(effect[None], errors[None],
                                      {"confidence_level": .95, "energy_distance_tolerances": [.1]})
        self.assertFalse(output["decisions"][0]["supported"][0])

    def test_constant_correlation_is_inconclusive(self):
        config = {"horizons": [10], "max_lag": 2, "confidence_level": .95,
                  "correlation_tolerances": [.1, .2]}
        output = correlation_summary(np.zeros((20, 10, 2)), bootstrap_weights(20, 5, 1), config)
        self.assertTrue(np.isnan(output["effect"][0]))
        self.assertTrue(output["decisions"][0]["inconclusive"][0])

    def test_three_way_decisions(self):
        result = classify([.01, .15, .30], [.05, .25, .40], [.2])[0]
        np.testing.assert_array_equal(result["supported"], [True, False, False])
        np.testing.assert_array_equal(result["inconclusive"], [False, True, False])
        np.testing.assert_array_equal(result["beyond_tolerance"], [False, False, True])

    def test_distribution_bounds_cover_all_arms_and_transform_squared_distance(self):
        effects = np.array([[.01, .04], [.02, .09]])
        errors = np.zeros((2, 10, 2))
        errors[1] = .01
        output = distribution_summary(effects, errors,
                    {"confidence_level": .95, "energy_distance_tolerances": [.1, .4]})
        np.testing.assert_allclose(output["effect"], np.sqrt([.02, .09]))
        np.testing.assert_allclose(output["upper"], np.sqrt([.03, .10]))


if __name__ == "__main__":
    unittest.main()
