"""Small statistical checks; no experiment simulation or PPO training."""

import itertools
import unittest

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from independence_comparison import _bds_effects, autohsic_test, bds_test, u_center


class IndependenceComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limit = threadpool_limits(limits=1)

    @classmethod
    def tearDownClass(cls):
        cls.limit.restore_original_limits()

    def test_hsic_agrees_with_unbiased_expanded_formula(self):
        rng = np.random.default_rng(34)
        x = rng.normal(size=(30, 2))
        k = np.exp(-cdist(x[1:], x[1:], 'sqeuclidean') / 2)
        l = np.exp(-cdist(x[:-1], x[:-1], 'sqeuclidean') / 2)
        np.fill_diagonal(k, 0)
        np.fill_diagonal(l, 0)
        n = len(k)
        expected = (np.sum(k*l) + k.sum()*l.sum()/((n-1)*(n-2))
                    - 2*np.dot(k.sum(axis=1), l.sum(axis=1))/(n-2)) / (n*(n-3))
        actual = np.sum(u_center(k)*u_center(l)) / (n*(n-3))
        self.assertAlmostEqual(actual, expected, places=13)
        np.testing.assert_allclose(u_center(k).sum(axis=1), 0, atol=1e-13)

    def test_hsic_detects_nonlinear_cross_lag_relationship(self):
        noise = np.random.default_rng(75).normal(size=181)
        # Nonlinear dependence of coordinate 2 at t on coordinate 1 at t-1.
        x = np.column_stack((noise[1:], noise[:-1]**2))
        answer = autohsic_test(x, 1, 499, 16)
        self.assertLessEqual(answer['pvalue'], .01)
        scaled = autohsic_test(7*x+3, 1, 499, 16)
        self.assertEqual(answer['pvalue'], scaled['pvalue'])

    def test_bds_matches_direct_embedding_pair_count(self):
        x = np.array([0., 2., 2.5, 4., 7., 8.])
        epsilon = 1.5*x.std(ddof=1)
        d = np.abs(x[:, None] - x) < epsilon
        orders = np.array(list(itertools.permutations(range(len(x)))))
        actual = _bds_effects(d[None], orders)[:, 0]
        expected = []
        for order in orders:
            z = x[order]
            vectors = np.column_stack((z[1:], z[:-1]))
            pairs = list(itertools.combinations(range(len(vectors)), 2))
            c1 = np.mean([abs(z[i+1]-z[j+1]) < epsilon for i, j in pairs])
            c2 = np.mean([np.max(np.abs(vectors[i]-vectors[j])) < epsilon for i, j in pairs])
            expected.append(np.sqrt(len(vectors))*(c2-c1**2))
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_bds_permutation_against_exhaustive_reference(self):
        x = np.array([0., 0., 0., 3., 3., 3.])[:, None]
        epsilon = 1.5*x.std(ddof=1)
        indicators = (np.abs(x[:, 0, None]-x[:, 0]) < epsilon)[None]
        orders = np.array(list(itertools.permutations(range(len(x)))))
        stats = _bds_effects(indicators, orders)[:, 0]
        observed = stats[0]
        exact = np.mean(np.abs(stats) >= abs(observed)-1e-12)
        answer = bds_test(x, 4999, 109)
        self.assertLess(abs(answer['pvalue']-exact), .025)

    def test_bds_adjusts_two_coordinates_and_detects_persistence(self):
        rng = np.random.default_rng(19)
        x = np.zeros((150, 2))
        for t in range(1, len(x)):
            x[t] = .97*x[t-1] + rng.normal(size=2)
        answer = bds_test(x, 499, 61)
        self.assertEqual(answer['pvalue'], min(1., 2*min(answer['coordinate_pvalue'])))
        self.assertLessEqual(answer['pvalue'], .01)

    def test_constant_vectors_do_not_trigger_rejection(self):
        x = np.ones((20, 2))
        self.assertEqual(autohsic_test(x, 5, 199, 17)['pvalue'], 1)
        self.assertEqual(bds_test(x, 199, 17)['pvalue'], 1)

    def test_iid_short_prefix_sanity(self):
        rng = np.random.default_rng(856)
        pvalues = []
        for r in range(60):
            x = rng.normal(size=(20, 2))
            pvalues.append([autohsic_test(x, 5, 499, r)['pvalue'], bds_test(x, 499, r)['pvalue']])
        counts = (np.array(pvalues) <= .05).sum(axis=0)
        # Loose reproducible size sanity check, not a finite-sample guarantee.
        self.assertTrue(np.all(counts <= 10), counts)


if __name__ == '__main__':
    unittest.main()
