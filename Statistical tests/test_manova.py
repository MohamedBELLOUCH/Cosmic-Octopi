"""Numerical MANOVA validation on synthetic arrays; no experiment is run."""
import unittest

import numpy as np

from analysis_manova import manova_test, pillai


class ManovaTests(unittest.TestCase):
    def test_pillai_matches_eigenvalue_definition(self):
        h = np.array([[3., .2], [.2, 2.]])
        e = np.array([[12., 1.], [1., 9.]])
        self.assertAlmostEqual(pillai(h, h+e), np.linalg.eigvals(np.linalg.solve(h+e, h)).sum())

    def test_pillai_matches_dense_one_way_manova(self):
        rng = np.random.default_rng(24)
        x = rng.normal(size=(50, 6, 2))
        mask = rng.random((50, 6)) < .6
        r, t = np.nonzero(mask)
        y = x[r, t]
        design = np.eye(6)[t]
        fitted = design @ np.linalg.lstsq(design, y, rcond=None)[0]
        error = (y-fitted).T @ (y-fitted)
        total = (y-y.mean(0)).T @ (y-y.mean(0))
        expected = pillai(total-error, total)
        actual = manova_test(x, mask, 19, 7)
        self.assertAlmostEqual(actual['statistic'], expected, places=12)

    def test_affine_response_invariance(self):
        rng = np.random.default_rng(38)
        x = rng.normal(size=(40, 5, 2))
        mask = rng.random((40, 5)) < .7
        a = manova_test(x, mask, 99, 10)
        b = manova_test(x @ np.array([[2., 1.], [.1, 3.]]) + [4, -2], mask, 99, 10)
        self.assertAlmostEqual(a['statistic'], b['statistic'], places=10)
        self.assertEqual(a['pvalue'], b['pvalue'])

    def test_equal_sample_means_are_not_rejected_for_variance_alone(self):
        x = np.random.default_rng(38).normal(size=(50, 4, 2))
        x *= np.array([1., 2., 4., 8.])[None, :, None]
        x -= x.mean(axis=0)
        result = manova_test(x, np.ones((50, 4), bool), 99, 10)
        self.assertLess(abs(result['statistic']), 1e-12)
        self.assertEqual(result['pvalue'], 1.0)

    def test_large_mean_difference_is_detected(self):
        x = np.random.default_rng(75).normal(size=(80, 4, 2))
        x[:, 2:, :] += 5
        result = manova_test(x, np.ones((80, 4), bool), 199, 10)
        self.assertLessEqual(result['pvalue'], .01)

    def test_singular_responses_are_not_silent_acceptance(self):
        with self.assertRaisesRegex(ValueError, 'Singular'):
            manova_test(np.ones((10, 3, 2)), np.ones((10, 3), bool), 19, 7)


if __name__ == '__main__':
    unittest.main()
