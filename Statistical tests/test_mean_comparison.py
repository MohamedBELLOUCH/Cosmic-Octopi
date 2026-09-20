import unittest

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import f

from mean_comparison import mats_statistic, mats_test, welch_james_test, holm


class MeanComparisonTests(unittest.TestCase):
    def test_welch_james_reduces_to_univariate_welch(self):
        means = np.array([0.2, 0.7, -0.1, 0.9])
        variances = np.array([1.0, 4.0, 0.6, 2.0])
        n = np.array([21, 32, 19, 26])
        w = n / variances
        k = len(n)
        a = np.sum((1-w/w.sum())**2 / (n-1))
        statistic = np.sum(w*(means-np.average(means, weights=w))**2)
        fstat = statistic / ((k-1)*(1 + 2*(k-2)*a/(k*k-1)))
        expected = f.sf(fstat, k-1, (k*k-1)/(3*a))
        actual = welch_james_test(means[:, None], variances[:, None, None], n)
        np.testing.assert_allclose(actual[:2], [statistic, expected], rtol=1e-11)

    def test_multivariate_welch_matches_weighted_formula(self):
        means = np.array([[0.2, 0.1], [0.7, -0.2], [-0.1, 0.3]])
        cov = np.array([[[1., .3], [.3, 2.]], [[3., -.2], [-.2, 1.]], [[.5, .1], [.1, .8]]])
        n = np.array([21, 32, 19])
        weights = np.linalg.inv(cov/n[:, None, None])
        inv_total = np.linalg.inv(weights.sum(axis=0))
        center = inv_total @ np.einsum('gij,gj->i', weights, means)
        delta = means-center
        statistic = np.einsum('gi,gij,gj', delta, weights, delta)
        b = np.eye(2) - inv_total @ weights
        a = .5*np.sum((np.einsum('gij,gji->g', b, b)+np.trace(b, axis1=1, axis2=2)**2)/(n-1))
        q = (len(n)-1)*2
        expected = f.sf(statistic/(q+2*a-6*a/(q+2)), q, q*(q+2)/(3*a))
        actual = welch_james_test(means, cov, n)
        np.testing.assert_allclose(actual[:2], [statistic, expected], rtol=1e-11)

    def test_mats_matches_projection_with_zero_variances(self):
        for zeros in [False, True]:
            means = np.array([[.2, .7], [.9, .1], [.3, -.2], [.2, .5]])
            variances = np.array([[.02, .1], [.03, .07], [.01, .2], [.02, .03]])
            if zeros:
                variances[[0, 3], 0] = 0
            t = np.kron(np.eye(4)-np.ones((4, 4))/4, np.eye(2))
            centered = t @ means.ravel()
            expected = centered @ np.linalg.pinv(t @ np.diag(variances.ravel()) @ t) @ centered
            np.testing.assert_allclose(mats_statistic(means, variances), expected, rtol=1e-11)

    def test_bootstrap_reproducibility_and_mean_shift(self):
        cov = np.array([[[1., .2], [.2, .6]], [[2., .1], [.1, 1.]], [[.5, 0.], [0., .9]]])
        n = np.array([20, 30, 25])
        equal = np.zeros((3, 2))
        self.assertEqual(mats_test(equal, cov, n, 199, 17)[1], 1)
        shifted = equal.copy()
        shifted[-1] = 10
        a = mats_test(shifted, cov, n, 199, 17)
        self.assertEqual(a, mats_test(shifted, cov, n, 199, 17))
        self.assertEqual(a[1], 1/200)
        scaled = mats_test(shifted*np.array([2., 3.]), cov*np.outer([2., 3.], [2., 3.]), n, 199, 17)
        self.assertAlmostEqual(a[0], scaled[0])

    def test_singular_welch_does_not_ignore_unestimable_difference(self):
        means = np.array([[0.], [1.], [2.]])
        cov = np.array([[[0.]], [[0.]], [[1.]]])
        with self.assertRaisesRegex(ValueError, 'zero estimated variance'):
            welch_james_test(means, cov, np.array([20, 20, 20]))

    def test_holm_preserves_order_and_bounds(self):
        np.testing.assert_allclose(holm(np.array([.5, .001, .02])), [.5, .003, .04])


if __name__ == '__main__':
    unittest.main()
