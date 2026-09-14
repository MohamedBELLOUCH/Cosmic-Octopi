"""Scalarize-then-robustify UCB, extracted from bandits.ipynb.

The selected scalarizer averages length-scalarized observations (Eq. 48).
Selection and vector bookkeeping are shared with RTS; scalar statistics differ.
"""

import numpy as np

from RTS_Scalarized_UCB import _ScalarizedUCBBase, length_scalarization


class STR_ScalarizedMultiObjectiveUCB(_ScalarizedUCBBase):
    def __init__(self, K, D, weights_list, z=None, rho=0.01, *, seed=None,
                 initialization="random", exploration_coefficient=2.0):
        super().__init__(K, D, weights_list, z, rho, seed=seed,
                         initialization=initialization,
                         exploration_coefficient=exploration_coefficient)
        for sub in self.scalar_ucbs:
            sub["sum_scalar"] = np.zeros(K)

    def update(self, arm, j, reward_vec):
        reward, sub = self._observe(arm, j, reward_vec)
        sub["sum_scalar"][arm] += length_scalarization(reward, self.weights_list[j], self.z)
        sub["mean_scalar"][arm] = sub["sum_scalar"][arm] / sub["nji"][arm]
