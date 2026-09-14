"""Robustify-then-scalarize UCB, extracted from bandits.ipynb.

Rewards are maximized. ``rho`` is retained for notebook compatibility but is
unused: length scalarization has no augmented-Chebyshev term. The default
global mean pooling follows the toy; mean_scope='scalarizer' implements the
scalarizer-specific empirical averages in Eqs. 46-47 of the paper.
"""

import numpy as np


def length_scalarization(reward_vec, w, z):
    return float(np.min(np.maximum(np.asarray(reward_vec) - z, 0.0) / w))


class _ScalarizedUCBBase:
    def __init__(self, K, D, weights_list, z=None, rho=0.01, *, seed=None,
                 initialization="random", exploration_coefficient=2.0):
        if type(K) is not int or K < 1 or type(D) is not int or D < 1:
            raise ValueError("K and D must be positive integers")
        weights = np.asarray(weights_list, dtype=float)
        if (weights.ndim != 2 or weights.shape[1] != D or not len(weights)
                or not np.all(np.isfinite(weights) & (weights > 0))):
            raise ValueError("Provide a nonempty list of finite positive D-dimensional weights")
        self.z = np.full(D, -0.01) if z is None else np.asarray(z, dtype=float).copy()
        if self.z.shape != (D,) or not np.all(np.isfinite(self.z)):
            raise ValueError("z must be a finite D-dimensional reference")
        if initialization not in ("random", "round_robin"):
            raise ValueError("initialization must be random or round_robin")
        if not np.isfinite(exploration_coefficient) or exploration_coefficient <= 0:
            raise ValueError("exploration_coefficient must be finite and positive")
        self.K, self.D, self.S = K, D, len(weights)
        self.rho = rho
        self.weights_list = [row.copy() for row in weights]
        self.initialization = initialization
        self.exploration_coefficient = exploration_coefficient
        self.rng = None if seed is None else np.random.default_rng(seed)
        self.ni = np.zeros(K, dtype=int)
        self.sum_vec = np.zeros((K, D))
        self.mean_vec = np.zeros((K, D))
        self.scalar_ucbs = [dict(nj=0, nji=np.zeros(K, dtype=int),
                                 mean_scalar=np.zeros(K)) for _ in weights]
        self.n = 0

    def select_action(self):
        """Return (arm, scalarizer); Eq. 43 initialization is optional."""
        if self.initialization == "round_robin" and self.n < self.K * self.S:
            return self.n // self.S, self.n % self.S
        j = int(np.random.randint(self.S) if self.rng is None else self.rng.integers(self.S))
        sub = self.scalar_ucbs[j]
        if np.any(sub["nji"] == 0):
            arm = int(np.argmin(sub["nji"]))
        else:
            bonus = np.sqrt(self.exploration_coefficient * np.log(max(sub["nj"], 1)) / sub["nji"])
            arm = int(np.argmax(sub["mean_scalar"] + bonus))
        return arm, j

    def _observe(self, arm, j, reward_vec):
        if not 0 <= arm < self.K or not 0 <= j < self.S:
            raise ValueError("Invalid arm or scalarizer index")
        reward = np.asarray(reward_vec, dtype=float)
        if reward.shape != (self.D,) or not np.all(np.isfinite(reward)):
            raise ValueError("reward_vec must be a finite D-dimensional observation")
        self.ni[arm] += 1
        self.sum_vec[arm] += reward
        self.mean_vec[arm] = self.sum_vec[arm] / self.ni[arm]
        sub = self.scalar_ucbs[j]
        sub["nji"][arm] += 1
        sub["nj"] += 1
        self.n += 1
        return reward, sub

    def recommended_arms(self):
        """Union of empirical exploitation maximizers, excluding unseen pairs.

        Select one maximizer per observed scalarizer, using the smallest arm
        index for ties, as in select_action. Exploration bonuses are excluded.
        """
        arms = set()
        for sub in self.scalar_ucbs:
            seen = np.flatnonzero(sub["nji"])
            if len(seen):
                arms.add(int(seen[np.argmax(sub["mean_scalar"][seen])]))
        return np.array(sorted(arms), dtype=int)


class RTS_ScalarizedMultiObjectiveUCB(_ScalarizedUCBBase):
    def __init__(self, K, D, weights_list, z=None, rho=0.01, *, seed=None,
                 initialization="random", exploration_coefficient=2.0,
                 mean_scope="global"):
        super().__init__(K, D, weights_list, z, rho, seed=seed,
                         initialization=initialization,
                         exploration_coefficient=exploration_coefficient)
        if mean_scope not in ("global", "scalarizer"):
            raise ValueError("mean_scope must be global or scalarizer")
        self.mean_scope = mean_scope
        for sub in self.scalar_ucbs:
            sub["sum_vec"] = np.zeros((K, D))
            sub["mean_vec"] = np.zeros((K, D))

    def update(self, arm, j, reward_vec):
        reward, sub = self._observe(arm, j, reward_vec)
        sub["sum_vec"][arm] += reward
        sub["mean_vec"][arm] = sub["sum_vec"][arm] / sub["nji"][arm]
        # Include this observation before scalarizing. The toy used the old
        # global mean and also left other scalarizers' pooled scores stale.
        if self.mean_scope == "global":
            for k, target in enumerate(self.scalar_ucbs):
                target["mean_scalar"][arm] = length_scalarization(
                    self.mean_vec[arm], self.weights_list[k], self.z)
        else:
            sub["mean_scalar"][arm] = length_scalarization(
                sub["mean_vec"][arm], self.weights_list[j], self.z)
