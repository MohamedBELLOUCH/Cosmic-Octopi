"""Online cost adapter for the unchanged notebook Pareto UCB implementation."""

from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Pareto UCB"))
from Pareto_UCB import EmpiricalParetoUCB, get_pareto_indices


class CostParetoUCB(EmpiricalParetoUCB):
    """Keep the original mean-plus-UCB rule, with reward = -normalized cost."""

    def __init__(self, K, seed):
        super().__init__(K, 2)
        self.rng = np.random.default_rng(seed)
        self.decisions = 0

    def select_action(self):
        if self.decisions < self.K:
            arm = self.decisions
        else:
            unseen = np.flatnonzero(self.ni == 0)
            # With one pending observation, finish initialization before
            # computing empirical confidence vectors for all arms.
            arm = int(unseen[0]) if len(unseen) else int(
                self.rng.choice(get_pareto_indices(self._compute_ucb_vectors())))
        self.decisions += 1
        return arm

    def update_cost(self, arm, cost):
        cost = np.asarray(cost, dtype=float)
        if cost.shape != (2,) or not np.isfinite(cost).all() or np.any(cost < 0):
            raise ValueError("Expected two finite nonnegative normalized costs")
        self.ni[arm] += 1
        self.sum_vec[arm] -= cost
        self.mean_vec[arm] = self.sum_vec[arm] / self.ni[arm]
        self.n += 1

    def recommended_arms(self):
        seen = np.flatnonzero(self.ni)
        return seen[get_pareto_indices(self.mean_vec[seen])]


class DelayedCostFeedback:
    """Attribute (overhead_t, instability_{t+1}) to the action at t.

    The first instability has no preceding action. At budget exhaustion, the
    last action remains pending; no fabricated observation or extra step is used.
    """

    def __init__(self, bandit, bounds):
        self.bandit = bandit
        self.bounds = np.asarray(bounds, dtype=float)
        self.pending = None
        self.observations = []

    def observe(self, iteration, arm, costs):
        costs = np.asarray(costs, dtype=float)
        if (costs.shape != (2,) or not np.isfinite(costs).all()
                or np.any(costs < -1e-9) or np.any(costs > self.bounds + 1e-9)):
            raise ValueError("Increment outside the shared normalization bounds")
        if self.pending is not None:
            previous_iteration, previous_arm, overhead = self.pending
            if iteration != previous_iteration + 1:
                raise ValueError("Feedback requires consecutive global iterations")
            pair = np.array([overhead, costs[1]])
            self.bandit.update_cost(previous_arm, pair / self.bounds)
            self.observations.append({"action_iteration": previous_iteration,
                                      "observation_iteration": iteration,
                                      "arm": previous_arm, "costs": pair,
                                      "normalized_costs": pair / self.bounds})
        self.pending = (iteration, arm, float(costs[0]))
