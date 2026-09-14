"""Cost-minimizing STR scalarized KG (Selective Pressure Allocation, Eqs. 51-52).

Each observation is length-scalarized before its mean and sample variance are
estimated. The Gaussian KG bonus is computed in these scalar cost coordinates.
"""

import numpy as np

from RTS_Scalarized_KG import _ScalarizedKGBase, length_cost


class STR_ScalarizedMultiObjectiveKG(_ScalarizedKGBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean_scalar_costs = np.zeros((self.S, self.K))
        self.m2_scalar_costs = np.zeros((self.S, self.K))

    def update(self, arm, j, cost_vec):
        cost, count = self._observe(arm, j, cost_vec)
        scalar = float(length_cost(cost, self.weights[j], self.reference))
        delta = scalar - self.mean_scalar_costs[j, arm]
        self.mean_scalar_costs[j, arm] += delta / count
        self.m2_scalar_costs[j, arm] += delta * (scalar - self.mean_scalar_costs[j, arm])

    def exploitation_scores(self, j):
        return -self.mean_scalar_costs[j].copy()

    def decision_statistics(self, j):
        means = self.mean_scalar_costs[j]
        errors, indices, bonus, remaining = self._exploration(j, means, self.m2_scalar_costs[j])
        return {"standard_errors": errors, "kg_indices": indices,
                "exploration_bonus": bonus, "remaining_iterations": remaining,
                "exploitation_scores": self.exploitation_scores(j), "scores": -means + bonus}

    def state_dict(self):
        return {**super().state_dict(), "mean_scalar_costs": self.mean_scalar_costs.copy(),
                "m2_scalar_costs": self.m2_scalar_costs.copy()}
