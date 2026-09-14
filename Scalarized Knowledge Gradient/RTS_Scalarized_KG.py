"""Cost-minimizing RTS scalarized KG (Selective Pressure Allocation, Eqs. 49-50).

Statistics are specific to an arm-direction pair. The exploration term uses
Yahyaa et al. (2014), Section 4.2: (T - k) * K * D * Gaussian KG index.
The standard error is the sample standard deviation divided by sqrt(n).
"""

import numpy as np
from scipy.special import ndtr


def length_cost(costs, weights, reference):
    """Equation 26: max_d [cost_d - reference_d]_+ / weight_d."""
    return np.max(np.maximum(np.asarray(costs) - reference, 0.0) / weights, axis=-1)


def gaussian_kg_index(means, standard_errors):
    """One-step Gaussian improvement over the best OTHER arm, for minimization.

    The first axis indexes arms; remaining axes are independent objectives.
    v = se * (phi(q) - q * Phi(-q)), q = abs(mean - competitor) / se.
    Zero uncertainty has zero value of information, including tied means.
    """
    means = np.asarray(means, dtype=float)
    errors = np.asarray(standard_errors, dtype=float)
    if (means.ndim not in (1, 2) or len(means) < 2 or errors.shape != means.shape
            or not np.isfinite(means).all() or not np.isfinite(errors).all()
            or (errors < 0).any()):
        raise ValueError("Expected finite means and nonnegative errors for at least two arms")
    competitors = np.broadcast_to(means[None, ...], (len(means), *means.shape)).copy()
    competitors[np.arange(len(means)), np.arange(len(means))] = np.inf
    gap = np.abs(means - competitors.min(axis=1))
    result = np.zeros_like(means)
    uncertain = errors > 0
    with np.errstate(over="ignore", divide="ignore"):
        q = gap[uncertain] / errors[uncertain]
    # Truncate negligible far-tail improvements near float64 underflow.
    finite_tail = q < 38
    gain = np.zeros_like(q)
    x = q[finite_tail]
    gain[finite_tail] = np.maximum(np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
                                    - x * ndtr(-x), 0.0)
    result[uncertain] = errors[uncertain] * gain
    return result


class _ScalarizedKGBase:
    def __init__(self, K, D, weights_list, horizon, reference=None, *, seed=None,
                 initial_pulls_per_pair=2, exploration_coefficient=1.0):
        if type(K) is not int or K < 2 or type(D) is not int or D < 1:
            raise ValueError("K >= 2 and D >= 1 must be integers")
        weights = np.asarray(weights_list, dtype=float)
        if (weights.ndim != 2 or weights.shape[1] != D or not len(weights)
                or not np.all(np.isfinite(weights) & (weights > 0))
                or not np.allclose(np.linalg.norm(weights, axis=1), 1.0)):
            raise ValueError("Directions must be finite positive unit D-vectors")
        if type(initial_pulls_per_pair) is not int or initial_pulls_per_pair < 2:
            raise ValueError("KG needs at least two observations per arm-direction pair")
        initialization = initial_pulls_per_pair * K * len(weights)
        if type(horizon) is not int or horizon <= initialization:
            raise ValueError("The online horizon must leave adaptive iterations after initialization")
        reference = np.full(D, -0.01) if reference is None else np.asarray(reference, dtype=float)
        if reference.shape != (D,) or not np.isfinite(reference).all():
            raise ValueError("Reference must be a finite D-vector in the cost coordinates")
        if not np.isfinite(exploration_coefficient) or exploration_coefficient <= 0:
            raise ValueError("Exploration coefficient must be finite and positive")
        self.K, self.D, self.S = K, D, len(weights)
        self.weights = weights.copy()
        self.reference = reference.copy()
        self.horizon = horizon
        self.initial_pulls_per_pair = initial_pulls_per_pair
        self.initialization_iterations = initialization
        self.exploration_coefficient = exploration_coefficient
        self.rng = np.random.default_rng(seed)
        self.n = 0
        self.counts = np.zeros((self.S, K), dtype=int)
        self.mean_costs = np.zeros((self.S, K, D))
        self.m2_costs = np.zeros((self.S, K, D))
        self.last_decision = None

    def select_action(self):
        if self.n >= self.horizon:
            raise RuntimeError("The online horizon is exhausted")
        if self.n < self.initialization_iterations:
            # Repeat Eq. 43's entire arm-direction schedule twice.
            pair = self.n % (self.K * self.S)
            self.last_decision = None
            return pair // self.S, pair % self.S
        j = int(self.rng.integers(self.S))
        self.last_decision = self.decision_statistics(j)
        return int(np.argmax(self.last_decision["scores"])), j

    def _observe(self, arm, j, cost_vec):
        if (not isinstance(arm, (int, np.integer)) or not 0 <= arm < self.K
                or not isinstance(j, (int, np.integer)) or not 0 <= j < self.S):
            raise ValueError("Invalid arm or direction index")
        cost = np.asarray(cost_vec, dtype=float)
        if cost.shape != (self.D,) or not np.isfinite(cost).all():
            raise ValueError("Expected a finite D-dimensional cost observation")
        if self.n >= self.horizon:
            raise RuntimeError("The online horizon is exhausted")
        self.counts[j, arm] += 1
        count = self.counts[j, arm]
        delta = cost - self.mean_costs[j, arm]
        self.mean_costs[j, arm] += delta / count
        self.m2_costs[j, arm] += delta * (cost - self.mean_costs[j, arm])
        self.n += 1
        return cost, count

    def _exploration(self, j, means, m2):
        counts = self.counts[j]
        if (counts < 2).any():
            raise ValueError("Every arm in this direction needs at least two observations")
        denominator = counts * (counts - 1)
        if means.ndim == 2:
            denominator = denominator[:, None]
        errors = np.sqrt(np.maximum(m2, 0.0) / denominator)
        indices = gaussian_kg_index(means, errors)
        # k is the upcoming, one-based request. The final request is greedy.
        remaining = max(self.horizon - (self.n + 1), 0)
        scale = remaining * self.K * self.D * self.exploration_coefficient
        return errors, indices, scale * indices, remaining

    def recommended_arms(self):
        """Union of observed arms with best empirical exploitation per direction."""
        arms = set()
        for j in range(self.S):
            seen = np.flatnonzero(self.counts[j])
            if len(seen):
                scores = self.exploitation_scores(j)
                arms.add(int(seen[np.argmax(scores[seen])]))
        return np.array(sorted(arms), dtype=int)

    def state_dict(self):
        return {"n": self.n, "counts": self.counts.copy(),
                "mean_costs": self.mean_costs.copy(), "m2_costs": self.m2_costs.copy()}


class RTS_ScalarizedMultiObjectiveKG(_ScalarizedKGBase):
    """Average costs, subtract componentwise KG bonuses, then scalarize."""

    def update(self, arm, j, cost_vec):
        self._observe(arm, j, cost_vec)

    def exploitation_scores(self, j):
        return -length_cost(self.mean_costs[j], self.weights[j], self.reference)

    def decision_statistics(self, j):
        means = self.mean_costs[j]
        errors, indices, bonus, remaining = self._exploration(j, means, self.m2_costs[j])
        return {"standard_errors": errors, "kg_indices": indices,
                "exploration_bonus": bonus, "remaining_iterations": remaining,
                "exploitation_scores": self.exploitation_scores(j),
                "scores": -length_cost(means - bonus, self.weights[j], self.reference)}
