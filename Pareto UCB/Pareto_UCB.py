"""Pareto UCB extracted from Scalarized UCB/bandits.ipynb.

The original EmpiricalParetoUCB and its three helpers are copied unchanged.
This toy implementation maximizes reward vectors. Cost conversion and the
(overhead_t, instability_{t+1}) feedback alignment belong to the future
model-free baseline environment; this module does not implement that adapter.
Importing it does not run the toy experiment.
"""

import numpy as np

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return np.all(a >= b) and np.any(a > b)


def get_pareto_indices(mean_rewards: np.ndarray) -> np.ndarray:
    """Indices of non-dominated arms."""
    K = len(mean_rewards)
    is_pareto = np.ones(K, dtype=bool)
    for i in range(K):
        for j in range(K):
            if i != j and dominates(mean_rewards[j], mean_rewards[i]):
                is_pareto[i] = False
                break
    return np.where(is_pareto)[0]


def hypervolume_2d(front: np.ndarray, ref: np.ndarray = np.array([0.0, 0.0])) -> float:
    """Exact 2D hypervolume for a non-dominated front (staircase integration)."""
    if len(front) == 0:
        return 0.0
    pts = np.unique(np.array(front), axis=0)
    pts = pts[np.argsort(pts[:, 0])]          # sort ascending by objective 1
    hv = 0.0
    prev_x = ref[0]
    for p in pts:
        hv += (p[0] - prev_x) * (p[1] - ref[1])
        prev_x = p[0]
    return hv


class EmpiricalParetoUCB:
    def __init__(self, K: int, D: int):
        self.K = K
        self.D = D
        self.ni = np.zeros(K, dtype=int)
        self.sum_vec = np.zeros((K, D))
        self.mean_vec = np.zeros((K, D))
        self.n = 0
        self.hypervolume_history = []

    def _compute_ucb_vectors(self) -> np.ndarray:
        """UCB vector = mean + standard UCB confidence (problem-independent)."""
        # Avoid division by zero for unpulled arms (handled by init)
        c = np.sqrt(2 * np.log(max(self.n, 1)) / np.maximum(self.ni, 1))
        ucb = self.mean_vec + c[:, np.newaxis]   # broadcast scalar confidence to all objectives
        return ucb

    def run(self, bandit_pull_fn, T: int, monitor_hv: bool = True) -> dict:
        """
        Run for horizon T.
        bandit_pull_fn(arm) must return a D-dimensional reward vector.
        """
        # ------------------- Initialization: play each arm once -------------------
        for arm in range(self.K):
            rew = bandit_pull_fn(arm)
            self.ni[arm] += 1
            self.sum_vec[arm] += rew
            self.mean_vec[arm] = self.sum_vec[arm] / self.ni[arm]
        self.n = self.K

        # ------------------- Main loop with optional hypervolume monitoring -------------------
        step = self.n
        while step < T:
            ucb = self._compute_ucb_vectors()
            pareto_idx = get_pareto_indices(ucb)          # current non-dominated UCB vectors
            arm = np.random.choice(pareto_idx)            # uniform selection from current Pareto set

            rew = bandit_pull_fn(arm)

            # Update statistics
            self.ni[arm] += 1
            self.sum_vec[arm] += rew
            self.mean_vec[arm] = self.sum_vec[arm] / self.ni[arm]
            self.n += 1
            step += 1

            current_pareto_idx = get_pareto_indices(self.mean_vec)
            current_front = self.mean_vec[current_pareto_idx]
            hv = hypervolume_2d(current_front)
            self.hypervolume_history.append(hv)

        return self.hypervolume_history.copy()
