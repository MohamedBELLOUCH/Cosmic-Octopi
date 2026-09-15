"""Focused checks for delayed credit assignment and Pareto cost orientation."""
import unittest
import numpy as np
from pareto_online import CostParetoUCB, DelayedCostFeedback


class FeedbackTests(unittest.TestCase):
    def test_lag_and_terminal_pending(self):
        agent = CostParetoUCB(3, seed=5)
        feedback = DelayedCostFeedback(agent, [100, 1])
        actions, observed_before_selection = [], []
        for t, costs in enumerate(([10, .1], [20, .4], [30, .8])):
            observed_before_selection.append(agent.n)
            actions.append(agent.select_action())
            feedback.observe(t, actions[-1], costs)
        self.assertEqual(actions, [0, 1, 2])
        self.assertEqual(observed_before_selection, [0, 0, 1])
        self.assertEqual(agent.n, 2)
        np.testing.assert_allclose(feedback.observations[0]['costs'], [10, .4])
        np.testing.assert_allclose(feedback.observations[1]['costs'], [20, .8])
        np.testing.assert_allclose(agent.mean_vec[:2], [[-.1, -.4], [-.2, -.8]])
        self.assertEqual(feedback.pending, (2, 2, 30.0))
        self.assertEqual(agent.ni.tolist(), [1, 1, 0])
        self.assertEqual(feedback.observations[1]['action_iteration'], 1)
        self.assertEqual(feedback.observations[1]['observation_iteration'], 2)

    def test_recommendation_minimizes_cost_and_excludes_unseen(self):
        agent = CostParetoUCB(4, seed=7)
        for arm, cost in enumerate(([.1, .4], [.4, .1], [.8, .8])):
            agent.update_cost(arm, cost)
        np.testing.assert_array_equal(agent.recommended_arms(), [0, 1])
        expected = agent.mean_vec + np.sqrt(2*np.log(agent.n)/np.maximum(agent.ni, 1))[:,None]
        np.testing.assert_allclose(agent._compute_ucb_vectors(), expected)

    def test_pending_initialization_arm_is_observed_before_adaptive_selection(self):
        agent = CostParetoUCB(2, seed=3)
        feedback = DelayedCostFeedback(agent, [10, 1])
        for t in range(4):
            arm = agent.select_action()
            if t == 2:
                self.assertEqual(arm, 1)
            feedback.observe(t, arm, [1+arm, .2])
        self.assertTrue(np.all(agent.ni > 0))
        self.assertEqual(agent.n, 3)


if __name__ == '__main__':
    unittest.main()
