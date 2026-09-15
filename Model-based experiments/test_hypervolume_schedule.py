"""Check that model-based hypervolume sampling does not alter BO steps."""

from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "MORBO"))
import script_holistic_gravity as morbo

spec = importlib.util.spec_from_file_location("_scheduled_qnparego", ROOT / "qNParEGO" / "runner.py")
qnparego = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qnparego)


def morbo_run(step):
    n = 8 + 4 * step
    return {
        "cost_realizations": np.ones((n, 2, 2)),
        "trust_region_length": 0.8,
        "center": np.zeros(3),
        "X_unit": np.zeros((n, 3)),
        "trust_region_length_history": [0.8],
        "direction_angle_history": [],
        "hypervolume_history": [0.1],
        "evaluation_counts": [8],
        "pending_X": np.zeros((4, 3)),
        "pending_angle": 0.2,
    }


class HypervolumeScheduleTests(unittest.TestCase):
    def test_morbo_and_qnparego_sample_steps_two_to_twenty(self):
        hp = {
            "batch_size": 4,
            "n_iterations": 20,
            "hypervolume_checkpoint_every": 2,
            "trust_region_length_minimum": 0.01,
            "trust_region_length_maximum": 1.6,
            "trust_region_success_multiplier": 1.5,
            "trust_region_failure_multiplier": 0.7,
            "trust_region_center_old_weight": 0.7,
            "trust_region_center_new_weight": 0.3,
        }
        directions = np.array([[0.5, 0.5]])
        reference = np.array([1.0, 1.0])

        for backend, factory in ((morbo, morbo_run), (qnparego, lambda _: {
                "cost_realizations": np.ones((8, 2, 2)),
                "X_unit": np.zeros((8, 3)),
                "direction_angle_history": [], "hypervolume_history": [0.1],
                "evaluation_counts": [8], "pending_X": np.zeros((4, 3)), "pending_angle": 0.2,
            })):
            run = factory(0)
            with patch.object(backend, "estimated_hypervolume", side_effect=range(1, 11)) as evaluate:
                for step in range(1, 21):
                    if backend is morbo:
                        run.update(morbo_run(step))
                        run["direction_angle_history"] = [0.2] * (step - 1)
                        run["hypervolume_history"] = [0.1] + list(range(1, step // 2))
                        run["evaluation_counts"] = [8, *range(16, 8 + 4 * step, 8)]
                    else:
                        run["X_unit"] = np.zeros((8 + 4 * step, 3))
                    backend.finish_iteration(run, "RTS", 0.2, hp, directions, reference)
            self.assertEqual(len(run["direction_angle_history"]), 20)
            self.assertEqual(evaluate.call_count, 10)
            self.assertEqual(run["evaluation_counts"], [8, *range(16, 89, 8)])
            self.assertEqual(len(run["hypervolume_history"]), 11)

    def test_default_preserves_every_step_sampling(self):
        hp = {
            "batch_size": 4,
            "n_iterations": 3,
            "trust_region_length_minimum": 0.01,
            "trust_region_length_maximum": 1.6,
            "trust_region_success_multiplier": 1.5,
            "trust_region_failure_multiplier": 0.7,
            "trust_region_center_old_weight": 0.7,
            "trust_region_center_new_weight": 0.3,
        }
        run = morbo_run(0)
        with patch.object(morbo, "estimated_hypervolume", side_effect=range(1, 4)) as evaluate:
            for step in range(1, 4):
                run.update(morbo_run(step))
                run["direction_angle_history"] = [0.2] * (step - 1)
                run["hypervolume_history"] = [0.1] + list(range(1, step))
                run["evaluation_counts"] = [8, *range(12, 8 + 4 * step, 4)]
                morbo.finish_iteration(run, "RTS", 0.2, hp, np.array([[0.5, 0.5]]), np.array([1.0, 1.0]))
        self.assertEqual(evaluate.call_count, 3)
        self.assertEqual(run["evaluation_counts"], [8, 12, 16, 20])


if __name__ == "__main__":
    unittest.main()
