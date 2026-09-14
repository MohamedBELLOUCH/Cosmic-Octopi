"""Check HV sampling against saved trajectories, without running experiments."""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import pickle
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Scalarized UCB"))
sys.path.insert(0, str(ROOT / "Scalarized Knowledge Gradient"))
import experiment_runner as ucb
import kg_runner as kg


class CheckpointTests(unittest.TestCase):
    def test_sampling_preserves_training_and_selects_saved_hypervolumes(self):
        for backend, directory in ((ucb, "Scalarized UCB"), (kg, "Scalarized Knowledge Gradient")):
            folder = ROOT / directory
            with (folder / "result_holistic_gravity.pkl").open("rb") as file:
                original = pickle.load(file)
            for interval, expected in (
                (1, list(range(1, 151))),
                (5, list(range(5, 151, 5))),
                (7, [*range(7, 151, 7), 150]),
                (200, [150]),
            ):
                with self.subTest(algorithm=directory, interval=interval):
                    setup = deepcopy(original["setup"])
                    setup["evaluation"]["checkpoint_every"] = interval

                    def recorded_training(settings, formulation, level, *args):
                        self.assertEqual(settings["bandit"]["n_online_iterations"], 150)
                        run = deepcopy(original["runs"][f"{formulation}/gamma_grav={level:g}"])
                        for key in ("checkpoint_iterations", "hypervolume_history", "evaluation_keys"):
                            run[key] = []
                        run["complete"] = False
                        return run

                    def recorded_evaluation(settings, level, subset, *args):
                        key = f"gamma_grav={level:g}/arms=" + ",".join(map(str, subset))
                        return original["evaluation_cache"][key]

                    with TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()), \
                            patch.object(backend, "train_online", side_effect=recorded_training) as train, \
                            patch.object(backend, "evaluate_subset", side_effect=recorded_evaluation) as evaluate, \
                            patch.object(backend, "save_result"):
                        result = backend.run(setup, Path(temporary) / "unused.pkl",
                                             folder / "script_holistic_gravity.py")
                    self.assertTrue(result["complete"])
                    self.assertEqual(train.call_count, 4)
                    needed_subsets = set()
                    for key, run in result["runs"].items():
                        baseline = original["runs"][key]
                        self.assertEqual(run["checkpoint_iterations"], expected)
                        np.testing.assert_array_equal(run["hypervolume_history"],
                                                      np.asarray(baseline["hypervolume_history"])[np.array(expected)-1])
                        np.testing.assert_array_equal(run["arm_history"], baseline["arm_history"])
                        self.assertEqual(len(run["arm_history"]), 150)
                        needed_subsets.update(baseline["evaluation_keys"][i-1] for i in expected)
                    self.assertEqual(set(result["evaluation_cache"]), needed_subsets)
                    self.assertEqual(evaluate.call_count, len(needed_subsets))

    def test_batch_configuration_is_deferred_and_uses_five(self):
        config = json.loads((ROOT / "Model-free experiments/setup.json").read_text())
        self.assertEqual(config["evaluation"]["checkpoint_every"], 5)
        self.assertEqual(config["common_bandit"]["n_online_iterations"], 150)
        self.assertEqual(config["n_independent_runs"], 10)


if __name__ == "__main__":
    unittest.main()
