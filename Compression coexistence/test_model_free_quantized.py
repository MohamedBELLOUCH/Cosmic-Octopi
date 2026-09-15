"""Transport and dispatch checks with mocked simulations; no experiment runs."""
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor
import unittest
from unittest.mock import patch

import numpy as np
import torch

import model_free_quantized as common
import holistic_runner_quantized as holistic
import reductionist_runner_quantized as reductionist
import evaluate_holistic_comparison_quantized as holistic_eval
import evaluate_reductionist_comparison_quantized as reductionist_eval


def inspect_worker_inputs():
    # Inspect a real spawned evaluator process without evaluating a sequence.
    _, setup, xi, sizes, _, _ = reductionist_eval._worker
    return setup["approach"], xi["Cheetah"], sizes


class QuantizedModelFreeTests(unittest.TestCase):
    def test_spawned_worker_receives_frozen_quantized_inputs(self):
        setup = reductionist.load_setup()
        xi, sizes = reductionist.simulator_inputs(setup)
        with ProcessPoolExecutor(max_workers=1, initializer=reductionist_eval.worker_init,
                                 initargs=(setup, np.array([[0., 1.]]), [123],
                                           dict(zip(common.CLASSES, sizes)))) as pool:
            approach, matrix, received = pool.submit(inspect_worker_inputs).result(timeout=60)
        self.assertEqual(approach, "reductionist")
        np.testing.assert_array_equal(matrix, xi["Cheetah"])
        np.testing.assert_array_equal(received, sizes)

    def test_payload_inputs_preserve_rng_and_benchmark_references(self):
        setup = holistic.load_setup()
        common._measure_payloads.cache_clear()
        state = torch.random.get_rng_state().clone()
        xi, sizes = common.simulator_inputs(setup)
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        measurements = common.payload_measurements(setup)
        for i, name in enumerate(common.CLASSES):
            self.assertEqual(sizes[i] * 1024, measurements[name]["quantized_bytes"])
            self.assertLess(measurements[name]["quantized_bytes"], measurements[name]["float32_bytes"])
            np.testing.assert_array_equal(xi[name], setup["Xi_matrices"][name])
        changed = deepcopy(setup)
        changed["reference_costs"][0] /= 4
        with self.assertRaisesRegex(ValueError, "reference_costs"):
            common.check_standard_settings(changed)

    def test_all_algorithms_receive_quantized_sizes_and_preserve_margin_units(self):
        for runner in (holistic, reductionist):
            setup = runner.load_setup()
            _, sizes = runner.simulator_inputs(setup)
            expected_arms = np.linspace(0, 1, 7) if runner is holistic else np.linspace(-10, 20, 7)
            for algorithm, backend in (("Scalarized UCB", runner.ucb_runner), ("Scalarized KG", runner.kg_runner)):
                saved, called = [], []

                def fake_train(cfg, formulation, level, xi, received_sizes, arms, weights):
                    np.testing.assert_array_equal(received_sizes, sizes)
                    np.testing.assert_allclose(arms, expected_arms)
                    self.assertEqual(cfg["reference_costs"], setup["reference_costs"])
                    self.assertEqual(cfg["bandit"]["n_online_iterations"], 150)
                    called.append(formulation)
                    return {"recommended_arms": [np.array([0, 6])], "environment_seed": 123}

                with patch.object(backend, "train_online", fake_train), \
                     patch.object(runner, "save", side_effect=lambda path, record: saved.append((path, record))):
                    runner.train(algorithm)
                self.assertEqual(called, ["RTS", "STR"])
                for path, record in saved:
                    self.assertEqual(path.parent, common.HERE)
                    self.assertTrue(path.name.endswith("quantized.pkl"))
                    self.assertEqual(record["compression"], "int8")

            def fake_online(cfg, xi, received_sizes, seed, select, observe, approach):
                np.testing.assert_array_equal(received_sizes, sizes)
                costs = []
                for k in range(150):
                    margin = select(0, k)
                    self.assertTrue(np.any(np.isclose(margin, expected_arms)))
                    increment = np.array([sizes[0], (k % 5) / 10.])
                    observe(0, k, margin, increment)
                    costs.append(increment)
                return np.sum(costs, axis=0)

            with patch.object(runner, "run_online", fake_online):
                trace = runner.train_pareto(setup, {}, sizes, expected_arms)
            self.assertEqual(trace["n_completed_observations"], 149)
            np.testing.assert_allclose([o["costs"][1] for o in trace["delayed_observations"]],
                                       trace["increment_costs"][1:, 1])

    def test_evaluation_dispatch_and_measured_comparison(self):
        for runner, evaluation in ((holistic, holistic_eval), (reductionist, reductionist_eval)):
            setup = runner.load_setup()
            xi, sizes = runner.simulator_inputs(setup)
            calls = []

            class FakeSimulator:
                def evaluate_sequence(self, config, matrices, payloads, sequence, mode, seed):
                    np.testing.assert_array_equal(payloads, sizes)
                    calls.append(sequence)
                    return np.array([2 * sizes[0], 0.25])

            state = (FakeSimulator(), setup, xi, sizes, np.array([[0., .5, 1.]]), [123])
            with patch.object(evaluation, "_worker", state, create=True):
                evaluation.worker_evaluate((0, 0))
            np.testing.assert_allclose(calls[0], [0, .5, 1] if runner is holistic else [-10, 5, 20])
            original = common.read_standard(setup)
            rows = deepcopy(original["summary"])
            for row in rows:
                row["hypervolume"] *= .9
            measured = common.compare_standard({"setup": setup, "summary": rows})
            # Reporting must not force the hypothesized improvement.
            self.assertTrue(all(row["difference"] < 0 for row in measured))


if __name__ == "__main__":
    unittest.main(verbosity=2)
