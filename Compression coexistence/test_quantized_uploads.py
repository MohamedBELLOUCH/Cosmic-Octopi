"""Small transport/integration checks; no PPO training or experiment output."""

import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def load_copy(name):
    spec = importlib.util.spec_from_file_location(name.replace(" ", "_"),
                                                HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


octopus = load_copy("octopus quantized")
oracle = load_copy("oracle quantized")
codec = octopus._codec


class QuantizedUploadsTest(unittest.TestCase):
    def test_codec_round_trip_and_validation(self):
        torch.manual_seed(321)
        model = codec.Network(17, 6, 64)
        if torch.cuda.is_available():
            model = model.cuda()
        before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        payload = codec.quantize_model(model)
        restored = codec.dequantize_model(payload)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for name, tensor in restored.state_dict().items():
            original = before[name]
            tolerance = original.abs().max().item() / 254.0 + 1e-7
            self.assertLessEqual((tensor - original).abs().max().item(), tolerance)
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertTrue(torch.equal(model.state_dict()[name].cpu(), original))
        # Decoded weights still support ordinary forward/backward training.
        (_, _), value = restored(torch.zeros(2, 17))
        value.sum().backward()
        self.assertIsNotNone(restored.value.out.weight.grad)
        for bad in [b"bad", payload[:-1], payload + b"extra"]:
            with self.assertRaises(ValueError):
                codec.dequantize_model(bad)
        with self.assertRaises(TypeError):
            codec.dequantize_model(model)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        zero_model = codec.dequantize_model(codec.quantize_model(model))
        self.assertTrue(all(torch.count_nonzero(p) == 0 for p in zero_model.parameters()))

    def test_real_octopus_upload_oracle_aggregation_and_overhead(self):
        torch.manual_seed(42)
        np.random.seed(42)
        xi = json.loads((HERE.parent / "Model calibration" / "Xi_matrix.json").read_text())
        schedule = [[1., 2., 3.] for _ in range(4)]
        hyper = {"temperature": 0.01, "scaling_constant": 5.0}
        agents = []
        try:
            # Real constructors, environments, upload methods and broadcast path.
            # Only three/five OU reward steps are used, with no PPO training.
            for index, episodes in enumerate([3, 5]):
                agents.append(octopus.Octopus(
                    index, [[0., 4.]], schedule,
                    [[episodes] * 3 for _ in range(4)], schedule, xi, hyper, 9.81))
            aggregator = oracle.Oracle(agents, 4, schedule, hyper)
            attributes = ["cheetah_model", "ant_model", "leg_model", "humanoid_model"]
            for class_id, attribute in enumerate(attributes):
                models = [getattr(agent, attribute) for agent in agents]
                payloads = [codec.quantize_model(model) for model in models]
                decoded = [codec.dequantize_model(payload) for payload in payloads]
                expected = oracle.federated_average(decoded, [3, 5])
                expected_bytes = sum(map(len, payloads))
                float_kib = sum(oracle.get_model_size_in_kb(model) for model in models)
                overhead, instability = aggregator.step(class_id, 0, 0.0)
                self.assertEqual(overhead, expected_bytes / 1024.0)
                self.assertEqual(aggregator.unquantized_overhead_increments_history[class_id][-1],
                                 float_kib)
                self.assertLess(overhead, float_kib)
                self.assertTrue(np.isfinite(instability))
                self.assertEqual(len(aggregator.contributing_octopi_history[class_id][-1]), 2)
                for agent in agents:
                    actual = getattr(agent, attribute)
                    for wanted, received in zip(expected.parameters(), actual.parameters()):
                        torch.testing.assert_close(received.cpu(), wanted)
                        self.assertEqual(received.dtype, torch.float32)
                # An iteration with no uploads must have zero overhead.
                overhead_next, _ = aggregator.step(class_id, 1, np.inf)
                self.assertEqual(overhead_next, 0.0)
                self.assertEqual(aggregator.uploaded_payload_sizes_history[class_id][-1], [])
                print(f"{attribute}: {float_kib / 2:.3f} KiB float32 -> "
                      f"{overhead / 2:.3f} KiB int8 (mean per upload); "
                      f"{100 * (1 - overhead / float_kib):.2f}% smaller", flush=True)
            # The oracle must report an upload failure, not count missing bytes
            # as an apparently successful compression saving.
            with torch.no_grad():
                next(agents[0].cheetah_model.parameters()).fill_(float("nan"))
            with self.assertRaisesRegex(RuntimeError, "upload failed"):
                aggregator.step(0, 2, -np.inf)
        finally:
            for agent in agents:
                for name in ["cheetah", "ant", "leg", "humanoid"]:
                    getattr(agent, f"{name}_marionette").close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
