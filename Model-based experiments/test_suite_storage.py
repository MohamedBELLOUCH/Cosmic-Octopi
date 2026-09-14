"""Test routing, single-file storage, resume and progress without optimization."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_model_based_storage", HERE / "script.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class StubBackend:
    __file__ = __file__

    def __init__(self, kind, calls, interrupt_at=None):
        self.kind, self.calls, self.interrupt_at = kind, calls, interrupt_at

    def source_fingerprints(self):
        return {}

    def save_result(self, path, result):
        with path.open("wb") as file:
            pickle.dump(result, file, protocol=pickle.HIGHEST_PROTOCOL)

    def run_condition(self, result, key, formulation, *args):
        if self.kind == "qNParEGO":
            parameter, level, setup, hp, xi, sizes, output, approach = args
        else:
            level, setup, hp, xi, sizes, output, *approaches = args
            approach = approaches[0] if approaches else self.kind
            parameter = setup["sweep"]["parameter"]
            assert self.kind == (approach if result["experiment"] == "gravity" else result["experiment"])
        assert approach == result["approach"]
        assert parameter == setup["sweep"]["parameter"]
        assert level in setup["sweep"]["values"]
        assert key == f"{formulation}/{parameter}={level:g}"
        assert hp["dimension"] == setup[approach]["horizon"]
        assert hp["n_iterations"] == 20
        previous = result["runs"].get(key)
        if previous and len(previous["direction_angle_history"]) == hp["n_iterations"]:
            return
        self.calls.append((result["algorithm"], approach, result["experiment"], setup["seed"], key))
        run = {"X_unit": np.zeros((4, hp["dimension"])), "direction_angle_history": [],
               "hypervolume_history": [], "evaluation_counts": [], "formulation": formulation}
        result["runs"][key] = run
        self.save_result(output, result)
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        run["X_unit"] = np.zeros((88, hp["dimension"]))
        run["direction_angle_history"] = [0.5] * 20
        run["hypervolume_history"] = list(np.linspace(0.1, 0.8, 21))
        run["evaluation_counts"] = list(range(8, 89, 4))
        self.save_result(output, result)


def files_below(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "setup.json").read_text())
        self.xi = {name: np.array(values) for name, values in self.config["Xi_matrices"].items()}
        self.sizes = np.array([1., 2., 3., 4.])
        self.calls = []

    def backends(self, interrupt_at=None):
        return {name: StubBackend(name, self.calls, interrupt_at)
                for name in ("holistic", "reductionist", "synchronization", "heterogeneity", "qNParEGO")}

    def test_all_120_experiments_have_one_output_and_progress(self):
        backends = self.backends()
        console = io.StringIO()
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(console):
            folder = Path(directory)
            (folder / "setup.json").write_text(json.dumps(self.config))
            (folder / "existing_result.pkl").write_bytes(b"unchanged")
            before = files_below(folder)
            with patch.object(suite, "EXPERIMENT_DIR", folder):
                result = suite.run_suite(self.config, backends, self.xi, self.sizes, show_progress=True)
                self.assertTrue(result["complete"])
                self.assertEqual(len(result["experiments"]), 120)
                self.assertEqual(len(self.calls), 480)
                self.assertEqual({call[3] for call in self.calls}, set(range(42, 52)))
                self.assertEqual(len({call[:3] for call in self.calls}), 12)
                self.assertIn("120/120", console.getvalue())
                self.assertIn("step 20/20", console.getvalue())
                self.assertIn("init 4/8", console.getvalue())
                self.assertIn("Run 10/10 qNParEGO", console.getvalue())
                after = files_below(folder)
                self.assertEqual(set(after)-set(before), {"model_based_results.pkl"})
                self.assertFalse(any(p.is_dir() for p in folder.iterdir()))
                for name, digest in before.items():
                    self.assertEqual(after[name], digest)
                with (folder / "model_based_results.pkl").open("rb") as file:
                    saved = pickle.load(file)
                self.assertEqual(len(saved["experiments"]), 120)
                self.assertTrue(all(len(r["runs"]) == 4 for r in saved["experiments"].values()))
                suite.run_suite(self.config, backends, self.xi, self.sizes, show_progress=True)
                self.assertEqual(len(self.calls), 480)
                self.assertEqual(files_below(folder), after)
                changed = deepcopy(self.config)
                changed["base_seed"] += 1
                with self.assertRaisesRegex(ValueError, "Settings/code differ"):
                    suite.run_suite(changed, backends, self.xi, self.sizes)
                self.assertEqual(files_below(folder), after)

    def test_interrupt_preserves_partial_condition_and_resumes(self):
        self.config["n_independent_runs"] = 1
        backends = self.backends(interrupt_at=6)
        console = io.StringIO()
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()), redirect_stderr(console):
            folder = Path(directory)
            with patch.object(suite, "EXPERIMENT_DIR", folder):
                with self.assertRaises(KeyboardInterrupt):
                    suite.run_suite(self.config, backends, self.xi, self.sizes, show_progress=True)
                self.assertEqual(set(files_below(folder)), {"model_based_results.pkl"})
                with (folder / "model_based_results.pkl").open("rb") as file:
                    interrupted = pickle.load(file)
                self.assertFalse(interrupted["complete"])
                self.assertEqual(len(interrupted["experiments"]), 2)
                self.assertEqual(sum(r["complete"] for r in interrupted["experiments"].values()), 1)
                for backend in backends.values():
                    self.assertEqual(backend.save_result.__func__, StubBackend.save_result)
                console.seek(0)
                console.truncate(0)
                result = suite.run_suite(self.config, backends, self.xi, self.sizes, show_progress=True)
                self.assertTrue(result["complete"])
                self.assertEqual(len(self.calls), 49)
                self.assertIn("1/12", console.getvalue())
                self.assertIn("12/12", console.getvalue())

    def test_serialization_failure_preserves_previous_file(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "model_based_results.pkl"
            output.write_bytes(b"previous checkpoint")
            with patch.object(suite.pickle, "dump", side_effect=OSError("write failed")):
                with self.assertRaisesRegex(OSError, "write failed"):
                    suite.save_bundle(output, {"new": "result"})
            self.assertEqual(output.read_bytes(), b"previous checkpoint")
            self.assertEqual(set(files_below(folder)), {"model_based_results.pkl"})


if __name__ == "__main__":
    unittest.main()
