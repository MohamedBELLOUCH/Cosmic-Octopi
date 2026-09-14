"""Exercise single-file storage/resume with stubs; never run simulations."""

from contextlib import redirect_stdout
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

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("_model_free_storage", HERE / "script.py")
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


class StubBackend:
    def __init__(self, calls, interrupt_at=None):
        self.calls = calls
        self.interrupt_at = interrupt_at
        self.resumed = False

    def source_fingerprints(self, entrypoint):
        return {str(entrypoint): "stub-source"}

    def run(self, setup, checkpoint, entrypoint, executor=None):
        self.calls.append(deepcopy(setup))
        if checkpoint.exists():
            with checkpoint.open("rb") as file:
                previous = pickle.load(file)
            assert previous["setup"] == setup and not previous["complete"]
            self.resumed = True
        result = {"setup": setup, "complete": False, "runs": {"stub": [1, 2, 3]}}
        with checkpoint.open("wb") as file:
            pickle.dump(result, file)
        if len(self.calls) == self.interrupt_at:
            raise KeyboardInterrupt
        result["complete"] = True
        return result


def files_below(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*") if p.is_file()}


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((HERE / "setup.json").read_text())

    def test_only_one_output_for_120_experiments_and_complete_resume_is_read_only(self):
        calls = []
        backends = {name: StubBackend(calls) for name in ("UCB", "KG")}
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            (folder / "setup.json").write_text(json.dumps(self.config))
            (folder / "existing_result.pkl").write_bytes(b"unchanged existing result")
            before = files_below(folder)
            with patch.object(suite, "HERE", folder):
                result = suite.run_suite(self.config, backends)
                self.assertTrue(result["complete"])
                self.assertEqual(len(result["experiments"]), 120)
                self.assertEqual(len(calls), 120)
                self.assertEqual({c["seed"] for c in calls}, set(range(42, 52)))
                self.assertEqual({c["evaluation"]["checkpoint_every"] for c in calls}, {5})
                after = files_below(folder)
                self.assertEqual(set(after) - set(before), {"model_free_results.pkl"})
                for name, digest in before.items():
                    self.assertEqual(after[name], digest)
                self.assertFalse(any(p.is_dir() for p in folder.iterdir()))
                with (folder / "model_free_results.pkl").open("rb") as file:
                    self.assertEqual(pickle.load(file), result)
                suite.run_suite(self.config, backends)
                self.assertEqual(len(calls), 120)
                self.assertEqual(files_below(folder), after)

                changed = deepcopy(self.config)
                changed["base_seed"] += 1
                with self.assertRaisesRegex(ValueError, "Settings/code differ"):
                    suite.run_suite(changed, backends)
                self.assertEqual(files_below(folder), after)

    def test_keyboard_interrupt_saves_partial_record_and_resumes(self):
        self.config["n_independent_runs"] = 1
        calls = []
        backend = StubBackend(calls, interrupt_at=2)
        backends = {name: backend for name in ("UCB", "KG")}
        with TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            folder = Path(directory)
            with patch.object(suite, "HERE", folder):
                with self.assertRaises(KeyboardInterrupt):
                    suite.run_suite(self.config, backends)
                self.assertEqual(set(files_below(folder)), {"model_free_results.pkl"})
                with (folder / "model_free_results.pkl").open("rb") as file:
                    interrupted = pickle.load(file)
                self.assertFalse(interrupted["complete"])
                self.assertEqual(len(interrupted["experiments"]), 2)
                self.assertEqual(sum(r["complete"] for r in interrupted["experiments"].values()), 1)
                result = suite.run_suite(self.config, backends)
                self.assertTrue(result["complete"])
                self.assertTrue(backend.resumed)
                self.assertEqual(len(result["experiments"]), 12)
                self.assertEqual(len(calls), 13)

    def test_failed_serialization_does_not_damage_previous_output(self):
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            output = folder / "model_free_results.pkl"
            output.write_bytes(b"previous valid checkpoint")
            with patch.object(suite.pickle, "dump", side_effect=OSError("disk write failed")):
                with self.assertRaisesRegex(OSError, "disk write failed"):
                    suite.save_bundle(output, {"new": "data"})
            self.assertEqual(output.read_bytes(), b"previous valid checkpoint")
            self.assertEqual(set(files_below(folder)), {"model_free_results.pkl"})


if __name__ == "__main__":
    unittest.main()
