"""Focused recovery/resource regressions; no SDK, model computation or jobs."""
import getpass
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hw00.watchdog import cleanup, release
from hw00.contract import Geometry, Tensor, output_schema
from hw00.store import Store


class CleanupTests(unittest.TestCase):
    def exercise(self, oversized=False, terminate_error=False, old=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            (root / "run.jobs").write_text("wsjob-new1\n")
            if oversized:
                with (root / "run.log").open("w") as log:
                    log.seek((8 << 20) + 1)
                    log.write("x")
            if old:
                (root / "run.log").write_text("Job id: wsjob-old1\n")
            baseline = {"items": [{"meta": {"name": "wsjob-old1"}}]} if old else {"items": []}
            seen = []
            def terminate(proc):
                if terminate_error:
                    raise ProcessLookupError("process vanished")
            def release_fn(root, name, submitted, query_fn):
                seen.extend(submitted)
                return [{"id": j, "released": True} for j in submitted]
            result = cleanup(root, "run", baseline, object(), query_fn=lambda *a: baseline if a[0] == "jobs" else {"items": []},
                             release_fn=release_fn, terminate_fn=terminate)
            self.assertEqual(seen, ["wsjob-new1"])
            self.assertTrue(result["cleanup_errors"])
            self.assertTrue((root / "run-systems-after.json").exists())

    def test_oversized_log_does_not_bypass_durable_job_cleanup(self):
        self.exercise(oversized=True)

    def test_termination_exception_does_not_bypass_job_cleanup(self):
        self.exercise(terminate_error=True)

    def test_old_log_id_does_not_block_new_job_cleanup(self):
        self.exercise(old=True)

    def test_owner_check_precedes_cancel_and_terminal_system_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            job = {"spec": {"user": {"username": "other-user"}}, "status": {"phase": "RUNNING"}}
            cancelled = []
            with self.assertRaises(RuntimeError):
                release(root, "run", {"wsjob-a"}, query_fn=lambda *a: job, cancel_fn=cancelled.append)
            self.assertEqual(cancelled, [])
            job["spec"]["user"]["username"] = getpass.getuser()
            def query(*args):
                return {"items": []} if args[0] == "systems" else job
            def cancel(jid):
                cancelled.append(jid)
                job["status"]["phase"] = "CANCELLED"
            result = release(root, "run", {"wsjob-a"}, query_fn=query, cancel_fn=cancel)
            self.assertEqual(cancelled, ["wsjob-a"])
            self.assertTrue(result[0]["released"])
            self.assertTrue(result[0]["cancelled"])


class CheckpointTests(unittest.TestCase):
    def test_all_layers_have_explicit_persistent_state_and_full_vocab(self):
        specs = [output_schema(s, 4, 2) for s in range(3)]
        all_names = [name for spec in specs for name in spec]
        self.assertEqual(sum(name.endswith("_key") for name in all_names), 16)
        self.assertEqual(sum(name.endswith("_recurrent") for name in all_names), 48)
        self.assertEqual(sum(name.endswith("_conv") for name in all_names), 48)
        self.assertEqual(specs[0]["layer_00_conv"].shape, (10240, 4))
        self.assertEqual(specs[2]["last_logits"].shape, (248320,))
        with self.assertRaises(ValueError):
            output_schema(0, 2049, 1)

    def test_restart_load_detects_identity_and_payload_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "fixture.bin"
            source.write_bytes(b"\x00\x80\x00\x00")
            store = Store(root / "store")
            schema = {"hidden": Tensor("bf16_le", (1, 2))}
            identity = dict(request="fixture", generation=0, stage=0, position=1, revision="test-only")
            with store.locked():
                key = store.commit("a"*32, identity, schema, {"hidden": source}, {"fixture_only": True})
                store.update({"snapshot": key})
            restored = Store(root / "store")
            self.assertEqual(restored.cursor()["snapshot"], key)
            _, paths = restored.load(key, identity, schema)
            with self.assertRaises(ValueError):
                restored.load(key, dict(identity, generation=1), schema)
            paths["hidden"].write_bytes(b"\x00"*4)
            with self.assertRaises(ValueError):
                restored.load(key, identity, schema)


if __name__ == "__main__":
    unittest.main()
