"""Check SQLite claims, cancellation and the single physical-device slot."""

import os
import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import Mock, patch
from uuid import uuid4

from atc_service.config import Settings
from atc_service.store import FileRecord, ResourceError, Row, Store
from atc_service.worker import _recover, _stop_group, run_worker


class SchedulingTest(unittest.TestCase):
    """Exercise the same queue and shutdown paths used by the service."""

    def test_claim_once_and_recover(self) -> None:
        """A concurrent claim creates one attempt which recovery finalizes."""
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(Path(directory), "test")
            store = Store(settings)
            store.migrate()
            store.issue_token("owner")
            model = FileRecord(uuid4(), "onnx", "model.onnx", 1, "digest")
            store.add_asset("owner", model)
            job = store.submit(
                "owner", "inspect_onnx", model.id, "test", None, True, "key"
            )
            with ThreadPoolExecutor(max_workers=4) as pool:
                claims = list(pool.map(store.claim, ["cpu"] * 4))
            self.assertEqual(sum(item is not None for item in claims), 1)
            _recover(settings, store)
            self.assertEqual(store.job("owner", job["id"])[0]["state"], "failed")

    def test_concurrent_submission_and_cancelled_result(self) -> None:
        """Concurrent retries share one job and cancellation wins over publication."""
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Settings(Path(directory), "test", max_pending_jobs=1))
            store.migrate()
            store.issue_token("owner")
            model = FileRecord(uuid4(), "onnx", "model.onnx", 1, "digest")
            store.add_asset("owner", model)

            def submit(key: str) -> Row:
                """Submit an inspection using an account-scoped idempotency key.

                Parameters
                ----------
                key
                    Client request identifier.

                Returns
                -------
                dict
                    Persistent job record.
                """
                return store.submit("owner", "inspect_onnx", model.id, "test", None, False, key)

            with ThreadPoolExecutor(max_workers=4) as pool:
                jobs = list(pool.map(submit, ["same"] * 4))
            self.assertEqual(len({job["id"] for job in jobs}), 1)
            with self.assertRaises(ResourceError) as limit:
                submit("different")
            self.assertEqual(limit.exception.status, 429)
            attempt = store.claim("cpu")
            assert attempt is not None
            store.cancel("owner", attempt["id"])
            self.assertFalse(store.finish(attempt, 0, None, [model]))
            job, outputs = store.job("owner", attempt["id"])
            self.assertEqual((job["state"], outputs), ("cancelled", []))

    @unittest.skipUnless(sys.platform == "linux", "Linux service scheduler")
    def test_inference_waits_for_profile(self) -> None:
        """Keep a waiting inference out of the device lane until profiling exits."""
        with tempfile.TemporaryDirectory() as directory:
            settings = Settings(Path(directory), "test")
            queue = Mock(spec=Store)
            queue.running.return_value = []
            jobs = [{"operation": "profile_om"}, {"operation": "run_npu"}]
            release = Event()
            finished = Event()
            history: list[str] = []

            def claim(lane: str) -> Row | None:
                """Supply two consecutive jobs to the device lane.

                Parameters
                ----------
                lane
                    Requested execution queue.

                Returns
                -------
                dict or None
                    Next device job, if queued.
                """
                if finished.is_set():
                    raise StopIteration
                return jobs.pop(0) if lane == "device" and jobs else None

            def execute(config: Settings, store: Store, slot: int, job: Row) -> None:
                """Hold the profiling slot across several scheduler ticks.

                Parameters
                ----------
                config, store, slot, job
                    Scheduler settings, queue, slot identity and claimed job.
                """
                history.append(job["operation"])
                if job["operation"] == "profile_om":
                    release.wait(5)
                    history.append("profile exited")
                else:
                    finished.set()

            queue.claim.side_effect = claim
            ticks = 0

            def tick(seconds: float) -> None:
                """Release the profiler after several polls.

                Parameters
                ----------
                seconds
                    Requested poll interval.
                """
                nonlocal ticks
                ticks += 1
                self.assertLess(ticks, 500, "Scheduler stopped making progress")
                if ticks >= 3:
                    release.set()
                finished.wait(0.01)

            with (
                patch("atc_service.worker.Store", return_value=queue),
                patch("atc_service.worker._execute", side_effect=execute),
                patch("atc_service.worker.sleep", side_effect=tick),
            ):
                with self.assertRaises(StopIteration):
                    run_worker(settings)
            self.assertEqual(history, ["profile_om", "profile exited", "run_npu"])

    @unittest.skipUnless(sys.platform == "linux", "POSIX process groups")
    def test_cancel_drains_descendants(self) -> None:
        """A child ignoring TERM must exit before the slot is released."""
        program = (
            "import os,signal,time;"
            "pid=os.fork();"
            "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
            "print('ready',flush=True);time.sleep(60)"
        )
        with subprocess.Popen(
            [sys.executable, "-c", program],
            stdout=subprocess.PIPE,
            start_new_session=True,
        ) as process:
            assert process.stdout is not None
            self.assertEqual(process.stdout.readline(), b"ready\n")
            self.assertEqual(process.stdout.readline(), b"ready\n")
            _stop_group(process)
            with self.assertRaises(ProcessLookupError):
                os.killpg(process.pid, 0)
