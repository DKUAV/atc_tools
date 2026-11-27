"""Execute queued jobs in restricted Docker containers."""

from __future__ import annotations

import json
import logging
import shutil
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from time import monotonic, sleep
from uuid import uuid4

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container
from docker.types import LogConfig, Mount, Ulimit

from atc_service.config import Settings
from atc_service.store import FileRecord, ResourceError, Row, Store

LOG = logging.getLogger("atc_service.worker")


def _capture(container: Container, path: Path, limit: int) -> None:
    """Save a live log with a bounded prefix and tail.

    Parameters
    ----------
    container
        Running or stopped container with a readable Docker log.
    path
        Attempt log path owned by the worker.
    limit
        Maximum number of bytes to retain.
    """
    marker = b"\n[Earlier task output truncated]\n"
    prefix_limit = limit * 3 // 4
    tail_limit = max(0, limit - prefix_limit - len(marker))
    length = 0
    tail = b""
    omitted = False
    with path.open("wb") as output:
        for chunk in container.logs(stream=True, follow=True):
            size = min(len(chunk), max(0, prefix_limit - length))
            if size:
                length += output.write(chunk[:size])
                output.flush()
            rest = tail + chunk[size:]
            if len(rest) > tail_limit:
                omitted = True
            tail = rest[-tail_limit:] if tail_limit else b""
        if omitted:
            output.write(marker[: limit - length - len(tail)])
        output.write(tail)


def _publish(settings: Settings, job: Row, output: Path) -> list[FileRecord]:
    """Validate the task manifest and stage registered artifacts.

    Parameters
    ----------
    settings
        Output bounds and persistent storage root.
    job
        Operation and simulation flag.
    output
        Task output directory after container termination.

    Returns
    -------
    list[FileRecord]
        Verified files ready to be registered in PostgreSQL.

    Raises
    ------
    ValueError
        An artifact is invalid, missing, or over the output limit.
    """
    expected = {"inspection.json": "inspection"}
    if job["operation"] == "compile_om":
        name, kind = (
            ("model.simulated", "simulated_om")
            if job["simulated"]
            else ("model.om", "om")
        )
        expected = {name: kind, "build.json": "build_info"}
    manifest = output / "result.json"
    if (
        manifest.is_symlink()
        or not manifest.is_file()
        or manifest.stat().st_size > 8192
    ):
        raise ValueError("Task result manifest is missing or invalid.")
    if json.loads(manifest.read_text(encoding="utf-8")) != {
        "version": 1,
        "artifacts": list(expected),
    }:
        raise ValueError("Task result does not match the expected output files.")
    records = []
    try:
        for filename, kind in expected.items():
            path = output / filename
            if path.is_symlink() or not path.is_file():
                raise ValueError("Task returned a missing or invalid artifact.")
            size = path.stat().st_size
            if (
                size == 0
                or size + sum(record.size for record in records)
                > settings.max_output_bytes
            ):
                raise ValueError("Task output is empty or exceeds the output limit.")
            digest = sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            record = FileRecord(uuid4(), kind, filename, size, digest.hexdigest())
            path.replace(settings.data_dir / "assets" / str(record.id))
            records.append(record)
    except (OSError, ValueError):
        for record in records:
            (settings.data_dir / "assets" / str(record.id)).unlink(missing_ok=True)
        raise
    return records


def _recover(settings: Settings, store: Store, client: docker.DockerClient) -> None:
    """Stop interrupted attempts and mark them terminal before accepting work.

    Parameters
    ----------
    settings
        Log location and retention cap.
    store
        Database containing the interrupted attempts.
    client
        Trusted Docker daemon client.
    """
    for job in store.running():
        attempt = job["attempt_id"]
        try:
            container = client.containers.get(f"atc-attempt-{attempt.hex}")
        except NotFound:
            pass
        else:
            container.stop(timeout=5)
            path = settings.data_dir / "runs" / str(attempt) / "task.log"
            if path.parent.is_dir():
                recovery_log = path.with_name("task.recovery")
                _capture(container, recovery_log, settings.max_log_bytes)
                if (
                    not path.exists()
                    or recovery_log.stat().st_size > path.stat().st_size
                ):
                    recovery_log.replace(path)
                else:
                    recovery_log.unlink()
            container.remove()
        store.finish(
            job, None, "Worker was interrupted. Submit the operation again.", []
        )
        LOG.warning("job_id=%s attempt_id=%s event=recovered", job["id"], attempt)


def _execute(
    settings: Settings,
    store: Store,
    client: docker.DockerClient,
    job: Row,
) -> None:
    """Stage one operation, execute it, and publish verified results.

    Parameters
    ----------
    settings
        Task resource limits and storage paths.
    store
        Persistent job state and cancellation requests.
    client
        Trusted Docker daemon client.
    job
        Claimed job and execution attempt.
    """
    attempt = job["attempt_id"]
    directory = settings.data_dir / "runs" / str(attempt)
    inputs = directory / "input"
    output = directory / "output"
    host = settings.docker_data_dir / "runs" / str(attempt)
    error = None
    records: list[FileRecord] = []
    exit_code = None
    container = None
    LOG.info("job_id=%s attempt_id=%s event=started", job["id"], attempt)
    try:
        directory.mkdir(mode=0o700, parents=True)
        inputs.mkdir()
        output.mkdir()
        # The sandbox UID can write the output bind mount. The host parent is private.
        output.chmod(0o777)
        shutil.copyfile(
            settings.data_dir / "assets" / str(job["input_id"]), inputs / "model.onnx"
        )
        (inputs / "request.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "operation": job["operation"],
                    "preset": job["preset"],
                    "simulated": job["simulated"],
                }
            ),
            encoding="utf-8",
        )
        container = client.containers.create(
            job["image"],
            name=f"atc-attempt-{attempt.hex}",
            labels={"atc.attempt": str(attempt)},
            mounts=[
                Mount("/work/input", str(host / "input"), type="bind", read_only=True),
                Mount("/work/output", str(host / "output"), type="bind"),
            ],
            environment={
                "SSH_MODE": "disabled",
                "HOME": "/tmp",
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            },
            user="10001:10001",
            network_disabled=True,
            read_only=True,
            tmpfs={"/tmp": "rw,nosuid,nodev,size=67108864"},
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            mem_limit=settings.memory_bytes,
            nano_cpus=2_000_000_000,
            pids_limit=128,
            ulimits=[
                Ulimit(
                    name="fsize",
                    soft=settings.max_output_bytes,
                    hard=settings.max_output_bytes,
                )
            ],
            log_config=LogConfig(
                type="json-file", config={"max-size": "4m", "max-file": "2"}
            ),
        )
        container.start()
        deadline = monotonic() + settings.timeout_seconds
        log_path = directory / "task.log"
        with ThreadPoolExecutor(max_workers=1) as collector:
            capture = collector.submit(
                _capture, container, log_path, settings.max_log_bytes
            )
            try:
                while True:
                    container.reload()
                    if container.status in ("exited", "dead"):
                        exit_code = int(container.attrs["State"]["ExitCode"])
                        if container.attrs["State"].get("OOMKilled"):
                            error = "Execution exceeded the memory limit."
                        break
                    if store.cancel_requested(job["id"]):
                        error = "Execution cancelled."
                        break
                    if monotonic() >= deadline:
                        error = "Execution exceeded the time limit."
                        break
                    sleep(0.25)
            finally:
                if exit_code is None:
                    container.stop(timeout=5)
            try:
                capture.result()
            except DockerException:
                LOG.warning(
                    "job_id=%s attempt_id=%s event=log_stream_interrupted",
                    job["id"],
                    attempt,
                )
                _capture(container, log_path, settings.max_log_bytes)
        if error is None:
            if exit_code == 0:
                records = _publish(settings, job, output)
            else:
                error = f"Execution exited with status {exit_code}. See the task log."
    except (DockerException, OSError, ValueError):
        LOG.exception(
            "job_id=%s attempt_id=%s event=execution_error", job["id"], attempt
        )
        error = "Execution or result collection failed. See the task log or contact an operator."
    finally:
        if container is not None:
            # A removal failure leaves the attempt running so recovery can confirm exit.
            container.remove(force=True)
    try:
        published = store.finish(job, exit_code, error, records)
    except ResourceError as failure:
        published = False
        store.finish(job, exit_code, str(failure), [])
    if not published:
        for record in records:
            (settings.data_dir / "assets" / str(record.id)).unlink(missing_ok=True)
    shutil.rmtree(inputs, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    LOG.info(
        "job_id=%s attempt_id=%s event=finished exit_code=%s",
        job["id"],
        attempt,
        exit_code,
    )


def run_worker(settings: Settings) -> None:
    """Run the first-release single-worker queue and reconcile restarts.

    Parameters
    ----------
    settings
        Database, Docker mount, and execution settings.

    Raises
    ------
    RuntimeError
        Another worker holds the deployment lock.
    """
    store = Store(settings)
    (settings.data_dir / "assets").mkdir(parents=True, exist_ok=True)
    client = docker.from_env(timeout=30)
    with store.connect() as lease:
        lease.autocommit = True
        locked = lease.execute(
            "SELECT pg_try_advisory_lock(792640302) AS locked"
        ).fetchone()
        if locked is None or not locked["locked"]:
            raise RuntimeError("Another worker is already active.")
        _recover(settings, store, client)
        while True:
            lease.execute("SELECT 1")
            job = store.claim()
            if job is None:
                sleep(0.25)
            else:
                _execute(settings, store, client, job)
