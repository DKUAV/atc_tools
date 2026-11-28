"""Execute queued jobs in restricted Docker containers."""

from __future__ import annotations

import json
import logging
import os
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from errno import ENODATA
from hashlib import sha256
from pathlib import Path
from stat import S_ISREG
from time import monotonic, sleep
from typing import BinaryIO
from uuid import uuid4

import docker
from docker.errors import DockerException, NotFound
from docker.models.containers import Container
from docker.types import LogConfig, Mount, Ulimit

from atc_service.config import Settings
from atc_service.store import FileRecord, ResourceError, Row, Store
from atc_service.workflows import DATASET_EXTENSIONS, planned_stages

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


def _open_output(path: Path) -> BinaryIO:
    """Open a regular top-level artifact without following symlinks or FIFOs.

    Parameters
    ----------
    path
        Service-selected filename inside the attempt output directory.

    Returns
    -------
    BinaryIO
        Open descriptor for the checked file.

    Raises
    ------
    ValueError
        The output is not a regular file.
    """
    descriptor = os.open(
        path, os.O_RDONLY | getattr(os, "O_NOFOLLOW") | getattr(os, "O_NONBLOCK")
    )
    try:
        if not S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Workflow output must be a regular file.")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _sync_workflow(
    settings: Settings,
    store: Store,
    job: Row,
    output: Path,
    stages: dict[str, str],
    published: dict[str, int],
) -> None:
    """Copy each allowed artifact as soon as its stage durably succeeds.

    Parameters
    ----------
    settings
        Output bounds and persistent storage root.
    store
        Job and asset database.
    job
        Active workflow attempt.
    output
        Per-attempt output directory.
    stages
        Previously stored stage states, updated in place.
    published
        Bytes already registered per filename, updated in place.
    """
    for attempt in range(3):
        try:
            with _open_output(output / "progress.json") as stream:
                data = stream.read(4097)
            break
        except FileNotFoundError:
            return
        except OSError as error:
            if error.errno != ENODATA or attempt == 2:
                raise
            # A Docker Desktop bind mount can invalidate an open handle on replace.
            sleep(0.01)
    if len(data) > 4096:
        raise ValueError("Workflow progress snapshot is too large.")
    snapshot = json.loads(data)
    state = snapshot.get("stages", snapshot) if isinstance(snapshot, dict) else snapshot
    allowed = set(planned_stages(job["workflow"]))
    if not isinstance(state, dict) or any(
        stage not in allowed
        or status not in ("running", "succeeded", "failed", "skipped")
        for stage, status in state.items()
    ):
        raise ValueError("Invalid workflow progress snapshot.")
    times = dict(job["stage_times"])
    for stage, entries in snapshot.get("times", {}).items():
        if stage not in allowed or not isinstance(entries, dict):
            raise ValueError("Invalid workflow stage timing.")
        for field, stamp in entries.items():
            if (
                field not in {"started_at", "finished_at"}
                or not isinstance(stamp, str)
                or datetime.fromisoformat(stamp).tzinfo is None
            ):
                raise ValueError("Invalid workflow stage timestamp.")
        times[stage] = entries
    for stage, status in stages.items():
        if status in {"succeeded", "failed", "skipped"}:
            state[stage] = status
    if state == stages:
        return

    name, kind = (
        ("model.simulated", "simulated_om") if job["simulated"] else ("model.om", "om")
    )
    bundles = {
        "compile": {name: kind, "build.json": "build_info"},
        "compile_baseline": {"baseline.om": "baseline_om"},
        "compare": {"accuracy.json": "accuracy"},
        "profile": {"profiling.zip": "profiling"},
        "infer": {"tensors.zip": "tensors"},
    }
    if job["operation"] == "quantize_onnx":
        bundles["quantize"] = {"model.quantized.onnx": "quantized_onnx"}
    if (
        not job["simulated"]
        and job["workflow"]["precision"] == "int8"
        and job["operation"] in {"compile_om", "retry_step"}
    ):
        bundles["compare"].update(
            {
                "accuracy-onnx-vs-int8.json": "accuracy",
                "accuracy-fp-vs-int8.json": "accuracy",
            }
        )
    awaiting = [
        stage
        for stage in bundles
        if state.get(stage) == "succeeded" and stages.get(stage) != "succeeded"
    ]
    for stage in awaiting:
        state[stage] = "running"
    for stage in awaiting:
        state[stage] = "succeeded"
        try:
            files = _publish_bundle(
                settings,
                store,
                job,
                output,
                state,
                bundles[stage],
                settings.max_output_bytes - sum(published.values()),
                times,
            )
        except (OSError, ValueError) as error:
            if (
                stage in {"compile", "quantize", "infer"}
                or job["operation"] != "compile_om"
            ):
                raise
            state[stage] = "failed"
            LOG.warning(
                "job_id=%s stage=%s event=output_not_published error=%s",
                job["id"],
                stage,
                error,
            )
        else:
            stages.update(state)
            published.update({record.filename: record.size for record in files})
    if state != stages:
        store.publish_progress(job, state, [], times=times)
        stages.update(state)
    job["stage_times"] = times


def _publish_bundle(
    settings: Settings,
    store: Store,
    job: Row,
    output: Path,
    state: dict[str, str],
    bundle: dict[str, str],
    remaining: int,
    times: Row,
) -> list[FileRecord]:
    """Copy and register all artifacts belonging to one completed stage.

    Parameters
    ----------
    settings
        Service asset storage path.
    store
        Database used to register the artifact bundle.
    job
        Active workflow attempt.
    output
        Executor's isolated output directory.
    state
        Current stage states conditional on publication.
    bundle
        Allowlisted filenames and asset kinds belonging to the stage.
    remaining
        Total artifact byte budget left for the job.
    times
        Timestamp metadata for the current workflow phases.

    Returns
    -------
    list[FileRecord]
        Newly published artifacts.
    """
    records = []
    paths = []
    try:
        for name, kind in bundle.items():
            identifier = uuid4()
            destination = settings.data_dir / "assets" / str(identifier)
            paths.append(destination)
            size = 0
            digest = sha256()
            with (
                _open_output(output / name) as source,
                destination.open("xb") as target,
            ):
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(chunk)
                    if size > remaining:
                        raise ValueError("Workflow artifacts exceed the output limit.")
                    digest.update(chunk)
                    target.write(chunk)
            if size == 0:
                raise ValueError("Workflow returned an empty artifact.")
            records.append(FileRecord(identifier, kind, name, size, digest.hexdigest()))
            remaining -= size
        summary = None
        if "accuracy.json" in bundle:
            with _open_output(output / "accuracy-summary.json") as stream:
                data = stream.read(4097)
            if len(data) > 4096:
                raise ValueError("Accuracy summary exceeds the metadata limit.")
            summary = json.loads(data)
            if not isinstance(summary, dict):
                raise ValueError("Invalid accuracy summary.")
        store.publish_progress(job, state, records, summary, times)
    except (OSError, ValueError):
        for path in paths:
            path.unlink(missing_ok=True)
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
        directory = settings.data_dir / "runs" / str(attempt)
        try:
            container = client.containers.get(f"atc-attempt-{attempt.hex}")
        except NotFound:
            pass
        else:
            container.stop(timeout=5)
            path = directory / "task.log"
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
        shutil.rmtree(directory / "input", ignore_errors=True)
        shutil.rmtree(directory / "output", ignore_errors=True)
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
    stages: dict[str, str] = dict(job["stages"])
    published: dict[str, int] = {}
    deferred = False
    LOG.info("job_id=%s attempt_id=%s event=started", job["id"], attempt)
    try:
        directory.mkdir(mode=0o700, parents=True)
        inputs.mkdir()
        output.mkdir()
        # The sandbox UID can write the output bind mount. The host parent is private.
        output.chmod(0o777)
        workflow = job["workflow"]
        action = workflow.get("action") if workflow is not None else None
        checks = workflow is not None and (
            workflow["compare"] or workflow["profile"] or action == "retry_step"
        )
        split = bool(checks and action in {None, "retry_step"})
        phase = (
            "verify"
            if split and job["lane"] == "device"
            else "build"
            if split
            else "all"
        )
        deferred = phase == "build"
        asset_dir = settings.data_dir / "assets"
        shutil.copyfile(
            asset_dir / str(job["input_id"]),
            inputs / ("reference.zip" if action == "compare_outputs" else "model.onnx"),
        )
        if workflow is not None:
            (output / "workspace").mkdir()
            _, own_outputs = store.job(job["owner"], job["id"])
            published.update(
                {record["filename"]: record["size"] for record in own_outputs}
            )
            sources_by_kind = {
                record["kind"]: str(record["id"]) for record in own_outputs
            }
            primary = workflow.get("primary_id") or (
                sources_by_kind.get("om") or sources_by_kind.get("simulated_om")
                if phase == "verify"
                else None
            )
            baseline = workflow.get("baseline_id") or sources_by_kind.get("baseline_om")
            if (
                phase == "verify" or action in {"run_npu", "profile_om"}
            ) and not primary:
                raise ValueError("The required finished OM is unavailable.")
            for filename, identifier in (
                ("primary.om", primary),
                ("baseline.om", baseline),
                ("actual.zip", workflow.get("actual_id")),
            ):
                if identifier:
                    shutil.copyfile(asset_dir / identifier, inputs / filename)
            sources = store.dataset_sources(job)
            if sources:
                dataset = inputs / "dataset"
                dataset.mkdir()
                for asset_id, source_name in sources.items():
                    suffix = Path(source_name).suffix.lower()
                    if suffix not in DATASET_EXTENSIONS:
                        raise ValueError("Invalid dataset file type.")
                    shutil.copyfile(
                        settings.data_dir / "assets" / asset_id,
                        dataset / (asset_id + suffix),
                    )
        use_npu = job["lane"] == "device" and not job["simulated"]
        if use_npu and settings.ascend_device is None:
            raise ValueError("The 910B2 runtime device is unavailable.")
        image = client.images.get(job["image"])
        (inputs / "request.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "operation": job["operation"],
                    "preset": job["preset"],
                    "simulated": job["simulated"],
                    "workflow": workflow,
                    "phase": phase,
                    "executor_image": image.id,
                    "limit_bytes": settings.max_output_bytes,
                }
            ),
            encoding="utf-8",
        )
        container = client.containers.create(
            image.id or job["image"],
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
                "OPENCV_IO_MAX_IMAGE_PIXELS": "16777216",
                **(
                    {"ASCEND_VISIBLE_DEVICES": str(settings.ascend_device)}
                    if use_npu
                    else {}
                ),
            },
            runtime="ascend" if use_npu else None,
            user="10001:10001",
            network_disabled=True,
            read_only=True,
            tmpfs={
                "/tmp": "rw,nosuid,nodev,size=67108864",
                **(
                    {
                        "/work/output/workspace": f"rw,nosuid,nodev,mode=1777,size={settings.scratch_bytes}"
                    }
                    if workflow is not None
                    else {}
                ),
            },
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
                    if workflow is not None:
                        total = 0
                        for path in output.iterdir():
                            try:
                                info = path.stat(follow_symlinks=False)
                            except FileNotFoundError:
                                continue
                            if S_ISREG(info.st_mode):
                                total += info.st_size
                        if total > settings.max_output_bytes + 8192:
                            raise ResourceError(
                                413, "Execution exceeded the aggregate output limit."
                            )
                        _sync_workflow(settings, store, job, output, stages, published)
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
                if workflow is None:
                    records = _publish(settings, job, output)
                else:
                    expected = planned_stages(workflow)
                    if deferred:
                        expected = expected[: expected.index("prepare_validation")]
                    if any(
                        stages.get(stage) not in {"succeeded", "failed", "skipped"}
                        for stage in expected
                    ):
                        raise ValueError(
                            "Workflow exited without a complete stage result."
                        )
                    if any(
                        stages.get(stage) != "succeeded"
                        for stage in {"compile", "infer", "quantize"} & set(expected)
                    ):
                        raise ValueError("A required workflow stage failed.")
                    if (
                        action == "retry_step"
                        and not deferred
                        and stages.get(workflow["retry_step"]) != "succeeded"
                    ):
                        raise ValueError("The retried verification step failed.")
            else:
                error = f"Execution exited with status {exit_code}. See the task log."
    except ResourceError as failure:
        error = str(failure)
    except (DockerException, OSError, ValueError):
        LOG.exception(
            "job_id=%s attempt_id=%s event=execution_error", job["id"], attempt
        )
        error = "Execution or result collection failed. See the task log or contact an operator."
    finally:
        if container is not None:
            # A removal failure leaves the attempt running so recovery can confirm exit.
            container.remove(force=True)
    # Remove phase-local inputs before releasing its slot or queueing verification.
    shutil.rmtree(inputs, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    try:
        accepted = (
            deferred and error is None and store.queue_verification(job)
        ) or store.finish(job, exit_code, error, records)
    except ResourceError as failure:
        accepted = False
        store.finish(job, exit_code, str(failure), [])
    if not accepted:
        for record in records:
            (settings.data_dir / "assets" / str(record.id)).unlink(missing_ok=True)
    LOG.info(
        "job_id=%s attempt_id=%s event=finished exit_code=%s",
        job["id"],
        attempt,
        exit_code,
    )


def _executor_checks(settings: Settings, client: docker.DockerClient) -> Row:
    """Check that all enabled local images and the configured runtime exist.

    Parameters
    ----------
    settings, client
        Worker deployment settings and Docker connection.

    Returns
    -------
    dict
        Public boolean readiness flags.
    """
    images = {"inspection_image": settings.inspect_image}
    if not settings.simulate_compile:
        if settings.compile_image:
            images["310b4_image"] = settings.compile_image
        if settings.compile_910b2_image:
            images["910b2_image"] = settings.compile_910b2_image
    checks: Row = {}
    for name, image in images.items():
        try:
            client.images.get(image)
        except NotFound:
            checks[name] = False
        else:
            checks[name] = True
    if (
        settings.compile_910b2_image
        and settings.ascend_device is not None
        and not settings.simulate_compile
    ):
        checks["ascend_runtime"] = "ascend" in client.info().get("Runtimes", {})
    return checks


def run_worker(settings: Settings) -> None:
    """Coordinate independent CPU, compile and device slots and recover interrupted work.

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
        cleanup_due = monotonic()
        slots = {
            "cpu": settings.cpu_slots,
            "compile": settings.compile_slots,
            "device": 1,
        }
        running: dict[str, set[Future[None]]] = {lane: set() for lane in slots}
        with ThreadPoolExecutor(max_workers=sum(slots.values())) as pool:
            while True:
                lease.execute("SELECT 1")
                if monotonic() >= cleanup_due:
                    cleanup_due = monotonic() + 5
                    try:
                        store.record_worker_status(_executor_checks(settings, client))
                        store.clear_browser_files()
                    except OSError:
                        LOG.exception("event=browser_cleanup_failed")
                for lane, capacity in slots.items():
                    finished = {attempt for attempt in running[lane] if attempt.done()}
                    for attempt in finished:
                        attempt.result()
                    running[lane].difference_update(finished)
                    while len(running[lane]) < capacity:
                        job = store.claim(lane)
                        if job is None:
                            break
                        running[lane].add(
                            pool.submit(_execute, settings, store, client, job)
                        )
                sleep(0.25)
