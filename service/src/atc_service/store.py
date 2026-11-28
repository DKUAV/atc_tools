"""Persist accounts, owned files, and job state in SQLite."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Generator
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from secrets import token_urlsafe
from typing import Any
from uuid import UUID, uuid4

from atc_service.config import Settings

Row = dict[str, Any]

sqlite3.register_adapter(UUID, str)
sqlite3.register_converter("uuid", lambda raw: UUID(raw.decode()))
sqlite3.register_converter("json", json.loads)
sqlite3.register_converter("boolean", lambda raw: bool(int(raw)))


def _named_row(cursor: sqlite3.Cursor, values: tuple[Any, ...]) -> Row:
    """Map projected SQL column names to their decoded values.

    Parameters
    ----------
    cursor, values
        Query metadata and values for one result.

    Returns
    -------
    dict
        Named row for service callers.
    """
    return {column[0]: value for column, value in zip(cursor.description, values)}


class ResourceError(ValueError):
    """Report a resource error that can be shown to the API caller."""

    def __init__(self, status: int, message: str) -> None:
        """Store the HTTP status and a public message.

        Parameters
        ----------
        status
            HTTP status for the rejected request.
        message
            Safe description of the error.
        """
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class FileRecord:
    """Describe a file ready to be registered as an owned asset."""

    id: UUID
    kind: str
    filename: str
    size: int
    sha256: str


class Store:
    """Keep transactions for job transitions and ownership in one place."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the store with the deployment connection and limits.

        Parameters
        ----------
        settings
            Connection and quota settings.
        """
        self.settings = settings

    @contextmanager
    def connect(
        self, *, write: bool = False
    ) -> Generator[sqlite3.Connection, None, None]:
        """Open a bounded SQLite transaction with foreign key enforcement.

        Parameters
        ----------
        write
            Reserve the writer before reading state for a transition.

        Yields
        ------
        sqlite3.Connection
            Connection scoped to this transaction.
        """
        with closing(
            sqlite3.connect(
                self.settings.data_dir / "service.sqlite3",
                timeout=5,
                isolation_level=None,
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
        ) as db:
            db.row_factory = _named_row
            db.execute("PRAGMA foreign_keys = ON")
            db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            with db:
                yield db

    @contextmanager
    def _publication(self) -> Generator[sqlite3.Connection, None, None]:
        """Distinguish a rejected publication from an uncertain commit.

        Yields
        ------
        sqlite3.Connection
            Transaction used to register owned files and their job state.

        Raises
        ------
        ResourceError
            A database error occurred before starting the commit. The caller
            can safely discard the unregistered files.
        sqlite3.DatabaseError
            The commit failed or its outcome is unknown. Keep the files in
            case registration became durable.
        """
        with ExitStack() as transaction:
            try:
                yield transaction.enter_context(self.connect(write=True))
            except sqlite3.DatabaseError as error:
                raise ResourceError(
                    503, "Could not register the files. Try again."
                ) from error

    def migrate(self) -> None:
        """Apply schema migrations under the SQLite writer lock."""
        self.settings.data_dir.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.settings.data_dir / "service.sqlite3")) as db:
            if db.execute("PRAGMA journal_mode=WAL").fetchone()[0] != "wal":
                raise RuntimeError(
                    "The service database requires WAL on local storage."
                )
        with self.connect(write=True) as db:
            version = db.execute("PRAGMA user_version").fetchone()["user_version"]
            if version == 0:
                schema = (
                    files("atc_service")
                    .joinpath("001_initial.sql")
                    .read_text(encoding="utf-8")
                )
                for statement in schema.split(";"):
                    if statement.strip():
                        db.execute(statement)
                db.execute("PRAGMA user_version = 1")
            elif version != 1:
                raise RuntimeError("Unsupported service database version.")

    def issue_token(self, username: str) -> str:
        """Provision an account and issue one bearer token.

        Parameters
        ----------
        username
            Validated account name.

        Returns
        -------
        str
            Plaintext token to disclose to the operator once.
        """
        token = token_urlsafe(32)
        with self.connect(write=True) as db:
            db.execute(
                "INSERT INTO accounts (username) VALUES (?) ON CONFLICT DO NOTHING",
                (username,),
            )
            db.execute(
                "INSERT INTO tokens VALUES (?, ?)",
                (sha256(token.encode()).digest(), username),
            )
        return token

    def revoke_token(self, token: str) -> bool:
        """Revoke a token presented by an operator.

        Parameters
        ----------
        token
            Plaintext token to revoke.

        Returns
        -------
        bool
            Whether a token was removed.
        """
        with self.connect(write=True) as db:
            return (
                db.execute(
                    "DELETE FROM tokens WHERE digest = ?",
                    (sha256(token.encode()).digest(),),
                ).rowcount
                != 0
            )

    def authenticate(self, token: str) -> str | None:
        """Resolve a bearer token to its account.

        Parameters
        ----------
        token
            Plaintext token received over the API.

        Returns
        -------
        str or None
            Account name if the token is active.
        """
        with self.connect() as db:
            row = db.execute(
                "SELECT username FROM tokens WHERE digest = ?",
                (sha256(token.encode()).digest(),),
            ).fetchone()
        return row["username"] if row else None

    def _insert_asset(
        self,
        db: sqlite3.Connection,
        owner: str,
        record: FileRecord,
        session_id: UUID | None,
    ) -> Row:
        """Register one complete file in the caller's transaction.

        Parameters
        ----------
        db
            Connection in the active SQLite write transaction.
        owner
            Account that owns the new file.
        record
            Metadata for the published file.
        session_id
            Browser session that owns the file, if applicable.

        Returns
        -------
        dict
            New asset record.

        Raises
        ------
        ResourceError
            The account's storage quota would be exceeded.
        """
        used = db.execute(
            "SELECT COALESCE(sum(size), 0) AS total FROM assets WHERE owner = ? AND NOT cleared",
            (owner,),
        ).fetchone()
        assert used is not None
        if used["total"] + record.size > self.settings.max_account_bytes:
            raise ResourceError(413, "Account storage quota exceeded.")
        row = db.execute(
            "INSERT INTO assets (id, owner, kind, filename, size, sha256, browser_session_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) RETURNING *",
            (
                record.id,
                owner,
                record.kind,
                record.filename,
                record.size,
                record.sha256,
                session_id,
            ),
        ).fetchone()
        assert row is not None
        return row

    def add_asset(
        self, owner: str, record: FileRecord, session_id: UUID | None = None
    ) -> Row:
        """Publish a complete uploaded file for its owner.

        Parameters
        ----------
        owner
            Authenticated account.
        record
            Metadata for the completed upload.
        session_id
            Current browser session for a dataset file.

        Returns
        -------
        dict
            New asset record.
        """
        with self._publication() as db:
            self._require_live_session(db, owner, session_id)
            return self._insert_asset(db, owner, record, session_id)

    def add_browser_model(
        self, owner: str, record: FileRecord, session_id: UUID
    ) -> Row:
        """Register a browser upload and its inspection in one transaction.

        Parameters
        ----------
        owner
            Authenticated account.
        record
            Completed ONNX upload.
        session_id
            Active browser session that will own the file.

        Returns
        -------
        dict
            Queued inspection job and its input ID.
        """
        with self._publication() as db:
            self._require_live_session(db, owner, session_id)
            self._insert_asset(db, owner, record, session_id)
            return self._enqueue(
                db,
                owner,
                "inspect_onnx",
                record.id,
                self.settings.runtime,
                None,
                False,
                None,
                session_id,
            )

    def asset(self, owner: str, asset_id: UUID, session_id: UUID | None = None) -> Row:
        """Read a file record scoped to its owner.

        Parameters
        ----------
        owner
            Authenticated account.
        asset_id
            Requested asset.
        session_id
            Browser session used to access the file, if applicable.

        Returns
        -------
        dict
            Owned asset record.

        Raises
        ------
        ResourceError
            The asset is unavailable to the account.
        """
        with self.connect() as db:
            row = db.execute(
                "SELECT a.* FROM assets a LEFT JOIN browser_sessions s ON s.id = a.browser_session_id "
                "WHERE a.id = ? AND a.owner = ? AND NOT a.cleared "
                "AND a.browser_session_id IS ? "
                "AND (s.id IS NULL OR (NOT s.closed AND s.expires_at > CURRENT_TIMESTAMP))",
                (asset_id, owner, session_id),
            ).fetchone()
        if row is None:
            raise ResourceError(404, "Asset not found.")
        return row

    def _require_live_session(
        self, db: sqlite3.Connection, owner: str, session_id: UUID | None
    ) -> None:
        """Hold a live browser session open while a file reference is created.

        Parameters
        ----------
        db
            Transaction creating the reference.
        owner
            Authenticated account.
        session_id
            Browser session, if applicable.

        Raises
        ------
        ResourceError
            The browser session is closed or expired.
        """
        if (
            session_id is not None
            and db.execute(
                "SELECT 1 FROM browser_sessions WHERE id = ? AND owner = ? "
                "AND NOT closed AND expires_at > CURRENT_TIMESTAMP",
                (session_id, owner),
            ).fetchone()
            is None
        ):
            raise ResourceError(401, "Your session ended. Sign in again.")

    def submit(
        self,
        owner: str,
        operation: str,
        input_id: UUID,
        image: str,
        preset: str | None,
        simulated: bool,
        key: str | None,
        session_id: UUID | None = None,
        workflow: Row | None = None,
    ) -> Row:
        """Validate ownership, apply quotas, and persist a new job.

        Parameters
        ----------
        owner
            Authenticated account.
        operation
            Allowlisted operation.
        input_id
            Uploaded ONNX asset ID.
        image
            Administrator-selected execution image.
        preset
            Validated preset ID.
        simulated
            Whether compilation is a simulation.
        key
            Optional account-scoped idempotency key.
        session_id
            Active browser session that owns the input.
        workflow
            Resolved browser workflow and its ordered dataset references.

        Returns
        -------
        dict
            New or previously submitted job.

        Raises
        ------
        ResourceError
            The input is unavailable, unvalidated, or over quota, or the key
            was used by a different request.
        """
        with self.connect(write=True) as db:
            self._require_live_session(db, owner, session_id)
            if key is not None:
                previous = db.execute(
                    "SELECT * FROM jobs WHERE owner = ? AND idempotency_key = ?",
                    (owner, key),
                ).fetchone()
                if previous is not None:
                    if (
                        previous["operation"],
                        previous["input_id"],
                        previous["preset"],
                        previous["browser_session_id"],
                        previous["workflow"],
                    ) != (operation, input_id, preset, session_id, workflow):
                        raise ResourceError(
                            409,
                            "Idempotency key already used with different parameters.",
                        )
                    return previous
            standalone_int8 = (
                operation == "compile_om"
                and workflow is not None
                and workflow.get("action") == "compile_om"
                and workflow["precision"] == "int8"
            )
            input_kinds = ["onnx", "quantized_onnx"]
            if operation == "compare_outputs":
                input_kinds = ["tensors"]
            elif standalone_int8:
                input_kinds = ["quantized_onnx"]
            asset = self._require_asset(db, owner, session_id, input_id, input_kinds)
            if (
                standalone_int8
                and not simulated
                and db.execute(
                    "SELECT 1 FROM job_outputs o JOIN jobs j ON j.id = o.job_id "
                    "WHERE o.asset_id = ? AND j.operation = 'quantize_onnx' "
                    "AND NOT j.simulated",
                    (input_id,),
                ).fetchone()
                is None
            ):
                raise ResourceError(
                    422, "Use an ONNX model produced by real quantization."
                )
            if (
                operation not in {"inspect_onnx", "compare_outputs"}
                and not asset["validated"]
            ):
                raise ResourceError(
                    409, "Run a successful ONNX inspection before using the model."
                )
            if workflow is not None:
                if workflow.get("primary_id"):
                    kind = "simulated_om" if simulated else "om"
                    primary = self._require_asset(
                        db, owner, session_id, UUID(workflow["primary_id"]), [kind]
                    )
                    if (
                        operation in {"run_npu", "profile_om"}
                        and db.execute(
                            "SELECT 1 FROM job_outputs o JOIN jobs j ON j.id = o.job_id "
                            "WHERE o.asset_id = ? AND j.input_id = ? AND "
                            "(json_extract(j.workflow, '$.target') = '910B2' OR j.preset LIKE '910b2-%')",
                            (primary["id"], input_id),
                        ).fetchone()
                        is None
                    ):
                        raise ResourceError(
                            422, "Use a 910B2 OM built from the selected model."
                        )
                if workflow.get("actual_id"):
                    self._require_asset(
                        db, owner, session_id, UUID(workflow["actual_id"]), ["tensors"]
                    )
                if workflow.get("baseline_id"):
                    self._require_asset(
                        db,
                        owner,
                        session_id,
                        UUID(workflow["baseline_id"]),
                        ["baseline_om"],
                    )
                ids = {
                    UUID(value)
                    for role in ("calibration", "validation")
                    for sample in workflow[role]
                    for value in sample
                }
                if ids:
                    owned = db.execute(
                        "SELECT count(*) AS total FROM assets WHERE id IN (SELECT value FROM json_each(?)) AND owner = ? "
                        "AND browser_session_id IS ? AND kind = 'dataset' AND NOT cleared",
                        (json.dumps(sorted(map(str, ids))), owner, session_id),
                    ).fetchone()
                    if owned is None or owned["total"] != len(ids):
                        raise ResourceError(
                            404,
                            "One or more dataset files are unavailable in this session.",
                        )
            return self._enqueue(
                db,
                owner,
                operation,
                input_id,
                image,
                preset,
                simulated,
                key,
                session_id,
                workflow,
            )

    def _require_asset(
        self,
        db: sqlite3.Connection,
        owner: str,
        session_id: UUID | None,
        asset_id: UUID,
        kinds: list[str],
    ) -> Row:
        """Validate an input asset within the locked submission transaction.

        Parameters
        ----------
        db
            Current transaction.
        owner, session_id
            Authorization scope of the request.
        asset_id, kinds
            Resource ID and accepted stored kinds.

        Returns
        -------
        dict
            Owned, available input asset.

        Raises
        ------
        ResourceError
            An input does not exist in the requested scope or kind.
        """
        asset = db.execute(
            "SELECT * FROM assets WHERE id = ? AND owner = ? AND kind IN (SELECT value FROM json_each(?)) "
            "AND NOT cleared AND browser_session_id IS ?",
            (asset_id, owner, json.dumps(kinds), session_id),
        ).fetchone()
        if asset is None:
            raise ResourceError(
                404,
                "An input asset is unavailable in this scope or has the wrong kind.",
            )
        return asset

    def _enqueue(
        self,
        db: sqlite3.Connection,
        owner: str,
        operation: str,
        input_id: UUID,
        image: str,
        preset: str | None,
        simulated: bool,
        key: str | None,
        session_id: UUID | None,
        workflow: Row | None = None,
    ) -> Row:
        """Queue a job after checking its prerequisites and ownership.

        Parameters
        ----------
        db
            Write transaction for this submission.
        owner
            Account submitting the job.
        operation
            Allowlisted operation.
        input_id
            Owned input asset.
        image
            Administrator-selected executor image.
        preset
            Validated preset ID.
        simulated
            Whether compilation is simulated.
        key
            Optional submission idempotency key.
        session_id
            Browser session that owns the files, if applicable.
        workflow
            Optional snapshot of the browser conversion settings.

        Returns
        -------
        dict
            Queued job record.

        Raises
        ------
        ResourceError
            The account has reached its pending job limit.
        """
        pending = db.execute(
            "SELECT count(*) AS total FROM jobs WHERE owner = ? AND state IN ('queued', 'running')",
            (owner,),
        ).fetchone()
        assert pending is not None
        if pending["total"] >= self.settings.max_pending_jobs:
            raise ResourceError(429, "Account pending job limit reached.")
        lane = "compile"
        if operation in {"inspect_onnx", "run_cpu", "compare_outputs"}:
            lane = "cpu"
        elif operation in {"run_npu", "profile_om", "retry_step"} and not (
            workflow and workflow.get("build_baseline")
        ):
            lane = "device"
        job = db.execute(
            "INSERT INTO jobs (id, owner, operation, input_id, image, preset, simulated, "
            "idempotency_key, browser_session_id, workflow, lane) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING *",
            (
                uuid4(),
                owner,
                operation,
                input_id,
                image,
                preset,
                simulated,
                key,
                session_id,
                json.dumps(workflow) if workflow is not None else None,
                lane,
            ),
        ).fetchone()
        assert job is not None
        return job

    def job(self, owner: str, job_id: UUID) -> tuple[Row, list[Row]]:
        """Read an owned job and its published output records.

        Parameters
        ----------
        owner
            Authenticated account.
        job_id
            Requested job ID.

        Returns
        -------
        tuple[dict, list[dict]]
            Job and output records.

        Raises
        ------
        ResourceError
            The job is unavailable to this account.
        """
        with self.connect() as db:
            job = db.execute(
                "SELECT * FROM jobs WHERE id = ? AND owner = ?", (job_id, owner)
            ).fetchone()
            if job is None:
                raise ResourceError(404, "Job not found.")
            outputs = db.execute(
                "SELECT a.* FROM assets a JOIN job_outputs o ON a.id = o.asset_id "
                "WHERE o.job_id = ? AND a.owner = ? ORDER BY a.filename",
                (job_id, owner),
            ).fetchall()
            return job, outputs

    def publish_progress(
        self,
        job: Row,
        stages: dict[str, str],
        records: list[FileRecord],
        summary: Row | None = None,
        times: Row | None = None,
    ) -> None:
        """Persist stage states and available artifacts for the running attempt.

        Parameters
        ----------
        job
            Claimed job and attempt ID.
        stages
            Validated stage snapshot.
        records
            Artifacts available for immediate download.
        summary
            Optional bounded accuracy verdict for the comparison artifact.
        times
            Executor timestamps for observed stage transitions.
        """
        with self._publication() as db:
            changed = db.execute(
                "UPDATE jobs SET stages = ?, accuracy_summary = COALESCE(?, accuracy_summary), "
                "stage_times = COALESCE(?, stage_times) "
                "WHERE id = ? AND attempt_id = ? AND state = 'running'",
                (
                    json.dumps(stages),
                    json.dumps(summary) if summary is not None else None,
                    json.dumps(times) if times is not None else None,
                    job["id"],
                    job["attempt_id"],
                ),
            )
            if changed.rowcount != 1:
                raise ResourceError(409, "The workflow attempt is no longer active.")
            self._publish_outputs(db, job, records)

    def _publish_outputs(
        self, db: sqlite3.Connection, job: Row, records: list[FileRecord]
    ) -> None:
        """Register output assets in a job transaction.

        Parameters
        ----------
        db
            Active publication transaction.
        job
            Job that produced the files.
        records
            Verified file metadata.
        """
        if not records:
            return
        for record in records:
            self._insert_asset(db, job["owner"], record, job["browser_session_id"])
            db.execute("INSERT INTO job_outputs VALUES (?, ?)", (job["id"], record.id))

    def dataset_sources(self, job: Row) -> dict[str, str]:
        """Find the filenames of the dataset assets used by a claimed workflow.

        Parameters
        ----------
        job
            Claimed job holding its session assets in use.

        Returns
        -------
        dict
            Original filename of each workflow asset ID.
        """
        workflow = job["workflow"]
        ids = {
            UUID(value)
            for role in ("calibration", "validation")
            for sample in workflow[role]
            for value in sample
        }
        if not ids:
            return {}
        with self.connect() as db:
            rows = db.execute(
                "SELECT id, filename FROM assets WHERE id IN (SELECT value FROM json_each(?)) AND owner = ? "
                "AND browser_session_id IS ? AND kind = 'dataset' AND NOT cleared",
                (
                    json.dumps(sorted(map(str, ids))),
                    job["owner"],
                    job["browser_session_id"],
                ),
            ).fetchall()
        return {str(row["id"]): row["filename"] for row in rows}

    def queue_verification(self, job: Row) -> bool:
        """Release the compile slot and queue verification against the published OM.

        Parameters
        ----------
        job
            Active compile attempt for the workflow.

        Returns
        -------
        bool
            True if the task still needs verification and was not cancelled.
        """
        with self.connect(write=True) as db:
            return (
                db.execute(
                    "UPDATE jobs SET lane = 'device', state = 'queued' WHERE id = ? AND attempt_id = ? "
                    "AND lane = 'compile' AND state = 'running' AND NOT cancel_requested",
                    (job["id"], job["attempt_id"]),
                ).rowcount
                == 1
            )

    def pin_executor(self, job: Row, image_id: str) -> None:
        """Persist the installed runtime identifier before starting an attempt.

        Parameters
        ----------
        job
            Claimed job and active attempt ID.
        image_id
            Base digest and installed wheel version.

        Raises
        ------
        ResourceError
            The worker no longer owns this attempt.
        """
        with self.connect(write=True) as db:
            changed = db.execute(
                "UPDATE jobs SET image = ? WHERE id = ? AND attempt_id = ? AND state = 'running'",
                (image_id, job["id"], job["attempt_id"]),
            )
            if changed.rowcount != 1:
                raise ResourceError(409, "The execution attempt is no longer active.")

    def build_info_for_om(self, job: Row, primary_id: str) -> UUID | None:
        """Find the owned build metadata published alongside a primary OM.

        Parameters
        ----------
        job
            Job whose account and file scope must own the compilation.
        primary_id
            Published primary OM used for verification or retry.

        Returns
        -------
        UUID or None
            Available build information asset from the originating compilation.
        """
        with self.connect() as db:
            row = db.execute(
                "SELECT a.id FROM job_outputs om JOIN job_outputs info USING (job_id) "
                "JOIN assets a ON a.id = info.asset_id WHERE om.asset_id = ? "
                "AND a.kind = 'build_info' AND NOT a.cleared AND a.owner = ? "
                "AND a.browser_session_id IS ?",
                (UUID(primary_id), job["owner"], job["browser_session_id"]),
            ).fetchone()
        return row["id"] if row else None

    def record_worker_status(self, checks: Row) -> None:
        """Publish the current scheduling coordinator's readiness checks.

        Parameters
        ----------
        checks
            Safe readiness flags for configured executors and runtime.
        """
        with self.connect(write=True) as db:
            db.execute(
                "INSERT INTO worker_status (checks) VALUES (?) ON CONFLICT (singleton) "
                "DO UPDATE SET checks = EXCLUDED.checks, heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')",
                (json.dumps(checks),),
            )

    def cancel(self, owner: str, job_id: UUID) -> Row:
        """Cancel a queued job or request cancellation of a running job.

        Parameters
        ----------
        owner
            Authenticated account.
        job_id
            Job to cancel.

        Returns
        -------
        dict
            Current job after requesting cancellation.

        Raises
        ------
        ResourceError
            No such owned job exists.
        """
        with self.connect(write=True) as db:
            row = db.execute(
                "UPDATE jobs SET cancel_requested = true, "
                "state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END, "
                "finished_at = CASE WHEN state = 'queued' THEN strftime('%Y-%m-%dT%H:%M:%fZ','now') ELSE finished_at END "
                "WHERE id = ? AND owner = ? AND state IN ('queued', 'running') RETURNING *",
                (job_id, owner),
            ).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT * FROM jobs WHERE id = ? AND owner = ?", (job_id, owner)
                ).fetchone()
            if row is None:
                raise ResourceError(404, "Job not found.")
            return row

    def claim(self, lane: str) -> Row | None:
        """Claim one queued job and register its execution attempt.

        Parameters
        ----------
        lane
            Queue with an available execution slot.

        Returns
        -------
        dict or None
            Claimed job or None when the queue is empty.
        """
        with self.connect(write=True) as db:
            job = db.execute(
                "SELECT id FROM jobs WHERE state = 'queued' AND lane = ? "
                "ORDER BY created_at, id LIMIT 1",
                (lane,),
            ).fetchone()
            if job is None:
                return None
            attempt = uuid4()
            db.execute(
                "INSERT INTO attempts (id, job_id) VALUES (?, ?)",
                (attempt, job["id"]),
            )
            return db.execute(
                "UPDATE jobs SET state = 'running', started_at = COALESCE(started_at, strftime('%Y-%m-%dT%H:%M:%fZ','now')), attempt_id = ? "
                "WHERE id = ? RETURNING *",
                (attempt, job["id"]),
            ).fetchone()

    def running(self) -> list[Row]:
        """Read interrupted jobs when an exclusive worker starts.

        Returns
        -------
        list[dict]
            Running jobs and their attempt IDs.
        """
        with self.connect() as db:
            return db.execute("SELECT * FROM jobs WHERE state = 'running'").fetchall()

    def cancel_requested(self, job_id: UUID) -> bool:
        """Check whether a running job has a pending cancellation.

        Parameters
        ----------
        job_id
            Running job ID.

        Returns
        -------
        bool
            Whether cancellation was requested.
        """
        with self.connect() as db:
            row = db.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            assert row is not None
            return row["cancel_requested"]

    def finish(
        self,
        job: Row,
        exit_code: int | None,
        error: str | None,
        records: list[FileRecord],
    ) -> bool:
        """Finalize the current attempt and register its complete outputs.

        Parameters
        ----------
        job
            Claimed job and attempt ID.
        exit_code
            Exit status reported by the execution process.
        error
            Safe failure summary, if the operation failed.
        records
            Verified files staged for publication.

        Returns
        -------
        bool
            Whether the job published its staged outputs.
        """
        with self._publication() as db:
            current = db.execute(
                "SELECT cancel_requested FROM jobs WHERE id = ? AND attempt_id = ? AND state = 'running'",
                (job["id"], job["attempt_id"]),
            ).fetchone()
            if current is None:
                return False
            state = (
                "cancelled"
                if current["cancel_requested"]
                else "failed"
                if error
                else "succeeded"
            )
            if state == "succeeded":
                self._publish_outputs(db, job, records)
                if job["operation"] == "inspect_onnx":
                    db.execute(
                        "UPDATE assets SET validated = true WHERE id = ?",
                        (job["input_id"],),
                    )
            db.execute(
                "UPDATE jobs SET state = ?, error = ?, exit_code = ?, finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE id = ?",
                (state, error, exit_code, job["id"]),
            )
            return state == "succeeded"

    def clear_browser_files(self) -> None:
        """Remove files from closed or expired sessions after their jobs finish.

        Notes
        -----
        File rows and job attempts remain available for history. An interrupted
        cleanup can retry the same unlink in the next transaction.
        """
        with self.connect(write=True) as db:
            sessions = db.execute(
                "SELECT s.id FROM browser_sessions s WHERE (s.closed OR s.expires_at <= CURRENT_TIMESTAMP) "
                "AND EXISTS (SELECT 1 FROM assets a WHERE a.browser_session_id = s.id AND NOT a.cleared) "
                "AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.browser_session_id = s.id "
                "AND j.state IN ('queued', 'running')) LIMIT 32"
            ).fetchall()
            if not sessions:
                return
            rows = db.execute(
                "SELECT a.id FROM assets a WHERE NOT a.cleared AND a.browser_session_id IN (SELECT value FROM json_each(?)) "
                "AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.browser_session_id = a.browser_session_id "
                "AND j.state IN ('queued', 'running')) LIMIT 100",
                (json.dumps([str(session["id"]) for session in sessions]),),
            ).fetchall()
            for row in rows:
                (self.settings.data_dir / "assets" / str(row["id"])).unlink(
                    missing_ok=True
                )
                db.execute(
                    "UPDATE assets SET cleared = true WHERE id = ?", (row["id"],)
                )
