"""Persist accounts, owned files, and job state in PostgreSQL."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
from secrets import token_urlsafe
from typing import Any
from uuid import UUID, uuid4

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from atc_service.config import Settings

Row = dict[str, Any]


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

    def connect(self) -> psycopg.Connection[Row]:
        """Open a connection that returns named columns.

        Returns
        -------
        psycopg.Connection
            Connection for one request or worker operation.
        """
        return psycopg.Connection[Row].connect(
            self.settings.database_url, row_factory=dict_row
        )

    def migrate(self) -> None:
        """Apply schema migrations under a database migration lock."""
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(792640301)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS schema_versions (version integer PRIMARY KEY)"
            )
            for version, filename in enumerate(
                (
                    "001_initial.sql",
                    "002_browser.sql",
                    "003_workflows.sql",
                    "004_delivery.sql",
                    "005_stage_times.sql",
                ),
                1,
            ):
                if (
                    db.execute(
                        "SELECT 1 FROM schema_versions WHERE version = %s", (version,)
                    ).fetchone()
                    is None
                ):
                    migration = files("atc_service").joinpath(filename).read_bytes()
                    db.execute(migration, prepare=False)
                    db.execute("INSERT INTO schema_versions VALUES (%s)", (version,))

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
        with self.connect() as db:
            db.execute(
                "INSERT INTO accounts (username) VALUES (%s) ON CONFLICT DO NOTHING",
                (username,),
            )
            db.execute(
                "INSERT INTO tokens VALUES (%s, %s)",
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
        with self.connect() as db:
            return (
                db.execute(
                    "DELETE FROM tokens WHERE digest = %s",
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
                "SELECT username FROM tokens WHERE digest = %s",
                (sha256(token.encode()).digest(),),
            ).fetchone()
        return row["username"] if row else None

    def _insert_asset(
        self,
        db: psycopg.Connection[Row],
        owner: str,
        record: FileRecord,
        session_id: UUID | None,
    ) -> Row:
        """Register one complete file in the caller's transaction.

        Parameters
        ----------
        db
            Database connection with the account row locked.
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
            "SELECT COALESCE(sum(size), 0) AS total FROM assets WHERE owner = %s AND NOT cleared",
            (owner,),
        ).fetchone()
        assert used is not None
        if used["total"] + record.size > self.settings.max_account_bytes:
            raise ResourceError(413, "Account storage quota exceeded.")
        row = db.execute(
            "INSERT INTO assets (id, owner, kind, filename, size, sha256, browser_session_id) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING *",
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
        with self.connect() as db:
            db.execute(
                "SELECT username FROM accounts WHERE username = %s FOR UPDATE", (owner,)
            )
            self._lock_session(db, owner, session_id)
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
        with self.connect() as db:
            db.execute(
                "SELECT username FROM accounts WHERE username = %s FOR UPDATE", (owner,)
            )
            self._lock_session(db, owner, session_id)
            self._insert_asset(db, owner, record, session_id)
            return self._enqueue(
                db,
                owner,
                "inspect_onnx",
                record.id,
                self.settings.inspect_image,
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
                "WHERE a.id = %s AND a.owner = %s AND NOT a.cleared "
                "AND a.browser_session_id IS NOT DISTINCT FROM %s "
                "AND (s.id IS NULL OR (NOT s.closed AND s.expires_at > now()))",
                (asset_id, owner, session_id),
            ).fetchone()
        if row is None:
            raise ResourceError(404, "Asset not found.")
        return row

    def _lock_session(
        self, db: psycopg.Connection[Row], owner: str, session_id: UUID | None
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
                "SELECT 1 FROM browser_sessions WHERE id = %s AND owner = %s "
                "AND NOT closed AND expires_at > clock_timestamp() FOR UPDATE",
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
        with self.connect() as db:
            db.execute(
                "SELECT username FROM accounts WHERE username = %s FOR UPDATE", (owner,)
            )
            self._lock_session(db, owner, session_id)
            if key is not None:
                previous = db.execute(
                    "SELECT * FROM jobs WHERE owner = %s AND idempotency_key = %s",
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
            input_kinds = ["onnx", "quantized_onnx"]
            if operation == "compare_outputs":
                input_kinds = ["tensors"]
            elif (
                operation == "compile_om"
                and workflow
                and workflow.get("action")
                and workflow["precision"] == "int8"
            ):
                input_kinds = ["quantized_onnx"]
            asset = self._require_asset(db, owner, session_id, input_id, input_kinds)
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
                            "WHERE o.asset_id = %s AND j.input_id = %s AND "
                            "(j.workflow->>'target' = '910B2' OR j.preset LIKE '910b2-%%')",
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
                        "SELECT count(*) AS total FROM assets WHERE id = ANY(%s) AND owner = %s "
                        "AND browser_session_id IS NOT DISTINCT FROM %s AND kind = 'dataset' AND NOT cleared",
                        (list(ids), owner, session_id),
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
        db: psycopg.Connection[Row],
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
            "SELECT * FROM assets WHERE id = %s AND owner = %s AND kind = ANY(%s) "
            "AND NOT cleared AND browser_session_id IS NOT DISTINCT FROM %s",
            (asset_id, owner, kinds, session_id),
        ).fetchone()
        if asset is None:
            raise ResourceError(
                404,
                "An input asset is unavailable in this scope or has the wrong kind.",
            )
        return asset

    def _enqueue(
        self,
        db: psycopg.Connection[Row],
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
            Transaction holding the account row lock.
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
            "SELECT count(*) AS total FROM jobs WHERE owner = %s AND state IN ('queued', 'running')",
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
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
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
                Jsonb(workflow) if workflow is not None else None,
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
                "SELECT * FROM jobs WHERE id = %s AND owner = %s", (job_id, owner)
            ).fetchone()
            if job is None:
                raise ResourceError(404, "Job not found.")
            outputs = db.execute(
                "SELECT a.* FROM assets a JOIN job_outputs o ON a.id = o.asset_id "
                "WHERE o.job_id = %s AND a.owner = %s ORDER BY a.filename",
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
        with self.connect() as db:
            changed = db.execute(
                "UPDATE jobs SET stages = %s, accuracy_summary = COALESCE(%s, accuracy_summary), "
                "stage_times = COALESCE(%s, stage_times) "
                "WHERE id = %s AND attempt_id = %s AND state = 'running'",
                (
                    Jsonb(stages),
                    Jsonb(summary) if summary is not None else None,
                    Jsonb(times) if times is not None else None,
                    job["id"],
                    job["attempt_id"],
                ),
            )
            if changed.rowcount != 1:
                raise ResourceError(409, "The workflow attempt is no longer active.")
            self._publish_outputs(db, job, records)

    def _publish_outputs(
        self, db: psycopg.Connection[Row], job: Row, records: list[FileRecord]
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
        db.execute(
            "SELECT username FROM accounts WHERE username = %s FOR UPDATE",
            (job["owner"],),
        )
        for record in records:
            self._insert_asset(db, job["owner"], record, job["browser_session_id"])
            db.execute(
                "INSERT INTO job_outputs VALUES (%s, %s)", (job["id"], record.id)
            )

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
                "SELECT id, filename FROM assets WHERE id = ANY(%s) AND owner = %s "
                "AND browser_session_id IS NOT DISTINCT FROM %s AND kind = 'dataset' AND NOT cleared",
                (list(ids), job["owner"], job["browser_session_id"]),
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
        with self.connect() as db:
            return (
                db.execute(
                    "UPDATE jobs SET lane = 'device', state = 'queued' WHERE id = %s AND attempt_id = %s "
                    "AND lane = 'compile' AND state = 'running' AND NOT cancel_requested",
                    (job["id"], job["attempt_id"]),
                ).rowcount
                == 1
            )

    def record_worker_status(self, checks: Row) -> None:
        """Publish the current scheduling coordinator's readiness checks.

        Parameters
        ----------
        checks
            Safe readiness flags for configured executors and runtime.
        """
        with self.connect() as db:
            db.execute(
                "INSERT INTO worker_status (checks) VALUES (%s) ON CONFLICT (singleton) "
                "DO UPDATE SET checks = EXCLUDED.checks, heartbeat_at = now()",
                (Jsonb(checks),),
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
        with self.connect() as db:
            row = db.execute(
                "UPDATE jobs SET cancel_requested = true, "
                "state = CASE WHEN state = 'queued' THEN 'cancelled' ELSE state END, "
                "finished_at = CASE WHEN state = 'queued' THEN now() ELSE finished_at END "
                "WHERE id = %s AND owner = %s AND state IN ('queued', 'running') RETURNING *",
                (job_id, owner),
            ).fetchone()
            if row is None:
                row = db.execute(
                    "SELECT * FROM jobs WHERE id = %s AND owner = %s", (job_id, owner)
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
        with self.connect() as db:
            job = db.execute(
                "SELECT id FROM jobs WHERE state = 'queued' AND lane = %s "
                "ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED",
                (lane,),
            ).fetchone()
            if job is None:
                return None
            attempt = uuid4()
            db.execute(
                "INSERT INTO attempts (id, job_id) VALUES (%s, %s)",
                (attempt, job["id"]),
            )
            return db.execute(
                "UPDATE jobs SET state = 'running', started_at = COALESCE(started_at, now()), attempt_id = %s "
                "WHERE id = %s RETURNING *",
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
                "SELECT cancel_requested FROM jobs WHERE id = %s", (job_id,)
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
            Exit status reported by the task container.
        error
            Safe failure summary, if the operation failed.
        records
            Verified files staged for publication.

        Returns
        -------
        bool
            Whether the job published its staged outputs.
        """
        with self.connect() as db:
            current = db.execute(
                "SELECT cancel_requested FROM jobs WHERE id = %s AND attempt_id = %s AND state = 'running' FOR UPDATE",
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
                        "UPDATE assets SET validated = true WHERE id = %s",
                        (job["input_id"],),
                    )
            db.execute(
                "UPDATE jobs SET state = %s, error = %s, exit_code = %s, finished_at = now() WHERE id = %s",
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
        with self.connect() as db:
            sessions = db.execute(
                "SELECT s.id FROM browser_sessions s WHERE (s.closed OR s.expires_at <= now()) "
                "AND EXISTS (SELECT 1 FROM assets a WHERE a.browser_session_id = s.id AND NOT a.cleared) "
                "AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.browser_session_id = s.id "
                "AND j.state IN ('queued', 'running')) FOR UPDATE OF s SKIP LOCKED LIMIT 32"
            ).fetchall()
            if not sessions:
                return
            rows = db.execute(
                "SELECT a.id FROM assets a WHERE NOT a.cleared AND a.browser_session_id = ANY(%s) "
                "AND NOT EXISTS (SELECT 1 FROM jobs j WHERE j.browser_session_id = a.browser_session_id "
                "AND j.state IN ('queued', 'running')) LIMIT 100",
                ([session["id"] for session in sessions],),
            ).fetchall()
            for row in rows:
                (self.settings.data_dir / "assets" / str(row["id"])).unlink(
                    missing_ok=True
                )
                db.execute(
                    "UPDATE assets SET cleared = true WHERE id = %s", (row["id"],)
                )
