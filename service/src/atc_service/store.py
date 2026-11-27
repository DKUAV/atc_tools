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
        """Apply the initial schema under a database migration lock."""
        with self.connect() as db:
            db.execute("SELECT pg_advisory_xact_lock(792640301)")
            db.execute(
                "CREATE TABLE IF NOT EXISTS schema_versions (version integer PRIMARY KEY)"
            )
            if (
                db.execute(
                    "SELECT version FROM schema_versions WHERE version = 1"
                ).fetchone()
                is None
            ):
                migration = (
                    files("atc_service")
                    .joinpath("001_initial.sql")
                    .read_text(encoding="utf-8")
                )
                db.execute(migration.encode("utf-8"), prepare=False)
                db.execute("INSERT INTO schema_versions (version) VALUES (1)")

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
        self, db: psycopg.Connection[Row], owner: str, record: FileRecord
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
            "SELECT COALESCE(sum(size), 0) AS total FROM assets WHERE owner = %s",
            (owner,),
        ).fetchone()
        assert used is not None
        if used["total"] + record.size > self.settings.max_account_bytes:
            raise ResourceError(413, "Account storage quota exceeded.")
        row = db.execute(
            "INSERT INTO assets (id, owner, kind, filename, size, sha256) "
            "VALUES (%s, %s, %s, %s, %s, %s) RETURNING *",
            (
                record.id,
                owner,
                record.kind,
                record.filename,
                record.size,
                record.sha256,
            ),
        ).fetchone()
        assert row is not None
        return row

    def add_asset(self, owner: str, record: FileRecord) -> Row:
        """Publish a complete uploaded file for its owner.

        Parameters
        ----------
        owner
            Authenticated account.
        record
            Metadata for the completed upload.

        Returns
        -------
        dict
            New asset record.
        """
        with self.connect() as db:
            db.execute(
                "SELECT username FROM accounts WHERE username = %s FOR UPDATE", (owner,)
            )
            return self._insert_asset(db, owner, record)

    def asset(self, owner: str, asset_id: UUID) -> Row:
        """Read a file record scoped to its owner.

        Parameters
        ----------
        owner
            Authenticated account.
        asset_id
            Requested asset.

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
                "SELECT * FROM assets WHERE id = %s AND owner = %s", (asset_id, owner)
            ).fetchone()
        if row is None:
            raise ResourceError(404, "Asset not found.")
        return row

    def submit(
        self,
        owner: str,
        operation: str,
        input_id: UUID,
        image: str,
        preset: str | None,
        simulated: bool,
        key: str | None,
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
                    ) != (operation, input_id, preset):
                        raise ResourceError(
                            409,
                            "Idempotency key already used with different parameters.",
                        )
                    return previous
            asset = db.execute(
                "SELECT validated FROM assets WHERE id = %s AND owner = %s AND kind = 'onnx'",
                (input_id, owner),
            ).fetchone()
            if asset is None:
                raise ResourceError(404, "ONNX asset not found.")
            if operation == "compile_om" and not asset["validated"]:
                raise ResourceError(
                    409, "Run a successful ONNX inspection before compiling."
                )
            pending = db.execute(
                "SELECT count(*) AS total FROM jobs WHERE owner = %s AND state IN ('queued', 'running')",
                (owner,),
            ).fetchone()
            assert pending is not None
            if pending["total"] >= self.settings.max_pending_jobs:
                raise ResourceError(429, "Account pending job limit reached.")
            job = db.execute(
                "INSERT INTO jobs (id, owner, operation, input_id, image, preset, simulated, idempotency_key) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) RETURNING *",
                (uuid4(), owner, operation, input_id, image, preset, simulated, key),
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
            outputs = (
                db.execute(
                    "SELECT a.* FROM assets a JOIN job_outputs o ON a.id = o.asset_id "
                    "WHERE o.job_id = %s AND a.owner = %s ORDER BY a.filename",
                    (job_id, owner),
                ).fetchall()
                if job["state"] == "succeeded"
                else []
            )
            return job, outputs

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

    def claim(self) -> Row | None:
        """Claim one queued job and register its execution attempt.

        Returns
        -------
        dict or None
            Claimed job or None when the queue is empty.
        """
        with self.connect() as db:
            job = db.execute(
                "SELECT id FROM jobs WHERE state = 'queued' "
                "ORDER BY created_at, id LIMIT 1 FOR UPDATE SKIP LOCKED"
            ).fetchone()
            if job is None:
                return None
            attempt = uuid4()
            db.execute(
                "INSERT INTO attempts (id, job_id) VALUES (%s, %s)",
                (attempt, job["id"]),
            )
            return db.execute(
                "UPDATE jobs SET state = 'running', started_at = now(), attempt_id = %s "
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
                db.execute(
                    "SELECT username FROM accounts WHERE username = %s FOR UPDATE",
                    (job["owner"],),
                )
                for record in records:
                    self._insert_asset(db, job["owner"], record)
                    db.execute(
                        "INSERT INTO job_outputs VALUES (%s, %s)",
                        (job["id"], record.id),
                    )
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
