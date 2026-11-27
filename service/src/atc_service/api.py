"""Expose account-scoped HTTP endpoints for the first conversion workflow."""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from atc_service.config import Settings
from atc_service.store import FileRecord, ResourceError, Store


class Asset(BaseModel):
    """Describe an uploaded file or an output available for download."""

    id: UUID
    kind: str
    filename: str
    size: int
    sha256: str
    validated: bool


class Submission(BaseModel):
    """Select one allowed operation, its owned input, and a compile preset."""

    model_config = ConfigDict(extra="forbid")

    operation: Literal["inspect_onnx", "compile_om"]
    input_id: UUID
    preset: str | None = Field(default=None, max_length=64)


class Job(BaseModel):
    """Return public job state and the files published by that job."""

    id: UUID
    operation: str
    input_id: UUID
    preset: str | None
    simulated: bool
    state: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    cancel_requested: bool
    attempt_id: UUID | None
    error: str | None
    exit_code: int | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    outputs: list[Asset] = Field(default_factory=list)


class LogPart(BaseModel):
    """Return a bounded section of an attempt log and the next byte offset."""

    attempt_id: UUID | None
    text: str
    next_offset: int


def create_app(settings: Settings) -> FastAPI:
    """Register routes using explicit settings and a PostgreSQL store.

    Parameters
    ----------
    settings
        Database, storage, and operation limits for this API.

    Returns
    -------
    FastAPI
        Application exposing the versioned API.
    """
    store = Store(settings)
    uploads = settings.data_dir / "uploads"
    assets = settings.data_dir / "assets"
    uploads.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("atc_service.api")
    bearer = HTTPBearer(auto_error=False)
    app = FastAPI(title="ATC Tools Service", version="0.1.0")

    def account(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> str:
        """Resolve the current account from its request credentials.

        Parameters
        ----------
        credentials
            Bearer credentials extracted from the request.

        Returns
        -------
        str
            Account bound to the presented token.

        Raises
        ------
        HTTPException
            The token is missing, invalid, or revoked.
        """
        owner = None
        if credentials is not None and len(credentials.credentials) <= 128:
            owner = store.authenticate(credentials.credentials)
        if owner is None:
            raise HTTPException(
                401,
                "A valid API token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return owner

    Owner = Annotated[str, Depends(account)]

    @app.middleware("http")
    async def log_request(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Assign a request ID and log the route and status.

        Parameters
        ----------
        request
            Incoming request.
        call_next
            Downstream HTTP application.

        Returns
        -------
        Response
            HTTP response annotated with the request ID.
        """
        request_id = uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        route = getattr(request.scope.get("route"), "path", "unmatched")
        logger.info(
            "request_id=%s method=%s route=%s status=%s",
            request_id,
            request.method,
            route,
            response.status_code,
        )
        return response

    @app.exception_handler(ResourceError)
    async def resource_error(request: Request, error: Exception) -> JSONResponse:
        """Translate a resource error to a public JSON response.

        Parameters
        ----------
        request
            Request associated with the rejected operation.
        error
            Resource error carrying a safe message.

        Returns
        -------
        JSONResponse
            Error status and message.
        """
        assert isinstance(error, ResourceError)
        return JSONResponse({"detail": str(error)}, status_code=error.status)

    @app.get("/healthz")
    def health() -> dict[str, str]:
        """Verify database connectivity.

        Returns
        -------
        dict
            Readiness status.
        """
        with store.connect() as db:
            db.execute("SELECT 1")
        return {"status": "ok"}

    @app.get("/v1/capabilities")
    def capabilities(owner: Owner) -> dict[str, object]:
        """Describe operations available in the current deployment.

        Parameters
        ----------
        owner
            Authenticated account.

        Returns
        -------
        dict
            Enabled operation names and compile presets.
        """
        preset = settings.compile_preset()
        return {
            "operations": ["inspect_onnx", "compile_om"]
            if preset
            else ["inspect_onnx"],
            "compile_presets": [preset] if preset else [],
            "max_upload_bytes": settings.max_upload_bytes,
        }

    @app.post("/v1/assets", status_code=201)
    async def upload(
        request: Request,
        owner: Owner,
        filename: Annotated[str, Query(min_length=6, max_length=200)],
    ) -> Asset:
        """Stream a single ONNX upload while enforcing size limits.

        Parameters
        ----------
        request
            Body with the ONNX file bytes.
        owner
            Authenticated account.
        filename
            Display filename used when downloading the asset.

        Returns
        -------
        Asset
            New owned asset and checksum.

        Raises
        ------
        HTTPException
            The upload has an invalid filename, media type, or size.
        """
        if not filename.lower().endswith(".onnx") or any(
            char in "/\\" or ord(char) < 32 or ord(char) == 127 for char in filename
        ):
            raise HTTPException(400, "Provide a plain .onnx filename.")
        media_type = (
            request.headers.get("content-type", "").partition(";")[0].strip().lower()
        )
        if media_type != "application/octet-stream":
            raise HTTPException(415, "Upload the file as application/octet-stream.")
        declared = request.headers.get("content-length")
        if declared and int(declared) > settings.max_upload_bytes:
            raise HTTPException(413, "Upload exceeds the configured size limit.")

        asset_id = uuid4()
        staging = uploads / str(asset_id)
        final = assets / str(asset_id)
        size = 0
        digest = sha256()
        try:
            with staging.open("xb") as output:
                async for chunk in request.stream():
                    size += len(chunk)
                    if size > settings.max_upload_bytes:
                        raise HTTPException(
                            413, "Upload exceeds the configured size limit."
                        )
                    digest.update(chunk)
                    await run_in_threadpool(output.write, chunk)
            if size == 0:
                raise HTTPException(400, "The model file is empty.")
            staging.replace(final)
            record = FileRecord(asset_id, "onnx", filename, size, digest.hexdigest())
            try:
                row = await run_in_threadpool(store.add_asset, owner, record)
            except ResourceError:
                final.unlink(missing_ok=True)
                raise
            return Asset.model_validate(row)
        finally:
            staging.unlink(missing_ok=True)

    @app.get("/v1/assets/{asset_id}")
    def asset(owner: Owner, asset_id: UUID) -> Asset:
        """Return metadata for an owned asset.

        Parameters
        ----------
        owner
            Authenticated account.
        asset_id
            Asset to read.

        Returns
        -------
        Asset
            Owned asset metadata.
        """
        return Asset.model_validate(store.asset(owner, asset_id))

    @app.get("/v1/assets/{asset_id}/download")
    def download(owner: Owner, asset_id: UUID) -> FileResponse:
        """Return the file registered under an owned asset ID.

        Parameters
        ----------
        owner
            Authenticated account.
        asset_id
            Asset to download.

        Returns
        -------
        FileResponse
            File attachment with its registered name.
        """
        row = store.asset(owner, asset_id)
        path = assets / str(asset_id)
        if not path.is_file():
            raise HTTPException(404, "Asset file is unavailable.")
        return FileResponse(
            path, filename=row["filename"], media_type="application/octet-stream"
        )

    @app.post("/v1/jobs", status_code=202)
    def submit(
        request: Request,
        body: Submission,
        owner: Owner,
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ) -> Job:
        """Register an operation using an administrator-controlled execution image.

        Parameters
        ----------
        request
            Request carrying the correlation identifier.
        body
            Validated job parameters.
        owner
            Authenticated account.
        idempotency_key
            Optional key that identifies a retried submission.

        Returns
        -------
        Job
            Job accepted for execution.

        Raises
        ------
        HTTPException
            The selected preset is disabled or incompatible.
        """
        image = settings.inspect_image
        simulated = False
        if body.operation == "compile_om":
            preset = settings.compile_preset()
            if preset is None or body.preset != preset["id"]:
                raise HTTPException(422, "Select an enabled compile preset.")
            simulated = bool(preset["simulated"])
            if not simulated:
                assert settings.compile_image is not None
                image = settings.compile_image
        elif body.preset is not None:
            raise HTTPException(422, "ONNX inspection takes no compile preset.")
        row = store.submit(
            owner,
            body.operation,
            body.input_id,
            image,
            body.preset,
            simulated,
            idempotency_key,
        )
        logger.info(
            "request_id=%s job_id=%s operation=%s",
            request.state.request_id,
            row["id"],
            row["operation"],
        )
        # An idempotent request may refer to an already completed job.
        row, outputs = store.job(owner, row["id"])
        return Job.model_validate({**row, "outputs": outputs})

    @app.get("/v1/jobs/{job_id}")
    def job(owner: Owner, job_id: UUID) -> Job:
        """Return an owned job's current state and results.

        Parameters
        ----------
        owner
            Authenticated account.
        job_id
            Job to query.

        Returns
        -------
        Job
            Current job and its output assets.
        """
        row, outputs = store.job(owner, job_id)
        return Job.model_validate({**row, "outputs": outputs})

    @app.post("/v1/jobs/{job_id}/cancel", status_code=202)
    def cancel(owner: Owner, job_id: UUID) -> Job:
        """Request cancellation of an owned job.

        Parameters
        ----------
        owner
            Authenticated account.
        job_id
            Job to cancel.

        Returns
        -------
        Job
            Updated job state.
        """
        store.cancel(owner, job_id)
        return job(owner, job_id)

    @app.get("/v1/jobs/{job_id}/logs")
    def read_log(
        owner: Owner,
        job_id: UUID,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> LogPart:
        """Read the current attempt log from a client-supplied byte offset.

        Parameters
        ----------
        owner
            Authenticated account.
        job_id
            Job whose log is requested.
        offset
            Offset returned by the previous log request.

        Returns
        -------
        LogPart
            Available log text and a cursor for the next poll.
        """
        row, _ = store.job(owner, job_id)
        attempt = row["attempt_id"]
        path = settings.data_dir / "runs" / str(attempt) / "task.log"
        try:
            with path.open("rb") as log:
                log.seek(min(offset, path.stat().st_size))
                data = log.read(64 * 1024)
                return LogPart(
                    attempt_id=attempt,
                    text=data.decode("utf-8", "replace"),
                    next_offset=log.tell(),
                )
        except FileNotFoundError:
            return LogPart(attempt_id=attempt, text="", next_offset=0)

    return app
