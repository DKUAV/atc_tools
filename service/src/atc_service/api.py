"""Expose account-scoped HTTP endpoints for the first conversion workflow."""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from secrets import compare_digest
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from atc_service.config import Settings
from atc_service.sessions import (
    BrowserSession,
    close_session,
    open_session,
    read_session,
)
from atc_service.store import FileRecord, ResourceError, Row, Store
from atc_service.workflows import (
    DATASET_EXTENSIONS,
    Conversion,
    browser_presets,
    planned_stages,
    resolve_workflow,
)

BROWSER_COOKIE = "atc_session"


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
    stages: dict[str, str] = Field(default_factory=dict)


class LogPart(BaseModel):
    """Return a bounded section of an attempt log and the next byte offset."""

    attempt_id: UUID | None
    text: str
    next_offset: int


class BrowserLogin(BaseModel):
    """Accept the locally provisioned account credentials."""

    model_config = ConfigDict(extra="forbid")

    username: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
    password: str = Field(min_length=1, max_length=128)


class BrowserTask(BaseModel):
    """Describe one account-owned row in the browser task history."""

    id: UUID
    filename: str
    operation: str
    preset: str | None
    state: str
    created_at: datetime


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
    presets = {preset.id: preset for preset in browser_presets(settings)}

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

    def browser_session(request: Request) -> BrowserSession:
        """Authenticate a browser request and verify its mutation header.

        Parameters
        ----------
        request
            Request with a session cookie and a CSRF token for writes.

        Returns
        -------
        BrowserSession
            The authenticated browser session.

        Raises
        ------
        HTTPException
            The CSRF token is missing or mismatched.
        """
        session = read_session(store, request.cookies.get(BROWSER_COOKIE, ""))
        if request.method != "GET" and not compare_digest(
            request.headers.get("x-atc-csrf", "").encode(), session.csrf.encode()
        ):
            raise HTTPException(403, "Invalid request verification token.")
        return session

    Session = Annotated[BrowserSession, Depends(browser_session)]

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
        if request.url.path == "/" or request.url.path.startswith("/web/"):
            response.headers["Cache-Control"] = "no-store"
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Content-Security-Policy"] = (
                "default-src 'self'; object-src 'none'; frame-ancestors 'none'; base-uri 'none'"
            )
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

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, error: Exception) -> Response:
        """Report validation errors without reflecting sign-in credentials.

        Parameters
        ----------
        request
            Incoming request that failed validation.
        error
            FastAPI request validation error.

        Returns
        -------
        Response
            Validation response with a plain message for sign-in.
        """
        assert isinstance(error, RequestValidationError)
        if request.url.path == "/web/api/login":
            return JSONResponse(
                {"detail": "Check the account name and password."}, status_code=422
            )
        return await request_validation_exception_handler(request, error)

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

    async def save_upload(
        request: Request,
        filename: str,
        register: Callable[[FileRecord], Row],
        kind: Literal["onnx", "dataset"] = "onnx",
    ) -> Row:
        """Stream one allowed input file while enforcing size limits.

        Parameters
        ----------
        request
            Body with the ONNX file bytes.
        filename
            Display filename used when downloading the asset.
        register
            Register the completed file for the authenticated caller.
        kind
            Allowlist and stored asset kind for the input.

        Returns
        -------
        dict
            Registered file or its queued inspection.

        Raises
        ------
        HTTPException
            The upload has an invalid filename, media type, or size.
        """
        extensions = {".onnx"} if kind == "onnx" else DATASET_EXTENSIONS
        if Path(filename).suffix.lower() not in extensions or any(
            char in "/\\" or ord(char) < 32 or ord(char) == 127 for char in filename
        ):
            raise HTTPException(
                400, "Provide a plain filename with a supported extension."
            )
        media_type = (
            request.headers.get("content-type", "").partition(";")[0].strip().lower()
        )
        if media_type != "application/octet-stream":
            raise HTTPException(415, "Upload the file as application/octet-stream.")
        declared = request.headers.get("content-length")
        if declared and (not declared.isascii() or not declared.isdecimal()):
            raise HTTPException(400, "Invalid Content-Length header.")
        if declared and (
            len(declared) > 20 or int(declared) > settings.max_upload_bytes
        ):
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
                raise HTTPException(400, "The uploaded file is empty.")
            staging.replace(final)
            record = FileRecord(asset_id, kind, filename, size, digest.hexdigest())
            try:
                return await run_in_threadpool(register, record)
            except ResourceError:
                final.unlink(missing_ok=True)
                raise
        finally:
            staging.unlink(missing_ok=True)

    @app.post("/v1/assets", status_code=201)
    async def upload(
        request: Request,
        owner: Owner,
        filename: Annotated[str, Query(min_length=6, max_length=200)],
    ) -> Asset:
        """Upload a model for a bearer-token caller.

        Parameters
        ----------
        request
            Request with the model bytes.
        owner
            Authenticated account.
        filename
            Original display filename.

        Returns
        -------
        Asset
            Newly registered model.
        """
        row = await save_upload(
            request, filename, lambda record: store.add_asset(owner, record)
        )
        return Asset.model_validate(row)

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

    def asset_download(
        owner: str, asset_id: UUID, session_id: UUID | None = None
    ) -> FileResponse:
        """Return the file registered under an owned asset ID.

        Parameters
        ----------
        owner
            Authenticated account.
        asset_id
            Asset to download.
        session_id
            Browser session allowed to serve this file, if applicable.

        Returns
        -------
        FileResponse
            File attachment with its registered name.
        """
        row = store.asset(owner, asset_id, session_id)
        path = assets / str(asset_id)
        if not path.is_file():
            raise HTTPException(404, "Asset file is unavailable.")
        return FileResponse(
            path, filename=row["filename"], media_type="application/octet-stream"
        )

    @app.get("/v1/assets/{asset_id}/download")
    def download(owner: Owner, asset_id: UUID) -> FileResponse:
        """Download an asset for a bearer-token caller.

        Parameters
        ----------
        owner
            Authenticated account.
        asset_id
            Owned asset to serve.

        Returns
        -------
        FileResponse
            Registered file as an attachment.
        """
        return asset_download(owner, asset_id)

    def queue_job(
        request: Request,
        body: Submission,
        owner: str,
        idempotency_key: str | None,
        session_id: UUID | None = None,
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
        session_id
            Browser session owning the input, if applicable.

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
            session_id,
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

    @app.post("/v1/jobs", status_code=202)
    def submit(
        request: Request,
        body: Submission,
        owner: Owner,
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ) -> Job:
        """Submit one allowlisted operation for a bearer-token caller.

        Parameters
        ----------
        request
            Request with the correlation identifier.
        body
            Operation and input selection.
        owner
            Authenticated account.
        idempotency_key
            Reusable key for a retried submission.

        Returns
        -------
        Job
            Accepted operation.
        """
        return queue_job(request, body, owner, idempotency_key)

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

    @app.post("/web/api/login", status_code=204)
    def login(request: Request, body: BrowserLogin) -> Response:
        """Set a browser session cookie after verifying the account password.

        Parameters
        ----------
        request
            Same-origin login request.
        body
            Account name and password.

        Returns
        -------
        Response
            Empty response with the new session cookie.
        """
        if request.headers.get("x-atc-login") != "1":
            raise HTTPException(403, "Missing sign-in request header.")
        token = open_session(
            store, body.username, body.password, request.cookies.get(BROWSER_COOKIE, "")
        )
        response = Response(status_code=204)
        response.set_cookie(
            BROWSER_COOKIE,
            token,
            path="/web/api",
            httponly=True,
            secure=settings.browser_secure_cookies,
            samesite="strict",
        )
        return response

    @app.get("/web/api/session")
    def session_info(session: Session) -> dict[str, object]:
        """Return the active account, verification token, and allowed presets.

        Parameters
        ----------
        session
            Authenticated browser session.

        Returns
        -------
        dict
            Account name, CSRF token, and available compile capabilities.
        """
        return {
            "username": session.owner,
            "csrf": session.csrf,
            **capabilities(session.owner),
            "workflow_presets": [preset.public() for preset in presets.values()],
        }

    @app.post("/web/api/logout", status_code=204)
    def logout(session: Session) -> Response:
        """End the browser session and revoke access to its temporary files.

        Parameters
        ----------
        session
            Authenticated browser session.

        Returns
        -------
        Response
            Empty response expiring the cookie.
        """
        close_session(store, session)
        response = Response(status_code=204)
        response.delete_cookie(
            BROWSER_COOKIE,
            path="/web/api",
            httponly=True,
            secure=settings.browser_secure_cookies,
            samesite="strict",
        )
        return response

    @app.post("/web/api/models", status_code=202)
    async def browser_model(
        request: Request,
        session: Session,
        filename: Annotated[str, Query(min_length=6, max_length=200)],
    ) -> Job:
        """Upload a local model and automatically enqueue its inspection.

        Parameters
        ----------
        request
            The uploaded ONNX bytes.
        session
            Browser session that will own the file.
        filename
            Original local filename.

        Returns
        -------
        Job
            Queued inspection and its model ID.
        """
        row = await save_upload(
            request,
            filename,
            lambda record: store.add_browser_model(session.owner, record, session.id),
        )
        logger.info(
            "request_id=%s job_id=%s operation=inspect_onnx",
            request.state.request_id,
            row["id"],
        )
        return Job.model_validate(row)

    @app.post("/web/api/dataset-files", status_code=201)
    async def browser_dataset(
        request: Request,
        session: Session,
        filename: Annotated[str, Query(min_length=5, max_length=200)],
    ) -> Asset:
        """Save one dataset file for the current browser session.

        Parameters
        ----------
        request
            Raw file bytes.
        session
            Owner of the new file.
        filename
            Local filename used to select its decoder.

        Returns
        -------
        Asset
            File ID for use in an ordered workflow sample.
        """
        row = await save_upload(
            request,
            filename,
            lambda record: store.add_asset(session.owner, record, session.id),
            "dataset",
        )
        return Asset.model_validate(row)

    @app.post("/web/api/conversions", status_code=202)
    def browser_conversion(
        request: Request,
        body: Conversion,
        session: Session,
        idempotency_key: Annotated[str, Header(min_length=1, max_length=128)],
    ) -> Job:
        """Run a supported compile preset using a model inspected in this session.

        Parameters
        ----------
        request
            Request carrying the correlation identifier.
        body
            The inspected model and selected preset.
        session
            Owner of the temporary working files.
        idempotency_key
            Stable submission identifier for client retries.

        Returns
        -------
        Job
            Accepted conversion and current stage state.
        """
        preset = presets.get(body.preset)
        if preset is None:
            if body.model_dump(exclude_defaults=True).keys() != {"model_id", "preset"}:
                raise HTTPException(
                    422, "Select an enabled workflow preset for these options."
                )
            operation = Submission(
                operation="compile_om", input_id=body.model_id, preset=body.preset
            )
            return queue_job(
                request, operation, session.owner, idempotency_key, session.id
            )
        workflow = resolve_workflow(body, preset)
        row = store.submit(
            session.owner,
            "compile_om",
            body.model_id,
            preset.image,
            preset.id,
            preset.simulated,
            idempotency_key,
            session.id,
            workflow,
        )
        logger.info(
            "request_id=%s job_id=%s preset=%s",
            request.state.request_id,
            row["id"],
            preset.id,
        )
        return browser_task(session, row["id"])

    @app.get("/web/api/tasks")
    def browser_tasks(session: Session) -> list[BrowserTask]:
        """List the latest browser tasks belonging to the active account.

        Parameters
        ----------
        session
            Authenticated browser session.

        Returns
        -------
        list[BrowserTask]
            The most recent browser tasks with their original filenames.
        """
        with store.connect() as db:
            rows = db.execute(
                "SELECT j.id, a.filename, j.operation, j.preset, j.state, j.created_at "
                "FROM jobs j JOIN assets a ON a.id = j.input_id "
                "WHERE j.owner = %s AND j.browser_session_id IS NOT NULL "
                "ORDER BY j.created_at DESC, j.id DESC LIMIT 50",
                (session.owner,),
            ).fetchall()
        return [BrowserTask.model_validate(row) for row in rows]

    @app.get("/web/api/tasks/{job_id}")
    def browser_task(session: Session, job_id: UUID) -> Job:
        """Show task state and only files accessible in the current session.

        Parameters
        ----------
        session
            Authenticated browser session.
        job_id
            Account-owned task to view.

        Returns
        -------
        Job
            Task status with current-session outputs.
        """
        row, outputs = store.job(session.owner, job_id)
        available = row["browser_session_id"] == session.id
        stages = row["stages"]
        if row["workflow"] is not None:
            stages = {
                stage: stages.get(stage, "pending")
                for stage in planned_stages(row["workflow"])
            }
            if row["state"] in {"failed", "cancelled"}:
                marked = False
                for stage, status in stages.items():
                    if status == "running":
                        stages[stage] = row["state"]
                        marked = True
                if (
                    row["state"] == "failed"
                    and not marked
                    and "failed" not in stages.values()
                ):
                    first_pending = next(
                        (
                            stage
                            for stage, status in stages.items()
                            if status == "pending"
                        ),
                        None,
                    )
                    if first_pending is not None:
                        stages[first_pending] = "failed"
        return Job.model_validate(
            {**row, "stages": stages, "outputs": outputs if available else []}
        )

    @app.get("/web/api/tasks/{job_id}/logs")
    def browser_log(
        session: Session,
        job_id: UUID,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> LogPart:
        """Read the current attempt log across sessions of the same account.

        Parameters
        ----------
        session
            Authenticated browser session.
        job_id
            Account-owned task to view.
        offset
            Byte offset from the previous response.

        Returns
        -------
        LogPart
            A bounded slice and the next cursor.
        """
        return read_log(session.owner, job_id, offset)

    @app.post("/web/api/tasks/{job_id}/cancel", status_code=202)
    def browser_cancel(session: Session, job_id: UUID) -> Job:
        """Cancel a queued task or request that its running attempt stop.

        Parameters
        ----------
        session
            Authenticated browser session.
        job_id
            Account-owned task to cancel.

        Returns
        -------
        Job
            Updated account-owned task state.
        """
        store.cancel(session.owner, job_id)
        return browser_task(session, job_id)

    @app.get("/web/api/files/{asset_id}")
    def browser_file(session: Session, asset_id: UUID) -> FileResponse:
        """Download an output from the current browser session.

        Parameters
        ----------
        session
            Authenticated browser session that owns the file.
        asset_id
            File to download.

        Returns
        -------
        FileResponse
            Registered file as an attachment.
        """
        return asset_download(session.owner, asset_id, session.id)

    web_dir = Path(__file__).with_name("web")
    app.mount("/web/static", StaticFiles(directory=web_dir), name="workbench-static")

    @app.get("/", include_in_schema=False)
    def workbench() -> FileResponse:
        """Serve the local conversion workbench.

        Returns
        -------
        FileResponse
            Workbench HTML with its same-origin asset references.
        """
        return FileResponse(web_dir / "index.html", media_type="text/html")

    return app
