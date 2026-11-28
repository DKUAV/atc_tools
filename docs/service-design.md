# ATC model service design

The laboratory runs one ARM64 application container. Its inherited entrypoint initializes CANN and public-key SSH. The service command starts the HTTP API and one scheduler with local SQLite storage. Container init reaps orphan processes.

## Execution and storage

SQLite uses WAL and per-connection foreign keys. Each state transition reserves a writer for a short transaction. A claim records the attempt and changes the job state before execution starts. The scheduler holds a local single-instance lock and finalizes interrupted attempts when the container restarts.

CPU and compilation slots are independently configurable. All device operations share one slot for the assigned NPU. Each slot runs subprocesses in a separate process group under a dedicated Linux identity. The scheduler grants read access to the staged inputs and write access to the task output. Service data and published artifacts remain private. CPU and memory limits apply to the whole container. This deployment serves trusted laboratory users.

## API and persistent records

Use versioned HTTP endpoints with typed requests, typed responses, and an OpenAPI contract for the later Web UI. The API exposes the same operation schemas for standalone tasks and workflow steps. Administrative presets define what users can select. [FastAPI request models](https://fastapi.tiangolo.com/tutorial/body/)

| API | Purpose |
| --- | --- |
| `GET /v1/capabilities` | List enabled targets, compatible presets, required inputs, and optional steps. |
| `POST /v1/assets` | Stream an upload to temporary storage, enforce limits, and register an owned asset ID after the file is complete. |
| `POST /v1/jobs` | Submit one operation with typed parameters and input asset IDs. Return a job ID after committing the request. |
| `POST /v1/workflows` | Submit named steps with inputs that reference owned asset IDs or named outputs of earlier steps. Return a workflow ID and step IDs. |
| Job and workflow resources | List and retrieve owned tasks, show step state and validation outcomes, request cancellation or retry, and read logs. |
| Asset resources | Read owned metadata and download published files by asset ID. |

Account records carry permissions. Asset records carry ownership, kind, checksum, and a storage key. Workflows own the step dependency graph. A job stores one operation, its inputs and resolved preset, its state, and its output references. An attempt records one execution of a job, including its installed tool version, assigned resources, timing, log location, and outcome. A retry creates a new attempt and retains the earlier results and logs.

Clients can resend a submission with the same idempotency key to recover the original job or workflow ID. Scope the key to the account and submission endpoint, and reject reuse of the key with a different payload.

Administrators provision accounts. The API uses server-side browser sessions with secure cookies and CSRF protection, plus revocable bearer tokens for scripts. Store password hashes and token verifiers. Authorize every file, log, task, and workflow access through the owning account. The API checks filenames, upload limits, and parameter types. Model and image parsing runs in a task subprocess, and inspection must confirm that a model is self-contained before it can be used by dependent steps. All execution commands and mounts originate from service-owned configuration. [OWASP session management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html) · [OWASP password storage](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)

## Operations and delivery

The scheduler stages inputs, writes the execution request and captures a bounded log. The executor publishes progress and a result manifest in its workspace. The scheduler validates outputs and the current attempt before registering files. Cancellation drains the process group before the device slot is reused.

Reserve the target NPU during profiling. The scheduler serializes service jobs and lab operators coordinate external jobs. See the [deployment guide](../deploy/README.md) for acceptance and release procedures.
