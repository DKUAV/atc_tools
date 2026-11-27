# ATC Tools Service Architecture

The service runs on one laboratory host. Docker Compose starts the API, PostgreSQL, and a worker. The worker starts isolated containers to inspect, convert, and verify models. The first implementation milestone delivers the upload-to-download path through the API. The Web UI will use the same task and resource interfaces. [Service requirements](service-requirements.md)

## Repository and deployment layout

Keep two Python packages in one repository. `atc_tools` owns the existing model operations and the entry point used by task containers. `atc_service` owns HTTP endpoints, accounts, file ownership, and scheduling. Give the service its own `pyproject.toml` so that the service image can install its dependencies independently of the model tools.

```text
src/atc_tools/              Existing CLI, model operations, and task entry point
scripts/                    Existing ATC and CANN scripts
service/pyproject.toml      Independent service dependencies
service/src/atc_service/    API, jobs, storage, and worker code
deploy/compose.yaml          Definitions for the three long-running services
deploy/Dockerfile.service    Common image for API and worker
deploy/Dockerfile.executor   Task entry point and tools layered onto execution images
```

| Container | Image and responsibility |
| --- | --- |
| `api` | The service image runs FastAPI, authenticates users, accepts uploads and submissions, and serves status, logs, and downloads. |
| `postgres` | A pinned official PostgreSQL image stores accounts, resource metadata, workflows, and scheduling state on a persistent volume. Database migrations are versioned with the service. |
| `worker` | The same service image runs a polling loop, reserves capacity, launches task containers, and publishes their results. |
| Task container | The worker creates one per running attempt from an approved execution image. It runs one operation and is removed after its result and logs are collected. |

Compose publishes the API through the laboratory intranet endpoint. The API and worker connect to PostgreSQL on the Compose network. They share a persistent host data directory for uploads and results. The Docker daemon control socket is available only to the trusted worker. Treat the worker as a privileged component because Docker control can grant access to the host. [Docker daemon security](https://docs.docker.com/engine/security/)

## Execution images and presets

Use an ONNX execution image for model inspection, CPU reference inference, and output comparison. Use the configured AMCT and ATC environments for quantization and compilation. Target presets record the SoC, compatible toolchain, execution image, and approved options. The service snapshots the resolved preset and image reference when it accepts a task so that a retry uses the same settings.

Base the 910B2 CANN execution image on the pinned `ascend-atc` image maintained in Luciole `images` under `src/ascend-atc`. The current reference variant is Linux ARM64 with CANN 8.5.0, ATC, AMCT in its own Python environment, and AIS Bench. Keep its vendor and AMCT environments isolated. Add `atc_tools`, the service task entry point, and the required scripts in a compatible environment. Disable the reference image's SSH service during task execution. Enable 310B4 only after a compatible compiler and operator packages have produced the expected OM in a smoke test. Verify the 910B2 driver, device mounts, and `msprof` in the selected image on the lab host.

The worker receives an administrator-configured host storage root and uses host paths when asking Docker to mount files. For example, `/srv/atc-tools` can be mounted as `/data` in the API and worker, with each task container receiving only its assigned paths under that host root. Mount inputs read-only and give each attempt its own writable output directory. Use a non-root task process, a constrained filesystem and network, and explicit CPU, memory, disk, and runtime limits. Grant the assigned NPU to the operations that require it. [Docker bind mounts](https://docs.docker.com/engine/storage/bind-mounts/) · [Container resource limits](https://docs.docker.com/engine/containers/resource_constraints/)

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

Account records carry permissions. Asset records carry ownership, kind, checksum, and a storage key. Workflows own the step dependency graph. A job stores one operation, its inputs and resolved preset, its state, and its output references. An attempt records one execution of a job, including its container identity, assigned resources, timing, log location, and outcome. A retry creates a new attempt and retains the earlier results and logs.

Clients can resend a submission with the same idempotency key to recover the original job or workflow ID. Scope the key to the account and submission endpoint, and reject reuse of the key with a different payload.

Administrators provision accounts. The API uses server-side browser sessions with secure cookies and CSRF protection, plus revocable bearer tokens for scripts. Store password hashes and token verifiers. Authorize every file, log, task, and workflow access through the owning account. The API checks filenames, upload limits, and parameter types. Model and image parsing runs in a task container, and inspection must confirm that a model is self-contained before it can be used by dependent steps. All execution commands and mounts originate from service-owned configuration. [OWASP session management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html) · [OWASP password storage](https://cheatsheetseries.owasp.org/cheatsheets/Password_Storage_Cheat_Sheet.html)

## Worker protocol and artifact delivery

1. The API validates resource ownership, allowable parameters, and workflow references. It rejects cycles, incompatible input and output types, and unsupported target combinations before committing the workflow. Ready jobs enter the queue and dependent jobs wait for published inputs.
2. The worker polls PostgreSQL for eligible jobs. It claims a job, creates an attempt, and reserves capacity in one short transaction. Row locks with `FOR UPDATE SKIP LOCKED` let concurrent workers skip jobs another worker is claiming. The worker releases the transaction before starting the container. [PostgreSQL row locking](https://www.postgresql.org/docs/current/sql-select.html#SQL-FOR-UPDATE-SHARE)
3. The worker stages the inputs and writes a versioned `request.json` with the operation, resolved options, and paths inside the container. The task entry point dispatches to a fixed operation and writes a `result.json` manifest with the outcome and relative artifact paths. Use current `atc_tools` functions and scripts for execution, including the existing tensor output manifest. [ATC compile script](../scripts/compile.sh) · [Tensor I/O](../src/atc_tools/model/tensor_io.py)
4. The worker captures standard output and error in the attempt log. It checks the exit status, validates the result manifest and artifact paths, and writes verified files to persistent storage. It registers each artifact and updates job state in a transaction guarded by the current attempt ID. Dependent jobs become eligible when their required asset records are published.
5. The API returns the primary OM, its build information, and any requested reports with independent step outcomes. Downloads become available as each artifact is published. A numerical comparison can complete successfully with an out-of-tolerance verdict. A failed optional step can be retried against the same OM.

## Scheduling, cancellation, and recovery

Start with one shared compile slot across 310B4 and 910B2. Give CPU work its own configured capacity. A 910B2 device offers one slot shared by inference and `msprof`. For an INT8 comparison, prepare both OMs and their validation inputs before reserving a device for the paired runs. Execute the floating-point and INT8 OMs on that device with the same validation inputs. Run `msprof` independently against the OM selected for delivery.

A job moves from waiting for dependencies to queued, running, and a terminal state. A failed prerequisite blocks its dependents. Unrelated steps can continue. A queued job can be cancelled in the database. For running jobs, the worker records the cancellation or timeout, stops the container, and waits for termination before releasing capacity.

Label task containers with their attempt IDs. On restart, the worker reconciles active attempts with those containers before issuing each resource slot again. An expired heartbeat triggers reconciliation. The worker must establish that the old container has stopped before a replacement starts. The current-attempt check also protects artifact publication from stale results.

## Logging and operation

Follow Minifly `src/minifly/utils/logger.py` for component loggers with configurable console and rotating file output. Initialize logging at each service entry point and use separate rotating files for API and worker. Attach request, job, and attempt IDs where available. Record authentication and task lifecycle events, and keep credentials, bearer tokens, and user file contents out of service logs.

The API exposes user-visible attempt logs after verifying task ownership. The build record includes the target SoC, toolchain version, image reference, resolved presets, and delivered file checksums. Prepare the images on a connected machine and import the pinned images into the laboratory host before starting `docker compose up -d`. Treat the database and artifact directory as persistent deployment data.

## Implementation and validation order

1. **First API workflow.** Implement accounts and tokens, streamed asset upload, capability discovery, job submission and polling, and authenticated artifact download. Run a real small ONNX inspection followed by a 310B4 compile step in a substitute image. Verify the returned OM placeholder and build manifest through the download API.
2. **Concurrency and recovery.** Use multiple accounts and competing workers to test ownership, atomic claims, slot limits, duplicate submission, cancellation, timeouts, restart reconciliation, and result publication. Test invalid uploads, over-limit files, and untrusted output manifests with synthetic fixtures.
3. **Target operations.** Add the real ATC and AMCT bindings, 910B2 comparison, and `msprof`. Exercise CPU inference and workflow combinations on a connected developer machine. On the lab host, compile for both targets and verify NPU assignment, both INT8 comparison reports, profiling, and isolation using the real toolchains.
4. **Web UI and handoff.** Build the UI against the tested API. Deliver an example ONNX, an example API request, and an offline deployment procedure. Verify that a user can submit a task, return to it later, and retrieve the OM and selected reports.

## Laboratory configuration to finalize

Record compatible image digests and CANN, AMCT, driver, and `msprof` versions for each enabled operation. Confirm the host storage root, volume ownership and quotas, device inventory, runtime permissions, HTTPS endpoint, and resource and retention limits. Keep these values in deployment configuration and verify them during the lab smoke test.
