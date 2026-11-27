# ATC Tools Service

The first backend milestone provides token-authenticated ONNX upload, isolated model inspection, a simulated 310B4 compile preset, and per-job status, cancellation, logs, and artifact downloads. See [service architecture](../docs/service-design.md) for the full deployment design.

## Run the local smoke deployment

Run these commands from the repository root on a Linux host with Docker and Compose.

```bash
export ATC_DB_PASSWORD="$(openssl rand -hex 24)"
export ATC_HOST_DATA_DIR="$PWD/work/service-data"
export ATC_SIMULATE_COMPILE=1
mkdir -p "$ATC_HOST_DATA_DIR"

docker compose -f deploy/compose.yaml --profile build build api executor
docker compose -f deploy/compose.yaml up -d --no-build
```

The Compose API port defaults to `127.0.0.1:8766` and the OpenAPI schema is available at `/openapi.json`. Use `ATC_API_PORT` to change the host port. On Docker Desktop for Windows, set `ATC_HOST_DATA_DIR` to the Windows bind source and `ATC_DOCKER_DATA_DIR` to the same directory as seen by the Linux daemon. For example, use `D:/atc-tools-data` as the Compose bind source and `/run/desktop/mnt/host/d/atc-tools-data` for task containers.

Issue two test tokens and run the integration smoke test. The script uploads a small generated ONNX, exercises both accounts, checks failure logs, and downloads a file explicitly marked as simulated.

```bash
export ATC_SMOKE_TOKEN_A="$(docker compose -f deploy/compose.yaml exec -T api atc-service issue-token alice)"
export ATC_SMOKE_TOKEN_B="$(docker compose -f deploy/compose.yaml exec -T api atc-service issue-token bob)"
uv run --locked python service/tests/smoke.py
unset ATC_SMOKE_TOKEN_A ATC_SMOKE_TOKEN_B
```

Use `docker compose -f deploy/compose.yaml exec api atc-service revoke-token` to revoke a token through a private prompt. Use `docker compose -f deploy/compose.yaml down` to stop the services and retain the database volume and files.

## Submit through the API

Send the token in an `Authorization: Bearer` header. Upload model bytes with `POST /v1/assets?filename=model.onnx` and the `application/octet-stream` content type. Submit an `inspect_onnx` job with the returned `input_id`. After it succeeds, submit a `compile_om` job with the same input and one of the presets returned by `GET /v1/capabilities`. A stable `Idempotency-Key` lets a client recover the same job after a request is interrupted.

Poll `GET /v1/jobs/{job_id}` for status and artifact IDs. Use `GET /v1/jobs/{job_id}/logs` with the returned `next_offset` to read the log incrementally. The download endpoint is `GET /v1/assets/{asset_id}/download`.

The default configuration enables inspection. `ATC_SIMULATE_COMPILE=1` adds a development-only preset that returns `model.simulated`. To enable the real 310B4 preset, select a validated CANN image with the task entry point using `ATC_COMPILE_IMAGE` and leave simulation disabled. The task runtime and host filesystem quota must be verified on the lab host before opening the service to users.
