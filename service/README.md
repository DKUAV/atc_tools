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

## Use the web workbench

Provision a local account password at the operator prompt. Setting a new password ends the account's existing browser sessions.

```bash
docker compose -f deploy/compose.yaml exec api atc-service set-browser-password alice
```

Open `/` on the same service origin. For local HTTP testing, set `ATC_BROWSER_SECURE_COOKIES=0` before starting Compose and use the loopback bind. On the lab network, serve the workbench over HTTPS with secure cookies and apply ingress rate and body size limits to the login endpoint. Keep the API's Compose port on loopback when the HTTPS proxy runs on the same host.

Set `ATC_SIMULATE_COMPILE=1` to try the 310B4 and 910B2 workflows without Ascend hardware. Both targets have original, mixed Float16, preferred FP16, and INT8 options. Calibration files appear for INT8. For 910B2, either precision comparison or `msprof` enables a separate validation set. Both sets use the same dimensions and numerical settings. The simulator performs ONNX inspection, dataset preparation and CPU reference inference when requested. Its `model.simulated`, `accuracy.json`, and `profiling.zip` identify the simulated results.

Each dataset accepts images, NumPy arrays, and raw binaries. For a multi-input ONNX, use `input_0`, `input_1`, and so on as sibling subfolders with matching sample basenames, or select files named like `sample.input_0.npy`. Each sample supplies all inputs in model order. Images decode to RGB or grayscale. NumPy arrays and raw data use the selected model layout. Raw data defaults to the model input shape and little-endian dtype. Each dataset accepts up to 128 samples.

Sign out after downloading the required results. The default browser session lifetime is 12 hours and can be changed with `ATC_BROWSER_SESSION_SECONDS`. Logout or expiry revokes access to session files. The worker removes their bytes after the session's pending jobs finish. Task history and logs remain visible at the next sign-in. See the [Web UI requirements](../docs/web-ui-requirements.md) and [Web UI design](../docs/web-ui-design.md).

With the simulator and local HTTP enabled, run `uv run --locked python service/tests/browser_smoke.py` to check browser endpoints, mixed-format datasets, optional stage failure, account isolation, and cleanup. Set `ATC_SMOKE_USER_A`, `ATC_SMOKE_USER_B`, `ATC_SMOKE_PASSWORD_A`, and `ATC_SMOKE_PASSWORD_B` for two provisioned test accounts. Optionally set `ATC_SMOKE_DATA_DIR` to the host data directory to verify physical file removal.

## Enable the laboratory toolchains

Build the service executor on an Ascend-compatible host using a locally provisioned `ascend-atc` base image. Use a base image with the matching CANN Toolkit and ops packages for each target.

```bash
docker build -f deploy/Dockerfile.ascend-executor \
  --build-arg ASCEND_ATC_IMAGE=ascend-atc:8.5.0-910b-ubuntu22.04-py3.11 \
  -t atc-executor:910b2 .
```

Set `ATC_COMPILE_IMAGE` for the 310B4 executor and `ATC_COMPILE_910B2_IMAGE` for the 910B2 executor. Leave `ATC_SIMULATE_COMPILE=0`. The configured images enable the three floating-point presets. List enabled targets in `ATC_AMCT_TARGETS` after verifying AMCT calibration and ATC deployment conversion for each target. Use `ATC_910B2_DEVICE=0` to enable checks on host device zero. The Docker daemon needs the `ascend` runtime, and executor UID `10001` needs access to the configured device and profiling facilities. Real toolchain and non-root runtime access require lab-host validation.

The supported precision options use ATC `--precision_mode_v2`. The INT8 preset uses the AMCT default activation algorithm with the number of supplied calibration batches. The comparison uses CPU ONNX and, for INT8, a same-device original-precision OM baseline. `msprof` runs the primary OM with the shared validation inputs and packages its collected results. [ATC precision modes](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/atctool/atlasatcparam_16_0069.html) · [AMCT configuration](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/amct/atlasamct_16_0163.html) · [CANN 8.5 runtime profiling](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0011.html)

## Submit through the API

Send the token in an `Authorization: Bearer` header. Upload model bytes with `POST /v1/assets?filename=model.onnx` and the `application/octet-stream` content type. Submit an `inspect_onnx` job with the returned `input_id`. After it succeeds, submit a `compile_om` job with the same input and one of the presets returned by `GET /v1/capabilities`. A stable `Idempotency-Key` lets a client recover the same job after a request is interrupted.

Poll `GET /v1/jobs/{job_id}` for status and artifact IDs. Use `GET /v1/jobs/{job_id}/logs` with the returned `next_offset` to read the log incrementally. The download endpoint is `GET /v1/assets/{asset_id}/download`.

The default configuration enables inspection. `ATC_SIMULATE_COMPILE=1` adds a development-only preset that returns `model.simulated`. To enable the real 310B4 preset, select a validated CANN image with the task entry point using `ATC_COMPILE_IMAGE` and leave simulation disabled. The task runtime and host filesystem quota must be verified on the lab host before opening the service to users.
