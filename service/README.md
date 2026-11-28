# ATC Tools Service

ATC Tools provides a browser workbench and bearer-token API for ONNX inspection, conversion, quantization, inference, comparison, and profiling. See the [service architecture](../docs/service-design.md) for ownership and execution boundaries.

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

To try the workflow with a small fixed-shape model, run from the repository root. Upload `work/demo.onnx` and select `work/demo.npy` when asked for validation or calibration data.

```bash
mkdir -p work
uv run --locked python - <<'PY'
import numpy as np
from onnx import TensorProto, helper, save

value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
node = helper.make_node("Relu", ["input"], ["output"])
graph = helper.make_graph([node], "demo", [value], [output])
save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9), "work/demo.onnx")
np.save("work/demo.npy", np.array([[-1, 2]], dtype=np.float32), allow_pickle=False)
PY
```

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

Open `profiling.json` inside the archive for the input count, profiler wall-clock duration and collection size. Use the native `msprof` files in the archive for device timing.

The Worker holds one coordinator lock and uses independent CPU, compiler and device queues. `ATC_CPU_SLOTS` defaults to two and `ATC_COMPILE_SLOTS` to one. One slot handles the configured 910B2 device. Workflows compile their primary OM and any floating-point baseline before requesting the device slot, and keep the same task ID across both phases. `ATC_SCRATCH_BYTES` bounds the temporary workspace of each workflow container. Use `/readyz` to check worker and image availability. [Offline bundle and rollout](../deploy/OFFLINE.md)

## Submit through the API

Send the token in an `Authorization: Bearer` header. Upload model bytes with `POST /v1/assets?filename=model.onnx` and the `application/octet-stream` content type. Submit an `inspect_onnx` job with the returned `input_id`. After it succeeds, submit a `compile_om` job with the same input and one of the presets returned by `GET /v1/capabilities`. A stable `Idempotency-Key` lets a client recover the same job after a request is interrupted.

Poll `GET /v1/jobs/{job_id}` for status and artifact IDs. Use `GET /v1/jobs/{job_id}/logs` with the returned `next_offset` to read the log incrementally. The download endpoint is `GET /v1/assets/{asset_id}/download`.

`GET /v1/capabilities` reports both legacy job presets and the full `workflow_presets` and `standalone_operations` lists. `GET /v1/jobs` lists recent script jobs. Upload each calibration or validation file with `POST /v1/dataset-files?filename=sample.npy` using `application/octet-stream`. Organize the returned asset IDs as batches ordered by the model inputs.

| Endpoint | Request and result |
| --- | --- |
| `POST /v1/workflows` | `model_id`, `preset`, optional `calibration`, `validation`, `compare`, `profile`, `shapes` and `preparation`. Returns a full conversion task. |
| `POST /v1/operations` | `operation`, `input_id` and the applicable fields from the operation table. |
| `POST /v1/jobs/{job_id}/retry` | `step` set to `compare` or `profile` and optional replacement `validation`. Returns a child task linked to the original OM. |

| `operation` | `input_id` | Other fields and output |
| --- | --- | --- |
| `quantize_onnx` | Inspected source ONNX | INT8 `preset` and `samples`. Produces a deployment ONNX. |
| `compile_om` | Inspected ONNX | `preset`. For standalone INT8, use the inspected deployment ONNX from quantization. Produces OM and build information. |
| `run_cpu` | Inspected ONNX | `samples`. Produces `tensors.zip` with CPU reference outputs. |
| `run_npu` | Inspected source ONNX | Enabled 910B2 `preset`, `primary_id` of a 910B2 OM from that model, and `samples`. Produces `tensors.zip`. |
| `compare_outputs` | Reference tensor archive | `actual_id` of the second archive. Produces accuracy JSON and an inline verdict. |
| `profile_om` | Inspected source ONNX | Enabled 910B2 `preset`, `primary_id` and `samples`. Produces a profiling archive. |

For example, submit `{"operation":"run_cpu","input_id":"<inspected-onnx-id>","samples":[["<dataset-asset-id>"]]}` to `/v1/operations`. `GET /openapi.json` exposes the typed request fields and responses. In the browser, a failed optional verification displays a retry action while the original OM remains available. Users can supply replacement validation files when retrying.

Run `uv run --locked python service/tests/delivery_smoke.py` with the same bearer-token environment used by `smoke.py` to exercise standalone operations, phased logs and retry end to end. The simulator returns `model.simulated` and marks substituted NPU results. Laboratory acceptance covers real ATC, AMCT and device behavior.
