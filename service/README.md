# ATC Tools Service

[English](README.md) | [中文](README_zh.md)

ATC Tools Service provides a browser workbench and a bearer-token HTTP API for ONNX model inspection, OM compilation, INT8 quantization, inference, accuracy comparison, and profiling.

> **New here?** Start with [Quick Start (Simulation Mode)](#quick-start-simulation-mode) to try the full workflow locally without Ascend hardware.  
> For architecture, ownership, and execution boundaries, see the [Service Architecture](../docs/service-design.md).

## Contents

- [Prepare Images](#prepare-images)
- [Quick Start (Simulation Mode)](#quick-start-simulation-mode)
- [Using the Web Workbench](#using-the-web-workbench)
- [Enabling Real Ascend Toolchains](#enabling-real-ascend-toolchains)
- [Using the HTTP API](#using-the-http-api)
- [Running Smoke Tests](#running-smoke-tests)
- [GitHub Actions](#github-actions)

---

## Prepare Images

Use the released Linux ARM64 application image. Copy deploy/lab.env.example to lab.env, specifying the exact application tag, a local data directory and the root-owned SSH public key directory containing authorized_keys. See the [offline deployment guide](../deploy/OFFLINE.md).

## Quick Start (Simulation Mode)

Set ATC_SIMULATE_COMPILE=1 in lab.env. For local HTTP browser tests set ATC_BROWSER_SECURE_COOKIES=0.

```bash
docker compose --env-file lab.env -f deploy/compose.yaml up -d --no-build --pull never --wait
```

### Common Management Commands

```bash
# Stop services (keeps the database volume and stored files)
docker compose --env-file lab.env -f deploy/compose.yaml down

# Issue an API bearer token for a user
docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token alice

# Revoke a token via a private prompt
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service revoke-token
```

---

## Using the Web Workbench

### 1. Create a Browser Account

Set a password for a local account using the interactive prompt. Setting a new password automatically ends all existing browser sessions for that account.

```bash
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service set-browser-password alice
```

Then open `http://127.0.0.1:8766/` in your browser and sign in.

> **Security notes**:
> - **Local HTTP testing**: Set `ATC_BROWSER_SECURE_COOKIES=0` before starting Compose and bind only to loopback (`127.0.0.1`).
> - **Lab / shared network**: Serve the workbench over HTTPS with secure cookies enabled (default `ATC_BROWSER_SECURE_COOKIES=1`). Keep the Compose API port bound to loopback when the HTTPS reverse proxy runs on the same host, and configure ingress rate limits and request body size limits on the login endpoint.

### 2. Try with a Small Demo Model

In simulation mode (`ATC_SIMULATE_COMPILE=1`), both **310B4** and **910B2** targets support `original`, `mixed Float16`, `preferred FP16`, and `INT8` presets:

- Selecting **INT8** enables the calibration dataset input.
- On **910B2**, enabling precision comparison or `msprof` adds a separate validation dataset input (using the same dimensions and numerical settings).
- The simulator runs real ONNX inspection, dataset preparation, and CPU reference inference, and marks simulated outputs clearly as `model.simulated`, `accuracy.json`, and `profiling.zip`.

Generate a minimal fixed-shape ONNX model (`work/demo.onnx`) and sample tensor (`work/demo.npy`) from the repository root:

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

Upload `work/demo.onnx` in the workbench and select `work/demo.npy` when prompted for validation or calibration data.

### 3. Dataset Format Rules

- **Supported formats**: Images (decoded to RGB or grayscale), NumPy arrays (`.npy`), and raw binaries (`.bin`).
- **Layout & dtype**: NumPy arrays and raw binaries use the selected model layout. Raw binaries default to the model input shape and little-endian dtype.
- **Multi-input models**: Supply every input in model order for each sample using either:
  - Sibling subfolders `input_0/`, `input_1/`, ... with matching sample basenames, or
  - Flat files named like `sample.input_0.npy`, `sample.input_1.npy`.
- **Size limit**: Up to **128 samples** per dataset.

### 4. Sessions & File Cleanup

- Default browser session lifetime is **12 hours** (configurable via `ATC_BROWSER_SESSION_SECONDS`).
- Always sign out after downloading your results.
- Logging out or letting a session expire immediately revokes access to session files; the worker deletes their bytes once pending jobs finish.
- Task history and logs remain visible on your next sign-in.

For full UI specifications, see [Web UI Requirements](../docs/web-ui-requirements.md) and [Web UI Design](../docs/web-ui-design.md).

---

## Enabling Real Ascend Toolchains

The image includes CANN. Set ATC_SIMULATE_COMPILE=0 and configure ATC_AMCT_TARGETS for validated SoCs. For 910B2 inference and profiling set ATC_RUNTIME=ascend and ATC_910B2_DEVICE. Reserve the device throughout each capture, including SSH and external jobs.

### 2. Precision, Quantization & Profiling Details

- **Floating-point presets**: Mapped to ATC `--precision_mode_v2` ([ATC precision modes](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/atctool/atlasatcparam_16_0069.html)).
- **INT8 quantization**: Uses the AMCT default activation algorithm with the number of supplied calibration batches ([AMCT configuration](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/amct/atlasamct_16_0163.html)).
- **Accuracy comparison**: Compares against CPU ONNX reference outputs; for INT8, also compares against a same-device original-precision OM baseline.
- **Profiling (`msprof`)**: Runs the primary OM with the shared validation inputs and archives the collected data ([CANN 8.5 runtime profiling](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0011.html)). Open `profiling.json` inside the archive for input count, profiler wall-clock duration, and collection size; use the native `msprof` files inside the archive for device timing.

Inference, device comparisons and profiling share one device slot. The readiness endpoint checks SQLite, scheduler heartbeat and enabled tools.

## Using the HTTP API

All API requests require an `Authorization: Bearer <token>` header. Pass a stable `Idempotency-Key` header to safely recover the same job if a submission request is interrupted. Full typed request/response schemas are available at `GET /openapi.json`.

### Typical Workflow

1. **Discover capabilities**: Call `GET /v1/capabilities` to see legacy presets, `workflow_presets`, and `standalone_operations`.
2. **Upload a model**: `POST /v1/assets?filename=model.onnx` with `Content-Type: application/octet-stream` → returns `input_id`.
3. **Upload dataset files** *(optional)*: `POST /v1/dataset-files?filename=sample.npy` (`application/octet-stream`), then organize the returned asset IDs as batches ordered by model inputs.
4. **Inspect & run**:
   - Submit an `inspect_onnx` job with `input_id`.
   - Once inspection succeeds, submit a `compile_om` job, a full workflow (`POST /v1/workflows`), or a standalone operation (`POST /v1/operations`).
5. **Track & download**:
   - List recent script jobs: `GET /v1/jobs`
   - Poll status & artifact IDs: `GET /v1/jobs/{job_id}`
   - Read logs incrementally: `GET /v1/jobs/{job_id}/logs` (pass the returned `next_offset`)
   - Download artifacts: `GET /v1/assets/{asset_id}/download`

### Workflow & Retry Endpoints

| Endpoint | Request fields & behavior |
| --- | --- |
| `POST /v1/workflows` | `model_id`, `preset`, plus optional `calibration`, `validation`, `compare`, `profile`, `shapes`, and `preparation`. Starts a full conversion task. |
| `POST /v1/operations` | `operation`, `input_id`, and operation-specific fields (see table below). |
| `POST /v1/jobs/{job_id}/retry` | `step` (`compare` or `profile`) and optional replacement `validation`. Creates a child task linked to the original OM. |

> In the browser workbench, if an optional verification step fails, a retry action is shown while the original OM stays downloadable. Users can also provide replacement validation files when retrying.

### Standalone Operations (`POST /v1/operations`)

| `operation` | `input_id` | Additional fields & output |
| --- | --- | --- |
| `quantize_onnx` | Inspected source ONNX | INT8 `preset` and `samples`. Produces a deployment ONNX. |
| `compile_om` | Inspected ONNX | `preset`. For standalone INT8, pass the inspected deployment ONNX from quantization. Produces OM and build info. |
| `run_cpu` | Inspected ONNX | `samples`. Produces `tensors.zip` with CPU reference outputs. |
| `run_npu` | Inspected source ONNX | Enabled 910B2 `preset`, `primary_id` (910B2 OM from that model), and `samples`. Produces `tensors.zip`. |
| `compare_outputs` | Reference tensor archive | `actual_id` (second tensor archive). Produces accuracy JSON and an inline verdict. |
| `profile_om` | Inspected source ONNX | Enabled 910B2 `preset`, `primary_id`, and `samples`. Produces a profiling archive. |

Example `POST /v1/operations` payload:

```json
{
  "operation": "run_cpu",
  "input_id": "<inspected-onnx-id>",
  "samples": [["<dataset-asset-id>"]]
}
```

---

## Running Smoke Tests

All three smoke test suites run against a local simulation deployment (`ATC_SIMULATE_COMPILE=1`).

### 1. API & Delivery Smoke Tests

Issue two test tokens and run `smoke.py` (uploads a generated ONNX, checks multi-account isolation, failure logs, and simulated downloads) and `delivery_smoke.py` (standalone operations, phased logs, and retry):

```bash
export ATC_SMOKE_TOKEN_A="$(docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token alice)"
export ATC_SMOKE_TOKEN_B="$(docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token bob)"

uv run --locked python service/tests/smoke.py
uv run --locked python service/tests/delivery_smoke.py

unset ATC_SMOKE_TOKEN_A ATC_SMOKE_TOKEN_B
```

> The simulator returns `model.simulated` and marks substituted NPU results. Real ATC, AMCT, and device behavior are covered by laboratory acceptance.

### 2. Browser Workbench Smoke Test

With simulation mode and local HTTP (`ATC_BROWSER_SECURE_COOKIES=0`) enabled, provision two test accounts (`set-browser-password`) and run `browser_smoke.py` to verify browser endpoints, mixed-format datasets, optional stage failure, account isolation, and file cleanup:

```bash
export ATC_SMOKE_USER_A="alice"
export ATC_SMOKE_PASSWORD_A="<alice-password>"
export ATC_SMOKE_USER_B="bob"
export ATC_SMOKE_PASSWORD_B="<bob-password>"
# Optional: verify physical file removal in the host data directory
export ATC_SMOKE_DATA_DIR="$PWD/work/service-data"

uv run --locked python service/tests/browser_smoke.py
```

---

## GitHub Actions

ARM64 CI verifies SSH, simulation and real floating-point and INT8 compilation for both SoCs. Release builds reuse the verified candidate wheel. [Offline deployment guide](../deploy/OFFLINE.md)
