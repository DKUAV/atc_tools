# ATC Tools Service Guide

[English](service-guide.md) | [中文](service-guide_zh.md)

ATC Tools Service provides a Web Workbench and a Bearer-Token HTTP API for ONNX model inspection, OM compilation, INT8 quantization, inference, accuracy comparison, and `msprof` profiling.

> **New here?** Start with [Quick Start (Simulation Mode)](#quick-start-simulation-mode) to try the full workflow locally without Ascend hardware.  
> For service architecture, ownership, and execution boundaries, see [Service Architecture](../design/service-design.md).

## Contents

- [Quick Start (Simulation Mode)](#quick-start-simulation-mode)
- [Using the Web Workbench](#using-the-web-workbench)
- [Enabling Real Ascend Toolchains](#enabling-real-ascend-toolchains)
- [Using the HTTP API](#using-the-http-api)
- [Running Smoke Tests](#running-smoke-tests)

---

## Quick Start (Simulation Mode)

Copy `deploy/lab.env.example` to `lab.env` and set `ATC_APP_IMAGE` (a released Linux ARM64 image tag), `ATC_HOST_DATA_DIR`, and `SSH_CONFIG_DIR` (a root-owned directory containing a nonempty `authorized_keys`). For offline image import and all `compose.yaml` variables, see the [Offline Deployment Guide](../../deploy/README.md).

To run locally without Ascend hardware, set the following in `lab.env`:

- `ATC_SIMULATE_COMPILE=1` — enables the hardware-free simulator;
- `ATC_BROWSER_SECURE_COOKIES=0` — allows browser sign-in over local loopback HTTP (`127.0.0.1` only).

Start and stop the service from the repository root:

```bash
# Start the service and wait for /readyz healthcheck
docker compose --env-file lab.env -f deploy/compose.yaml up -d --no-build --pull never --wait

# Stop the service (keeps SQLite data and stored files)
docker compose --env-file lab.env -f deploy/compose.yaml down
```

---

## Using the Web Workbench

### 1. Create a Browser Account

Set or reset a password for a local account (updating a password invalidates all existing browser sessions for that user; API tokens are not required for the Web Workbench):

```bash
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service set-browser-password alice
```

Open `http://127.0.0.1:8766/` in your browser and sign in.

> **Security notes**:
> - **Local HTTP testing**: Set `ATC_BROWSER_SECURE_COOKIES=0` before starting Compose and bind only to `127.0.0.1`.
> - **Lab / intranet deployment**: Serve behind an HTTPS reverse proxy with `ATC_BROWSER_SECURE_COOKIES=1` (default). Keep the Compose API port bound to loopback when the proxy runs on the same host, and configure rate limits and body-size limits on the login endpoint at the ingress layer.

### 2. Try with a Minimal Demo Model

In simulation mode (`ATC_SIMULATE_COMPILE=1`), both `310B4` and `910B2` support `original`, `mixed Float16`, `preferred FP16`, and `INT8` presets:

- Selecting **INT8** reveals the calibration dataset upload field.
- On **910B2**, enabling accuracy comparison or `msprof` profiling reveals a separate validation dataset upload field (sharing dimension and numeric settings with calibration).
- The simulator runs real ONNX inspection, dataset preparation, and CPU reference inference, while clearly labeling simulated artifacts as `model.simulated`, `accuracy.json`, and `profiling.zip`.

Generate a minimal fixed-shape model (`work/demo.onnx`) and sample tensor (`work/demo.npy`) from the repository root:

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

Upload `work/demo.onnx` in the workbench and select `work/demo.npy` whenever calibration or validation data is requested.

### 3. Dataset Format Rules

- **Supported formats**: Images (decoded to RGB or grayscale), NumPy arrays (`.npy`), and raw binaries (`.bin`).
- **Layout and dtype**: `.npy` and `.bin` follow the selected model layout; raw `.bin` files default to the model input shape and little-endian dtype.
- **Multi-input models**: Provide all inputs in model order for each sample using either:
  - Sibling subdirectories `input_0/`, `input_1/`, ... with matching sample filenames; or
  - Flat files named `sample.input_0.npy`, `sample.input_1.npy`, etc.
- **Sample limit**: Up to **128 samples** per dataset.

### 4. Sessions and File Cleanup

- Browser sessions expire after **12 hours** by default (configurable via `ATC_BROWSER_SESSION_SECONDS`).
- Sign out after downloading your artifacts. Logging out or letting a session expire immediately revokes access to session files, and the worker deletes them once pending jobs finish.
- Job history and logs remain visible on your next sign-in.

For UI specifications, see [Web UI Requirements](../design/web-ui-requirements.md) and [Web UI Design](../design/web-ui-design.md).

---

## Enabling Real Ascend Toolchains

The application image ships with CANN pre-installed. To switch from simulation to real Ascend execution, update `lab.env`:

- Set `ATC_SIMULATE_COMPILE=0` and configure `ATC_AMCT_TARGETS` with your validated INT8 targets (e.g. `310B4,910B2`).
- Set `ATC_RUNTIME=ascend` and `ATC_910B2_DEVICE=<device_id>` to enable 910B2 inference, on-device accuracy comparison, and `msprof` profiling.
- Reserve exclusive access to the target NPU during profiling windows, including SSH sessions and other containers.

### Precision, Quantization, and Profiling Behavior

- **Floating-point presets**: Mapped to ATC's `--precision_mode_v2` flag ([ATC Precision Modes](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/atctool/atlasatcparam_16_0069.html)).
- **INT8 quantization**: Uses the AMCT default activation algorithm with the batch count derived from the uploaded calibration dataset ([AMCT Configuration](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/amct/atlasamct_16_0163.html)).
- **Accuracy comparison**: Compares against CPU ONNX reference outputs; for INT8 builds, also compares against an original-precision OM baseline on the same device.
- **Profiling (`msprof`)**: Runs the primary OM with the shared validation dataset and archives the collected data ([CANN 8.5 Runtime Profiling](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/850/devaids/profiling/atlasprofiling_16_0011.html)). Inspect `profiling.json` inside the archive for input count, profiler wall-clock duration, and collection size; use the native `msprof` files for device-level timing.

Inference, on-device accuracy comparison, and `msprof` share a single serialized device slot. The `/readyz` readiness endpoint verifies SQLite, scheduler heartbeat, and enabled toolchains.

---

## Using the HTTP API

### 1. Issue or Revoke a Bearer Token

```bash
# Issue an API Bearer Token
docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token alice

# Revoke a token interactively
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service revoke-token
```

Include `Authorization: Bearer <token>` on every API request. Pass a stable `Idempotency-Key` header so interrupted submissions can be safely recovered. Full typed schemas are available at `GET /openapi.json`.

### 2. Typical Workflow

1. **Discover capabilities**: `GET /v1/capabilities` to inspect legacy presets, `workflow_presets`, and `standalone_operations`.
2. **Upload a model**: `POST /v1/assets?filename=model.onnx` (`Content-Type: application/octet-stream`) → returns `input_id`.
3. **Upload dataset files** *(optional)*: `POST /v1/dataset-files?filename=sample.npy` (`application/octet-stream`), then group asset IDs as per-batch lists in model input order.
4. **Inspect and run**:
   - Submit an `inspect_onnx` job with `input_id` first.
   - Once inspection succeeds, submit a full workflow (`POST /v1/workflows`), a standalone operation (`POST /v1/operations`), or a legacy `compile_om` job.
5. **Track and download**:
   - List recent jobs: `GET /v1/jobs`
   - Poll status and artifact IDs: `GET /v1/jobs/{job_id}`
   - Stream logs incrementally: `GET /v1/jobs/{job_id}/logs` (pass the previous `next_offset`)
   - Download artifacts: `GET /v1/assets/{asset_id}/download`

### 3. Workflow and Retry Endpoints

| Endpoint | Request fields & behavior |
| --- | --- |
| `POST /v1/workflows` | `model_id`, `preset`, plus optional `calibration`, `validation`, `compare`, `profile`, `shapes`, and `preparation`. Starts a full conversion task. |
| `POST /v1/operations` | `operation`, `input_id`, and operation-specific fields (see below). |
| `POST /v1/jobs/{job_id}/retry` | `step` (`compare` or `profile`) and optional replacement `validation`. Creates a child task linked to the original OM. |

> In the Web Workbench, if an optional verification stage fails, a retry action is offered while keeping the original OM downloadable; replacement validation files can also be uploaded during retry.

### 4. Standalone Operations (`POST /v1/operations`)

| `operation` | `input_id` | Additional fields & output |
| --- | --- | --- |
| `quantize_onnx` | Inspected source ONNX | INT8 `preset` and `samples`. Produces a deployment ONNX. |
| `compile_om` | Inspected ONNX | `preset` (for standalone INT8, pass the inspected deployment ONNX from quantization). Produces OM and build metadata. |
| `run_cpu` | Inspected ONNX | `samples`. Produces `tensors.zip` with CPU reference outputs. |
| `run_npu` | Inspected source ONNX | Enabled 910B2 `preset`, `primary_id` (910B2 OM for that model), and `samples`. Produces `tensors.zip`. |
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

All three smoke test suites target a local simulation deployment (`ATC_SIMULATE_COMPILE=1`).

### 1. API and Delivery Smoke Tests

Issue two test tokens and run `smoke.py` (ONNX upload, multi-account isolation, failure logs, and simulated downloads) followed by `delivery_smoke.py` (standalone operations, phased logs, and retries):

```bash
export ATC_SMOKE_TOKEN_A="$(docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token alice)"
export ATC_SMOKE_TOKEN_B="$(docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token bob)"

uv run --locked python tests/service/smoke.py
uv run --locked python tests/service/delivery_smoke.py

unset ATC_SMOKE_TOKEN_A ATC_SMOKE_TOKEN_B
```

> The simulator returns `model.simulated` and marks substituted NPU outputs. Real ATC, AMCT, and NPU behavior is validated during lab acceptance (see [Offline Deployment Guide](../../deploy/README.md)).

### 2. Web Workbench Smoke Test

With simulation mode and local HTTP (`ATC_BROWSER_SECURE_COOKIES=0`) enabled, provision two accounts via `set-browser-password` and run `browser_smoke.py` to test browser endpoints, mixed-format datasets, optional stage failures, account isolation, and file cleanup:

```bash
export ATC_SMOKE_USER_A="alice"
export ATC_SMOKE_PASSWORD_A="<alice-password>"
export ATC_SMOKE_USER_B="bob"
export ATC_SMOKE_PASSWORD_B="<bob-password>"
# Optional: verify physical file removal in the host data directory
export ATC_SMOKE_DATA_DIR="$PWD/work/service-data"

uv run --locked python tests/service/browser_smoke.py
```
