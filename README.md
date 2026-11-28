# Ascend ATC Offline Deployment Tools

[English](README.md) | [中文](README_zh.md)

Compile ONNX models to Ascend OM with INT8 calibration, on-device inference, and numerical comparison against CPU reference outputs.

The walkthrough below validates a floating-point baseline on 910B first, then quantizes and cross-compiles for 310 as needed. For full offline preparation, quantization details, alternative inference backends, and troubleshooting, see the [Deployment Guide](docs/guides/deployment-guide.md).

## 1. Preparation

### 1.1 Environment Setup

#### Verify the Environment

Confirm the target chip's **exact SoC model**, the board-side CANN version, and matching compiler toolchain and operator (`ops`) packages. If unsure, run the environment probe script:

```bash
bash scripts/probe_env.sh
```

#### Install the Package

Activate a Python 3.10+ virtual environment at the repository root and install in editable mode:

```bash
python3 -m pip install -e .
atc_tools --help
```

The default installation includes Python dependencies for ONNX inspection, CPU reference inference, direct image input, and the YOLO example. The package version is derived from Git tags and commits. [Dependencies](pyproject.toml)

The CLI entry point is `atc_tools`: `onnx` handles model inspection, CPU reference generation, and calibration preparation; `om` runs on-device inference and output comparison; `yolo` provides single-image detection.

The same wheel also provides the `atc_service` command. Running the service requires the `service` optional dependencies:

```bash
python3 -m pip install -e '.[service]'
atc_service --help
```

For service startup, Web Workbench, and HTTP API usage, see the [Service Guide](docs/guides/service-guide.md).

#### Install AMCT

INT8 quantization requires AMCT, which is not bundled with CANN by default. To download it on an internet-connected machine:

```bash
bash scripts/download_amct.sh <CANN_version>
```

Copy `work/amct/<CANN_version>/` to the same path on the target machine, activate a dedicated AMCT Python environment with its required dependencies, and install. [AMCT Dependencies](docs/guides/deployment-guide.md#downloading-and-installing-amct)

```bash
bash scripts/install_amct.sh <CANN_version>
```

### 1.2 Data Preparation

- Prepare the ONNX model (including external weights if any), validation inputs (`.npy`, `.bin`, `.jpg`, or `.png`), and an environment for generating CPU reference outputs.
- For offline deployment, prepare container images, models, Python wheelhouse, and required CANN packages in advance.
- For INT8 quantization, also prepare a representative calibration dataset.

## 2. Inspect the Model

### 2.1 Model Inspection and CPU Reference Outputs

```bash
# Inspect model inputs, outputs, and external weights.
atc_tools onnx inspect --model work/model.onnx

# Generate CPU reference outputs for comparison.
atc_tools onnx reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

Replace `images` with the actual input name reported by `onnx inspect`. For multi-input models, repeat `--input 'name=path'`. Both CPU reference and OM inference accept the same input file (e.g. `--input images=work/test.jpg`). JPG and PNG files are automatically preprocessed using YOLO rules; for custom preprocessing, provide `.npy` or `.bin` tensors matching the model shape and dtype. [Input Formats & Image Preprocessing](docs/guides/deployment-guide.md#preparing-model-inputs-and-cpu-reference-outputs)

## 3. Local Compilation and Verification

### 3.1 Floating-Point Model

#### Compile the Model

Replace the example SoC with your exact target model and use a fresh output directory:

```bash
SOC_VERSION=Ascend910B2
bash scripts/compile.sh --model work/model.onnx --soc "${SOC_VERSION}" \
  --output-dir "work/${SOC_VERSION}-float"
```

To explicitly select FP16 execution mode, pass `--precision_mode_v2=fp16` via `--atc-arg` and write to a separate output directory. [ATC Precision Modes](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0069.html)

```bash
bash scripts/compile.sh --model work/model.onnx --soc "${SOC_VERSION}" \
  --atc-arg=--precision_mode_v2=fp16 \
  --output-dir "work/${SOC_VERSION}-fp16"
```

OM input and output data types can also be customized:

- For FP16 inputs, append `--atc-arg=--input_fp16_nodes=images` (replace `images` with the actual ONNX input name) and supply `float16` `.npy` or `.bin` inputs for inference. [ATC Input Type](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/devaids/atctool/atlasatcparam_16_0043.html)
- For FP16 outputs, append `--atc-arg=--output_type=FP16`. [ATC Output Type](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0030.html)

#### Validate Model Outputs

Run the compiled OM on the target device and compare its outputs against the CPU reference:

```bash
# Run the OM and compare its outputs with the CPU reference.
bash scripts/run_model.sh --model work/"$SOC_VERSION"-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/"$SOC_VERSION"-float-outputs \
  --reference work/reference
```

To validate the FP16 build, point `--model` to `work/${SOC_VERSION}-fp16/model.om` and choose a new `--output-dir`.

The compilation directory records the OM, command, log, and exit code; the inference directory stores raw output tensors while comparison metrics print to the terminal. Default float tolerances are for debugging only—define acceptance thresholds based on your model and task. See the [Deployment Guide](docs/guides/deployment-guide.md#compiling-and-running-a-float-baseline-on-910b) for renamed I/O nodes, dynamic shapes, and tolerance tuning.

### 3.2 Quantized Model (INT8)

The steps below produce an INT8 OM from `work/model.onnx`. Single-input image models can use [AMCT-ONNX one-command quantization](#using-amct-onnx) directly; for AMCT-ACL or pre-generated tensors, prepare the calibration data first.

#### Prepare Calibration Data

Select samples representative of production traffic and preprocess them according to model requirements. Place each batch in its own directory as `input_0.npy`, `input_1.npy`, etc. in ONNX input order, matching the original shape and dtype. Two batches are shown below; increase as needed and keep an independent validation set.

```bash
atc_tools onnx calibration --model work/model.onnx \
  --sample work/samples/0001 --sample work/samples/0002 \
  --output-dir work/calibration
```

This validates the tensors and writes per-batch `.bin` files, concatenated per-input `.bin` files, and `work/calibration/calibration.json`. [Calibration Data Format](docs/guides/deployment-guide.md#amct-onnx-pre-calibration)

#### Using AMCT-ACL

ATC invokes AMCT-ACL inline during compilation to calibrate and emit the OM. Use a CANN environment that supports inline calibration with matching AMCT-ACL installed and a visible calibration NPU. [ATC Compression Config](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0059.html)

```bash
cp configs/atc-compression.cfg work/atc-compression.cfg
```

Edit the `calibration` section in `work/atc-compression.cfg`:

| Field | Value |
| --- | --- |
| `input_data_dir` | `input_data_dir` from `calibration.json`, e.g. `work/calibration/input_0.bin`. |
| `input_shape` | `atc_input_shape` from `calibration.json` (batch dimension includes all concatenated samples). |
| `infer_soc`, `infer_device_id` | Exact SoC model and logical device ID for calibration, e.g. `Ascend910B2` and `0`. |

```bash
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_VERSION" \
  --compression-optimize-conf work/atc-compression.cfg \
  --output-dir "work/${SOC_VERSION}-int8-acl"
```

#### Using AMCT-ONNX

Activate the dedicated Python environment for AMCT-ONNX and verify `amct_onnx calibration --help` works. [AMCT-ONNX Environment](docs/guides/deployment-guide.md#downloading-and-installing-amct)

**One-command quantization from an image directory.** Place `.jpg`, `.jpeg`, or `.png` images flat inside a directory. The command reads input metadata from the ONNX model, preprocesses images in filename order, and invokes AMCT-ONNX:

```bash
/path/to/atc-tools-venv/bin/atc_tools onnx quantize --model work/model.onnx \
  --dataset work/images --output-dir work/quantized
```

Replace `/path/to/atc-tools-venv` with the virtual environment where `atc_tools` is installed; it invokes `amct_onnx` from the active `PATH`.

This shortcut supports fixed-size, batch-1, single-input RGB models (NCHW or NHWC) using YOLO letterbox preprocessing (pad with `114`, RGB, float inputs divided by `255`). For custom preprocessing, generate tensors first and use the tensor workflow below.

All images are used by default; pass `--batch-num 50` to limit to the first 50 images, or pass `--config work/amct-onnx.cfg` to customize skipped layers, algorithms, and `batch_num` (these two flags are mutually exclusive). Omitting `--output-dir` writes to `<model-name>-int8/` next to the model. [AMCT-ONNX Config Reference](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0168.html)

Each calibration pass runs with batch size 1. Start with `--batch-num 16` to check memory usage and validation accuracy before scaling up. If the process is killed, inspect kernel OOM logs and container/job memory limits. [Quantization Memory Troubleshooting](docs/guides/deployment-guide.md#amct-onnx-process-killed)

**Quantization from preprocessed tensors.** Copy the template config and set `batch_num` to the number of calibration batches (`2` in this example):

```bash
cp configs/amct-onnx.cfg work/amct-onnx.cfg
```

Fill in `input_shape`, `data_types`, and `data_dir` from `calibration.json` (example below uses a single FP32 input `images` with shape `[1,3,640,640]`):

```bash
bash scripts/quantize_model.sh --model work/model.onnx \
  --input-shape 'images:1,3,640,640' --data-types float32 \
  --data-dir work/calibration/input_0 --config work/amct-onnx.cfg \
  --output-dir work/quantized
```

Both flows output `model_deploy_model.onnx` (for ATC compilation) and `model_fake_quant_model.onnx` (for accuracy simulation). One-command quantization saves AMCT logs to `amct-onnx.log` in the output directory.

```bash
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_VERSION" --output-dir "work/${SOC_VERSION}-int8-onnx"
```

#### Validate Quantized Outputs

Run the quantized OM on the same device using the validation input from Section 3.1 (replace the model path when validating an AMCT-ACL build):

```bash
bash scripts/run_model.sh --model "work/${SOC_VERSION}-int8-onnx/model.om" \
  --device 0 --input images=work/input.npy \
  --output-dir "work/${SOC_VERSION}-int8-outputs" \
  --reference "work/${SOC_VERSION}-float-outputs"
```

Tensor diffs help diagnose quantization drift, while final acceptance should compare task-level metrics (such as mAP for object detection) on an independent validation set with identical pre/post-processing. [Quantization Acceptance](docs/guides/deployment-guide.md#compiling-for-310-and-validating-on-the-target)

## 4. Cross-Compiling for 310

After validating on 910B, you can cross-compile OM files for a 310 target on a Linux build host—standard ONNX-to-OM ATC conversion does not require the target NPU. Confirm the board's exact SoC model and CANN version, select a compatible Toolkit, and install the matching `ops` package for CANN 8.5.0+. [ATC `--soc_version` Reference](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0036.html)

**Build on the compile host.** Replace `Ascend310B4` with your exact target SoC. Use `--cann-env` to pick a specific `set_env.sh` when multiple CANN versions are installed:

```bash
TARGET_SOC=Ascend310B4
bash scripts/compile.sh --model work/model.onnx --soc "$TARGET_SOC" \
  --output-dir "work/${TARGET_SOC}-float"
```

For INT8, compile `work/quantized/model_deploy_model.onnx` from Section 3.2 with the same target `--soc` and a separate output directory. If the quantized ONNX is incompatible with the target CANN version, re-quantize in a compatible AMCT environment. [310 Deployment Guide](docs/guides/deployment-guide.md#compiling-for-310-and-validating-on-the-target)

**Accept on the target board.** Copy `model.om`, validation inputs, and CPU reference outputs to the 310 device, run inference as in Section 3.1, and compare float vs. INT8 accuracy and latency on the same validation set. [Inference & Comparison](#validate-model-outputs) · [Quantization Acceptance](#validate-quantized-outputs)

## 5. Service Deployment and Usage

In addition to the `atc_tools` CLI, this repository provides a containerized `atc_service` offering a Web Workbench and HTTP API for ONNX inspection, OM compilation, INT8 quantization, inference, accuracy comparison, and `msprof` profiling. For full configuration, real Ascend setup, and smoke tests, see the [Service Guide](docs/guides/service-guide.md); for offline bundle import and upgrades, see the [Offline Deployment Guide](deploy/README.md).

### 5.1 Deploy and Start/Stop the Service

Copy [deploy/lab.env.example](deploy/lab.env.example) to `lab.env` and set `ATC_APP_IMAGE`, `ATC_HOST_DATA_DIR`, and `SSH_CONFIG_DIR` (a root-owned directory containing a nonempty `authorized_keys`).

- **Local simulation mode**: Set `ATC_SIMULATE_COMPILE=1` in `lab.env` (no Ascend hardware required); set `ATC_BROWSER_SECURE_COOKIES=0` for local loopback HTTP testing.
- **Real Ascend environment**: Keep `ATC_SIMULATE_COMPILE=0`, set `ATC_AMCT_TARGETS` for validated INT8 targets, and set `ATC_RUNTIME=ascend` plus `ATC_910B2_DEVICE` to enable 910B2 inference and profiling. [Enabling Real Ascend Toolchains](docs/guides/service-guide.md#enabling-real-ascend-toolchains)

```bash
# Start the service (--wait blocks until /readyz healthcheck passes)
docker compose --env-file lab.env -f deploy/compose.yaml up -d --no-build --pull never --wait

# Stop the service (preserves data directory)
docker compose --env-file lab.env -f deploy/compose.yaml down
```

### 5.2 Using the Web Workbench

Provision or reset a browser account password first (API tokens are not required when using the Web Workbench only):

```bash
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service set-browser-password alice
```

Open `http://127.0.0.1:8766/` and sign in. For lab or intranet deployments, serve behind an HTTPS reverse proxy and keep `ATC_BROWSER_SECURE_COOKIES=1`.

1. Upload an ONNX model and wait for automatic inspection to finish.
2. Choose the target SoC (e.g. `310B4`, `910B2`) and precision preset (`original`, `mixed Float16`, `preferred FP16`, `INT8`).
3. Upload datasets as needed: calibration data for `INT8`, and validation data when accuracy comparison or `msprof` profiling is enabled on `910B2` (supports images, `.npy`, and `.bin`; up to 128 samples per dataset).
4. Submit the job, follow phased logs and status, and download artifacts (OM, `accuracy.json`, `profiling.zip`). Failed optional verification steps can be retried directly.

Browser sessions expire after 12 hours by default; sign out after downloading artifacts so session files can be cleaned up promptly. [Web Workbench Details](docs/guides/service-guide.md#using-the-web-workbench)

### 5.3 Submitting Jobs via HTTP API

Issue a Bearer Token before calling the HTTP API (tokens can also be revoked interactively):

```bash
# Issue an API Bearer Token
docker compose --env-file lab.env -f deploy/compose.yaml exec -T app atc_service issue-token alice

# Revoke a token (interactive selection)
docker compose --env-file lab.env -f deploy/compose.yaml exec app atc_service revoke-token
```

Include `Authorization: Bearer <token>` on all API requests. Pass a stable `Idempotency-Key` header to safely recover interrupted submissions. Full schemas are served at `GET /openapi.json`. Submit an `inspect_onnx` job after uploading a model before starting workflows or standalone operations.

| Step | Endpoint | Description |
| --- | --- | --- |
| Discover capabilities | `GET /v1/capabilities` | List available presets, `workflow_presets`, and `standalone_operations` |
| Upload model | `POST /v1/assets?filename=model.onnx` | Upload model (`application/octet-stream`) and obtain `input_id` |
| Upload dataset | `POST /v1/dataset-files?filename=sample.npy` | Upload calibration or validation sample files as needed |
| Submit workflow | `POST /v1/workflows` | Run end-to-end conversion, quantization, and verification from an inspected `model_id` |
| Submit operation | `POST /v1/operations` | Run a single step (`quantize_onnx`, `compile_om`, `run_cpu`, `run_npu`, `compare_outputs`, `profile_om`) |
| Poll & logs | `GET /v1/jobs/{job_id}` / `GET /v1/jobs/{job_id}/logs` | Poll job status and artifact IDs, or stream logs incrementally via `next_offset` |
| Retry & download | `POST /v1/jobs/{job_id}/retry` / `GET /v1/assets/{asset_id}/download` | Retry `compare` or `profile` stages, or download output artifacts |

For request payloads and step-by-step examples, see [Using the HTTP API](docs/guides/service-guide.md#using-the-http-api).

## 6. More

- [Service Guide](docs/guides/service-guide.md) & [Offline Deployment Guide](deploy/README.md)
- [YOLO Single-Image Example](docs/guides/deployment-guide.md#yolo-single-image-example)
- [ais_bench Inference](docs/guides/deployment-guide.md#running-om-with-the-ais_bench-python-api)
- [Troubleshooting](docs/guides/deployment-guide.md#troubleshooting) & [Local Development Checks](docs/guides/deployment-guide.md#local-development-checks)
