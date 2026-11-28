# Ascend ATC Offline Deployment Guide

[English](deployment-guide.md) | [中文](deployment-guide_zh.md)

This guide covers offline environment preparation, ONNX model inspection and CPU reference generation, 910B floating-point baseline validation, INT8 quantization (AMCT-ONNX / AMCT-ACL), 310 cross-compilation and acceptance, the YOLO example, and troubleshooting.

> The current inference runner supports **single-device, fixed-shape, contiguous** tensors. Multi-input models are supported by repeating `--input`. The YOLO example is an optional, replaceable vision reference.

## Contents

- [Before Entering the Lab](#before-entering-the-lab)
- [Preparing Model Inputs and CPU Reference Outputs](#preparing-model-inputs-and-cpu-reference-outputs)
- [Compiling and Running a Float Baseline on 910B](#compiling-and-running-a-float-baseline-on-910b)
- [INT8 Quantization](#int8-quantization)
- [Compiling for 310 and Validating on the Target](#compiling-for-310-and-validating-on-the-target)
- [YOLO Single-Image Example](#yolo-single-image-example)
- [Troubleshooting](#troubleshooting)
- [Local Development Checks](#local-development-checks)

---

## Before Entering the Lab

### Preparation Checklist

- **Models and data**: Prepare the ONNX model and any external weight files, validation inputs, and CPU reference outputs. For INT8 quantization, also prepare a representative calibration dataset, native quantization config files, or a pre-exported quantized deployment ONNX.
- **Hardware and CANN versions**: Confirm the **exact SoC model** for both 910B and the target 310 board, plus the board-side CANN version. For example, `Ascend310`, `Ascend310P3`, and `Ascend310B4` are distinct targets requiring matching CANN toolchains and operator (`ops`) packages.
- **Container images**: For 910B validation, you can reuse `2.10.0.post7-cann9.2.0-910b-ubuntu22.04-py3.12` (verify it exists locally with `docker image inspect` and mount devices/tools per lab rules). Prepare separate images or CANN packages for 310 compilation.
- **Shell and Python requirements**:
  - All shell entry points use Bash and enhanced util-linux `getopt`. Invoke them via `bash scripts/xxx.sh` from either Bash or Zsh. Every script supports `-h` / `--help`; scripts accepting `--cann-env` can load a specific `set_env.sh`.
  - Python tools require **Python 3.10+**. Direct JPG/PNG input and the YOLO example require OpenCV. OM inference requires vendor `pyACL` or `ais_bench` matching the board CANN and Python ABI. AMCT uses its own dependency matrix.

### Offline Python Wheelhouse

On an internet-connected development host, install in editable mode with `uv sync`. For air-gapped targets, build a wheelhouse on a connected Linux host with the **same CPU architecture and Python version** (including `hatchling` and `hatch-vcs` for editable builds):

```bash
mkdir -p work
uv export --no-dev --no-emit-project \
  --format requirements-txt --output-file work/requirements-tools.txt
python3 -m pip download --only-binary=:all: -r work/requirements-tools.txt -d work/wheelhouse
python3 -m pip download --only-binary=:all: hatchling hatch-vcs -d work/wheelhouse
```

Copy the repository and `work/wheelhouse` to the offline target and install from the repository root:

```bash
python3 -m pip install --no-index --find-links work/wheelhouse -e .
atc_tools --version
```

> The package version is derived from Git tags and commits during installation. Keep `.git` when copying the workspace offline (or use a pre-built source distribution) to preserve version metadata; otherwise it falls back to `0.0.0`. Install your chosen OM inference backend (`pyACL` or `ais_bench`) separately.

### Transfer Verification and Environment Probe

Generate a SHA-256 manifest before transferring files, then verify integrity on the target with `sha256sum -c SHA256SUMS`:

```bash
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
```

Run the environment probe and save its output before compiling or running models (use `--python` to point to a specific interpreter):

```bash
bash scripts/probe_env.sh |& tee work/environment.log
```

**CANN environment resolution order** (shared by probe, compile, quantize, and run scripts):
1. Explicit `--cann-env /path/to/set_env.sh`;
2. Existing CANN environment variables already loaded in the shell;
3. Automatic fallback to `/usr/local/Ascend/cann/set_env.sh`, then legacy `/usr/local/Ascend/ascend-toolkit/set_env.sh`.

The probe reports available tools, environment variables, NPU devices, and Python modules. Compile hosts mainly need ATC and `ops` packages; inference hosts also require visible devices and a working `import acl`.

---

## Preparing Model Inputs and CPU Reference Outputs

Run these commands from the repository root. The example assumes a single input named `images`; replace with the actual names and shapes reported by inspection:

```bash
atc_tools onnx inspect --model work/model.onnx
atc_tools onnx reference --model work/model.onnx \
  --input images=work/input.npy --output-dir work/reference
```

`onnx inspect` prints opset versions and domains, inputs/outputs, dynamic dimensions, and external weight status. For both CPU reference and OM inference, pass `--input 'name=path'` once per input. Multi-input models can mix the following formats:

| Format | Requirements |
| --- | --- |
| `.npy` | Preprocessed NumPy tensor whose shape and dtype match the model input. |
| `.bin` | Raw C-contiguous tensor bytes matching model dtype and host endianness. File size must equal element count × element byte size (e.g. `array.tofile("input.bin")`). |
| `.jpg`, `.jpeg`, `.png` | Fixed-shape single-image RGB input with clear NCHW or NHWC layout. Automatically letterbox-resized with `114` padding; float inputs are divided by `255`, while `uint8` keeps raw pixel values (`opencv-python-headless` is included by default). |

For custom image preprocessing, generate `.npy` or `.bin` files beforehand. Reference outputs are written as `output_0.npy`, etc. alongside `outputs.json` recording names, shapes, and dtypes.

---

## Compiling and Running a Float Baseline on 910B

If Ascend Docker Runtime is configured, launch the local container image from the repository root (adjust the image tag and `ASCEND_VISIBLE_DEVICES` for your host):

```bash
IMAGE=ghcr.io/dkuav/luciole-ascend:2.10.0.post7-cann9.2.0-910b-ubuntu22.04-py3.12
docker image inspect "$IMAGE" >/dev/null
docker run --pull=never --rm -it --runtime=ascend -e ASCEND_VISIBLE_DEVICES=0 \
  -v "$PWD:/work" -w /work --entrypoint /bin/bash "$IMAGE" -l
```

Inside the container, run `bash scripts/probe_env.sh` first, then compile and validate:

```bash
SOC_910B=Ascend910B1  # Replace with the exact device SoC
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --output-dir work/910b-float
bash scripts/run_model.sh --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-float-outputs \
  --reference work/reference
```

**Options and artifacts**:
- **Compilation**: `compile.sh` passes `--framework=5` (ONNX). Use `--input-shape 'images:1,3,640,640'` for shape overrides and repeat `--atc-arg=--flag=value` for extra native ATC arguments. Always use a fresh output directory; it stores `exit-code.txt`, `tool.log`, and a replayable `command.txt`. Shell wrappers preserve spaces in paths, though ATC/AMCT and their config files still impose vendor character restrictions—prefer ASCII paths without spaces.
- **Inference and node mapping**: `run_model.sh` saves raw OM output tensors. When `--reference` is provided, tensors are matched by name; use `--output-map 'onnx_name=om_name'` if ATC renamed an output. If an input name changed, check the runner's diff message and pass the same file under the OM input name.
- **Comparison rules**: Floating-point tensors are checked with `abs(actual - ref) <= atol + rtol * abs(ref)`. Default `--atol 0.001 --rtol 0.01` is for debugging only—set acceptance tolerances according to task requirements. Mixed float widths are allowed (both dtypes are printed); integer and boolean outputs require exact equality. Mismatches return a non-zero exit code, and saved outputs can be re-compared independently.

### Running OM with the ais_bench Python API

As an alternative to `pyACL`, you can execute OM models using Ascend `ais_bench` and matching `aclruntime`. For offline hosts, prepare wheels matching the CANN version, CPU architecture, and Python ABI (e.g. `cp312` + `linux_aarch64`), and source the CANN `set_env.sh` beforehand. [Official Installation & Wheels](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/README.md)

```bash
atc_tools om run-ais-bench --model work/910b-float/model.om --device 0 \
  --input images=work/input.npy --output-dir work/910b-ais-bench-outputs
atc_tools om compare --reference work/reference \
  --actual work/910b-ais-bench-outputs
```

This runs one fixed-shape inference (repeat `--input` for multi-input models; inputs are passed to `InferSession` in OM input order) and writes `output_0.npy` + `outputs.json` compatible with CPU reference or `pyACL` comparisons. For warm-up, loop benchmarks, and throughput stats, use `python3 -m ais_bench` directly. [Official Python API](https://gitee.com/ascend/tools/blob/master/ais-bench_workload/tool/ais_bench/API_GUIDE.md)

---

## INT8 Quantization

### Downloading and Installing AMCT

Both scripts accept a CANN version argument. The download script fetches ARM64 and x86_64 archives containing `amct_onnx` and `amct_acl` from Ascend community repositories, along with custom-op headers for ONNX Runtime 1.16.0 and 1.20.0 (verified for `8.5.0`, `9.0.0`, `9.1.0`, and `9.2.0`):

```bash
# On an internet-connected machine
bash scripts/download_amct.sh 9.2.0

# On the target Linux machine, after copying work/amct/9.2.0/
source /path/to/amct-venv/bin/activate
bash scripts/install_amct.sh 9.2.0
```

The target machine must have the matching CANN Toolkit, a C++ compiler, and Python development headers installed, and the installer must run as a user with write permissions to the Toolkit directory. Create a dedicated Python virtual environment with `pip`, `setuptools`, and the dependencies below (pre-download wheels with `pip download` for offline use). [Huawei AMCT-ONNX Compatibility Matrix](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0014.html)

| Python Version | Recommended Dependencies |
| --- | --- |
| Python 3.10 / 3.11 | `onnx==1.14.0`, `onnxruntime==1.16.0`, `numpy==1.24.4`, `protobuf==3.20.3` |
| Python 3.12 | `onnx==1.18.0`, `onnxruntime==1.20.0`, `numpy>=2.0.0`, `protobuf==3.20.3` |

`install_amct.sh` selects the matching architecture archive from `work/amct/<version>/`, installs `amct_onnx` and compiles custom ops in the active Python environment, and installs `amct_acl` into the loaded Toolkit directory—fully offline. [Huawei AMCT Installation Guide](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/amct/atlasamct_16_0025.html)

> The installer reuses the active CANN environment (or auto-loads the default path) and checks the Toolkit version. Source the target `set_env.sh` first when installing into a non-default CANN location.

### AMCT-ONNX Pre-Calibration

Verify `amct_onnx calibration --help` and a sample run on a connected host first. If you need to recalibrate offline, bring the validated AMCT environment (custom-op libraries must strictly match the ONNX Runtime version).

**1. Prepare calibration data and config** (in the `atc_tools` environment):

Choose representative production samples. Place each calibration batch in a separate directory as `input_0.npy`, `input_1.npy`, etc. in ONNX input order (consistent shape, dtype, and preprocessing across all batches), and keep a separate validation set:

```bash
atc_tools onnx calibration --model work/model.onnx \
  --sample work/samples/0001 --sample work/samples/0002 --output-dir work/calibration
cp configs/amct-onnx.cfg work/amct-onnx.cfg
```

Set `batch_num` in `work/amct-onnx.cfg` to the number of calibration batches. Add `skip_layers: "node_name"` for sensitive layers that should stay in floating point. For custom algorithms or per-layer overrides, consult `calibration_config_onnx.proto` for your AMCT version and archive the final config and logs.

**2. Quantize and compile OM** (in the AMCT environment):

```bash
bash scripts/quantize_model.sh --model work/model.onnx \
  --input-shape 'images:1,3,640,640' --data-types float32 \
  --data-dir work/calibration/input_0 --config work/amct-onnx.cfg \
  --output-dir work/quantized
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_910B" --output-dir work/910b-int8
```

> `work/calibration/calibration.json` lists the exact `input_shape`, `data_types`, and `data_dir` values. For multi-input models, preserve manifest order and quote semicolon-delimited values in the shell. Quantization emits two models:
> - `model_deploy_model.onnx`: input for ATC INT8 compilation;
> - `model_fake_quant_model.onnx`: for CPU accuracy simulation (pass `--custom-ops-library /path/to/library.so` to `atc_tools onnx reference` with matching ONNX Runtime if custom ops are required).

### ATC Inline Calibration

In a CANN environment supporting inline calibration, verify `amct_acl` is installed and the calibration NPU is visible. Copy [the ATC compression template](../../configs/atc-compression.cfg), fill in `input_data_dir` and `atc_input_shape` from `calibration.json` (the batch dimension includes all concatenated batches), and set `infer_soc` and `infer_device_id` to the calibration device; the target OM SoC is still controlled by `--soc`:

```bash
cp configs/atc-compression.cfg work/atc-compression.cfg
# Edit work/atc-compression.cfg with calibration paths, shape, and device info
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_910B" \
  --compression-optimize-conf work/atc-compression.cfg --output-dir work/910b-int8-inline
```

For per-layer quantization overrides, add `config_file` inside `calibration` pointing to an AMCT-ACL simplified config for your CANN version. Note that AMCT-ACL and AMCT-ONNX use different config schemas—verify sample layout and precision constraints against your CANN release docs. [ATC Compression Config](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0059.html) · [AMCT-ONNX Environment](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/920beta2/devaids/amct/atlasamct_16_0014.html)

---

## Compiling for 310 and Validating on the Target

Switch to a build environment matching the target chip and board-side CANN version. Replace `Ascend310B4` with your exact board SoC:

```bash
SOC_310=Ascend310B4
bash scripts/compile.sh --model work/model.onnx --soc "$SOC_310" \
  --output-dir work/310-float
bash scripts/compile.sh --model work/quantized/model_deploy_model.onnx \
  --soc "$SOC_310" --output-dir work/310-int8
```

Transfer both OM files, the same validation inputs, and CPU reference outputs to the target 310 host, and run `run_model.sh` as in the 910B section. Build outputs and `exit-code.txt` verify compilation; target inference logs and tensors verify deployment. If the quantized ONNX is incompatible with the target CANN, re-quantize using a compatible AMCT environment.

**Quantization accuracy and latency acceptance**:
- **Accuracy**: Compare float OM vs. INT8 OM tensors on the same target board first. For vision tasks like YOLO, evaluate both OMs on an independent labeled validation set with identical pre/post-processing, record mAP and delta, and decide whether sensitive layers should be skipped.
- **Latency**: Pass the same `--warmup 10 --repeat 50` to both OMs and compare median `acl.mdl.execute` times (measures synchronous model execution only, excluding pre/post-processing).

---

## YOLO Single-Image Example

Install OpenCV, `ais_bench`, and `aclruntime`, and source the CANN environment. The command reads input dimensions and dtype from the OM and supports a single fixed-shape RGB input with one detection output head. [ais_bench Setup](#running-om-with-the-ais_bench-python-api)

```bash
atc_tools yolo infer --model work/910b-float/model.om \
  --image work/test.jpg --device 0 --layout v8 --output-dir work/detections
```

Select `--layout` according to the model output tensor:
- `v5`: `[1, N, 5+C]` (with objectness; applies per-class NMS)
- `v8`: `[1, 4+C, N]` (applies per-class NMS)
- `xyxy`: `[1, N, 6]` (box coordinates, confidence, class ID directly)

Results are written to `detections.json` and `annotated.jpg` with boxes mapped back to original image coordinates.

To generate CPU reference inputs or post-process existing prediction tensors separately:

```bash
atc_tools yolo preprocess --image work/test.jpg \
  --width 640 --height 640 --output work/input.npy
# After OM inference, check outputs.json to identify the prediction tensor
atc_tools yolo postprocess --image work/test.jpg \
  --width 640 --height 640 --predictions work/910b-float-outputs/output_0.npy \
  --layout v8 --output-dir work/detections
```

`yolo preprocess` applies letterbox resizing with `114` padding, converts to RGB, scales float inputs by `1/255`, and writes an NCHW tensor (`float32` by default; supports `--dtype float16`). Confirm this matches your model's training preprocessing.

---

## Troubleshooting

| Symptom | What to Check |
| --- | --- |
| ATC reports missing operator or unsupported SoC | Verify exact `soc_version`, CANN Toolkit and `ops` package versions, ONNX opset, and detailed errors in `tool.log`. |
| ATC works, but `pyACL` import or device selection fails | Check Python ABI compatibility, CANN runtime library paths, logical device IDs, and container NPU device mounts. |
| Calibration data size mismatch | Verify multi-input order, dtype, per-batch shape, and batch count; ATC inline calibration requires the concatenated `atc_input_shape`. |
| OM input byte size or format does not match input tensor | Check ATC input shape, format, and AIPP flags; the runner expects contiguous tensors matching the OM descriptor. |
| Output numerical diff exceeds tolerance | Compare float ONNX vs. float OM with identical inputs first, then isolate quantization drift via CPU fake-quant vs. NPU outputs. |

### AMCT-ONNX Process Killed

For one-command image quantization, `--batch-num` controls the total number of images read while each forward pass keeps batch size `1`. If the process is killed, retry with 16 representative images in a fresh output directory (or set `batch_num: 16` when using `--config`; Ascend docs also recommend starting with 16 or 32 samples). [AMCT-ONNX Tuning](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/amct/atlasamct_16_0146.html)

Inspect `amct-onnx.log` in the output directory: if the last line logs saving `modified_model.onnx`, termination likely occurred while saving the intermediate graph or entering calibration. On `SIGKILL`, check host kernel OOM logs for that timestamp and inspect container/scheduler memory limits (under cgroup v2, check `memory.max` on both the current and parent cgroups). [Linux cgroup v2 Memory](https://docs.kernel.org/admin-guide/cgroup-v2.html#memory)

```bash
journalctl -k -b --no-pager | grep -Ei 'oom-kill|out of memory|killed process' | tail -15
```

---

## Local Development Checks

```bash
uv run ruff check
uv run ruff format --check
uv run ty check
uv run --locked pytest -q tests
```

Local unit tests use ONNX Runtime CPU and a mock `pyACL` to validate data formats, parameter wiring, and resource cleanup. Real ATC compilation, AMCT quantization, and NPU inference must still be validated on target Ascend hardware following this guide.
