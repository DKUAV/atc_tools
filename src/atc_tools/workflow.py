"""Execute an allowlisted conversion and publish its completed stage results."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import onnxruntime as ort

from atc_tools.inference.compare import compare
from atc_tools.model.onnx import inspect_onnx
from atc_tools.model.tensor_io import Array, Tensor, read_outputs
from atc_tools.model.workflow_inputs import model_inputs, prepare_dataset

SCRIPTS = Path("/opt/atc-tools/scripts")
TOOL_ERRORS = (OSError, ValueError, RuntimeError, subprocess.CalledProcessError)
T = TypeVar("T")


def _shape_argument(tensors: list[Tensor]) -> str:
    """Format concrete model inputs for the ATC and AMCT shape options.

    Parameters
    ----------
    tensors
        Named model inputs in ONNX order.

    Returns
    -------
    str
        Semicolon-delimited list of named input shapes.

    Raises
    ------
    ValueError
        An input name contains unsupported separators.
    """
    if any(
        tensor.shape and any(char in tensor.name for char in ":;,\n\r\0")
        for tensor in tensors
    ):
        raise ValueError(
            "An input name is incompatible with the tool's input shape syntax."
        )
    return ";".join(
        f"{tensor.name}:{','.join(map(str, tensor.shape))}"
        for tensor in tensors
        if tensor.shape
    )


def _compile(
    model: Path, directory: Path, target: str, precision: str, shape: str
) -> Path:
    """Compile one ONNX model via the existing constrained ATC entry point.

    Parameters
    ----------
    model
        Floating or quantized deployment ONNX file.
    directory
        New compiler output directory.
    target
        Allowlisted SoC suffix.
    precision
        Preset's precision mode or INT8 for a calibrated model.
    shape
        Selected fixed input shapes.

    Returns
    -------
    Path
        Produced OM file.
    """
    options = ["--input-shape", shape] if shape else []
    if precision != "int8":
        options.extend(("--atc-arg", f"--precision_mode_v2={precision}"))
    subprocess.run(
        [
            "bash",
            str(SCRIPTS / "compile.sh"),
            "--model",
            str(model),
            "--soc",
            f"Ascend{target}",
            "--output-dir",
            str(directory),
            *options,
        ],
        check=True,
    )
    return directory / "model.om"


def _calibrate(
    model: Path,
    samples: list[Path],
    tensors: list[Tensor],
    scratch: Path,
    simulated: bool,
) -> Path:
    """Prepare ordered AMCT batch files and run the default calibration policy.

    Parameters
    ----------
    model
        Original embedded-weight ONNX file.
    samples
        Prepared calibration sample directories.
    tensors
        Concrete input metadata in model order.
    scratch
        Private workflow workspace.
    simulated
        Skip the vendor quantizer during CPU-only simulation.

    Returns
    -------
    Path
        Calibrated deployment ONNX, or the original for simulation.

    Raises
    ------
    ValueError
        An input does not have a batch dimension.
    """
    if any(not tensor.shape for tensor in tensors):
        raise ValueError("AMCT calibration requires inputs with a batch dimension.")
    if simulated:
        print(
            f"Simulating AMCT calibration with {len(samples)} prepared batch(es).",
            flush=True,
        )
        return model
    directories = [scratch / f"amct-input-{index}" for index in range(len(tensors))]
    for index, directory in enumerate(directories):
        directory.mkdir()
        tensor = tensors[index]
        for sample in samples:
            array = np.load(sample / f"input_{index}.npy", allow_pickle=False)
            array.astype(tensor.dtype.newbyteorder("<"), copy=False).tofile(
                directory / f"{sample.name}.bin"
            )
    config = scratch / "calibration.cfg"
    config.write_text(
        f"batch_num: {len(samples)}\nactivation_offset: true\n", encoding="utf-8"
    )
    output = scratch / "quantized"
    subprocess.run(
        [
            "bash",
            str(SCRIPTS / "quantize_model.sh"),
            "--model",
            str(model),
            "--input-shape",
            _shape_argument(tensors),
            "--data-dir",
            ";".join(map(str, directories)),
            "--data-types",
            ";".join(tensor.dtype.name for tensor in tensors),
            "--config",
            str(config),
            "--output-dir",
            str(output),
        ],
        check=True,
    )
    return output / "model_deploy_model.onnx"


def _vendor_om(model: Path, samples: Path, outputs: Path | None = None) -> list[str]:
    """Build the vendor Python command for static OM inference.

    Parameters
    ----------
    model
        OM to execute on logical device zero.
    samples
        Directory containing prepared validation batches.
    outputs
        Optional directory for per-sample inference results.

    Returns
    -------
    list[str]
        Argument vector containing only service-selected paths.
    """
    command = [
        "python3.11",
        "-m",
        "atc_tools.inference.workflow_om",
        str(model),
        str(samples),
    ]
    if outputs is not None:
        command.append(str(outputs))
    return command


def _compare_outputs(
    reference: dict[str, Array], actual: dict[str, Array]
) -> dict[str, Any]:
    """Compare one pair of output sets using the shared numerical checker.

    Parameters
    ----------
    reference
        Baseline model outputs in declaration order.
    actual
        Primary OM outputs in declaration order.

    Returns
    -------
    dict
        Matching policy, outcome, and per-tensor diagnostics.
    """
    if len(reference) != len(actual):
        return {
            "passed": False,
            "error": f"Output count differs: {len(reference)} vs {len(actual)}",
        }
    by_name = reference.keys() == actual.keys()
    right = list(reference) if by_name else list(actual)
    outcomes = []
    for left, name in zip(reference, right, strict=True):
        passed, detail = compare(reference[left], actual[name], atol=0.001, rtol=0.01)
        outcomes.append(
            {"reference": left, "actual": name, "passed": passed, "detail": detail}
        )
    return {
        "passed": all(entry["passed"] for entry in outcomes),
        "mapping": "name" if by_name else "position",
        "outputs": outcomes,
    }


def _accuracy(
    model: Path,
    om: Path,
    samples: list[Path],
    tensors: list[Tensor],
    scratch: Path,
    workflow: dict[str, Any],
    simulated: bool,
    destination: Path,
) -> None:
    """Compare the primary OM with CPU ONNX and, for INT8, a same-device baseline.

    Parameters
    ----------
    model, om
        Original ONNX and primary compiled result.
    samples
        Shared prepared validation tensors.
    tensors
        Concrete model input specifications.
    scratch
        Private workspace for inference outputs and the floating OM.
    workflow
        Selected target and precision.
    simulated
        Whether NPU execution is unavailable.
    destination
        Accuracy report path.
    """
    int8 = workflow["precision"] == "int8"
    primary = scratch / "primary-outputs"
    baseline = scratch / "floating-outputs"
    if not simulated:
        if int8:
            floating = _compile(
                model,
                scratch / "floating",
                workflow["target"],
                "origin",
                _shape_argument(tensors) if workflow["shapes"] else "",
            )
            subprocess.run(
                _vendor_om(floating, samples[0].parent, baseline), check=True
            )
        subprocess.run(_vendor_om(om, samples[0].parent, primary), check=True)
    session_options = ort.SessionOptions()
    session_options.intra_op_num_threads = 1
    session_options.inter_op_num_threads = 1
    try:
        cpu = ort.InferenceSession(
            str(model), sess_options=session_options, providers=["CPUExecutionProvider"]
        )
        names = [item.name for item in cpu.get_outputs()]
        rows = []
        for sample in samples:
            feeds = {
                tensor.name: np.load(sample / f"input_{index}.npy", allow_pickle=False)
                for index, tensor in enumerate(tensors)
            }
            reference: dict[str, Array] = {}
            for name, array in zip(names, cpu.run(names, feeds), strict=True):
                if not isinstance(array, np.ndarray):
                    raise ValueError(f"Expected a dense ONNX output for {name}.")
                reference[name] = array
            if simulated:
                rows.append(
                    {
                        "sample": int(sample.name) + 1,
                        "cpu_outputs": {
                            name: list(array.shape) for name, array in reference.items()
                        },
                    }
                )
                continue
            observed = read_outputs(primary / sample.name)
            row = {
                "sample": int(sample.name) + 1,
                "onnx_vs_om": _compare_outputs(reference, observed),
            }
            if int8:
                row["floating_om_vs_int8_om"] = _compare_outputs(
                    read_outputs(baseline / sample.name), observed
                )
            rows.append(row)
    except Exception as error:
        # ONNX Runtime exception classes are provided by the native extension.
        raise RuntimeError(f"Accuracy inference failed: {error}") from error
    destination.write_text(
        json.dumps(
            {"simulated": simulated, "atol": 0.001, "rtol": 0.01, "samples": rows},
            indent=2,
        ),
        encoding="utf-8",
    )


def _profile(
    om: Path,
    samples: list[Path],
    scratch: Path,
    simulated: bool,
    destination: Path,
    limit_bytes: int,
) -> None:
    """Collect an msprof archive for the primary OM and prepared validation data.

    Parameters
    ----------
    om
        Primary conversion result.
    samples
        Prepared validation sample directories.
    scratch
        Private profiler output directory.
    simulated
        Emit an explicitly marked archive with no NPU timing data.
    destination
        Downloadable ZIP archive.
    limit_bytes
        Maximum total profiling data bytes to archive.
    """
    if not simulated:
        application = shlex.join(_vendor_om(om, samples[0].parent))
        subprocess.run(
            ["msprof", f"--application={application}", f"--output={scratch}"],
            check=True,
        )
    with ZipFile(destination, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "profiling.json",
            json.dumps({"simulated": simulated, "samples": len(samples)}),
        )
        if not simulated:
            total = 0
            for path in sorted(scratch.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    total += path.stat().st_size
                    if total > limit_bytes:
                        raise ValueError("Profiling data exceeds the output budget.")
                    archive.write(path, path.relative_to(scratch))
            if not total:
                raise ValueError("msprof produced no profiling data.")


def run_workflow(inputs: Path, output: Path, request: dict[str, Any]) -> None:
    """Run required and optional workflow stages in the sandbox.

    Parameters
    ----------
    inputs, output
        Fixed read-only input and writable output directories.
    request
        Validated worker protocol including the workflow snapshot.
    """
    workflow = request["workflow"]
    target, precision = workflow["target"], workflow["precision"]
    if target not in ("310B4", "910B2") or precision not in (
        "origin",
        "mixed_float16",
        "fp16",
        "int8",
    ):
        raise ValueError("Unknown target or precision preset.")
    model = inputs / "model.onnx"
    scratch = output / "workspace"
    scratch.mkdir()
    work_tmp = scratch / "tmp"
    work_tmp.mkdir()
    os.environ["TMPDIR"] = str(work_tmp)
    simulated = bool(request["simulated"])
    limit = request["limit_bytes"]
    states: dict[str, str] = {}

    def record(stage: str, status: str) -> None:
        """Atomically publish one stage transition.

        Parameters
        ----------
        stage
            Fixed workflow step ID.
        status
            New progress state.
        """
        states[stage] = status
        temporary = output / "progress.next"
        temporary.write_text(json.dumps(states), encoding="utf-8")
        temporary.replace(output / "progress.json")

    def step(name: str, action: Callable[[], T]) -> T:
        """Run one stage and persist its outcome.

        Parameters
        ----------
        name
            Fixed stage identifier.
        action
            Implementation of the stage.

        Returns
        -------
        T
            Result returned by the stage.
        """
        print(f"Stage {name} started.", flush=True)
        record(name, "running")
        try:
            result = action()
        except TOOL_ERRORS:
            record(name, "failed")
            raise
        record(name, "succeeded")
        print(f"Stage {name} finished.", flush=True)
        return result

    metadata = inspect_onnx(model, require_embedded=True)
    tensors = model_inputs(metadata, workflow["shapes"])

    def prepare(role: str) -> list[Path]:
        """Prepare a dataset with the common transformation settings.

        Parameters
        ----------
        role
            Calibration or validation dataset.

        Returns
        -------
        list[Path]
            Prepared samples in order.
        """
        return prepare_dataset(
            inputs / "dataset",
            workflow[role],
            tensors,
            workflow["preparation"],
            scratch / role,
            limit,
        )

    deployment = model
    if precision == "int8":
        calibration = step("prepare_calibration", lambda: prepare("calibration"))
        deployment = step(
            "quantize",
            lambda: _calibrate(model, calibration, tensors, scratch, simulated),
        )

    def compile_primary() -> Path:
        """Compile the primary model and write its reproducibility metadata.

        Returns
        -------
        Path
            Ready-to-publish model file.
        """
        path = output / ("model.simulated" if simulated else "model.om")
        if simulated:
            path.write_text(
                f"Simulated Ascend{target} compilation ({precision}).\n",
                encoding="utf-8",
            )
        else:
            _compile(
                deployment,
                scratch / "compile",
                target,
                precision,
                _shape_argument(tensors)
                if precision == "int8" or workflow["shapes"]
                else "",
            ).replace(path)
        build = {
            "target": f"Ascend{target}",
            "preset": request["preset"],
            "precision": precision,
            "simulated": simulated,
            "preparation": workflow["preparation"],
            "input_shapes": {tensor.name: tensor.shape for tensor in tensors},
            "calibration_batches": len(workflow["calibration"]),
            "validation_batches": len(workflow["validation"]),
        }
        (output / "build.json").write_text(
            json.dumps(build, indent=2), encoding="utf-8"
        )
        return path

    om = step("compile", compile_primary)
    checks = [name for name in ("compare", "profile") if workflow[name]]
    if not checks:
        return
    try:
        validation = step("prepare_validation", lambda: prepare("validation"))
    except TOOL_ERRORS as error:
        print(f"Validation preparation failed: {error}", flush=True)
        for name in checks:
            record(name, "skipped")
        return
    for name in checks:
        try:
            if name == "compare":
                step(
                    name,
                    lambda: _accuracy(
                        model,
                        om,
                        validation,
                        tensors,
                        scratch,
                        workflow,
                        simulated,
                        output / "accuracy.json",
                    ),
                )
            else:
                step(
                    name,
                    lambda: _profile(
                        om,
                        validation,
                        scratch / "profiling",
                        simulated,
                        output / "profiling.zip",
                        limit,
                    ),
                )
        except TOOL_ERRORS as error:
            print(f"Optional stage {name} failed: {error}", flush=True)
