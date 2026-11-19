"""Quantize a single-image ONNX model with the AMCT-ONNX command line."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from google.protobuf.message import DecodeError
from onnx import TensorProto, helper

from atc_tools.model.image_input import image_tensor, letterbox, read_image
from atc_tools.model.onnx import inspect_onnx
from atc_tools.model.tensor_io import Tensor, image_geometry


def quantize_images(
    model: Path,
    dataset: Path,
    output_dir: Path,
    config: Path | None = None,
    batch_num: int | None = None,
) -> Path:
    """Prepare one binary tensor per image and run AMCT-ONNX calibration.

    Parameters
    ----------
    model
        ONNX model with one static, batch-one RGB input.
    dataset
        Flat directory of calibration images.
    output_dir
        New directory for the ONNX artifacts and the AMCT log.
    config
        Native AMCT configuration. Its settings determine the calibration count.
    batch_num
        Number of images selected in filename order when no config is supplied.
        Defaults to all images.

    Returns
    -------
    Path
        AMCT deployment ONNX to pass to ATC.

    Raises
    ------
    OSError
        An input or output path could not be accessed.
    ValueError
        The model input, image dataset, or batch count is incompatible.
    RuntimeError
        AMCT is unavailable, fails, or does not produce both ONNX artifacts.
    """
    executable = shutil.which("amct_onnx")
    if executable is None:
        raise RuntimeError(
            "amct_onnx is not on PATH. Activate the AMCT-ONNX environment."
        )
    if config is not None and batch_num is not None:
        raise ValueError("Choose either --config or --batch-num.")
    if config is not None and not config.is_file():
        raise FileNotFoundError(f"Calibration config not found: {config}")

    images = sorted(
        path
        for path in dataset.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.is_file()
    )
    if not images:
        raise ValueError(f"No JPG or PNG images found in {dataset}")
    if batch_num is not None:
        if not 1 <= batch_num <= len(images):
            raise ValueError(f"--batch-num must be between 1 and {len(images)}.")
        images = images[:batch_num]

    model = model.absolute()
    metadata = inspect_onnx(model)
    if missing := metadata["missing_external_weights"]:
        raise ValueError(f"Missing external ONNX weights: {', '.join(missing)}.")
    inputs = metadata["inputs"]
    if len(inputs) != 1:
        raise ValueError(f"Expected one ONNX input, found {len(inputs)}.")
    entry = inputs[0]
    shape = entry["shape"]
    if shape is None or any(type(dim) is not int for dim in shape):
        raise ValueError(
            f"{entry['name']}: expected a static tensor shape, got {shape}."
        )
    dtype = helper.tensor_dtype_to_np_dtype(TensorProto.DataType.Value(entry["type"]))
    tensor = Tensor(entry["name"], tuple(shape), dtype)
    channels_first, height, width = image_geometry(tensor)

    output_dir = output_dir.absolute()
    output_dir.mkdir(parents=True)
    prefix = output_dir / "model"
    log = output_dir / "amct-onnx.log"
    if config is None:
        settings = ["--batch_num", str(len(images))]
    else:
        saved_config = output_dir / "amct-onnx.cfg"
        shutil.copyfile(config, saved_config)
        settings = ["--calibration_config", str(saved_config)]

    print(f"Preparing {len(images)} calibration images.", flush=True)
    with TemporaryDirectory(prefix="calibration-", dir=output_dir) as directory:
        for index, image in enumerate(images):
            canvas, _, _, _ = letterbox(read_image(image), width, height)
            array = image_tensor(canvas, channels_first, tensor.dtype)
            (Path(directory) / f"{index:08d}.bin").write_bytes(array.tobytes(order="C"))

        command = [
            str(Path(executable).absolute()),
            "calibration",
            "--model",
            str(model),
            "--input_shape",
            f"{tensor.name}:{','.join(map(str, tensor.shape))}",
            "--data_dir",
            directory,
            "--data_types",
            tensor.dtype.name,
            "--save_path",
            str(prefix),
            *settings,
        ]
        print(f"Running AMCT-ONNX. Log: {log}", flush=True)
        try:
            with log.open("wb") as stream:
                subprocess.run(
                    command,
                    cwd=output_dir,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"AMCT-ONNX exited with status {error.returncode}. See {log}."
            ) from error

    deploy_model = output_dir / "model_deploy_model.onnx"
    fake_quant_model = output_dir / "model_fake_quant_model.onnx"
    for artifact in (deploy_model, fake_quant_model):
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise RuntimeError(f"AMCT-ONNX did not produce {artifact}. See {log}.")
    return deploy_model


def run_command(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    """Execute image-based ONNX quantization from the CLI.

    Parameters
    ----------
    args
        Parsed model, dataset, output, and calibration options.
    parser
        Parser used to report quantization errors.

    Raises
    ------
    SystemExit
        Model preparation or AMCT-ONNX calibration failed.
    """
    output_dir = args.output_dir or args.model.with_name(f"{args.model.stem}-int8")
    try:
        deploy_model = quantize_images(
            args.model, args.dataset, output_dir, args.config, args.batch_num
        )
    except (OSError, RuntimeError, ValueError, DecodeError) as error:
        parser.exit(1, f"Quantization failed: {error}\n")
    print(f"Deployment ONNX: {deploy_model}")
