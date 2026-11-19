"""Parse and dispatch the atc_tools command line."""

from __future__ import annotations

import argparse
from importlib import import_module
from importlib.metadata import version
from pathlib import Path


def _add_tensor_input(parser: argparse.ArgumentParser) -> None:
    """Register named tensor inputs on an ONNX or OM command.

    Parameters
    ----------
    parser
        Command parser that accepts one file per model input.
    """
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        metavar="NAME=FILE",
        help="Model input file (.npy, .bin, .jpg, .jpeg, or .png), repeatable.",
    )


def _add_onnx_commands(parser: argparse.ArgumentParser) -> None:
    """Register ONNX inspection and preparation commands.

    Parameters
    ----------
    parser
        Parser for the ONNX command group.
    """
    parser.set_defaults(handler_module="atc_tools.model.onnx", extra="prepare")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Show ONNX input and output details")
    reference = commands.add_parser("reference", help="Generate CPU reference outputs")
    calibration = commands.add_parser(
        "calibration", help="Prepare calibration binary files"
    )
    for command in (inspect, reference, calibration):
        command.set_defaults(command_parser=command)
        command.add_argument("--model", required=True, type=Path)
    for command in (reference, calibration):
        command.add_argument("--output-dir", required=True, type=Path)
    _add_tensor_input(reference)
    reference.add_argument("--custom-ops-library", type=Path)
    calibration.add_argument("--sample", required=True, action="append", type=Path)

    quantize = commands.add_parser(
        "quantize", help="Quantize a single-input ONNX model from an image directory"
    )
    quantize.set_defaults(
        command_parser=quantize,
        handler_module="atc_tools.model.quantize",
        extra="prepare,yolo",
    )
    quantize.add_argument(
        "--model", required=True, type=Path, help="Floating-point ONNX"
    )
    quantize.add_argument(
        "--dataset", required=True, type=Path, help="Directory of JPG and PNG images"
    )
    quantize.add_argument(
        "--output-dir", type=Path, help="New directory (default: <model-stem>-int8)"
    )
    settings = quantize.add_mutually_exclusive_group()
    settings.add_argument(
        "--config", type=Path, help="Native AMCT-ONNX calibration .cfg"
    )
    settings.add_argument(
        "--batch-num", type=int, help="Number of images to calibrate (default: all)"
    )


def _add_om_commands(parser: argparse.ArgumentParser) -> None:
    """Register OM inference and tensor comparison commands.

    Parameters
    ----------
    parser
        Parser for the OM command group.
    """
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run OM inference through pyACL")
    ais_bench = commands.add_parser(
        "run-ais-bench", help="Run OM inference through ais_bench"
    )
    compare = commands.add_parser("compare", help="Compare named model outputs")
    for command, module in (
        (run, "atc_tools.inference.pyacl"),
        (ais_bench, "atc_tools.inference.ais_bench"),
    ):
        command.set_defaults(command_parser=command, handler_module=module)
        command.add_argument("--model", required=True, type=Path)
        command.add_argument("--device", type=int, default=0)
        _add_tensor_input(command)
        command.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--warmup", type=int, default=0)
    run.add_argument("--repeat", type=int, default=1)
    compare.set_defaults(
        command_parser=compare, handler_module="atc_tools.inference.compare"
    )
    compare.add_argument("--reference", required=True, type=Path)
    compare.add_argument("--actual", required=True, type=Path)
    compare.add_argument("--atol", type=float, default=0.001)
    compare.add_argument("--rtol", type=float, default=0.01)
    compare.add_argument(
        "--output-map", action="append", default=[], metavar="REFERENCE=ACTUAL"
    )


def _add_yolo_commands(parser: argparse.ArgumentParser) -> None:
    """Register YOLO inference and image processing commands.

    Parameters
    ----------
    parser
        Parser for the YOLO command group.
    """
    parser.set_defaults(handler_module="atc_tools.detection.yolo", extra="yolo")
    commands = parser.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser(
        "preprocess", help="Write a normalized RGB NCHW tensor"
    )
    postprocess = commands.add_parser(
        "postprocess", help="Write boxes as JSON and draw an image"
    )
    infer = commands.add_parser(
        "infer", help="Run a YOLO OM through ais_bench and render detections"
    )
    for command in (preprocess, postprocess, infer):
        command.set_defaults(command_parser=command)
        command.add_argument("--image", required=True, type=Path)
    for command in (preprocess, postprocess):
        command.add_argument("--width", required=True, type=int)
        command.add_argument("--height", required=True, type=int)
    preprocess.add_argument("--output", required=True, type=Path)
    preprocess.add_argument(
        "--dtype", choices=("float32", "float16"), default="float32"
    )
    postprocess.add_argument("--predictions", required=True, type=Path)
    infer.add_argument("--model", required=True, type=Path, help="YOLO OM file")
    infer.add_argument(
        "--device", type=int, default=0, help="Logical NPU device ID (default: 0)"
    )
    for command in (postprocess, infer):
        command.add_argument("--layout", required=True, choices=("v5", "v8", "xyxy"))
        command.add_argument("--output-dir", required=True, type=Path)
        command.add_argument("--confidence", type=float, default=0.25)
        command.add_argument("--iou", type=float, default=0.45)


def main(argv: list[str] | None = None) -> int:
    """Run the selected atc_tools command.

    Parameters
    ----------
    argv
        Arguments to parse. Uses the process arguments when omitted.

    Returns
    -------
    int
        Exit status of the selected command.

    Raises
    ------
    SystemExit
        Argument parsing, dependency loading, or command execution stopped the CLI.
    """
    parser = argparse.ArgumentParser(
        prog="atc_tools", description="Model preparation and inference for Ascend CANN."
    )
    parser.set_defaults(extra=None)
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {version('atc_tools')}"
    )
    groups = parser.add_subparsers(required=True)
    _add_onnx_commands(
        groups.add_parser("onnx", help="Inspect and prepare ONNX models")
    )
    _add_om_commands(groups.add_parser("om", help="Run OM models and compare outputs"))
    _add_yolo_commands(
        groups.add_parser("yolo", help="Run single-image YOLO detection")
    )

    args = parser.parse_args(argv)
    command_parser: argparse.ArgumentParser = args.command_parser
    try:
        handler = import_module(args.handler_module)
    except (ImportError, OSError) as error:
        hint = f" Install the {args.extra!r} extra." if args.extra else ""
        command_parser.exit(1, f"Cannot load {command_parser.prog}: {error}.{hint}\n")
    status: int | None = handler.run_command(args, command_parser)
    return 0 if status is None else status


if __name__ == "__main__":
    raise SystemExit(main())
