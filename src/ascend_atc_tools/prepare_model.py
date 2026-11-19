"""Inspect ONNX I/O, generate CPU reference outputs and format calibration data."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf.message import DecodeError, Message

from ascend_atc_tools.tensor_io import (
    Array,
    Tensor,
    load_inputs,
    load_tensor,
    save_outputs,
)


def external_locations(message: Message) -> Iterator[str]:
    """Yield external tensor locations from the model and its nested messages.

    Parameters
    ----------
    message
        ONNX graph or a nested protobuf message.

    Yields
    ------
    str
        The location string recorded in an external tensor.
    """
    if isinstance(
        message, onnx.TensorProto
    ) and onnx.external_data_helper.uses_external_data(message):
        for entry in message.external_data:
            if entry.key == "location":
                yield entry.value
    for field, value in message.ListFields():
        if field.message_type is None:
            continue
        if isinstance(value, Message):
            yield from external_locations(value)
        else:
            for child in value:
                yield from external_locations(child)


def inspect_onnx(path: Path) -> dict[str, Any]:
    """Read graph metadata without loading weights or initializing ONNX Runtime.

    Parameters
    ----------
    path
        ONNX model path, used to resolve external tensor locations.

    Returns
    -------
    dict[str, Any]
        JSON-compatible graph and external weight metadata.
    """
    model = onnx.load_model(path, load_external_data=False)
    initializers = {tensor.name for tensor in model.graph.initializer}
    initializers.update(tensor.values.name for tensor in model.graph.sparse_initializer)

    def metadata(node: onnx.ValueInfoProto) -> dict[str, Any]:
        """Describe the type information stored for an ONNX graph value.

        Parameters
        ----------
        node
            Graph input or output declaration.

        Returns
        -------
        dict[str, Any]
            Value name, element type, and symbolic or static shape.
        """
        if not node.type.HasField("tensor_type"):
            return {
                "name": node.name,
                "type": node.type.WhichOneof("value"),
                "shape": None,
            }
        tensor = node.type.tensor_type
        shape = None
        if tensor.HasField("shape"):
            shape = []
            for dim in tensor.shape.dim:
                if dim.HasField("dim_value"):
                    shape.append(dim.dim_value)
                else:
                    shape.append(dim.dim_param or None)
        return {
            "name": node.name,
            "type": onnx.TensorProto.DataType.Name(tensor.elem_type),
            "shape": shape,
        }

    locations = sorted(set(external_locations(model)))
    return {
        "opsets": {
            opset.domain or "ai.onnx": opset.version for opset in model.opset_import
        },
        "inputs": [
            metadata(node)
            for node in model.graph.input
            if node.name not in initializers
        ],
        "outputs": [metadata(node) for node in model.graph.output],
        "external_weights": locations,
        "missing_external_weights": [
            location for location in locations if not (path.parent / location).is_file()
        ],
    }


def describe_inputs(session: ort.InferenceSession) -> list[Tensor]:
    """Return static ONNX Runtime input names, dimensions, and NumPy dtypes.

    Parameters
    ----------
    session
        Active session for the ONNX graph being prepared.

    Returns
    -------
    list[Tensor]
        Tensor descriptions in model input order.

    Raises
    ------
    ValueError
        An input uses dynamic dimensions or an unsupported type.
    """
    tensors = []
    for node in session.get_inputs():
        if any(type(dim) is not int or dim <= 0 for dim in node.shape):
            raise ValueError(
                f"Input {node.name} must have a positive static shape, got {node.shape}"
            )
        name = node.type.removeprefix("tensor(").removesuffix(")")
        names = {"float": "float32", "double": "float64", "bool": "bool"}
        if not node.type.startswith("tensor(") or name not in {
            "float",
            "float16",
            "double",
            "bool",
            "int8",
            "uint8",
            "int16",
            "uint16",
            "int32",
            "uint32",
            "int64",
            "uint64",
        }:
            raise ValueError(f"Unsupported input type for {node.name}: {node.type}")
        tensors.append(
            Tensor(node.name, tuple(node.shape), np.dtype(names.get(name, name)))
        )
    if not tensors:
        raise ValueError("The model has no runtime inputs.")
    return tensors


def write_calibration(
    tensors: list[Tensor], samples: list[Path], output: Path
) -> dict[str, str]:
    """Write one AMCT directory and one concatenated ATC .bin per input.

    Parameters
    ----------
    tensors
        Input descriptions in ONNX order.
    samples
        One directory per batch with indexed .npy files in ONNX input order.
    output
        New directory for the calibration artifacts.

    Returns
    -------
    dict[str, str]
        Values for the AMCT and ATC calibration options.

    Raises
    ------
    ValueError
        An input lacks a batch dimension or a sample has the wrong shape or dtype.
    """
    if any(not tensor.shape for tensor in tensors):
        raise ValueError("Calibration inputs must have a leading batch dimension.")
    output.mkdir(parents=True)
    dirs = [output / f"input_{index}" for index in range(len(tensors))]
    for directory in dirs:
        directory.mkdir()
    for batch_index, sample in enumerate(samples):
        for index, (tensor, directory) in enumerate(zip(tensors, dirs, strict=True)):
            array = load_tensor(sample / f"input_{index}.npy", tensor)
            data = array.tobytes(order="C")
            (directory / f"{batch_index:08d}.bin").write_bytes(data)
            with directory.with_suffix(".bin").open("ab") as stream:
                stream.write(data)
    return {
        "input_shape": ";".join(
            f"{tensor.name}:{','.join(map(str, tensor.shape))}" for tensor in tensors
        ),
        "atc_input_shape": ";".join(
            f"{tensor.name}:{','.join(map(str, (tensor.shape[0] * len(samples), *tensor.shape[1:])))}"
            for tensor in tensors
        ),
        "data_types": ";".join(tensor.dtype.name for tensor in tensors),
        "data_dir": ";".join(str(directory) for directory in dirs),
        "input_data_dir": ",".join(
            str(directory.with_suffix(".bin")) for directory in dirs
        ),
    }


def main() -> None:
    """Run the selected ONNX preparation command."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser(
        "inspect", help="Show ONNX input and output details"
    )
    reference_parser = commands.add_parser(
        "reference", help="Generate CPU reference outputs"
    )
    calibration_parser = commands.add_parser(
        "calibration", help="Prepare calibration binary files"
    )
    for command in (inspect_parser, reference_parser, calibration_parser):
        command.add_argument("--model", required=True, type=Path)
    reference_parser.add_argument(
        "--input",
        required=True,
        action="append",
        metavar="NAME=FILE",
        help="Model input file (.npy, .bin, .jpg, .jpeg, or .png), repeatable.",
    )
    reference_parser.add_argument("--custom-ops-library", type=Path)
    calibration_parser.add_argument(
        "--sample", required=True, action="append", type=Path
    )
    for command in (reference_parser, calibration_parser):
        command.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            print(json.dumps(inspect_onnx(args.model), indent=2))
            return
        options = ort.SessionOptions()
        if getattr(args, "custom_ops_library", None):
            options.register_custom_ops_library(str(args.custom_ops_library.resolve()))
        session = ort.InferenceSession(
            str(args.model), options, providers=["CPUExecutionProvider"]
        )
        tensors = describe_inputs(session)
        if args.command == "reference":
            inputs = load_inputs(tensors, args.input)
            names = [node.name for node in session.get_outputs()]
            results = session.run(names, inputs)
            outputs: dict[str, Array] = {}
            for name, result in zip(names, results, strict=True):
                if not isinstance(result, np.ndarray):
                    raise ValueError(f"Output {name} is not a dense tensor.")
                outputs[name] = result
            save_outputs(args.output_dir, outputs)
        else:
            config = write_calibration(tensors, args.sample, args.output_dir)
            contents = (
                json.dumps({"samples": len(args.sample), **config}, indent=2) + "\n"
            )
            (args.output_dir / "calibration.json").write_text(
                contents, encoding="utf-8"
            )
            print(contents, end="")
    except (OSError, RuntimeError, TypeError, ValueError, DecodeError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    except Exception as error:
        # ONNX Runtime reports native failures as extension-specific exceptions.
        if not type(error).__module__.startswith("onnxruntime."):
            raise
        parser.exit(1, f"ONNX Runtime failed: {error}\n")


if __name__ == "__main__":
    main()
