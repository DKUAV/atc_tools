"""Run a static OM through ais_bench and save named NumPy outputs."""

from __future__ import annotations

import argparse
import importlib
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from tensor_io import Array, Tensor, load_inputs, save_outputs


def infer(session: Any, arguments: list[str]) -> dict[str, Array]:
    """Run one set of named inputs and release the ais_bench session.

    Parameters
    ----------
    session
        An ais_bench InferSession for a fixed-shape OM.
    arguments
        One NAME=FILE argument per model input, in any order.

    Returns
    -------
    dict[str, Array]
        Output arrays keyed by the OM tensor names.

    Raises
    ------
    ValueError
        An input or the model metadata is incompatible with static inference.
    RuntimeError
        ais_bench failed or returned an unexpected set of outputs.
    """
    try:
        inputs: list[Tensor] = []
        for desc in session.get_inputs():
            shape = tuple(desc.shape)
            type_name = desc.datatype.name
            dtype = np.dtype("float64" if type_name == "double64" else type_name)
            if any(dim <= 0 for dim in shape) or prod(shape) * dtype.itemsize != desc.realsize:
                raise ValueError(f"Expected a static dense input for {desc.name}, got {shape} {dtype}")
            inputs.append(Tensor(desc.name, shape, dtype))
        if not inputs or len({tensor.name for tensor in inputs if tensor.name}) != len(inputs):
            raise ValueError("Expected unique, nonempty OM input names.")

        names: list[str] = [desc.name for desc in session.get_outputs()]
        if not names or not all(names) or len(set(names)) != len(names):
            raise ValueError("Expected unique, nonempty OM output names.")
        feeds = load_inputs(inputs, arguments)
        results: list[Array] = session.infer(list(feeds.values()), mode="static")
        if len(results) != len(names):
            raise RuntimeError(f"Expected {len(names)} outputs, got {len(results)}.")
        return dict(zip(names, results))
    finally:
        session.free_resource()


def main() -> None:
    """Run a single OM inference using the optional vendor packages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--input", required=True, action="append", metavar="NAME=FILE",
        help="Model input file (.npy, .bin, .jpg, .jpeg, or .png), repeatable.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("Device ID must be nonnegative.")
    if not args.model.is_file():
        parser.error(f"OM file not found: {args.model}")
    if args.output_dir.exists():
        parser.error(f"Choose a new output directory: {args.output_dir}")
    try:
        session_type = importlib.import_module("ais_bench.infer.interface").InferSession
    except (AttributeError, ImportError, OSError) as error:
        parser.exit(1, f"Cannot load ais_bench. Check ais_bench, aclruntime, and CANN: {error}\n")
    try:
        session = session_type(args.device, str(args.model))
        outputs = infer(session, args.input)
        save_outputs(args.output_dir, outputs)
        print(f"ais_bench outputs: {args.output_dir}")
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"ais_bench inference failed: {error}\n")


if __name__ == "__main__":
    main()
