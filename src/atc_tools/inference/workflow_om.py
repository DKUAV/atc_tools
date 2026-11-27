"""Run prepared workflow samples with the vendor AIS Bench Python environment."""

from __future__ import annotations

import argparse
from pathlib import Path

from atc_tools.inference.ais_bench import create_session, describe_inputs
from atc_tools.model.tensor_io import load_tensor, save_outputs


def main() -> int:
    """Run all prepared samples on the single visible NPU.

    Returns
    -------
    int
        Zero after all samples are processed.

    Raises
    ------
    ValueError
        Prepared inputs or OM outputs do not match the runtime metadata.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("samples", type=Path)
    parser.add_argument("outputs", type=Path, nargs="?")
    args = parser.parse_args()
    session = create_session(args.model, 0)
    try:
        tensors = describe_inputs(session)
        names = [desc.name for desc in session.get_outputs()]
        for directory in sorted(args.samples.iterdir()):
            if len(list(directory.glob("*.npy"))) != len(tensors):
                raise ValueError("Prepared sample and OM input counts differ.")
            arrays = [
                load_tensor(directory / f"input_{index}.npy", tensor)
                for index, tensor in enumerate(tensors)
            ]
            outputs = session.infer(arrays, mode="static")
            if args.outputs is not None:
                save_outputs(
                    args.outputs / directory.name,
                    dict(zip(names, outputs, strict=True)),
                )
    finally:
        session.free_resource()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
