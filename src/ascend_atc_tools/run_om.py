"""Run one synchronous inference on a static, dense OM using pyACL."""

from __future__ import annotations

import argparse
import ctypes
import importlib
from contextlib import ExitStack
from math import prod
from pathlib import Path
from statistics import median
from time import perf_counter
from types import ModuleType

import numpy as np

from ascend_atc_tools.tensor_io import Array, Tensor, load_inputs, save_outputs

ACL_DEVICE = 0
ACL_HOST = 1
ACL_MEM_MALLOC_NORMAL_ONLY = 2
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_DENSE_FORMATS = (0, 1, 2)  # NCHW, NHWC, ND.
ACL_DTYPES = (
    (0, "float32"),
    (1, "float16"),
    (2, "int8"),
    (3, "int32"),
    (4, "uint8"),
    (6, "int16"),
    (7, "uint16"),
    (8, "uint32"),
    (9, "int64"),
    (10, "uint64"),
    (11, "float64"),
    (12, "bool"),
)


def check(status: int, operation: str) -> None:
    """Check a pyACL status code.

    Parameters
    ----------
    status
        Status returned by pyACL. Zero indicates success.
    operation
        Name included in the failure message.

    Raises
    ------
    RuntimeError
        The status is nonzero.
    """
    if status != 0:
        raise RuntimeError(f"{operation} failed (ACL error {status})")


def require_handle(handle: int | None, operation: str) -> int:
    """Validate a pyACL object handle.

    Parameters
    ----------
    handle
        Handle returned by pyACL.
    operation
        Name included in the failure message.

    Returns
    -------
    int
        The handle, when nonzero.

    Raises
    ------
    RuntimeError
        The handle is null.
    """
    if not handle:
        raise RuntimeError(f"{operation} returned a null handle")
    return handle


def describe(acl: ModuleType, desc: int, direction: str) -> list[Tensor]:
    """Read one direction of the model descriptor and check its dense layout.

    Parameters
    ----------
    acl
        pyACL module.
    desc
        Model descriptor handle.
    direction
        Either "input" or "output".

    Returns
    -------
    list[Tensor]
        Model tensors in descriptor order.

    Raises
    ------
    ValueError
        The model contains dynamic, padded, or unsupported tensors.
    """
    mdl = acl.mdl
    dtypes = dict(ACL_DTYPES)
    tensors = []
    count = getattr(mdl, f"get_num_{direction}s")(desc)
    for index in range(count):
        dims, status = getattr(mdl, f"get_{direction}_dims")(desc, index)
        check(status, f"acl.mdl.get_{direction}_dims({index})")
        name = getattr(mdl, f"get_{direction}_name_by_index")(desc, index)
        shape = tuple(dims["dims"])
        code = getattr(mdl, f"get_{direction}_data_type")(desc, index)
        layout = getattr(mdl, f"get_{direction}_format")(desc, index)
        size = getattr(mdl, f"get_{direction}_size_by_index")(desc, index)
        if not name or any(tensor.name == name for tensor in tensors):
            raise ValueError(
                f"Missing or duplicate {direction} name at index {index}: {name}"
            )
        if (
            code not in dtypes
            or layout not in ACL_DENSE_FORMATS
            or any(dim <= 0 for dim in shape)
        ):
            raise ValueError(
                f"Unsupported {direction} {name}: shape={shape}, dtype={code}, format={layout}"
            )
        tensor = Tensor(name, shape, np.dtype(dtypes[code]))
        expected = prod(shape) * tensor.dtype.itemsize
        if expected != size:
            raise ValueError(
                f"{direction} {name}: dense tensor is {expected} bytes, OM requires {size}"
            )
        tensors.append(tensor)
    return tensors


def dataset(
    acl: ModuleType, resources: ExitStack, tensors: list[Tensor]
) -> tuple[int, list[tuple[int, int]]]:
    """Allocate an ACL dataset and register cleanup for its buffers.

    Parameters
    ----------
    acl
        pyACL module.
    resources
        Stack that owns each successful allocation.
    tensors
        Static tensor descriptors in dataset order.

    Returns
    -------
    tuple[int, list[tuple[int, int]]]
        Dataset handle and the address and size of each buffer.

    Raises
    ------
    RuntimeError
        An allocation or an ACL dataset operation failed.
    """
    handle = require_handle(acl.mdl.create_dataset(), "acl.mdl.create_dataset")
    resources.callback(
        lambda: check(acl.mdl.destroy_dataset(handle), "acl.mdl.destroy_dataset")
    )
    buffers = []
    for tensor in tensors:
        size = prod(tensor.shape) * tensor.dtype.itemsize
        address, status = acl.rt.malloc(size, ACL_MEM_MALLOC_NORMAL_ONLY)
        check(status, "acl.rt.malloc")
        resources.callback(
            lambda address=address: check(acl.rt.free(address), "acl.rt.free")
        )
        buffer = require_handle(
            acl.create_data_buffer(address, size), "acl.create_data_buffer"
        )
        resources.callback(
            lambda buffer=buffer: check(
                acl.destroy_data_buffer(buffer), "acl.destroy_data_buffer"
            )
        )
        _, status = acl.mdl.add_dataset_buffer(handle, buffer)
        check(status, "acl.mdl.add_dataset_buffer")
        buffers.append((address, size))
    return handle, buffers


def infer(
    acl: ModuleType,
    model: Path,
    device: int,
    arguments: list[str],
    warmup: int,
    repeat: int,
) -> tuple[dict[str, Array], list[float]]:
    """Execute one OM and return its last outputs and measured execute times in ms.

    Parameters
    ----------
    acl
        pyACL module.
    model
        Path to the validated OM file.
    device
        Logical NPU device ID.
    arguments
        Input NAME=FILE pairs.
    warmup, repeat
        Number of warm-up and measured executions. At least one measured run
        is required.

    Returns
    -------
    tuple[dict[str, Array], list[float]]
        Named arrays from the final run and synchronous execute times in ms.

    Raises
    ------
    RuntimeError
        An ACL operation failed.
    ValueError
        A model descriptor or input is unsupported or inconsistent.

    Notes
    -----
    ACL_HOST uses page-locked staging memory for transfers. ACL_DEVICE runs on
    the device-side CPU and directly accesses its ACL allocations.
    """
    with ExitStack() as resources:
        check(acl.init(), "acl.init")
        resources.callback(lambda: check(acl.finalize(), "acl.finalize"))
        check(acl.rt.set_device(device), "acl.rt.set_device")
        resources.callback(
            lambda: check(acl.rt.reset_device(device), "acl.rt.reset_device")
        )
        context, status = acl.rt.create_context(device)
        check(status, "acl.rt.create_context")
        resources.callback(
            lambda: check(acl.rt.destroy_context(context), "acl.rt.destroy_context")
        )
        mode, status = acl.rt.get_run_mode()
        check(status, "acl.rt.get_run_mode")
        if mode not in (ACL_HOST, ACL_DEVICE):
            raise RuntimeError(f"Unknown ACL run mode: {mode}")

        model_id, status = acl.mdl.load_from_file(str(model))
        check(status, "acl.mdl.load_from_file")
        resources.callback(lambda: check(acl.mdl.unload(model_id), "acl.mdl.unload"))
        desc = require_handle(acl.mdl.create_desc(), "acl.mdl.create_desc")
        resources.callback(
            lambda: check(acl.mdl.destroy_desc(desc), "acl.mdl.destroy_desc")
        )
        check(acl.mdl.get_desc(desc, model_id), "acl.mdl.get_desc")
        input_specs = describe(acl, desc, "input")
        output_specs = describe(acl, desc, "output")
        if not input_specs or not output_specs:
            raise ValueError("The OM must have at least one input and one output.")
        inputs = load_inputs(input_specs, arguments)
        input_dataset, input_buffers = dataset(acl, resources, input_specs)
        output_dataset, output_buffers = dataset(acl, resources, output_specs)

        staging = 0
        capacity = max(size for _, size in input_buffers + output_buffers)
        if mode == ACL_HOST:
            staging, status = acl.rt.malloc_host(capacity)
            check(status, "acl.rt.malloc_host")
            resources.callback(
                lambda: check(acl.rt.free_host(staging), "acl.rt.free_host")
            )

        for spec, (address, size) in zip(input_specs, input_buffers):
            source = inputs[spec.name].ctypes.data
            if mode == ACL_HOST:
                ctypes.memmove(staging, source, size)
                check(
                    acl.rt.memcpy(
                        address, size, staging, size, ACL_MEMCPY_HOST_TO_DEVICE
                    ),
                    "H2D copy",
                )
            else:
                ctypes.memmove(address, source, size)

        for _ in range(warmup):
            check(
                acl.mdl.execute(model_id, input_dataset, output_dataset),
                "acl.mdl.execute",
            )
        elapsed_ms = []
        for _ in range(repeat):
            start = perf_counter()
            check(
                acl.mdl.execute(model_id, input_dataset, output_dataset),
                "acl.mdl.execute",
            )
            elapsed_ms.append((perf_counter() - start) * 1000)

        outputs: dict[str, Array] = {}
        for spec, (address, size) in zip(output_specs, output_buffers):
            if mode == ACL_HOST:
                check(
                    acl.rt.memcpy(
                        staging, capacity, address, size, ACL_MEMCPY_DEVICE_TO_HOST
                    ),
                    "D2H copy",
                )
                address = staging
            contents = ctypes.string_at(address, size)
            outputs[spec.name] = np.frombuffer(contents, dtype=spec.dtype).reshape(
                spec.shape
            )
        return outputs, elapsed_ms


def main() -> None:
    """Load the vendor pyACL module and save outputs for a single execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        metavar="NAME=FILE",
        help="Model input file (.npy, .bin, .jpg, .jpeg, or .png), repeatable.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()
    if args.device < 0:
        parser.error("Device ID must be nonnegative.")
    if args.warmup < 0 or args.repeat < 1:
        parser.error("Warmup must be nonnegative and repeat must be positive.")
    if not args.model.is_file():
        parser.error(f"OM file not found: {args.model}")
    if args.output_dir.exists():
        parser.error(f"Choose a new output directory: {args.output_dir}")
    try:
        acl = importlib.import_module("acl")
        if not all(hasattr(acl, name) for name in ("init", "rt", "mdl")):
            raise ImportError("The imported acl module is not the vendor pyACL module.")
        outputs, elapsed_ms = infer(
            acl, args.model, args.device, args.input, args.warmup, args.repeat
        )
        save_outputs(args.output_dir, outputs)
        print(
            f"ACL execute ({len(elapsed_ms)} runs): median={median(elapsed_ms):.3f} ms, "
            f"min={min(elapsed_ms):.3f} ms, max={max(elapsed_ms):.3f} ms; outputs: {args.output_dir}"
        )
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        parser.exit(1, f"OM inference failed: {error}\n")


if __name__ == "__main__":
    main()
