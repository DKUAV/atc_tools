"""Read named inputs and write indexed output tensors with a manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[np.generic]
else:
    Array = np.ndarray


@dataclass(frozen=True)
class Tensor:
    """Describe a dense tensor by name, shape, and NumPy data type."""

    name: str
    shape: tuple[int, ...]
    dtype: np.dtype[np.generic]


def parse_pairs(items: list[str]) -> dict[str, str]:
    """Parse unique NAME=VALUE arguments.

    Parameters
    ----------
    items
        Command-line name and value pairs.

    Returns
    -------
    dict[str, str]
        Values keyed by name.

    Raises
    ------
    ValueError
        An argument is malformed or repeats a name.
    """
    pairs: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition("=")
        if not separator or not name or not value or name in pairs:
            raise ValueError(f"Expected a unique NAME=VALUE argument: {item}")
        pairs[name] = value
    return pairs


def load_tensor(path: Path, tensor: Tensor) -> Array:
    """Read a contiguous .npy array and check its exact shape and data type.

    Parameters
    ----------
    path
        NumPy array path.
    tensor
        Required shape and dtype, with the name used for diagnostics.

    Returns
    -------
    Array
        Validated array in C order.

    Raises
    ------
    ValueError
        The array is incompatible with the tensor specification.
    """
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Expected a .npy file: {path}")
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        array.close()
        raise ValueError(f"Expected one tensor in {path}")
    if array.shape != tensor.shape or array.dtype != tensor.dtype:
        raise ValueError(
            f"{tensor.name}: expected {tensor.shape} {tensor.dtype}, "
            f"got {array.shape} {array.dtype} in {path}"
        )
    return np.asarray(array, order="C")


def load_inputs(tensors: list[Tensor], arguments: list[str]) -> dict[str, Array]:
    """Read exactly one named .npy, .bin, .jpg, or .png file per model input.

    Parameters
    ----------
    tensors
        Expected input descriptors in model order.
    arguments
        NAME=FILE arguments in any order.

    Returns
    -------
    dict[str, Array]
        Named input arrays in model order.

    Raises
    ------
    ValueError
        The input names or an input tensor do not match the model.
    """
    paths = parse_pairs(arguments)
    expected = {tensor.name for tensor in tensors}
    if paths.keys() != expected:
        raise ValueError(
            f"Input names do not match: missing={sorted(expected - paths.keys())}, "
            f"unexpected={sorted(paths.keys() - expected)}"
        )
    return {tensor.name: _load_input(Path(paths[tensor.name]), tensor) for tensor in tensors}


def _load_input(path: Path, tensor: Tensor) -> Array:
    """Read a named inference input according to its file extension.

    Parameters
    ----------
    path
        NumPy, raw binary, JPEG, or PNG file.
    tensor
        Model input shape and dtype.

    Returns
    -------
    Array
        Input array with the required shape and dtype in C order.

    Raises
    ------
    OSError
        The file could not be read.
    ValueError
        The file or tensor specification is incompatible with the input format.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return load_tensor(path, tensor)
    if suffix == ".bin":
        size = prod(tensor.shape) * tensor.dtype.itemsize
        with path.open("rb") as stream:
            data = bytearray(size)
            if stream.readinto(data) != size or stream.read(1):
                raise ValueError(f"{tensor.name}: expected exactly {size} bytes in {path}")
        return np.frombuffer(data, dtype=tensor.dtype).reshape(tensor.shape)
    if suffix not in (".jpg", ".jpeg", ".png"):
        raise ValueError(f"{tensor.name}: expected a .npy, .bin, .jpg, .jpeg, or .png file: {path}")

    shape = tensor.shape
    if (
        len(shape) != 4 or shape[0] != 1 or any(dim <= 0 for dim in shape)
        or (shape[1] == 3) == (shape[3] == 3)
        or (tensor.dtype.kind != "f" and tensor.dtype != np.dtype("uint8"))
    ):
        raise ValueError(
            f"{tensor.name}: image input requires an unambiguous batch-one RGB NCHW or NHWC "
            f"tensor with a float or uint8 dtype, got {shape} {tensor.dtype}. Use .npy or .bin."
        )
    try:
        from image_input import image_tensor, letterbox, read_image
    except ImportError as error:
        raise ValueError("JPG/PNG input requires OpenCV. Install opencv-python-headless.") from error

    channels_first = shape[1] == 3
    height, width = shape[2:] if channels_first else shape[1:3]
    canvas, _, _, _ = letterbox(read_image(path), width, height)
    return image_tensor(canvas, channels_first, tensor.dtype)


def save_outputs(directory: Path, outputs: dict[str, Array]) -> None:
    """Save outputs in a new directory and write their manifest last.

    Parameters
    ----------
    directory
        Destination for the arrays and their manifest.
    outputs
        Named arrays in the order in which they should be saved.

    Raises
    ------
    FileExistsError
        The output directory already exists.
    """
    directory.mkdir(parents=True)
    entries = []
    for index, (name, array) in enumerate(outputs.items()):
        filename = f"output_{index}.npy"
        np.save(directory / filename, array, allow_pickle=False)
        entries.append(
            {"name": name, "file": filename, "shape": list(array.shape), "dtype": array.dtype.name}
        )
    (directory / "outputs.json").write_text(json.dumps(entries, indent=2) + "\n", encoding="utf-8")


def read_outputs(directory: Path) -> dict[str, Array]:
    """Load the arrays listed in an output manifest.

    Parameters
    ----------
    directory
        Directory containing outputs.json and the arrays it describes.

    Returns
    -------
    dict[str, Array]
        Validated arrays keyed by manifest name.

    Raises
    ------
    ValueError
        The manifest is empty, has invalid entries, or disagrees with its arrays.
    """
    entries = json.loads((directory / "outputs.json").read_text(encoding="utf-8"))
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Expected a nonempty output manifest in {directory}")
    outputs: dict[str, Array] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("file") != f"output_{index}.npy":
            raise ValueError(f"Invalid output entry {index} in {directory}")
        name, shape, dtype = entry.get("name"), entry.get("shape"), entry.get("dtype")
        if (
            not isinstance(name, str) or not name or name in outputs
            or not isinstance(shape, list) or not all(type(dim) is int and dim >= 0 for dim in shape)
            or not isinstance(dtype, str)
        ):
            raise ValueError(f"Invalid output metadata at index {index} in {directory}")
        outputs[name] = load_tensor(directory / entry["file"], Tensor(name, tuple(shape), np.dtype(dtype)))
    return outputs
