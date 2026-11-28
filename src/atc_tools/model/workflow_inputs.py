"""Resolve ONNX inputs and apply shared dataset preprocessing in the sandbox."""

from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from onnx import TensorProto, helper

from atc_tools.model.image_input import read_image
from atc_tools.model.tensor_io import Array, Tensor


def model_inputs(
    metadata: dict[str, Any], shapes: dict[str, list[int]]
) -> list[Tensor]:
    """Resolve model input dtypes and dimensions against any requested overrides.

    Parameters
    ----------
    metadata
        ONNX graph metadata, with inputs in declared order.
    shapes
        Concrete shapes keyed by input index.

    Returns
    -------
    list[Tensor]
        Concrete and size-bounded model inputs.

    Raises
    ------
    ValueError
        An input is unresolved, unsupported, too large, or conflicts with ONNX.
    """
    declared = metadata["inputs"]
    if not 1 <= len(declared) <= 16 or set(shapes) - {
        str(i) for i in range(len(declared))
    }:
        raise ValueError("Expected 1 to 16 model inputs and matching shape selections.")
    tensors = []
    for index, input_info in enumerate(declared):
        name = input_info["name"]
        if not name:
            raise ValueError("Expected ONNX model inputs to have names.")
        original = input_info["shape"]
        shape = shapes.get(str(index), original)
        if (
            shape is None
            or any(type(dim) is not int or dim <= 0 for dim in shape)
            or original is not None
            and (
                len(shape) != len(original)
                or any(
                    isinstance(dim, int) and dim > 0 and dim != resolved
                    for dim, resolved in zip(original, shape, strict=True)
                )
            )
        ):
            raise ValueError(f"Provide a compatible fixed shape for {name}.")
        try:
            dtype = np.dtype(
                helper.tensor_dtype_to_np_dtype(
                    TensorProto.DataType.Value(input_info["type"])
                )
            )
        except (KeyError, ValueError, TypeError) as error:
            raise ValueError(f"Unsupported tensor type for {name}.") from error
        if dtype.kind not in "fbiu" or prod(shape) * dtype.itemsize > 128 * 1024**2:
            raise ValueError(f"Unsupported tensor type or size for {name}.")
        tensors.append(Tensor(name, tuple(shape), dtype))
    return tensors


def _channel_axis(tensor: Tensor, layout: str) -> int:
    """Identify the channel axis for HWC/CHW and their batched forms.

    Parameters
    ----------
    tensor
        Concrete model input specification.
    layout
        Selected or automatically resolved input layout.

    Returns
    -------
    int
        Channel axis of the full tensor.

    Raises
    ------
    ValueError
        The input layout cannot be determined unambiguously.
    """
    rank = len(tensor.shape)
    if rank not in (3, 4):
        raise ValueError(f"Input {tensor.name} needs a rank 3 or 4 spatial layout.")
    first, last = rank - 3, rank - 1
    if layout == "nchw":
        return first
    if layout == "nhwc":
        return last
    matches = [axis for axis in (first, last) if tensor.shape[axis] in (1, 3, 4)]
    if len(matches) != 1:
        raise ValueError(f"Select NCHW or NHWC for input {tensor.name}.")
    return matches[0]


def _geometry(
    array: Array, shape: tuple[int, ...], axis: int, method: str, fill: float
) -> Array:
    """Resize or pad the spatial dimensions while preserving batch and channels.

    Parameters
    ----------
    array
        Source tensor in the selected layout.
    shape
        Target tensor shape.
    axis
        Channel axis in the full tensor.
    method
        Resize, pad, or proportional resize with padding.
    fill
        Padding value in the source value range.

    Returns
    -------
    Array
        Tensor with adjusted height and width.

    Raises
    ------
    ValueError
        Source batch or channels differ, padding would crop, or geometry would
        lose precision in an integer source tensor.
    """
    source = np.moveaxis(array, axis, -1)
    axes = [index for index in range(len(shape)) if index != axis]
    height, width = (shape[index] for index in axes[-2:])
    src_h, src_w, channels = source.shape[-3:]
    if channels != shape[axis] or (len(shape) == 4 and source.shape[0] != shape[0]):
        raise ValueError("Source batch or channel count does not match the model.")
    new_h, new_w = height, width
    if method == "pad":
        new_h, new_w = src_h, src_w
    elif method == "letterbox":
        scale = min(height / src_h, width / src_w)
        new_h, new_w = max(1, round(src_h * scale)), max(1, round(src_w * scale))
    if new_h > height or new_w > width:
        raise ValueError("Pad requires the input to fit inside the target.")
    resize = (src_h, src_w) != (new_h, new_w)
    if not resize and (new_h, new_w) == (height, width):
        return array
    top, left = (height - new_h) // 2, (width - new_w) // 2
    batch = source[None, ...] if source.ndim == 3 else source
    dtype = np.dtype(np.float64)
    if not resize and source.dtype.kind in "biu":
        bounds = np.iinfo(source.dtype.name) if source.dtype.kind != "b" else None
        lo, hi = (bounds.min, bounds.max) if bounds is not None else (0, 1)
        integer_fill = int(fill)
        if fill == integer_fill and lo <= integer_fill <= hi:
            dtype = source.dtype
    if dtype.kind == "f" and source.dtype.kind in "iu" and source.dtype.itemsize == 8:
        with np.errstate(invalid="ignore"):
            restored = source.astype(np.float64).astype(source.dtype)
        if np.any(restored != source):
            raise ValueError("Geometry would lose precision in the integer samples.")
    result = np.full((batch.shape[0], height, width, channels), fill, dtype=dtype)
    for index, sample in enumerate(batch):
        if resize:
            resize_source = sample.astype(np.float64)
            sample = cv2.resize(
                resize_source, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        result[index, top : top + new_h, left : left + new_w] = sample.reshape(
            (new_h, new_w, channels)
        )
    return np.moveaxis(result[0] if source.ndim == 3 else result, -1, axis)


def prepare_tensor(
    path: Path, tensor: Tensor, index: int, settings: dict[str, Any]
) -> Array:
    """Decode and transform one dataset file into a model input tensor.

    Parameters
    ----------
    path
        Sandboxed dataset file with an approved extension.
    tensor
        Concrete destination input descriptor.
    index
        Input position used to look up any raw binary format override.
    settings
        Shared layout, geometry and numerical settings.

    Returns
    -------
    Array
        Finite, contiguous tensor in the model dtype and shape.

    Raises
    ------
    ValueError
        The source data or the transformations do not match the model.
    """
    axis = None
    if path.suffix == ".npy":
        with path.open("rb") as stream:
            if stream.read(6) != b"\x93NUMPY":
                raise ValueError("Expected a valid .npy tensor.")
        array = np.load(path, allow_pickle=False, mmap_mode="r")
        if not isinstance(array, np.ndarray) or array.dtype.kind not in "fbiu":
            raise ValueError("Expected a numerical .npy tensor.")
    elif path.suffix == ".bin":
        raw = settings["raw_inputs"].get(str(index), {})
        shape = tuple(raw.get("shape", tensor.shape))
        dtype = np.dtype(raw.get("dtype", tensor.dtype)).newbyteorder("<")
        size = prod(shape) * dtype.itemsize
        if size != path.stat().st_size or size > 128 * 1024**2:
            raise ValueError(
                f"Raw byte count does not match the source shape and dtype for {tensor.name}."
            )
        array = np.fromfile(path, dtype=dtype).reshape(shape)
    else:
        axis = _channel_axis(tensor, settings["layout"])
        channels = tensor.shape[axis]
        if channels not in (1, 3):
            raise ValueError(
                "Image inputs support one grayscale channel or three RGB channels."
            )
        image = read_image(path)
        array = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[..., None]
            if channels == 1
            else image[..., ::-1]
        )
        array = np.moveaxis(array, -1, axis - (len(tensor.shape) == 4))
    if len(tensor.shape) == 4 and tensor.shape[0] == 1 and array.ndim == 3:
        array = array[None, ...]
    if (
        array.ndim != len(tensor.shape)
        or array.nbytes > 128 * 1024**2
        or any(dim == 0 for dim in array.shape)
    ):
        raise ValueError(f"Source rank or size is incompatible with {tensor.name}.")
    if settings["geometry"] != "keep":
        axis = _channel_axis(tensor, settings["layout"]) if axis is None else axis
        array = _geometry(
            array, tensor.shape, axis, settings["geometry"], settings["fill"]
        )
    if array.shape != tensor.shape:
        raise ValueError(
            f"Input {tensor.name} needs shape {tensor.shape}, got {array.shape}."
        )
    mode = settings["values"]
    if (
        mode == "unit"
        or mode == "imagenet"
        or mode == "custom"
        and settings["unit_first"]
    ):
        array = array.astype(np.float64) / 255.0
    if mode in ("imagenet", "custom"):
        axis = _channel_axis(tensor, settings["layout"]) if axis is None else axis
        mean, std = (
            ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            if mode == "imagenet"
            else (settings["mean"], settings["std"])
        )
        if (
            len(mean) != tensor.shape[axis]
            or len(mean) != len(std)
            or any(value <= 0 for value in std)
        ):
            raise ValueError(
                f"Mean and standard deviation must match the channels of {tensor.name}."
            )
        broadcast = [1] * array.ndim
        broadcast[axis] = len(mean)
        array = (array - np.reshape(mean, broadcast)) / np.reshape(std, broadcast)
    if not np.isfinite(array).all():
        raise ValueError(f"Input {tensor.name} contains non-finite values.")
    if tensor.dtype.kind in "biu" and array.dtype != tensor.dtype:
        lo, hi = (
            (0, 1)
            if tensor.dtype.kind == "b"
            else (np.iinfo(tensor.dtype.name).min, np.iinfo(tensor.dtype.name).max)
        )
        if array.dtype.kind == "f":
            invalid = (
                np.less(array, float(lo)).any()
                or np.greater_equal(array, float(hi + 1)).any()
                or (array != np.trunc(array)).any()
            )
        else:
            invalid = np.less(array, lo).any() or np.greater(array, hi).any()
        if invalid:
            raise ValueError(
                f"Input {tensor.name} does not fit {tensor.dtype} without precision loss."
            )
    with np.errstate(over="ignore"):
        result = np.array(array, dtype=tensor.dtype, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"Input {tensor.name} overflows {tensor.dtype}.")
    return result


def prepare_dataset(
    files: Path,
    samples: list[list[str]],
    tensors: list[Tensor],
    settings: dict[str, Any],
    destination: Path,
    limit_bytes: int,
) -> list[Path]:
    """Prepare ordered samples once for all consumers of the dataset.

    Parameters
    ----------
    files
        Read-only directory of UUID-named uploads.
    samples
        Asset IDs in sample and model input order.
    tensors
        Destination model input specifications.
    settings
        Preparation shared by calibration and validation.
    destination
        Private directory receiving one NumPy file per prepared input.
    limit_bytes
        Maximum aggregate prepared tensor bytes.

    Returns
    -------
    list[Path]
        Prepared samples in submission order.

    Raises
    ------
    ValueError
        A sample is incomplete or exceeds resource and format limits.
    """
    if (
        len(samples)
        * sum(prod(tensor.shape) * tensor.dtype.itemsize for tensor in tensors)
        > limit_bytes
    ):
        raise ValueError("Prepared samples exceed the dataset byte limit.")
    sources = {path.stem: path for path in files.iterdir()}
    folders = []
    for position, sample in enumerate(samples):
        if len(sample) != len(tensors) or any(
            asset_id not in sources for asset_id in sample
        ):
            raise ValueError(f"Sample {position + 1} needs one file per model input.")
        folder = destination / f"{position:08d}"
        folder.mkdir(parents=True)
        for index, (asset_id, tensor) in enumerate(zip(sample, tensors, strict=True)):
            try:
                array = prepare_tensor(sources[asset_id], tensor, index, settings)
            except (OSError, ValueError, cv2.error) as error:
                raise ValueError(
                    f"Sample {position + 1}, input {tensor.name}: {error}"
                ) from error
            np.save(folder / f"input_{index}.npy", array, allow_pickle=False)
        folders.append(folder)
    return folders
