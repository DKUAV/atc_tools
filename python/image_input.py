"""Decode images and prepare RGB tensors with the YOLO resize and padding."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


def read_image(path: Path) -> NDArray[np.uint8]:
    """Read a BGR image, including paths with non-ASCII characters.

    Parameters
    ----------
    path
        Encoded image path.

    Returns
    -------
    NDArray[np.uint8]
        Decoded image in BGR channel order.

    Raises
    ------
    OSError
        The file could not be read.
    ValueError
        The file could not be decoded as an image.
    """
    encoded = np.fromfile(path, dtype=np.uint8)
    if not encoded.size:
        raise ValueError(f"Cannot decode image: {path}")
    try:
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise ValueError(f"Cannot decode image: {path}") from error
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return np.asarray(image, dtype=np.uint8)


def letterbox(
    image: NDArray[np.uint8], width: int, height: int
) -> tuple[NDArray[np.uint8], float, int, int]:
    """Resize and pad to fixed dimensions and return the coordinate transform.

    Parameters
    ----------
    image
        Original BGR image.
    width, height
        Fixed model input dimensions.

    Returns
    -------
    tuple[NDArray[np.uint8], float, int, int]
        Padded image, resize scale, and left and top padding.
    """
    source_height, source_width = image.shape[:2]
    scale = min(width / source_width, height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    left = (width - resized_width) // 2
    top = (height - resized_height) // 2
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas, scale, left, top


def image_tensor(
    canvas: NDArray[np.uint8], channels_first: bool, dtype: np.dtype[np.generic]
) -> NDArray[np.generic]:
    """Convert a padded BGR image to a contiguous batch-one RGB tensor.

    Parameters
    ----------
    canvas
        Image produced by letterbox.
    channels_first
        Place channels first for NCHW or last for NHWC.
    dtype
        Model input dtype. Float pixels are scaled to [0, 1].

    Returns
    -------
    NDArray[np.generic]
        Model input tensor.
    """
    rgb = canvas[:, :, ::-1]
    if channels_first:
        rgb = rgb.transpose(2, 0, 1)
    if dtype.kind == "f":
        precision = np.float32 if dtype.itemsize < 4 else dtype
        tensor = np.ascontiguousarray(rgb[None], dtype=precision)
        np.divide(tensor, 255, out=tensor)
        return tensor.astype(dtype, copy=False)
    return np.ascontiguousarray(rgb[None], dtype=dtype)
