"""Verify shared dataset transformations using the supported file formats."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from atc_tools.model.tensor_io import Tensor
from atc_tools.model.workflow_inputs import model_inputs, prepare_tensor


def test_shared_preprocessing_formats(tmp_path: Path) -> None:
    """Apply geometry before numeric normalization for PNG, NumPy, and raw data.

    Parameters
    ----------
    tmp_path
        Temporary directory for source files.
    """
    image = np.array([[[60, 40, 20], [120, 100, 80]]], dtype=np.uint8)
    png = tmp_path / "input.png"
    assert cv2.imwrite(str(png), image)
    rgb = np.moveaxis(image[..., ::-1], -1, 0)[None, ...].astype("<f4")
    npy, raw = tmp_path / "input.npy", tmp_path / "input.bin"
    np.save(npy, rgb, allow_pickle=False)
    rgb.tofile(raw)
    tensor = Tensor("image", (1, 3, 3, 4), np.dtype("float32"))
    config = {
        "layout": "nchw",
        "geometry": "pad",
        "fill": 102,
        "values": "custom",
        "mean": [0.1, 0.2, 0.3],
        "std": [0.5, 0.25, 0.125],
        "unit_first": True,
        "raw_inputs": {"0": {"shape": [1, 3, 1, 2], "dtype": "float32"}},
    }
    expected = np.full((1, 3, 3, 4), 102, dtype=np.float64)
    expected[..., 1:2, 1:3] = rgb
    mean = np.array(config["mean"]).reshape((1, 3, 1, 1))
    std = np.array(config["std"]).reshape((1, 3, 1, 1))
    expected = (expected / 255 - mean) / std
    for source in (png, npy, raw):
        prepared = prepare_tensor(source, tensor, 0, config)
        assert prepared.dtype == np.float32 and prepared.flags.c_contiguous
        np.testing.assert_allclose(prepared.astype(np.float64), expected, rtol=1e-6)


@pytest.mark.parametrize(
    ("dtype_name", "value"), [("int64", 2**53 + 1), ("uint64", 2**63 + 1)]
)
def test_geometry_preserves_large_integers_or_reports_loss(
    tmp_path: Path, dtype_name: str, value: int
) -> None:
    """Keep exact source values or reject a lossy geometry conversion.

    Parameters
    ----------
    tmp_path
        Temporary directory for the source tensor.
    dtype_name, value
        Integer dtype and a value that float64 cannot represent exactly.
    """
    source = tmp_path / "input.npy"
    dtype = np.dtype(dtype_name)
    np.save(source, np.array([[[[value]]]], dtype=dtype), allow_pickle=False)
    tensor = Tensor("input", (1, 1, 1, 3), dtype)
    settings = {"layout": "nchw", "geometry": "pad", "fill": 0, "values": "raw"}
    expected = np.array([[[[0, value, 0]]]], dtype=dtype)
    np.testing.assert_array_equal(prepare_tensor(source, tensor, 0, settings), expected)
    with pytest.raises(ValueError, match="Geometry would lose precision"):
        prepare_tensor(source, tensor, 0, {**settings, "fill": 0.5})
    larger = Tensor("input", (1, 1, 2, 2), dtype)
    for geometry in ("resize", "letterbox"):
        with pytest.raises(ValueError, match="Geometry would lose precision"):
            prepare_tensor(source, larger, 0, {**settings, "geometry": geometry})


def test_model_dynamic_dimensions_preserve_fixed_axes() -> None:
    """Require concrete dimensions while preserving the fixed ONNX axes."""
    metadata = {
        "inputs": [{"name": "image", "type": "FLOAT", "shape": [1, 3, "h", "w"]}]
    }
    assert model_inputs(metadata, {"0": [1, 3, 32, 40]})[0].shape == (1, 3, 32, 40)
    with pytest.raises(ValueError, match="compatible fixed shape"):
        model_inputs(metadata, {})
    with pytest.raises(ValueError, match="compatible fixed shape"):
        model_inputs(metadata, {"0": [1, 1, 32, 40]})
