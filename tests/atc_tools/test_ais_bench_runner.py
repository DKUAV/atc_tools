"""Check the generic and YOLO ais_bench commands without an NPU."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np

from atc_tools.cli import main
from atc_tools.inference.ais_bench import infer
from atc_tools.model.tensor_io import read_outputs


class AisBenchRunnerTest(unittest.TestCase):
    """Check input ordering, output manifests, and session cleanup."""

    def test_cli_writes_outputs_in_model_order(self) -> None:
        """Read mixed NPY and BIN inputs in model order and emit comparable outputs."""
        with tempfile.TemporaryDirectory(prefix="atc-ais-bench-") as directory:
            root = Path(directory)
            model = root / "test.om"
            model.touch()
            left = np.array([1, 2], dtype=np.float32)
            right = np.array([4, 6], dtype=np.float32)
            np.save(root / "left.npy", left)
            right.tofile(root / "right.bin")
            dtype = SimpleNamespace(name="float32")
            session = Mock()
            session.get_inputs.return_value = [
                SimpleNamespace(name=name, shape=[2], datatype=dtype, realsize=8)
                for name in ("left", "right")
            ]
            session.get_outputs.return_value = [
                SimpleNamespace(name="difference"),
                SimpleNamespace(name="sum"),
            ]
            session.infer.return_value = [left - right, left + right]
            constructor = Mock(return_value=session)
            module = SimpleNamespace(InferSession=constructor)
            args = [
                "om",
                "run-ais-bench",
                "--model",
                str(model),
                "--device",
                "2",
                "--input",
                f"right={root / 'right.bin'}",
                "--input",
                f"left={root / 'left.npy'}",
                "--output-dir",
                str(root / "outputs"),
            ]
            with patch.dict(sys.modules, {"ais_bench.infer.interface": module}):
                self.assertEqual(main(args), 0)
            constructor.assert_called_once_with(2, str(model))
            ordered_feeds = session.infer.call_args.args[0]
            np.testing.assert_array_equal(ordered_feeds[0], left)
            np.testing.assert_array_equal(ordered_feeds[1], right)
            self.assertEqual(session.infer.call_args.kwargs, {"mode": "static"})
            outputs = read_outputs(root / "outputs")
            self.assertEqual(list(outputs), ["difference", "sum"])
            np.testing.assert_array_equal(outputs["difference"], left - right)
            np.testing.assert_array_equal(outputs["sum"], left + right)
            session.free_resource.assert_called_once_with()

    def test_bad_input_releases_session_before_inference(self) -> None:
        """Reject an incompatible input dtype and still release the session."""
        with tempfile.TemporaryDirectory(prefix="atc-ais-bench-") as directory:
            array = Path(directory) / "input.npy"
            np.save(array, np.zeros(2, dtype=np.float64))
            session = Mock()
            session.get_inputs.return_value = [
                SimpleNamespace(
                    name="x",
                    shape=[2],
                    datatype=SimpleNamespace(name="float32"),
                    realsize=8,
                )
            ]
            session.get_outputs.return_value = [SimpleNamespace(name="y")]
            with self.assertRaisesRegex(ValueError, "float32.*float64"):
                infer(session, [f"x={array}"])
            session.infer.assert_not_called()
            session.free_resource.assert_called_once_with()

    def test_incorrect_bin_size_releases_session(self) -> None:
        """Reject truncated and oversized binary tensors before inference."""
        with tempfile.TemporaryDirectory(prefix="atc-ais-bench-") as directory:
            path = Path(directory) / "input.bin"
            for size in (7, 9):
                with self.subTest(size=size):
                    path.write_bytes(bytes(size))
                    session = Mock()
                    session.get_inputs.return_value = [
                        SimpleNamespace(
                            name="x",
                            shape=[2],
                            datatype=SimpleNamespace(name="float32"),
                            realsize=8,
                        )
                    ]
                    session.get_outputs.return_value = [SimpleNamespace(name="y")]
                    with self.assertRaisesRegex(ValueError, "expected exactly 8 bytes"):
                        infer(session, [f"x={path}"])
                    session.infer.assert_not_called()
                    session.free_resource.assert_called_once_with()

    def test_yolo_infer_renders_detections_in_image_coordinates(self) -> None:
        """Use OM dimensions and dtype, release the session, and save detections."""
        with tempfile.TemporaryDirectory(prefix="atc-yolo-") as directory:
            root = Path(directory)
            image_path = root / "image.png"
            model_path = root / "model.om"
            output_dir = root / "detections"
            model_path.touch()
            encoded, image = cv2.imencode(
                ".png", np.full((2, 4, 3), (0, 51, 255), dtype=np.uint8)
            )
            self.assertTrue(encoded)
            image.tofile(image_path)
            session = Mock()
            session.get_inputs.return_value = [
                SimpleNamespace(
                    name="images",
                    shape=[1, 6, 8, 3],
                    datatype=SimpleNamespace(name="float16"),
                    realsize=288,
                )
            ]
            session.get_outputs.return_value = [SimpleNamespace(name="predictions")]
            session.infer.return_value = [
                np.array([[[2, 1, 6, 5, 0.75, 1]]], dtype=np.float32)
            ]
            constructor = Mock(return_value=session)
            args = [
                "yolo",
                "infer",
                "--model",
                str(model_path),
                "--image",
                str(image_path),
                "--device",
                "2",
                "--layout",
                "xyxy",
                "--output-dir",
                str(output_dir),
            ]
            with patch.dict(
                sys.modules,
                {
                    "ais_bench.infer.interface": SimpleNamespace(
                        InferSession=constructor
                    )
                },
            ):
                self.assertEqual(main(args), 0)
            constructor.assert_called_once_with(2, str(model_path))
            tensor = session.infer.call_args.args[0][0]
            self.assertEqual(session.infer.call_args.kwargs, {"mode": "static"})
            self.assertEqual(tensor.dtype, np.dtype("float16"))
            expected = np.full((1, 6, 8, 3), 114 / 255, dtype=np.float16)
            expected[:, 1:5, :, :] = np.array([1, 0.2, 0], dtype=np.float16)
            np.testing.assert_array_equal(tensor, expected)
            session.free_resource.assert_called_once_with()
            self.assertEqual(
                json.loads(
                    (output_dir / "detections.json").read_text(encoding="utf-8")
                ),
                [{"xyxy": [1.0, 0.0, 3.0, 2.0], "score": 0.75, "class_id": 1}],
            )
            self.assertGreater((output_dir / "annotated.jpg").stat().st_size, 0)
