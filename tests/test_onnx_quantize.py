"""Verify that image calibration passes the expected binaries to AMCT-ONNX."""

from __future__ import annotations

import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import STDOUT
from typing import BinaryIO
from unittest.mock import patch

import cv2
import numpy as np
import onnx
from onnx import TensorProto, helper

from atc_tools.cli import main
from atc_tools.model.tensor_io import Tensor, load_inputs


class ImageQuantizationTest(unittest.TestCase):
    """Exercise the CLI using an actual model and image files."""

    def test_image_directory_with_default_count_batch_limit_and_config(self) -> None:
        """Preserve preprocessing, sample order, config content, and both artifacts."""
        with tempfile.TemporaryDirectory(prefix="atc-quantize-") as directory:
            root = Path(directory)
            dataset = root / "photos"
            dataset.mkdir()
            tensor = Tensor("image", (1, 3, 6, 8), np.dtype("float32"))
            source = helper.make_tensor_value_info(
                tensor.name, TensorProto.FLOAT, tensor.shape
            )
            result = helper.make_tensor_value_info(
                "output", TensorProto.FLOAT, tensor.shape
            )
            graph = helper.make_graph(
                [helper.make_node("Identity", ["image"], ["output"])],
                "image",
                [source],
                [result],
            )
            model = root / "scene.onnx"
            onnx.save(helper.make_model(graph), model)

            first = dataset / "a space.PNG"
            last = dataset / "z.jpg"
            for path, color in ((last, (41, 101, 233)), (first, (9, 71, 166))):
                image = np.full((2, 4, 3), color, dtype=np.uint8)
                success, encoded = cv2.imencode(path.suffix.lower(), image)
                self.assertTrue(success)
                path.write_bytes(encoded.tobytes())
            (dataset / "notes.txt").write_text("Ignored", encoding="utf-8")
            (dataset / "folder.png").mkdir()
            expected = [
                load_inputs([tensor], [f"image={path}"])["image"].tobytes()
                for path in (first, last)
            ]
            config = root / "custom.cfg"
            config.write_text(
                "batch_num: 1\nactivation_offset: false\n", encoding="utf-8"
            )

            def run_amct(
                command: list[str],
                *,
                cwd: Path,
                stdout: BinaryIO,
                stderr: int,
                check: bool,
            ) -> None:
                """Check AMCT arguments and create its two model files."""
                self.assertTrue(check)
                self.assertEqual(stderr, STDOUT)
                self.assertEqual(command[1], "calibration")
                options = dict(zip(command[2::2], command[3::2], strict=True))
                self.assertEqual(Path(options["--model"]), model)
                self.assertEqual(options["--input_shape"], "image:1,3,6,8")
                self.assertEqual(options["--data_types"], "float32")
                bins = sorted(Path(options["--data_dir"]).iterdir())
                for index, path in enumerate(bins):
                    self.assertEqual(path.name, f"{index:08d}.bin")
                self.assertEqual([path.read_bytes() for path in bins], expected[:count])
                if use_config:
                    copied_config = Path(options["--calibration_config"])
                    self.assertEqual(copied_config, cwd / "amct-onnx.cfg")
                    self.assertEqual(copied_config.read_bytes(), config.read_bytes())
                    self.assertNotIn("--batch_num", options)
                else:
                    self.assertEqual(options["--batch_num"], str(count))
                    self.assertNotIn("--calibration_config", options)
                prefix = Path(options["--save_path"])
                self.assertEqual(prefix, cwd / "model")
                for name in ("model_deploy_model.onnx", "model_fake_quant_model.onnx"):
                    (cwd / name).write_bytes(model.read_bytes())
                stdout.write(b"Calibrated\n")

            cases = [
                ([], root / "scene-int8", 2, False),
                (
                    ["--batch-num", "1", "--output-dir", str(root / "limited")],
                    root / "limited",
                    1,
                    False,
                ),
                (
                    ["--config", str(config), "--output-dir", str(root / "custom")],
                    root / "custom",
                    2,
                    True,
                ),
            ]
            for options, output, count, use_config in cases:
                with (
                    self.subTest(options=options),
                    patch("shutil.which", return_value=sys.executable),
                    patch("subprocess.run", side_effect=run_amct) as process,
                    redirect_stdout(StringIO()) as report,
                ):
                    self.assertEqual(
                        main(
                            [
                                "onnx",
                                "quantize",
                                "--model",
                                str(model),
                                "--dataset",
                                str(dataset),
                                *options,
                            ]
                        ),
                        0,
                    )
                    process.assert_called_once()
                    self.assertIn(
                        str(output / "model_deploy_model.onnx"), report.getvalue()
                    )
                    self.assertEqual(
                        (output / "amct-onnx.log").read_bytes(), b"Calibrated\n"
                    )
                    self.assertEqual(list(output.glob("calibration-*")), [])
