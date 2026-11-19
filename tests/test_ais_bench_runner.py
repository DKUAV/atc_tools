"""Check the ais_bench runner's CLI contract without an NPU."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from run_om_ais_bench import infer, main
from tensor_io import read_outputs


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
                "run_om_ais_bench.py",
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
            with (
                patch.object(sys, "argv", args),
                patch.dict(sys.modules, {"ais_bench.infer.interface": module}),
            ):
                main()
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
