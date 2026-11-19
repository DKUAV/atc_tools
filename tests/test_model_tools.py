"""Check calibration bytes, output comparison, and pyACL resource cleanup."""

from __future__ import annotations

import ctypes
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from compare_outputs import compare
from prepare_model import describe_inputs, inspect_onnx, write_calibration
from run_om import ACL_DEVICE, ACL_HOST, infer
from tensor_io import load_inputs, read_outputs, save_outputs


class ModelPreparationTest(unittest.TestCase):
    """Exercise the format contract using a real two-input ONNX model."""

    def test_inspection_without_external_weights(self) -> None:
        """Report symbolic dimensions, opsets, and missing weight files."""
        with tempfile.TemporaryDirectory(prefix="atc-test-") as directory:
            root = Path(directory)
            value = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 2])
            result = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 2])
            weights = numpy_helper.from_array(np.ones(2, dtype=np.float32), name="w")
            graph = helper.make_graph(
                [helper.make_node("Add", ["x", "w"], ["y"])], "add", [value], [result], [weights]
            )
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
            path = root / "external.onnx"
            onnx.save_model(model, path, save_as_external_data=True, location="weights.bin", size_threshold=0)
            self.assertEqual(inspect_onnx(path)["missing_external_weights"], [])
            (root / "weights.bin").unlink()
            metadata = inspect_onnx(path)
            self.assertEqual(metadata["opsets"], {"ai.onnx": 13})
            self.assertEqual(metadata["inputs"][0]["shape"], ["batch", 2])
            self.assertEqual(metadata["missing_external_weights"], ["weights.bin"])

    def test_reference_and_calibration(self) -> None:
        """Preserve input order, batch boundaries, and exact float32 bytes."""
        with tempfile.TemporaryDirectory(prefix="atc-test-") as directory:
            root = Path(directory)
            inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 2]) for name in ("x", "y")]
            result = helper.make_tensor_value_info("sum", TensorProto.FLOAT, [1, 2])
            graph = helper.make_graph([helper.make_node("Add", ["x", "y"], ["sum"])], "add", inputs, [result])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8)
            onnx.save(model, root / "model.onnx")
            session = ort.InferenceSession(str(root / "model.onnx"), providers=["CPUExecutionProvider"])
            specs = describe_inputs(session)
            first = np.array([[1, 3]], dtype=np.float32)
            second = np.array([[6, 7]], dtype=np.float32)
            for index, (x, y) in enumerate(((first, second), (second, first))):
                batch = root / str(index)
                batch.mkdir()
                np.save(batch / "input_0.npy", x)
                np.save(batch / "input_1.npy", y)
            named = load_inputs(specs, [f"y={root / '0/input_1.npy'}", f"x={root / '0/input_0.npy'}"])
            output = session.run(["sum"], named)[0]
            self.assertIsInstance(output, np.ndarray)
            if isinstance(output, np.ndarray):
                save_outputs(root / "reference", {"sum": output})
            np.testing.assert_array_equal(read_outputs(root / "reference")["sum"], first + second)
            with self.assertRaisesRegex(ValueError, "missing"):
                load_inputs(specs, [f"x={root / '0/input_0.npy'}"])
            output_dir = root / "calibration"
            metadata = write_calibration(specs, [root / "0", root / "1"], output_dir)
            self.assertEqual(metadata["input_shape"], "x:1,2;y:1,2")
            self.assertEqual(metadata["atc_input_shape"], "x:2,2;y:2,2")
            self.assertEqual((output_dir / "input_0.bin").read_bytes(), first.tobytes() + second.tobytes())
            self.assertEqual((output_dir / "input_1/00000001.bin").read_bytes(), first.tobytes())

    def test_comparison_rejects_nonfinite_and_large_integer_mismatches(self) -> None:
        """Avoid rounding away integer errors or accepting non-finite values."""
        large = np.array([2**63 - 1], dtype=np.int64)
        self.assertFalse(compare(large, large - 1, 1, 1)[0])
        self.assertFalse(compare(np.array([np.nan]), np.array([np.nan]), 1, 1)[0])
        result = compare(np.array([0.0, 1.0]), np.array([0.0, 1.1]), 0.01, 0.01)
        self.assertFalse(result[0])
        self.assertIn("(1,)", result[1])


class FakeAcl(ModuleType):
    """Provide addressable allocation and a one-input, one-output ACL model."""

    def __init__(self, mode: int, execute_status: int = 0) -> None:
        """Configure the run mode and an optional ACL execute failure."""
        super().__init__("acl")
        self.init = MagicMock(return_value=0)
        self.finalize = MagicMock(return_value=0)
        self.allocations: dict[int, ctypes.Array[ctypes.c_char]] = {}
        self.rt = MagicMock()
        for operation in ("set_device", "reset_device", "destroy_context"):
            getattr(self.rt, operation).return_value = 0
        self.rt.create_context.return_value = (1, 0)
        self.rt.get_run_mode.return_value = (mode, 0)
        self.rt.malloc.side_effect = self.allocate
        self.rt.malloc_host.side_effect = self.allocate
        self.rt.free.side_effect = self.free
        self.rt.free_host.side_effect = self.free
        self.rt.memcpy.side_effect = self.copy
        self.create_data_buffer = MagicMock(side_effect=lambda address, size: address)
        self.destroy_data_buffer = MagicMock(return_value=0)
        self.mdl = MagicMock()
        for operation in ("unload", "destroy_desc", "get_desc", "destroy_dataset"):
            getattr(self.mdl, operation).return_value = 0
        self.mdl.load_from_file.return_value = (0, 0)
        self.mdl.create_desc.return_value = 1
        self.mdl.create_dataset.side_effect = (2, 3)
        self.mdl.add_dataset_buffer.side_effect = lambda dataset, buffer: (dataset, 0)
        self.mdl.execute.side_effect = self.execute
        self.execute_status = execute_status
        for direction, name in (("input", "x"), ("output", "y")):
            getattr(self.mdl, f"get_num_{direction}s").return_value = 1
            getattr(self.mdl, f"get_{direction}_name_by_index").return_value = name
            getattr(self.mdl, f"get_{direction}_dims").return_value = ({"dims": [1, 2]}, 0)
            getattr(self.mdl, f"get_{direction}_data_type").return_value = 0
            getattr(self.mdl, f"get_{direction}_format").return_value = 2
            getattr(self.mdl, f"get_{direction}_size_by_index").return_value = 8

    def allocate(self, size: int, policy: int = 0) -> tuple[int, int]:
        """Allocate backing memory for an ACL pointer."""
        memory = ctypes.create_string_buffer(size)
        address = ctypes.addressof(memory)
        self.allocations[address] = memory
        return address, 0

    def free(self, address: int) -> int:
        """Release backing memory and detect double frees."""
        del self.allocations[address]
        return 0

    def copy(self, destination: int, capacity: int, source: int, size: int, kind: int) -> int:
        """Copy bytes between mock ACL allocations."""
        if size > capacity:
            return 1
        ctypes.memmove(destination, source, size)
        return 0

    def execute(self, model: int, inputs: int, outputs: int) -> int:
        """Return the input bytes or the requested execution error."""
        if not self.execute_status:
            source, destination = list(self.allocations)[:2]
            ctypes.memmove(destination, source, 8)
        return self.execute_status


class InferenceTest(unittest.TestCase):
    """Exercise both transfer modes and error cleanup without a physical NPU."""

    def test_success_and_cleanup(self) -> None:
        """Copy input to output and release every allocation in both run modes."""
        with tempfile.TemporaryDirectory(prefix="atc-test-") as directory:
            path = Path(directory) / "x.npy"
            value = np.array([[2, 4]], dtype=np.float32)
            np.save(path, value)
            for mode in (ACL_HOST, ACL_DEVICE):
                with self.subTest(mode=mode):
                    acl = FakeAcl(mode)
                    outputs, times = infer(acl, Path("model.om"), 0, [f"x={path}"], 1, 2)
                    np.testing.assert_array_equal(outputs["y"], value)
                    self.assertEqual(len(times), 2)
                    self.assertEqual(acl.mdl.execute.call_count, 3)
                    self.assertEqual(acl.rt.memcpy.call_count, 2 if mode == ACL_HOST else 0)
                    self.assertFalse(acl.allocations)
                    acl.finalize.assert_called_once_with()

    def test_execute_failure_cleans_up(self) -> None:
        """Return the original execute error and release resources on failure."""
        with tempfile.TemporaryDirectory(prefix="atc-test-") as directory:
            path = Path(directory) / "x.npy"
            np.save(path, np.ones((1, 2), dtype=np.float32))
            acl = FakeAcl(ACL_HOST, execute_status=17)
            with self.assertRaisesRegex(RuntimeError, "ACL error 17"):
                infer(acl, Path("model.om"), 0, [f"x={path}"], 0, 1)
            self.assertFalse(acl.allocations)
            self.assertEqual(acl.destroy_data_buffer.call_count, 2)
            acl.mdl.unload.assert_called_once_with(0)
            acl.rt.reset_device.assert_called_once_with(0)
            acl.finalize.assert_called_once_with()
