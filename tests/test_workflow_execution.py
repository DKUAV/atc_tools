"""Verify the real INT8 workflow contract with controlled vendor commands."""

import json
import subprocess
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pytest
from onnx import TensorProto, helper, save

from atc_tools.model.tensor_io import save_outputs
from atc_tools.workflow import run_workflow


def test_real_workflow_build_and_verify(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use separate compile and device phases and publish both INT8 verdicts.

    Parameters
    ----------
    tmp_path, monkeypatch
        Private task paths and replacements for the Ascend subprocesses.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    inputs, build, verify = (tmp_path / name for name in ("input", "build", "verify"))
    for folder in (inputs, inputs / "dataset", build, verify):
        folder.mkdir()
    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "workflow",
        [value],
        [output],
    )
    save(
        helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
        ),
        inputs / "model.onnx",
    )
    np.save(
        inputs / "dataset" / "batch.npy",
        np.asarray([[1, 1]], dtype=np.float32),
        allow_pickle=False,
    )
    commands: list[str] = []

    def run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        """Produce the expected vendor artifacts from the fixed command contract.

        Parameters
        ----------
        command
            Argument vector supplied by the workflow.
        **_kwargs
            Subprocess control options.

        Returns
        -------
        subprocess.CompletedProcess
            Successful synthetic exit status.
        """
        if command[0] == "bash":
            destination = Path(command[command.index("--output-dir") + 1])
            destination.mkdir()
            if command[1].endswith("quantize_model.sh"):
                commands.append("quantize")
                assert "input:1,2" in command and "float32" in command
                quantized = (inputs / "model.onnx").read_bytes()
                (destination / "model_deploy_model.onnx").write_bytes(quantized)
            else:
                assert "Ascend910B2" in command
                is_baseline = "--precision_mode_v2=origin" in command
                commands.append("baseline" if is_baseline else "compile")
                (destination / "model.om").write_bytes(
                    b"baseline" if is_baseline else b"int8"
                )
        elif command[0] == "python3.11":
            commands.append("infer")
            model = Path(command[3]).read_bytes()
            outputs = Path(command[-1]) / "00000000"
            save_outputs(
                outputs,
                {
                    "output": np.asarray(
                        [[1, 1] if model == b"baseline" else [3, 3]], dtype=np.float32
                    )
                },
            )
        else:
            assert command[0] == "msprof"
            commands.append("profile")
            profile = Path(
                next(
                    arg.split("=", 1)[1]
                    for arg in command
                    if arg.startswith("--output=")
                )
            )
            profile.mkdir()
            (profile / "summary.csv").write_text(
                "name,duration\npass,1\n", encoding="utf-8"
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr("atc_tools.workflow.subprocess.run", run)
    request = {
        "simulated": False,
        "preset": "910b2-int8",
        "limit_bytes": 1024**2,
        "workflow": {
            "target": "910B2",
            "precision": "int8",
            "calibration": [["batch"]],
            "validation": [["batch"]],
            "compare": True,
            "profile": True,
            "shapes": {},
            "preparation": {
                "layout": "auto",
                "geometry": "keep",
                "values": "raw",
                "raw_inputs": {},
            },
        },
    }
    run_workflow(inputs, build, {**request, "phase": "build"})
    assert commands == ["quantize", "compile", "baseline"]
    (inputs / "primary.om").write_bytes((build / "model.om").read_bytes())
    (inputs / "baseline.om").write_bytes((build / "baseline.om").read_bytes())
    run_workflow(inputs, verify, {**request, "phase": "verify"})
    assert commands == ["quantize", "compile", "baseline", "infer", "infer", "profile"]
    assert (
        json.loads((verify / "accuracy-onnx-vs-int8.json").read_text(encoding="utf-8"))[
            "failed_samples"
        ]
        == 1
    )
    assert (
        json.loads((verify / "accuracy-fp-vs-int8.json").read_text(encoding="utf-8"))[
            "failed_samples"
        ]
        == 1
    )
    assert (
        json.loads((verify / "accuracy-summary.json").read_text(encoding="utf-8"))[
            "passed"
        ]
        is False
    )
    with ZipFile(verify / "profiling.zip") as archive:
        profile = json.loads(archive.read("profiling.json"))
        assert profile["samples"] == 1
        assert profile["simulated"] is False
        assert profile["collection_wall_seconds"] >= 0
        assert profile["collection_files"] == 1
        assert profile["collection_bytes"] == len(archive.read("summary.csv"))
