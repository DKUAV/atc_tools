"""Run a single service task in an isolated execution container."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from google.protobuf.message import DecodeError

from atc_tools.model.onnx import inspect_onnx


def main() -> int:
    """Execute the request in the input mount and write the output manifest.

    Returns
    -------
    int
        Zero for a complete result or one for a failed operation.
    """
    inputs = Path("/work/input")
    output = Path("/work/output")
    model = inputs / "model.onnx"
    try:
        request = json.loads((inputs / "request.json").read_text(encoding="utf-8"))
        if request["version"] != 1:
            raise ValueError("Unsupported task protocol.")
        if request.get("workflow") is not None:
            from atc_tools.workflow import run_workflow

            run_workflow(inputs, output, request)
            return 0
        if request["operation"] == "inspect_onnx":
            print("Inspecting ONNX graph and embedded weights.", flush=True)
            metadata = inspect_onnx(model, require_embedded=True)
            (output / "inspection.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            artifacts = ["inspection.json"]
            print(
                f"Inspected {len(metadata['inputs'])} inputs and {len(metadata['outputs'])} outputs."
            )
        elif request["operation"] == "compile_om":
            simulated = request["simulated"]
            print(
                "Simulating Ascend310B4 compilation."
                if simulated
                else "Compiling for Ascend310B4.",
                flush=True,
            )
            if simulated:
                artifact = "model.simulated"
                (output / artifact).write_text(
                    "Simulated Ascend310B4 compilation output for service testing.\n",
                    encoding="utf-8",
                )
            else:
                artifact = "model.om"
                script = Path("/opt/atc-tools/scripts/compile.sh")
                target = output / "compile"
                subprocess.run(
                    [
                        "bash",
                        str(script),
                        "--model",
                        str(model),
                        "--soc",
                        "Ascend310B4",
                        "--output-dir",
                        str(target),
                    ],
                    check=True,
                )
                (target / artifact).replace(output / artifact)
            build = {
                "target": "Ascend310B4",
                "preset": request["preset"],
                "simulated": simulated,
            }
            (output / "build.json").write_text(
                json.dumps(build, indent=2), encoding="utf-8"
            )
            artifacts = [artifact, "build.json"]
        else:
            raise ValueError("Unsupported task operation.")
        (output / "result.json").write_text(
            json.dumps({"version": 1, "artifacts": artifacts}), encoding="utf-8"
        )
    except (
        OSError,
        ValueError,
        RuntimeError,
        DecodeError,
        subprocess.CalledProcessError,
    ) as error:
        print(f"Task failed: {error}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
