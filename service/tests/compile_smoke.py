"""Compile MobileNetV2 with original precision and INT8 through the API."""

import argparse
import json
import os
from hashlib import sha256
from io import BytesIO
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
from smoke import await_job, request


def download(token: str, asset: dict[str, Any]) -> bytes:
    """Download a nonempty artifact and check its size and checksum.

    Parameters
    ----------
    token
        Job owner's bearer token.
    asset
        Public metadata for an artifact produced by the job.

    Returns
    -------
    bytes
        Verified artifact contents.
    """
    data = request(token, f"/v1/assets/{asset['id']}/download")
    assert data and len(data) == asset["size"], asset
    assert sha256(data).hexdigest() == asset["sha256"], asset
    return data


def save_log(token: str, job_id: str, path: Path) -> None:
    """Save all available pages of the job log without printing credentials.

    Parameters
    ----------
    token
        Job owner's bearer token.
    job_id
        Job whose log should be retained.
    path
        Output log path inside the CI artifact directory.
    """
    offset = 0
    with path.open("w", encoding="utf-8") as log:
        while True:
            part = json.loads(request(token, f"/v1/jobs/{job_id}/logs?offset={offset}"))
            log.write(part["text"])
            if part["next_offset"] <= offset:
                break
            offset = part["next_offset"]


def main() -> None:
    """Inspect and compile the pinned model, saving diagnostics for both targets."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("reports", type=Path)
    args = parser.parse_args()
    token = os.environ["ATC_SMOKE_TOKEN_A"]
    args.reports.mkdir(parents=True, exist_ok=True)
    model = args.model.read_bytes()
    uploaded = json.loads(
        request(
            token,
            "/v1/assets?filename=mobilenetv2-12.onnx",
            data=model,
            media_type="application/octet-stream",
            status=201,
        )
    )
    assert uploaded["sha256"] == sha256(model).hexdigest(), uploaded
    model_id = uploaded["id"]
    inspect_job = json.loads(
        request(
            token,
            "/v1/jobs",
            data=json.dumps(
                {"operation": "inspect_onnx", "input_id": model_id}
            ).encode(),
            status=202,
        )
    )
    try:
        await_job(token, inspect_job, "succeeded")
    finally:
        save_log(token, inspect_job["id"], args.reports / "inspection.log")

    # Fix the model's dynamic batch axis and generate repeatable calibration inputs.
    shape = (1, 3, 224, 224)
    rng = np.random.default_rng(42)
    calibration = []
    for index in range(4):
        sample = BytesIO()
        np.save(sample, rng.standard_normal(shape, dtype=np.float32))
        asset = json.loads(
            request(
                token,
                f"/v1/dataset-files?filename=calibration-{index}.npy",
                data=sample.getvalue(),
                media_type="application/octet-stream",
                status=201,
            )
        )
        calibration.append([asset["id"]])

    for precision, target in product(("origin", "int8"), ("310B4", "910B2")):
        preset = f"{target.lower()}-{precision}"
        samples = calibration if precision == "int8" else []
        job = json.loads(
            request(
                token,
                "/v1/workflows",
                data=json.dumps(
                    {
                        "model_id": model_id,
                        "preset": preset,
                        "shapes": {"0": shape},
                        "calibration": samples,
                    }
                ).encode(),
                status=202,
            )
        )
        try:
            assert not job["simulated"], job
            result = await_job(token, job, "succeeded", timeout=960)
            stages = result["stages"]
            assert stages["compile"] == "succeeded", result
            if precision == "int8":
                assert (
                    stages["prepare_calibration"] == stages["quantize"] == "succeeded"
                ), result
            files = {asset["kind"]: asset for asset in result["outputs"]}
            artifact = download(token, files["om"])
            info = download(token, files["build_info"])
            (args.reports / f"{preset}-build.json").write_bytes(info)
            build = json.loads(info)
            assert build["target"] == f"Ascend{target}" and not build["simulated"], (
                build
            )
            assert build["preset"] == preset, build
            assert build["precision"] == precision, build
            assert build["calibration_batches"] == len(samples), build
            assert build["model_sha256"] == sha256(artifact).hexdigest(), build
            assert build["input_shapes"] == {"input": list(shape)}, build
            print(f"MobileNetV2 compiled with {preset}, OM size {len(artifact)} bytes.")
        finally:
            save_log(token, job["id"], args.reports / f"{preset}-compile.log")


if __name__ == "__main__":
    main()
