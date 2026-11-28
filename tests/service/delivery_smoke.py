"""Exercise standalone operations, phased workflow logs, and OM-backed retry."""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from io import BytesIO
from typing import Any
from uuid import uuid4
from zipfile import ZipFile

import numpy as np
from onnx import TensorProto, helper
from smoke import await_job, request


def main() -> None:
    """Verify simulated API composition with real ONNX Runtime CPU references."""
    token, foreign = (
        os.environ[name] for name in ("ATC_SMOKE_TOKEN_A", "ATC_SMOKE_TOKEN_B")
    )
    assert json.loads(request("", "/readyz"))["status"] == "ready"

    def submit(
        path: str, body: dict[str, Any], status: int = 202, key: str | None = None
    ) -> dict[str, Any]:
        """Submit JSON and decode the expected response.

        Parameters
        ----------
        path, body, status, key
            Endpoint, request body, expected HTTP status, and optional submit key.

        Returns
        -------
        dict
            Public task or error response.
        """
        return json.loads(
            request(token, path, data=json.dumps(body).encode(), status=status, key=key)
        )

    def file(asset: dict[str, Any]) -> bytes:
        """Download an owned artifact and verify its advertised checksum.

        Parameters
        ----------
        asset
            Published metadata from the job.

        Returns
        -------
        bytes
            Artifact contents.
        """
        content = request(token, f"/v1/assets/{asset['id']}/download")
        assert sha256(content).hexdigest() == asset["sha256"]
        return content

    def upload_sample(name: str, data: bytes) -> str:
        """Register one script dataset member and return its owned ID.

        Parameters
        ----------
        name, data
            Dataset filename and encoded tensor bytes.

        Returns
        -------
        str
            Uploaded dataset asset ID.
        """
        return json.loads(
            request(
                token,
                f"/v1/dataset-files?filename={name}",
                data=data,
                media_type="application/octet-stream",
                status=201,
            )
        )["id"]

    source = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    result = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "delivery",
        [source],
        [result],
    )
    model_bytes = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    ).SerializeToString()
    model = json.loads(
        request(
            token,
            "/v1/assets?filename=delivery.onnx",
            data=model_bytes,
            media_type="application/octet-stream",
            status=201,
        )
    )["id"]
    await_job(
        token,
        submit("/v1/jobs", {"operation": "inspect_onnx", "input_id": model}),
        "succeeded",
    )
    samples = []
    for index in (1, 2):
        buffer = BytesIO()
        np.save(
            buffer, np.asarray([[index, index]], dtype=np.float32), allow_pickle=False
        )
        samples.append(upload_sample(f"sample{index}.npy", buffer.getvalue()))

    def atomic(operation: str, input_id: str = model, **fields: Any) -> dict[str, Any]:
        """Submit one standalone operation and wait for its artifacts.

        Parameters
        ----------
        operation, input_id, fields
            Operation name, primary input, and typed operation options.

        Returns
        -------
        dict
            Final status of the completed job.
        """
        return await_job(
            token,
            submit(
                "/v1/operations",
                {"operation": operation, "input_id": input_id, **fields},
            ),
            "succeeded",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        cpu = pool.submit(atomic, "run_cpu", samples=[[samples[0]]])
        quantization = pool.submit(
            atomic, "quantize_onnx", preset="910b2-int8", samples=[[samples[1]]]
        )
        reference = cpu.result()["outputs"][0]
        quantized = quantization.result()["outputs"][0]
    assert quantized["kind"] == "quantized_onnx" and file(quantized) == model_bytes
    assert json.loads(ZipFile(BytesIO(file(reference))).read("manifest.json")) == {
        "simulated": False
    }
    await_job(
        token,
        submit("/v1/jobs", {"operation": "inspect_onnx", "input_id": quantized["id"]}),
        "succeeded",
    )
    compiled = atomic("compile_om", quantized["id"], preset="310b4-int8")
    assert {item["kind"] for item in compiled["outputs"]} == {
        "simulated_om",
        "build_info",
    }

    base = {
        "model_id": model,
        "preset": "910b2-int8",
        "calibration": [[samples[0]]],
        "compare": True,
        "profile": True,
    }
    workflow = await_job(
        token,
        submit("/v1/workflows", {**base, "validation": [[samples[0]]]}),
        "succeeded",
    )
    assert workflow["stages"]["compile_baseline"] == "succeeded"
    assert (
        workflow["stage_times"]["compile"]["finished_at"]
        <= workflow["stage_times"]["compare"]["started_at"]
    )
    assert workflow["accuracy_summary"]["simulated"]
    logs = json.loads(request(token, f"/v1/jobs/{workflow['id']}/logs"))["text"]
    assert "Stage compile finished" in logs and "Stage profile finished" in logs
    om = next(item for item in workflow["outputs"] if item["kind"] == "simulated_om")
    build = json.loads(
        file(next(item for item in workflow["outputs"] if item["kind"] == "build_info"))
    )
    assert build["model_sha256"] == om["sha256"] and "/atc_tools-" in build["executor_image"]
    npu = atomic(
        "run_npu", primary_id=om["id"], preset="910b2-int8", samples=[[samples[0]]]
    )
    assert (
        atomic(
            "profile_om",
            primary_id=om["id"],
            preset="910b2-int8",
            samples=[[samples[0]]],
        )["outputs"][0]["kind"]
        == "profiling"
    )
    assert atomic(
        "compare_outputs", reference["id"], actual_id=npu["outputs"][0]["id"]
    )["accuracy_summary"]["simulated"]
    different = atomic("run_cpu", samples=[[samples[1]]])["outputs"][0]
    verdict = atomic("compare_outputs", reference["id"], actual_id=different["id"])[
        "accuracy_summary"
    ]
    assert verdict == {
        "simulated": False,
        "samples": 1,
        "failed": {"reference_vs_actual": 1},
        "passed": False,
    }

    bad = upload_sample("bad.npy", b"invalid")
    broken = await_job(
        token,
        submit("/v1/workflows", {**base, "compare": False, "validation": [[bad]]}),
        "succeeded",
    )
    assert (
        broken["retryable_steps"] == ["profile"]
        and broken["stages"]["profile"] == "skipped"
    )
    retry_path, retry_key = f"/v1/jobs/{broken['id']}/retry", uuid4().hex
    selection = {"step": "profile", "validation": [[samples[0]]]}
    request(foreign, retry_path, data=json.dumps(selection).encode(), status=404)
    retried = submit(retry_path, selection, key=retry_key)
    assert retried["id"] == submit(retry_path, selection, key=retry_key)["id"]
    child = await_job(token, retried, "succeeded")
    assert child["parent_job_id"] == broken["id"]
    assert set(child["stages"]) == {"prepare_validation", "profile"}
    assert [item["kind"] for item in child["outputs"]] == ["profiling"]
    assert any(
        item["id"] == child["id"] for item in json.loads(request(token, "/v1/jobs"))
    )
    print(
        "Verified standalone operations, paired phases, hashes, precision verdicts, and isolated retry with a replacement dataset."
    )


if __name__ == "__main__":
    main()
