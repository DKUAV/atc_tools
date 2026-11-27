"""Check the deployed API, database, and worker with two independent accounts."""

from __future__ import annotations

import json
import os
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4

from onnx import TensorProto, helper


def request(
    token: str,
    route: str,
    *,
    data: bytes | None = None,
    media_type: str = "application/json",
    status: int = 200,
    key: str | None = None,
) -> bytes:
    """Call the smoke-test API without printing credentials.

    Parameters
    ----------
    token
        User bearer token, or an empty string for an anonymous request.
    route
        Path and query to append to the configured service URL.
    data
        POST body. An omitted body selects GET.
    media_type
        Content type of the POST body.
    status
        Expected HTTP status.
    key
        Optional submission idempotency key.

    Returns
    -------
    bytes
        Response body for further assertions.
    """
    headers = {"Content-Type": media_type}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if key:
        headers["Idempotency-Key"] = key
    url = os.environ.get("ATC_SMOKE_URL", "http://127.0.0.1:8766") + route
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(
            Request(url, data=data, headers=headers), timeout=10
        ) as response:
            assert response.status == status, (route, response.status)
            return response.read()
    except HTTPError as error:
        body = error.read()
        assert error.code == status, (route, error.code, body)
        return body


def await_job(token: str, job: dict[str, Any], expected: str) -> dict[str, Any]:
    """Wait for a submitted job to reach its expected terminal state.

    Parameters
    ----------
    token
        Job owner's bearer token.
    job
        Accepted job response.
    expected
        Expected terminal job state.

    Returns
    -------
    dict
        Completed job record.
    """
    deadline = monotonic() + 60
    while monotonic() < deadline:
        result = json.loads(request(token, f"/v1/jobs/{job['id']}"))
        if result["state"] in ("succeeded", "failed", "cancelled"):
            assert result["state"] == expected, result
            return result
        sleep(0.25)
    raise AssertionError(f"Job did not finish: {job['id']}")


def main() -> None:
    """Verify upload ownership, task states, logs, and artifact delivery."""
    alice = os.environ["ATC_SMOKE_TOKEN_A"]
    bob = os.environ["ATC_SMOKE_TOKEN_B"]
    request("", "/v1/capabilities", status=401)
    available = json.loads(request(alice, "/v1/capabilities"))
    assert {
        "id": "310b4-simulation",
        "target": "310B4",
        "simulated": True,
    } in available["compile_presets"]

    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    result = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "smoke",
        [value],
        [result],
    )
    model = helper.make_model(graph).SerializeToString()
    upload_route = "/v1/assets?filename=small.onnx"
    opaque = "application/octet-stream"
    valid = json.loads(
        request(alice, upload_route, data=model, media_type=opaque, status=201)
    )
    assert request(alice, f"/v1/assets/{valid['id']}/download") == model
    request(bob, f"/v1/assets/{valid['id']}/download", status=404)
    inspection = json.dumps(
        {"operation": "inspect_onnx", "input_id": valid["id"]}
    ).encode()
    compile_body = json.dumps(
        {
            "operation": "compile_om",
            "input_id": valid["id"],
            "preset": "310b4-simulation",
        }
    ).encode()
    request(alice, "/v1/jobs", data=compile_body, status=409)
    request(bob, "/v1/jobs", data=inspection, status=404)
    key = uuid4().hex
    first = json.loads(request(alice, "/v1/jobs", data=inspection, key=key, status=202))
    assert (
        json.loads(request(alice, "/v1/jobs", data=inspection, key=key, status=202))[
            "id"
        ]
        == first["id"]
    )

    bad = json.loads(
        request(bob, upload_route, data=b"invalid ONNX", media_type=opaque, status=201)
    )
    bad_body = json.dumps({"operation": "inspect_onnx", "input_id": bad["id"]}).encode()
    second = json.loads(request(bob, "/v1/jobs", data=bad_body, status=202))
    await_job(bob, second, "failed")
    bad_log = json.loads(request(bob, f"/v1/jobs/{second['id']}/logs"))
    assert "Task failed:" in bad_log["text"]

    inspected = await_job(alice, first, "succeeded")
    assert (
        json.loads(
            request(alice, f"/v1/assets/{inspected['outputs'][0]['id']}/download")
        )["inputs"][0]["name"]
        == "input"
    )
    request(bob, f"/v1/jobs/{first['id']}/logs", status=404)
    request(bob, f"/v1/jobs/{first['id']}/cancel", data=b"", status=404)
    compile_job = json.loads(request(alice, "/v1/jobs", data=compile_body, status=202))
    compiled = await_job(alice, compile_job, "succeeded")
    outputs = {asset["kind"]: asset for asset in compiled["outputs"]}
    assert set(outputs) == {"simulated_om", "build_info"}
    assert b"Simulated Ascend310B4" in request(
        alice, f"/v1/assets/{outputs['simulated_om']['id']}/download"
    )
    log = json.loads(request(alice, f"/v1/jobs/{compiled['id']}/logs"))
    assert "Simulating Ascend310B4" in log["text"]
    assert (
        json.loads(
            request(
                alice, f"/v1/jobs/{compiled['id']}/logs?offset={log['next_offset']}"
            )
        )["text"]
        == ""
    )
    print(
        "Verified ONNX inspection, isolation, duplicate submission, failure logs, and simulated OM download."
    )


if __name__ == "__main__":
    main()
