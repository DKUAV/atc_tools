"""Exercise the deployed workbench API with separate browser sessions."""

import json
import os
from http.cookiejar import CookieJar
from io import BytesIO
from pathlib import Path
from time import monotonic, sleep
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener
from uuid import uuid4

import cv2
import numpy as np
from onnx import TensorProto, helper


class Browser:
    """Keep the cookie and CSRF state of one test browser."""

    def __init__(self) -> None:
        """Start a test browser with an empty HTTP cookie jar."""
        self.opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(CookieJar()))
        self.url = os.environ.get("ATC_SMOKE_URL", "http://127.0.0.1:8766")
        self.csrf = ""

    def call(
        self,
        path: str,
        status: int = 200,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        """Request a resource and verify the returned HTTP status.

        Parameters
        ----------
        path
            Path on the workbench origin.
        status
            Expected response status.
        data
            POST body, or None for GET.
        headers
            Request headers including any CSRF token.

        Returns
        -------
        bytes
            Response body.
        """
        try:
            with self.opener.open(
                Request(self.url + path, data, headers or {}), timeout=10
            ) as response:
                assert response.status == status, (path, response.status)
                return response.read()
        except HTTPError as error:
            assert error.code == status, (path, error.code)
            return error.read()

    def login(self, name: str, password: str) -> None:
        """Authenticate the browser and read its CSRF token.

        Parameters
        ----------
        name
            Provisioned test account.
        password
            Account password.
        """
        self.call(
            "/web/api/login",
            204,
            json.dumps({"username": name, "password": password}).encode(),
            {"Content-Type": "application/json", "X-ATC-Login": "1"},
        )
        session = json.loads(self.call("/web/api/session"))
        assert session["username"] == name
        assert session["compile_presets"][0]["id"] == "310b4-simulation"
        self.csrf = session["csrf"]

    def wait(self, job: dict[str, Any]) -> dict[str, Any]:
        """Wait for a successful test job.

        Parameters
        ----------
        job
            Submitted task response.

        Returns
        -------
        dict
            Completed task with its published files.
        """
        deadline = monotonic() + 60
        while monotonic() < deadline:
            current = json.loads(self.call(f"/web/api/tasks/{job['id']}"))
            if current["state"] not in ("queued", "running"):
                assert current["state"] == "succeeded", current
                return current
            sleep(0.25)
        raise AssertionError(f"Task did not finish: {job['id']}")


def main() -> None:
    """Check conversion, session isolation, idempotency, and artifact cleanup."""
    alice, bob = Browser(), Browser()
    alice_name = os.environ["ATC_SMOKE_USER_A"]
    alice_password = os.environ["ATC_SMOKE_PASSWORD_A"]
    assert b"workbench.js" in alice.call("/")
    assert b"compile-form" in alice.call("/web/static/workbench.js")
    alice.call("/web/api/session", 401)
    alice.call(
        "/web/api/login",
        403,
        json.dumps({"username": alice_name, "password": alice_password}).encode(),
        {"Content-Type": "application/json"},
    )
    alice.login(alice_name, alice_password)
    bob.login(os.environ["ATC_SMOKE_USER_B"], os.environ["ATC_SMOKE_PASSWORD_B"])
    alice.call("/v1/capabilities", 401)

    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])
    result = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "smoke",
        [value],
        [result],
    )
    model = helper.make_model(graph).SerializeToString()
    upload = "/web/api/models?filename=workbench-smoke.onnx"
    alice.call(upload, 403, model, {"Content-Type": "application/octet-stream"})
    inspection = alice.wait(
        json.loads(
            alice.call(
                upload,
                202,
                model,
                {"Content-Type": "application/octet-stream", "X-ATC-CSRF": alice.csrf},
            )
        )
    )
    report = inspection["outputs"][0]["id"]
    assert json.loads(alice.call(f"/web/api/files/{report}"))["inputs"][0] == {
        "name": "input",
        "type": "FLOAT",
        "shape": [1, 2],
    }
    bob.call(f"/web/api/tasks/{inspection['id']}", 404)
    bob.call(f"/web/api/files/{report}", 404)
    same_account = Browser()
    same_account.login(alice_name, alice_password)
    same_account.call(f"/web/api/files/{report}", 404)

    body = json.dumps(
        {"model_id": inspection["input_id"], "preset": "310b4-simulation"}
    ).encode()
    path = "/web/api/conversions"
    headers = {"Content-Type": "application/json", "Idempotency-Key": uuid4().hex}
    same_account.call(path, 404, body, {**headers, "X-ATC-CSRF": same_account.csrf})
    headers["X-ATC-CSRF"] = alice.csrf
    first = json.loads(alice.call(path, 202, body, headers))
    duplicate = json.loads(alice.call(path, 202, body, headers))
    assert first["id"] == duplicate["id"]
    job = alice.wait(first)
    artifact = next(
        file["id"] for file in job["outputs"] if file["kind"] == "simulated_om"
    )
    assert b"Simulated Ascend310B4" in alice.call(f"/web/api/files/{artifact}")

    alice.call("/web/api/logout", 204, b"", {"X-ATC-CSRF": alice.csrf})
    alice.call(f"/web/api/files/{artifact}", 401)
    same_account.call(f"/web/api/files/{artifact}", 404)
    detail = json.loads(same_account.call(f"/web/api/tasks/{job['id']}"))
    assert detail["state"] == "succeeded" and not detail["outputs"]
    assert b"Simulating Ascend310B4" in same_account.call(
        f"/web/api/tasks/{job['id']}/logs"
    )
    assert any(
        row["id"] == job["id"]
        for row in json.loads(same_account.call("/web/api/tasks"))
    )
    if data_dir := os.environ.get("ATC_SMOKE_DATA_DIR"):
        ids = [inspection["input_id"], report, *(file["id"] for file in job["outputs"])]
        deadline = monotonic() + 30
        while monotonic() < deadline and any(
            (Path(data_dir) / "assets" / uuid).exists() for uuid in ids
        ):
            sleep(0.25)
        assert all(not (Path(data_dir) / "assets" / uuid).exists() for uuid in ids)
    same_account.call("/web/api/logout", 204, b"", {"X-ATC-CSRF": same_account.csrf})
    bob.call("/web/api/logout", 204, b"", {"X-ATC-CSRF": bob.csrf})
    print("Browser workflow, owner isolation, idempotency and cleanup passed.")

    alice.login(alice_name, alice_password)
    bob.login(os.environ["ATC_SMOKE_USER_B"], os.environ["ATC_SMOKE_PASSWORD_B"])
    check_dataset_workflow(alice, bob)
    alice.call("/web/api/logout", 204, b"", {"X-ATC-CSRF": alice.csrf})
    bob.call("/web/api/logout", 204, b"", {"X-ATC-CSRF": bob.csrf})


def check_dataset_workflow(alice: Browser, bob: Browser) -> None:
    """Verify mixed-format preparation and independently selectable 910B2 checks.

    Parameters
    ----------
    alice, bob
        Authenticated but isolated browser accounts.
    """
    options = json.loads(alice.call("/web/api/session"))["workflow_presets"]
    assert {item["target"] for item in options} == {"310B4", "910B2"}
    selected = [item for item in options if item["id"] == "910b2-int8"]
    assert len(selected) == 1 and selected[0]["checks"] and selected[0]["simulated"]
    value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 5])
    result = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 4, 5])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "datasets",
        [value],
        [result],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=9
    )
    upload = {"Content-Type": "application/octet-stream", "X-ATC-CSRF": alice.csrf}
    inspection = alice.wait(
        json.loads(
            alice.call(
                "/web/api/models?filename=datasets.onnx",
                202,
                model.SerializeToString(),
                upload,
            )
        )
    )

    success, image = cv2.imencode(".png", np.full((2, 3, 3), 128, dtype=np.uint8))
    assert success
    batch = np.full((1, 3, 2, 3), 128, dtype="<f4")
    validation = BytesIO()
    np.save(validation, batch, allow_pickle=False)

    def put(filename: str, data: bytes) -> str:
        """Upload a sample file and return its asset ID.

        Parameters
        ----------
        filename
            Nominal supported file name.
        data
            Encoded sample bytes.

        Returns
        -------
        str
            ID of the session-owned dataset file.
        """
        return json.loads(
            alice.call(f"/web/api/dataset-files?filename={filename}", 201, data, upload)
        )["id"]

    calibration_png = put("sample.png", image.tobytes())
    calibration_bin = put("sample.bin", batch.tobytes())
    validation_npy = put("sample.npy", validation.getvalue())
    bob.call(f"/web/api/files/{validation_npy}", 404)
    body = {
        "model_id": inspection["input_id"],
        "preset": "910b2-int8",
        "compare": True,
        "profile": True,
        "calibration": [[calibration_png], [calibration_bin]],
        "validation": [[validation_npy]],
        "preparation": {
            "geometry": "pad",
            "layout": "nchw",
            "fill": 114,
            "values": "unit",
            "raw_inputs": {"0": {"shape": [1, 3, 2, 3], "dtype": "float32"}},
        },
    }
    headers = {
        "Content-Type": "application/json",
        "X-ATC-CSRF": alice.csrf,
        "Idempotency-Key": uuid4().hex,
    }
    path = "/web/api/conversions"
    alice.call(
        path, 422, json.dumps({**body, "preset": "310b4-int8"}).encode(), headers
    )
    alice.call(path, 422, json.dumps({**body, "calibration": []}).encode(), headers)
    job = alice.wait(
        json.loads(alice.call(path, 202, json.dumps(body).encode(), headers))
    )
    assert job["stages"] == {
        "prepare_calibration": "succeeded",
        "quantize": "succeeded",
        "compile": "succeeded",
        "prepare_validation": "succeeded",
        "compare": "succeeded",
        "profile": "succeeded",
    }
    outputs = {asset["filename"]: asset["id"] for asset in job["outputs"]}
    assert set(outputs) == {
        "model.simulated",
        "build.json",
        "accuracy.json",
        "profiling.zip",
    }
    report = json.loads(alice.call(f"/web/api/files/{outputs['accuracy.json']}"))
    assert report["simulated"] and report["samples"][0]["cpu_outputs"]["output"] == [
        1,
        3,
        4,
        5,
    ]
    assert b"PK" == alice.call(f"/web/api/files/{outputs['profiling.zip']}")[:2]
    build = json.loads(alice.call(f"/web/api/files/{outputs['build.json']}"))
    assert build["target"] == "Ascend910B2" and build["calibration_batches"] == 2

    bad = put("invalid.npy", b"invalid format")
    headers["Idempotency-Key"] = uuid4().hex
    optional_failure = {
        **body,
        "preset": "910b2-origin",
        "compare": False,
        "calibration": [],
        "validation": [[bad]],
    }
    result_job = alice.wait(
        json.loads(
            alice.call(path, 202, json.dumps(optional_failure).encode(), headers)
        )
    )
    assert result_job["stages"] == {
        "compile": "succeeded",
        "prepare_validation": "failed",
        "profile": "skipped",
    }
    assert {asset["filename"] for asset in result_job["outputs"]} == {
        "model.simulated",
        "build.json",
    }
    print(
        "910B2 and INT8 simulation, image/NumPy/raw input, and optional failure preservation passed."
    )


if __name__ == "__main__":
    main()
