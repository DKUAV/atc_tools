"""Exercise file registration and asset provenance against the smoke database."""

import os
from dataclasses import replace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from atc_service.config import Settings
from atc_service.store import ResourceError, Store
from atc_service.workflows import Operation, browser_presets, resolve_operation


def check_quantized_provenance(store: Store, owner: str) -> None:
    """Reject a previously simulated quantization in a real compilation request.

    Parameters
    ----------
    store
        Smoke deployment containing the inspected asset from delivery_smoke.py.
    owner
        Account that produced the simulated quantization.
    """
    with store.connect() as db:
        asset = db.execute(
            "SELECT a.id FROM assets a JOIN job_outputs o ON o.asset_id = a.id "
            "JOIN jobs j ON j.id = o.job_id WHERE a.owner = %s AND a.validated "
            "AND a.kind = 'quantized_onnx' AND j.operation = 'quantize_onnx' AND j.simulated",
            (owner,),
        ).fetchone()
    assert asset is not None, "Run the simulated delivery smoke first."
    real = replace(
        store.settings,
        simulate_compile=False,
        compile_image=store.settings.inspect_image,
        amct_targets=("310B4",),
    )
    body = Operation(operation="compile_om", input_id=asset["id"], preset="310b4-int8")
    image, simulated, workflow = resolve_operation(
        body, {preset.id: preset for preset in browser_presets(real)}, real
    )
    try:
        Store(real).submit(
            owner,
            body.operation,
            body.input_id,
            image,
            body.preset,
            simulated,
            None,
            workflow=workflow,
        )
    except ResourceError as error:
        assert error.status == 422 and "real quantization" in str(error), str(error)
    else:
        raise AssertionError("Real INT8 compilation accepted a simulated ONNX asset.")


def check_upload_rollback(store: Store, token: str) -> None:
    """Check that a rejected database insert leaves no unregistered upload.

    Parameters
    ----------
    store
        Disposable database and filesystem used by the smoke deployment.
    token
        Credential for the upload request to the API in the current container.
    """
    with store.connect() as db:
        db.execute("""
            CREATE FUNCTION reject_smoke_upload() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                RAISE EXCEPTION 'Smoke upload rejection' USING ERRCODE = '23514';
            END;
            $$
        """)
        db.execute(
            "CREATE TRIGGER reject_smoke_upload BEFORE INSERT ON assets "
            "FOR EACH ROW WHEN (NEW.filename = 'rollback-smoke.onnx') "
            "EXECUTE FUNCTION reject_smoke_upload()"
        )
    try:
        assets = store.settings.data_dir / "assets"
        before = set(assets.iterdir())
        upload = Request(
            "http://127.0.0.1:8080/v1/assets?filename=rollback-smoke.onnx",
            data=b"unregistered model",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
        )
        try:
            with urlopen(upload, timeout=10) as response:
                raise AssertionError(f"Rejected upload returned {response.status}.")
        except HTTPError as error:
            with error:
                assert error.code == 503, error.read()
        assert not set(assets.iterdir()) - before, "Rejected upload left an asset file."
    finally:
        with store.connect() as db:
            db.execute("DROP TRIGGER reject_smoke_upload ON assets")
            db.execute("DROP FUNCTION reject_smoke_upload()")


def main() -> None:
    """Run PostgreSQL checks after the simulated delivery workflow."""
    store = Store(Settings.from_env())
    check_quantized_provenance(store, os.environ["ATC_SMOKE_USER_A"])
    check_upload_rollback(store, os.environ["ATC_SMOKE_TOKEN_A"])
    print("Verified simulated quantization provenance and rejected upload cleanup.")


if __name__ == "__main__":
    main()
