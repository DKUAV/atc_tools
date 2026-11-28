"""Check installed runtime provenance across workflow attempts and upgrades."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
from uuid import uuid4

from atc_service.config import Settings
from atc_service.store import ResourceError, Store
from atc_service.worker import _executor_identity


class WorkerImageTest(unittest.TestCase):
    """Verify resumed work uses the tools recorded for the original OM."""

    def test_retry_requires_the_original_runtime(self) -> None:
        """Require a matching runtime in the original build manifest."""
        with TemporaryDirectory() as root:
            settings = Mock(spec=Settings, data_dir=Path(root))
            store = Mock(spec=Store)
            build_id = uuid4()
            store.build_info_for_om.return_value = build_id
            assets = Path(root) / "assets"
            assets.mkdir()
            build = assets / str(build_id)
            original = "sha256:" + "a" * 64
            job = {"image": "executor:lab"}
            pinned = patch("atc_service.worker.runtime_identity", return_value=original)
            pinned.start()
            self.addCleanup(pinned.stop)
            for data in (
                None,
                b"{}",
                b"[",
                b" " * 65537,
                b'{"executor_image":"different"}',
            ):
                with self.subTest(data=data and data[:20]):
                    if data is not None:
                        build.write_bytes(data)
                    with self.assertRaisesRegex(
                        ResourceError, "installed tools differ"
                    ):
                        _executor_identity(settings, store, job, "om")
                    store.pin_executor.assert_not_called()
            build.write_text(json.dumps({"executor_image": original}), encoding="utf-8")
            self.assertEqual(_executor_identity(settings, store, job, "om"), original)
            store.build_info_for_om.assert_called_with(job, "om")
            store.pin_executor.assert_called_once_with(job, original)
