"""Check Docker executor selection across workflow attempts and upgrades."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, call
from uuid import uuid4

from atc_service.config import Settings
from atc_service.store import ResourceError, Store
from atc_service.worker import _executor_image
from docker.errors import ImageNotFound


class WorkerImageTest(unittest.TestCase):
    """Check initial image pinning and verification against previously built OMs."""

    def test_retag_after_build(self) -> None:
        """A verification attempt uses the persisted ID even after its tag moves."""
        tag = "executor:lab"
        original = "sha256:" + "a" * 64
        replacement = "sha256:" + "b" * 64
        images = {
            tag: SimpleNamespace(id=original),
            original: SimpleNamespace(id=original),
        }
        client = Mock()
        client.images.get.side_effect = images.__getitem__
        store = Mock(spec=Store)
        settings = Mock(spec=Settings)
        job = {"image": tag}
        self.assertEqual(_executor_image(settings, store, client, job, None), original)
        store.pin_executor.assert_called_once_with(job, original)
        images[tag] = SimpleNamespace(id=replacement)
        resumed = {"image": store.pin_executor.call_args.args[1]}
        self.assertEqual(
            _executor_image(settings, store, client, resumed, "om"), original
        )
        self.assertEqual(client.images.get.call_args_list, [call(tag), call(original)])
        store.build_info_for_om.assert_not_called()
        client.images.get.side_effect = ImageNotFound("pruned image")
        with self.assertRaisesRegex(ResourceError, "executor image is unavailable"):
            _executor_image(settings, store, client, resumed, "om")
        client.images.get.assert_called_with(original)

    def test_legacy_build_confirms_only_the_configured_image(self) -> None:
        """Legacy metadata can confirm a trusted tag but cannot choose another image."""
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
            client = Mock()
            client.images.get.return_value.id = original
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
                        ResourceError, "verify the original executor"
                    ):
                        _executor_image(settings, store, client, job, "om")
                    client.images.get.assert_called_once_with("executor:lab")
                    client.images.get.reset_mock()
                    store.pin_executor.assert_not_called()
            build.write_text(json.dumps({"executor_image": original}), encoding="utf-8")
            self.assertEqual(
                _executor_image(settings, store, client, job, "om"), original
            )
            store.build_info_for_om.assert_called_with(job, "om")
            store.pin_executor.assert_called_once_with(job, original)
