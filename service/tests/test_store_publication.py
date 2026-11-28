"""Keep asset files when the outcome of a database commit is uncertain."""

import unittest
from unittest.mock import MagicMock, Mock, patch

from atc_service.config import Settings
from atc_service.store import Store
from psycopg import OperationalError


class PublicationTest(unittest.TestCase):
    """Check whether a publication failure occurred before or during commit."""

    def test_commit_error_stays_distinct_from_rollback(self) -> None:
        """Propagate a commit error so callers do not discard the asset file."""
        store = Store(Mock(spec=Settings))
        connection = MagicMock()
        connection.__enter__.return_value = connection
        connection.__exit__.side_effect = OperationalError("Connection lost at commit")
        with patch.object(store, "connect", return_value=connection):
            with self.assertRaisesRegex(OperationalError, "Connection lost at commit"):
                with store._publication() as active:
                    self.assertIs(active, connection)
        connection.__exit__.assert_called_once()
        self.assertEqual(connection.__exit__.call_args.args[-3:], (None, None, None))
