from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypeAlias
import unittest

from _local_package import load_local_package

load_local_package()
from omh.plugin_bundle.omh.memory_prefetch_receipt import (
    prefetch_receipt_path, read_prefetch_receipt, validate_prefetch_receipt,
)
from omh.plugin_bundle.omh.memory_provider import OmhMemoryProvider

Json: TypeAlias = str | int | float | bool | None | list["Json"] | dict[str, "Json"]
decode: Callable[[str], Json] = json.loads


def receipt(home: Path) -> dict[str, Json]:
    home.mkdir()
    provider = OmhMemoryProvider(home)
    provider.initialize("session-a", platform="cli", cwd=str(home.parent))
    _ = provider.prefetch()
    value = decode(json.dumps(provider.latest_prefetch_receipt()))
    assert isinstance(value, dict)
    return value


class PrefetchReceiptLimitsTests(unittest.TestCase):
    def test_new_session_does_not_reuse_served_receipt_or_query(self) -> None:
        with TemporaryDirectory() as temporary:
            home = Path(temporary) / "omh"
            home.mkdir()
            provider = OmhMemoryProvider(home)
            provider.initialize("session-a", platform="cli", cwd=temporary)
            provider.queue_prefetch("Previous session query")
            _ = provider.prefetch()
            self.assertIsNotNone(provider.latest_prefetch_receipt())

            provider.initialize("session-b", platform="cli", cwd=temporary)

            self.assertIsNone(provider.latest_prefetch_receipt())
            self.assertIsNone(provider.recall_status())
            _ = provider.prefetch()
            current = decode(json.dumps(provider.latest_prefetch_receipt()))
            assert isinstance(current, dict)
            self.assertEqual(current["session_id"], "session-b")
            lens = current["lens"]
            assert isinstance(lens, dict)
            self.assertFalse(lens["query_supplied"])
            self.assertEqual(lens["query_digest"], "")

    def test_changed_body_cannot_reuse_a_valid_receipt_identity(self) -> None:
        with TemporaryDirectory() as temporary:
            home = Path(temporary) / "omh"
            original = receipt(home)
            self.assertEqual(validate_prefetch_receipt(original), [])
            original["prepared_at"] = "2000-01-01T00:00:00Z"
            self.assertIn("receipt_id", validate_prefetch_receipt(original))
            _ = prefetch_receipt_path(home).write_text(json.dumps(original))
            self.assertIsNone(read_prefetch_receipt(home))

    def test_oversized_metadata_is_refused(self) -> None:
        with TemporaryDirectory() as temporary:
            home = Path(temporary) / "omh"
            oversized = receipt(home)
            oversized["session_id"] = "s" * (64 * 1024 + 1)
            self.assertIn("size", validate_prefetch_receipt(oversized))
            _ = prefetch_receipt_path(home).write_text(json.dumps(oversized))
            self.assertIsNone(read_prefetch_receipt(home))

    def test_deep_json_is_unavailable_not_an_uncaught_parser_error(self) -> None:
        with TemporaryDirectory() as temporary:
            home = Path(temporary) / "omh"
            path = prefetch_receipt_path(home)
            path.parent.mkdir(parents=True)
            _ = path.write_text("[" * 2000 + "]" * 2000)
            self.assertIsNone(read_prefetch_receipt(home))


if __name__ == "__main__":
    unittest.main()
