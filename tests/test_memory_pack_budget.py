"""The whole prefetch pack fits under the cap Hermes spills it at.

Hermes keeps only the first and last 500 characters of a provider prefetch
result longer than `DEFAULT_MAX_CHARS` (hermes-agent
`tools/hook_output_spill.py`, applied in `agent/memory_manager.py`), so the
records in the middle of an oversized pack vanish while the receipt, the
`served.json` counts and the reminder ledger still say they were served. Each
section had its own cap and the pack had none. These pin:

- the copied cap against the Hermes source, read with `ast`, never imported;
- a pack budget at least 1,000 characters under that cap;
- a worst-case store whose sections overflow the cap at their own caps, served
  within the budget with every record, every closing tag and the brief's
  `<do>` tail intact, and a receipt naming exactly what the text carries;
- what gives, in order: system blocks, then index lines, then demote rows.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from _local_package import load_local_package

load_local_package()
from omh.plugin_bundle.omh import memory_provider  # noqa: E402
from omh.plugin_bundle.omh.memory_blocks import (  # noqa: E402
    DEFAULT_SYSTEM_RENDER_BUDGET_CHARS,
    approve_memory_block,
    build_memory_block,
    read_memory_blocks,
    render_block_index,
    render_memory_blocks,
    select_memory_blocks,
    write_memory_block,
)
from omh.plugin_bundle.omh.memory_dreaming import read_latest_consolidation  # noqa: E402
from omh.plugin_bundle.omh.memory_prefetch_receipt import validate_prefetch_receipt  # noqa: E402
from omh.plugin_bundle.omh.memory_provider import (  # noqa: E402
    HERMES_PREFETCH_SPILL_MAX_CHARS,
    PACK_BUDGET_CHARS,
    REFERENCE_INDEX_PACK_MAX_CHARS,
    SYSTEM_BLOCKS_PACK_FLOOR_CHARS,
    render_consolidation_brief,
)
from project_identity_fixture import memory_paths  # noqa: E402
from test_memory_demote_tool import entry, write_memory  # noqa: E402
from test_memory_open_state import _approved, _open_record  # noqa: E402
from test_memory_prefetch_canonical import provider, rendered_ids  # noqa: E402
from test_memory_provider_hermes_order import hermes_turn  # noqa: E402
from test_memory_session_digest import DIGEST, ago, set_session_marks, stamp, write_record  # noqa: E402

SECTION_TAGS = ("memory_blocks", "memory_block_index", "memory_records", "memory_digest", "memory_consolidation")
QUERY = "deploy release cache"


def _hermes_spill_source() -> Path | None:
    """`tools/hook_output_spill.py` of a Hermes checkout, located without importing Hermes."""
    roots = [Path(os.environ["HERMES_AGENT_ROOT"])] if os.environ.get("HERMES_AGENT_ROOT") else []
    roots.append(Path.home() / ".hermes" / "hermes-agent")
    for root in roots:
        candidate = root / "tools" / "hook_output_spill.py"
        if candidate.is_file():
            return candidate
    return None


class HermesSpillCapPinTests(unittest.TestCase):
    """`HERMES_PREFETCH_SPILL_MAX_CHARS` is copied from Hermes; this is where the copy is checked."""

    def test_the_copied_cap_matches_the_hermes_source(self) -> None:
        source = _hermes_spill_source()
        if source is None:
            self.skipTest("no Hermes checkout at $HERMES_AGENT_ROOT or ~/.hermes/hermes-agent; tools/hook_output_spill.py cannot be compared")
        values = [
            node.value
            for node in ast.parse(source.read_text(encoding="utf-8")).body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            and any(isinstance(target, ast.Name) and target.id == "DEFAULT_MAX_CHARS" for target in (node.targets if isinstance(node, ast.Assign) else [node.target]))
        ]
        self.assertEqual(len(values), 1, f"{source} no longer assigns DEFAULT_MAX_CHARS once at module level")
        self.assertEqual(ast.literal_eval(values[0]), HERMES_PREFETCH_SPILL_MAX_CHARS)

    def test_the_pack_budget_stays_clear_of_the_cap(self) -> None:
        self.assertEqual(HERMES_PREFETCH_SPILL_MAX_CHARS, 10_000)
        self.assertLessEqual(PACK_BUDGET_CHARS, HERMES_PREFETCH_SPILL_MAX_CHARS - 1000)


def _words(seed: str, length: int) -> str:
    """Distinct prose of exactly ``length`` characters, so no two summaries collapse as duplicates."""
    text = ""
    index = 0
    while len(text) < length:
        text += f"{seed} {index} keeps the deploy release cache note for later sessions. "
        index += 1
    return text[:length].rstrip() + "."


class WorstCasePackTests(unittest.TestCase):
    """Every section at its own cap: big system blocks, a long index, full records, a reminder, a digest and a brief with demote rows."""

    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.store = self.root / ".omh"
        self.hermes = self.root / ".hermes"
        paths = memory_paths(self.store, self.hermes)
        # Records: more than the selector keeps, each near the summary cap.
        self.records = [_approved(paths, _words(f"Record {index}", 480))["record_id"] for index in range(8)]
        # The reminder: an open record past its deadline, summary at the cap.
        self.open_record = _open_record(paths, _words("Open question about the deploy cache", 500), days_open=34)["record_id"]
        # System blocks far past the section cap; a long reference index.
        for index in range(5):
            write_memory_block(self.store, approve_memory_block(build_memory_block(f"system{index}", _words(f"Block {index}", 1990), description=_words(f"System block {index}", 120))))
        for index in range(30):
            write_memory_block(
                self.store,
                approve_memory_block(build_memory_block(f"reference{index}", "value", description=_words(f"Reference runbook {index}", 140), tier="reference")),
            )
        # The digest: rows written after the last recorded close.
        set_session_marks(self.store, last_session_closed_at=ago(hours=3))
        for index in range(3):
            write_record(self.store, f"rec-digest-{index}", created_at=stamp(ago(hours=2)), source="hermes_model", scope={"kind": "user-global", "ref": "default"})
        # The brief: Hermes memory nearly full, so its demote rows ride along.
        write_memory(self.hermes, "MEMORY.md", *(entry(index) for index in range(10)))
        write_memory(self.hermes, "USER.md", *(entry(index) for index in range(6)))
        self.live = provider(self.root)

    def sections(self) -> dict[str, str]:
        """Each section the provider can render on its own, at its own cap, as before the pack budget."""
        blocks = read_memory_blocks(self.store)
        evaluations = select_memory_blocks(blocks, omh_home=self.store).evaluations
        return {
            "system": render_memory_blocks([block for block in blocks if block.tier == "system"], budget_chars=DEFAULT_SYSTEM_RENDER_BUDGET_CHARS, evaluations=evaluations),
            "index": render_block_index([block for block in blocks if block.tier == "reference"], evaluations=evaluations),
            "brief": render_consolidation_brief(read_latest_consolidation(self.store), hermes_home=self.hermes),
        }

    def test_the_fixture_overflows_the_spill_cap_at_the_old_section_caps(self) -> None:
        pack = hermes_turn(self.live, 1, QUERY)
        old = self.sections()
        records = re.search(r"<memory_records>.*?</memory_records>", pack, re.DOTALL)
        reminder = re.search(r"^omh reminder:.*$", pack, re.MULTILINE)
        digest = DIGEST.search(pack)
        assert records is not None and reminder is not None and digest is not None
        self.assertIn("<demote ", old["brief"])
        parts = (old["system"], old["index"], records.group(0), reminder.group(0), digest.group(0), old["brief"])
        self.assertGreater(len("\n".join(parts)), HERMES_PREFETCH_SPILL_MAX_CHARS, "not a worst case: the old caps would not have spilled")

    def test_the_pack_fits_whole_and_the_receipt_names_exactly_what_it_carries(self) -> None:
        pack = hermes_turn(self.live, 1, QUERY)
        self.assertLessEqual(len(pack), PACK_BUDGET_CHARS)
        receipt = self.live.latest_prefetch_receipt()
        assert receipt is not None
        self.assertEqual(validate_prefetch_receipt(receipt), [])
        self.assertEqual(receipt["rendering"]["pack_chars"], len(pack))
        # Every section is closed and parses on its own.
        for tag in SECTION_TAGS:
            self.assertEqual(pack.count(f"<{tag}"), pack.count(f"</{tag}>"), tag)
            for match in re.finditer(rf"<{tag}[ >].*?</{tag}>", pack, re.DOTALL):
                ElementTree.fromstring(match.group(0))
        # The records section is never cut, and the receipt lists those ids alone.
        ids = rendered_ids(pack)
        self.assertTrue(ids)
        self.assertEqual([row["record_id"] for row in receipt["rendering"]["rendered_records"]], ids)
        for record_id in ids:
            self.assertIn(f'id="{record_id}"', pack)
        # Blocks: the count is the elements in the text.
        system = re.search(r"<memory_blocks>.*?</memory_blocks>", pack, re.DOTALL)
        self.assertIsNotNone(system)
        assert system is not None
        self.assertEqual(receipt["rendering"]["rendered_block_count"], system.group(0).count("<value>"))
        self.assertGreaterEqual(receipt["rendering"]["rendered_block_count"], 1)
        self.assertIn('reason="render_budget_exhausted"', system.group(0))
        # The index is capped and says how much it left out.
        index = re.search(r"<memory_block_index>.*?</memory_block_index>", pack, re.DOTALL)
        assert index is not None
        self.assertLessEqual(len(index.group(0)), REFERENCE_INDEX_PACK_MAX_CHARS)
        self.assertRegex(index.group(0), r'<omitted count="\d+" />')
        # The reminder is whole, and disclosed because it was served.
        self.assertEqual(pack.count("omh reminder:"), 1)
        self.assertIn(f"omh memory retire {self.open_record}", pack)
        self.assertEqual(receipt["reminder"]["record_id"], self.open_record)
        # The brief keeps its demote rows here and its tail always.
        brief = re.search(r"<memory_consolidation .*?</memory_consolidation>", pack, re.DOTALL)
        assert brief is not None
        self.assertIn("<demote ", brief.group(0))
        self.assertIn("<do>Then tell the user in one short line", brief.group(0))
        self.assertTrue(pack.endswith("</memory_consolidation>"))

    def test_what_gives_is_system_blocks_then_index_lines_then_demote_rows(self) -> None:
        # A long-id reminder or a bigger brief is what squeezes a real pack;
        # a smaller budget squeezes this one the same way, section by section.
        # Rendered, never served: the reminder and the digest stay pending.
        self.live.queue_prefetch(QUERY)

        def at(budget: int) -> dict[str, object]:
            with patch.object(memory_provider, "PACK_BUDGET_CHARS", budget):
                pack = self.live.render_pack()
            system = re.search(r"<memory_blocks>.*?</memory_blocks>", pack, re.DOTALL)
            index = re.search(r"<memory_block_index>.*?</memory_block_index>", pack, re.DOTALL)
            self.assertLessEqual(len(pack), budget)
            self.assertIn("<do>Then tell the user in one short line", pack)
            self.assertTrue(pack.endswith("</memory_consolidation>"))
            self.assertIn("omh reminder:", pack)
            return {
                "pack": pack,
                "system": len(system.group(0)) if system else 0,
                "blocks": system.group(0).count("<value>") if system else 0,
                "index": len(index.group(0)) if index else 0,
                "rows": pack.count("<demote "),
            }

        base = at(PACK_BUDGET_CHARS)
        self.assertEqual(base["rows"], 3)
        rest = len(base["pack"]) - base["system"] - 1
        at_floor = rest + 1 + SYSTEM_BLOCKS_PACK_FLOOR_CHARS
        # Down to the floor only system blocks give.
        shrunk = at(at_floor)
        self.assertLessEqual(shrunk["system"], SYSTEM_BLOCKS_PACK_FLOOR_CHARS)
        self.assertEqual((shrunk["index"], shrunk["rows"]), (base["index"], base["rows"]))
        # Below it the index gives next, and the rows stay.
        tighter = at(at_floor - 200)
        self.assertLessEqual(tighter["index"], base["index"] - 200)
        self.assertEqual((tighter["rows"], tighter["blocks"]), (base["rows"], shrunk["blocks"]))
        # With the index gone, demote rows give before anything else does.
        tightest = at(at_floor - base["index"] - 300)
        self.assertEqual(tightest["index"], 0)
        self.assertLess(tightest["rows"], base["rows"])
        self.assertEqual(tightest["blocks"], shrunk["blocks"])
        for squeezed in (shrunk, tighter, tightest):
            self.assertEqual(rendered_ids(squeezed["pack"]), rendered_ids(base["pack"]))


if __name__ == "__main__":
    unittest.main()
