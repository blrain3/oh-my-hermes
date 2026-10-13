"""OMH memory on every surface Hermes starts: the provider's surface classes.

Captures made on the Hermes TUI and Desktop were never recalled there: the
provider treated any `platform` other than `cli` as a shared surface, and a
shared surface denies every identity-unbound record. Hermes passes `tui` and
`desktop` for its operator consoles and never passes a principal context, so
prefetch served 0 of 2 model captures on `tui` and 2 of 2 on `cli`. The owner
decision of 2026-10-13 classifies each surface in provider `initialize`, from
Hermes' own kwargs, failing closed for anything unknown:

- `operator_console`: `cli`, `tui`, `desktop` -- reads and writes like the CLI;
- `shared`: every gateway, `api_server`, `webhook`, `relay`, `acp`, an empty
  platform and anything unknown -- principal-bound reads only, no capture or
  demote, and no blocks, reminder or brief in the pack;
- `background`: a non-primary `agent_context` (cron) -- shared reads, no writes.

Everything here runs through the provider's real `initialize` and the tool's
real handler; nothing patches the classifier.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from _local_package import load_local_package

load_local_package()
from omh.plugin_bundle.omh.memory_blocks import approve_memory_block, build_memory_block, write_memory_block  # noqa: E402
from omh.plugin_bundle.omh.memory_provider import OmhMemoryProvider  # noqa: E402
from omh.plugin_bundle.omh.memory_session_lens import (  # noqa: E402
    SURFACE_BACKGROUND,
    SURFACE_OPERATOR_CONSOLE,
    SURFACE_SHARED,
    publish_session_lens,
    read_session_lens,
    session_lens_path,
)
from omh.plugin_bundle.omh.tools import memory_tool  # noqa: E402
from omh.plugin_bundle.omh.tools.memory_tool import omh_memory_handler  # noqa: E402
from project_identity_fixture import memory_paths, seed_project_identity  # noqa: E402
from test_memory_demote_tool import FACT, entry, sha, write_memory  # noqa: E402
from test_memory_open_state import _open_record  # noqa: E402
from test_memory_principal import PRINCIPAL_A, _context  # noqa: E402
from test_memory_prefetch_canonical import rendered_ids  # noqa: E402
from test_memory_provider_hermes_order import hermes_turn  # noqa: E402

# (platform, agent_context, shared_surface kwarg or None, expected class)
SURFACE_CASES = (
    ("cli", "primary", None, SURFACE_OPERATOR_CONSOLE),
    ("tui", "primary", None, SURFACE_OPERATOR_CONSOLE),
    ("desktop", "primary", None, SURFACE_OPERATOR_CONSOLE),
    ("telegram", "primary", None, SURFACE_SHARED),
    ("discord", "primary", None, SURFACE_SHARED),
    ("slack", "primary", None, SURFACE_SHARED),
    ("whatsapp", "primary", None, SURFACE_SHARED),
    ("api_server", "primary", None, SURFACE_SHARED),
    ("webhook", "primary", None, SURFACE_SHARED),
    ("relay", "primary", None, SURFACE_SHARED),
    ("acp", "primary", None, SURFACE_SHARED),
    ("", "primary", None, SURFACE_SHARED),
    ("bogus", "primary", None, SURFACE_SHARED),
    ("TUI", "primary", None, SURFACE_SHARED),
    ("cron", "cron", None, SURFACE_BACKGROUND),
    ("subagent", "subagent", None, SURFACE_BACKGROUND),
    ("cli", "cron", None, SURFACE_BACKGROUND),
    ("cli", "", None, SURFACE_OPERATOR_CONSOLE),
    # The kwarg tightens, never loosens.
    ("cli", "primary", True, SURFACE_SHARED),
    ("tui", "primary", True, SURFACE_SHARED),
    ("cron", "cron", True, SURFACE_BACKGROUND),
    ("telegram", "primary", False, SURFACE_SHARED),
    ("cron", "cron", False, SURFACE_BACKGROUND),
)
FLAGS = {SURFACE_OPERATOR_CONSOLE: (False, True), SURFACE_SHARED: (True, True), SURFACE_BACKGROUND: (True, False)}
GATEWAYS = ("telegram", "discord", "slack", "api_server")


class _Homes(unittest.TestCase):
    """A user OMH home, a Hermes home, a repository and a directory outside any repository."""

    def setUp(self) -> None:
        base = Path(self.enterContext(TemporaryDirectory())).resolve()
        self.home = base / "omh-home"
        self.hermes = base / "hermes-home"
        (self.hermes / "memories").mkdir(parents=True)
        self.repo = base / "repo"
        self.identity = seed_project_identity(self.repo)
        self.outside = base / "outside"
        self.outside.mkdir()
        self.enterContext(patch.dict(os.environ, {"OMH_HOME": str(self.home), "HERMES_HOME": str(self.hermes)}))

    def start(self, session_id: str, *, platform: str, cwd: Path, agent_context: str | None = None, **extra: object) -> OmhMemoryProvider:
        """`initialize` with the kwargs `agent/agent_init.py::_memory_provider_init_kwargs` builds."""
        live = OmhMemoryProvider(self.home)
        kwargs: dict[str, object] = {
            "session_id": session_id,
            "platform": platform,
            "hermes_home": str(self.hermes),
            "agent_context": agent_context if agent_context is not None else (platform if platform in ("cron", "subagent") else "primary"),
            "cwd": str(cwd),
            **extra,
        }
        live.initialize(**kwargs)
        return live

    def tool(self, action: str, session_id: str, cwd: Path, **args: object) -> dict:
        return json.loads(omh_memory_handler({"action": action, **args}, cwd=str(cwd), session_id=session_id))

    def store_bytes(self) -> dict[str, bytes]:
        memory = self.home / "memory"
        return {str(path.relative_to(memory)): path.read_bytes() for path in sorted(memory.rglob("*")) if path.is_file()} if memory.exists() else {}


class SurfaceClassTests(_Homes):
    def test_each_platform_and_agent_context_gets_its_class_from_initialize(self) -> None:
        for index, (platform, agent_context, shared_kwarg, expected) in enumerate(SURFACE_CASES):
            with self.subTest(platform=platform, agent_context=agent_context, shared_surface=shared_kwarg):
                session_id = f"surface-{index}"
                extra = {} if shared_kwarg is None else {"shared_surface": shared_kwarg}
                self.start(session_id, platform=platform, cwd=self.repo, agent_context=agent_context, **extra)
                lens = read_session_lens(self.home, session_id)
                assert lens is not None
                self.assertEqual(lens["surface_class"], expected)
                self.assertEqual((lens["shared_surface"], lens["writes_enabled"]), FLAGS[expected])
                read_back = memory_tool._session_lens({"session_id": session_id})
                self.assertEqual((read_back.bound, read_back.surface_class), (True, expected))

    def test_a_kanban_worker_is_an_operator_console(self) -> None:
        # `hermes -p <profile> --cli ... chat -q "work kanban task <id>"`
        # (`hermes_cli/kanban_db_dispatch.py`, `_worker_argv`) builds its agent
        # in `hermes_cli/cli_agent_setup_mixin.py` with `platform="cli"`, and
        # `_memory_provider_init_kwargs` makes that `agent_context="primary"`
        # with the CLI-only callbacks. Workers received records before; they
        # still do.
        seed = self.start("seed", platform="cli", cwd=self.repo)
        captured = self.tool("capture", "seed", self.repo, summary="Kanban workers run the release suite before moving a card.")
        self.assertEqual(captured["status"], "remembered", captured)
        seed.shutdown()
        worker = self.start(
            "kanban-worker",
            platform="cli",
            cwd=self.repo,
            warning_callback=lambda message: None,
            status_callback=lambda line: None,
            agent_identity="worker",
            agent_workspace="hermes",
        )
        self.assertEqual(read_session_lens(self.home, "kanban-worker")["surface_class"], SURFACE_OPERATOR_CONSOLE)
        self.assertEqual(rendered_ids(hermes_turn(worker, 1, "work kanban task release suite card")), [captured["record_id"]])

    def test_a_lens_an_earlier_provider_wrote_is_read_by_its_flags(self) -> None:
        for session_id, shared, writes, expected in (
            ("old-cli", False, True, SURFACE_OPERATOR_CONSOLE),
            ("old-shared", True, True, SURFACE_SHARED),
            ("old-cron", True, False, SURFACE_BACKGROUND),
        ):
            with self.subTest(session_id=session_id):
                publish_session_lens(
                    self.home,
                    session_id,
                    {"shared_surface": shared, "writes_enabled": writes, "principal_context": None, "homes": [str(self.home)], "project_identity": "", "executor_target": "hermes"},
                    reset=True,
                )
                self.assertEqual(memory_tool._session_lens({"session_id": session_id}).surface_class, expected)

    def test_a_class_its_own_flags_contradict_is_not_a_lens(self) -> None:
        self.start("forged", platform="discord", cwd=self.repo)
        path = session_lens_path(self.home, "forged")
        lens = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**lens, "surface_class": SURFACE_OPERATOR_CONSOLE}), encoding="utf-8")
        self.assertIsNone(read_session_lens(self.home, "forged"))
        refused = self.tool("capture", "forged", self.repo, summary="The deploy freeze ends on 2026-11-02.")
        self.assertEqual((refused["status"], refused["reason"]), ("refused", "session_lens_unavailable"))


class ConsoleRecallTests(_Homes):
    """The P0: a capture made on the CLI is recalled on the TUI and on Desktop, inside a repository and outside one."""

    def test_tui_and_desktop_recall_a_cli_capture_inside_a_repository(self) -> None:
        self.start("cli-repo", platform="cli", cwd=self.repo)
        captured = self.tool("capture", "cli-repo", self.repo, summary="The release branch is cut every Thursday at noon.")
        self.assertEqual(captured["status"], "remembered", captured)
        stored = json.loads((self.home / "memory" / "records" / f"{captured['record_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["scope"], {"kind": "project", "ref": self.identity})
        for platform in ("tui", "desktop", "cli"):
            with self.subTest(platform=platform):
                live = self.start(f"{platform}-repo-recall", platform=platform, cwd=self.repo)
                self.assertEqual(rendered_ids(hermes_turn(live, 1, "when is the release branch cut")), [captured["record_id"]])
                recalled = self.tool("recall", f"{platform}-repo-recall", self.repo, query="release branch cut")
                self.assertEqual([row["record_id"] for row in recalled["records"]], [captured["record_id"]])
                live.shutdown()
        for platform in GATEWAYS:
            with self.subTest(platform=platform):
                live = self.start(f"{platform}-repo-recall", platform=platform, cwd=self.repo)
                self.assertEqual(rendered_ids(hermes_turn(live, 1, "when is the release branch cut")), [])

    def test_tui_and_desktop_recall_a_cli_capture_outside_a_repository(self) -> None:
        self.start("cli-outside", platform="cli", cwd=self.outside)
        captured = self.tool("capture", "cli-outside", self.outside, summary="The operator prefers replies without emoji.")
        self.assertEqual(captured["status"], "remembered", captured)
        stored = json.loads((self.home / "memory" / "records" / f"{captured['record_id']}.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["scope"], {"kind": "user-global", "ref": "default"})
        for platform in ("tui", "desktop", "cli"):
            for cwd in (self.outside, self.repo):
                with self.subTest(platform=platform, cwd=cwd.name):
                    session_id = f"{platform}-{cwd.name}-recall"
                    live = self.start(session_id, platform=platform, cwd=cwd)
                    self.assertEqual(rendered_ids(hermes_turn(live, 1, "does the operator want emoji in replies")), [captured["record_id"]])
                    recalled = self.tool("recall", session_id, cwd, query="replies emoji")
                    self.assertEqual([row["record_id"] for row in recalled["records"]], [captured["record_id"]])
                    live.shutdown()


class OutsideRepositoryScopeTests(_Homes):
    """F4: with no project resolved, a console reads user-global and its own thread, and no project record at all."""

    def test_outside_a_repository_no_project_or_foreign_thread_record_leaks_in(self) -> None:
        self.start("cli-repo", platform="cli", cwd=self.repo)
        project = self.tool("capture", "cli-repo", self.repo, summary="The staging cluster drains nodes before each upgrade.")
        self.assertEqual(project["status"], "remembered", project)
        self.start("cli-outside", platform="cli", cwd=self.outside)
        personal = self.tool("capture", "cli-outside", self.outside, summary="The staging cluster notes are kept in the operator's paper notebook.")
        self.assertEqual(personal["status"], "remembered", personal)

        outside = self.start("tui-outside", platform="tui", cwd=self.outside)
        self.assertEqual(rendered_ids(hermes_turn(outside, 1, "staging cluster upgrade drains")), [personal["record_id"]])
        receipt = outside.latest_prefetch_receipt()
        assert receipt is not None
        self.assertEqual(
            receipt["lens"]["scope_allowlist"],
            [{"kind": "thread", "ref": "tui-outside"}, {"kind": "user-global", "ref": "default"}],
        )
        self.assertEqual(receipt["lens"]["project_identity_state"], "unresolved")
        refused = self.tool("confirm", "tui-outside", self.outside, record_id=project["record_id"])
        self.assertEqual((refused["status"], refused["reason"]), ("refused", "record_not_in_scope"))

        # Inside a repository the project label is required again: another
        # repository's record stays out, the personal one still comes in.
        other = self.repo.parent / "other-repo"
        seed_project_identity(other, "prj:" + "c" * 64)
        elsewhere = self.start("tui-other-repo", platform="tui", cwd=other)
        self.assertEqual(rendered_ids(hermes_turn(elsewhere, 1, "staging cluster upgrade drains")), [personal["record_id"]])

        # A shared surface outside a repository still reads nothing.
        shared = self.start("discord-outside", platform="discord", cwd=self.outside)
        self.assertEqual(rendered_ids(hermes_turn(shared, 1, "staging cluster upgrade drains")), [])
        self.assertEqual(shared.latest_prefetch_receipt()["lens"]["scope_allowlist"], [])


class WriteRefusalTests(_Homes):
    """`capture` and `demote` write on an operator console only, and a refusal writes nothing."""

    def setUp(self) -> None:
        super().setUp()
        self.memory_file = write_memory(self.hermes, "MEMORY.md", FACT, entry(1))

    def assert_refused_without_writing(self, session_id: str, reason: str) -> None:
        before, hermes_before = self.store_bytes(), self.memory_file.read_bytes()
        captured = self.tool("capture", session_id, self.repo, summary="The deploy freeze ends on 2026-11-02.")
        self.assertEqual((captured["status"], captured["reason"]), ("refused", reason), captured)
        self.assertTrue(captured["next_action"].startswith("Nothing was saved."), captured)
        self.assertIsNone(captured["record_id"])
        self.assertNotIn("display_line", captured)
        demoted = self.tool("demote", session_id, self.repo, entry_sha256=sha(FACT)[:12])
        self.assertEqual((demoted["status"], demoted["reason"]), ("refused", reason), demoted)
        self.assertTrue(demoted["next_action"].startswith("Keep the Hermes entry"), demoted)
        self.assertNotIn("remove_from_hermes", demoted)
        self.assertEqual(self.store_bytes(), before)
        self.assertEqual(self.memory_file.read_bytes(), hermes_before)

    def test_a_gateway_refuses_capture_and_demote(self) -> None:
        for platform in GATEWAYS:
            with self.subTest(platform=platform):
                self.start(f"{platform}-writes", platform=platform, cwd=self.repo)
                self.assert_refused_without_writing(f"{platform}-writes", "shared_surface_writes_refused")

    def test_a_console_asked_to_be_shared_refuses_them_too(self) -> None:
        self.start("cli-shared", platform="cli", cwd=self.repo, shared_surface=True)
        self.assert_refused_without_writing("cli-shared", "shared_surface_writes_refused")

    def test_a_background_run_refuses_every_write(self) -> None:
        self.start("cron-writes", platform="cron", cwd=self.repo)
        self.assert_refused_without_writing("cron-writes", "writes_disabled")

    def test_an_unbound_session_refuses_every_write(self) -> None:
        self.start("live", platform="cli", cwd=self.repo)
        for session_id in ("never-initialized", ""):
            with self.subTest(session_id=session_id):
                self.assert_refused_without_writing(session_id, "session_lens_unavailable")

    def test_a_console_writes(self) -> None:
        for platform in ("cli", "tui", "desktop"):
            with self.subTest(platform=platform):
                self.start(f"{platform}-writes", platform=platform, cwd=self.repo)
                captured = self.tool("capture", f"{platform}-writes", self.repo, summary=f"The {platform} operator keeps release notes in the wiki.")
                self.assertEqual(captured["status"], "remembered", captured)
        demoted = self.tool("demote", "desktop-writes", self.repo, entry_sha256=sha(FACT)[:12])
        self.assertEqual(demoted["status"], "demoted", demoted)


class PackGateTests(_Homes):
    """Blocks, the reminder and the brief reach an operator console only."""

    def setUp(self) -> None:
        super().setUp()
        # The operator's own store doubles as the repository's, so the open
        # record is in the project lens; Hermes memory nearly full raises a
        # brief whose demote rows quote it.
        self.home = self.repo / ".omh"
        self.enterContext(patch.dict(os.environ, {"OMH_HOME": str(self.home)}))
        paths = memory_paths(self.home, self.hermes)
        self.open_record = _open_record(paths, "cache question: redis or memcached", days_open=34)["record_id"]
        write_memory_block(self.home, approve_memory_block(build_memory_block("operator", "The operator's always-on note.")))
        write_memory_block(self.home, approve_memory_block(build_memory_block("runbook", "value", description="How the operator deploys.", tier="reference")))
        self.entries = [entry(index) for index in range(10)]
        write_memory(self.hermes, "MEMORY.md", *self.entries)

    def test_shared_and_background_packs_carry_no_blocks_reminder_or_brief(self) -> None:
        for platform in (*GATEWAYS, "cron"):
            with self.subTest(platform=platform):
                pack = hermes_turn(self.start(f"{platform}-pack", platform=platform, cwd=self.repo), 1, "cache question")
                for marker in ("<memory_blocks", "<memory_block_index", "omh reminder:", "<memory_consolidation", "<demote ", "always-on note"):
                    self.assertNotIn(marker, pack)
                for text in self.entries:
                    self.assertNotIn(text[:40], pack)

    def test_a_shared_surface_is_never_asked_even_about_a_record_its_principal_reads(self) -> None:
        # The lens does admit a principal-bound record here; the reminder is
        # still the operator's question, not the group chat's.
        paths = memory_paths(self.home, self.hermes)
        bound = _open_record(paths, "principal question: which cache tier", days_open=34, scope_kind="user", principal_context=_context(PRINCIPAL_A))
        live = self.start("shared_fixture", platform="discord", cwd=self.repo, principal_context=_context(PRINCIPAL_A))
        live.on_turn_start(1, "which cache tier", principal_context=_context(PRINCIPAL_A))
        pack = live.prefetch("which cache tier", session_id="shared_fixture")
        self.assertEqual(rendered_ids(pack), [bound["record_id"]])
        self.assertNotIn("omh reminder:", pack)
        self.assertIsNone(live.latest_open_reminder())

    def test_a_console_pack_carries_the_brief_with_its_demote_rows(self) -> None:
        for platform in ("tui", "desktop", "cli"):
            with self.subTest(platform=platform):
                live = self.start(f"{platform}-pack", platform=platform, cwd=self.repo)
                pack = hermes_turn(live, 1, "cache question")
                self.assertIn("<memory_blocks>", pack)
                self.assertIn("always-on note", pack)
                self.assertIn("<memory_block_index>", pack)
                self.assertIn("<memory_consolidation", pack)
                self.assertEqual(pack.count("<demote "), 3)
                self.assertIn(self.open_record, rendered_ids(pack))
                # Asked on the first console that served it, then on its cadence.
                self.assertEqual("omh reminder:" in pack, platform == "tui")
                live.shutdown()


if __name__ == "__main__":
    unittest.main()
