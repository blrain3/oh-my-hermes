"""The "since your last session" digest: what OMH memory changed between two sessions.

The user never saw what memory did between sessions. The provider serves one
bounded ``<memory_digest>`` section in the first pack of a new session, counted
from structured fields only. These pin:

- the counts are exact over records' ``created_at``/``source``/``derived_from``,
  the write journal's ``removed_entry_demoted``, and the retirement journal;
- the window starts at the last recorded session close (else the last served
  pack, at most seven days back) and ends at the session start, and an
  unreadable close stamp or receipt means no digest rather than a wider one;
- the report (`omh quality-evidence memory`) and the digest parse stamps and
  recognise a freeing remove with one implementation;
- the section rides only the first served pack of a session Hermes started --
  ``initialize`` or its own ``reset`` -- and never a turn-lease reload, a
  compression, a gateway rebuild of the same conversation, a cron or subagent
  context, or a pack with every count at zero; it never lights the recall line;
- a shared surface counts only records its own recall lens admits;
- a corrupt stamp or a store fault costs the digest, never the hook.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from _local_package import load_local_package

load_local_package()
from omh.paths import OmhPaths  # noqa: E402
from omh.plugin_bundle.omh import memory_provider  # noqa: E402
from omh.plugin_bundle.omh.hermes_memory import HERMES_MEMORY_DELIMITER, demotion_origin_ref  # noqa: E402
from omh.plugin_bundle.omh.memory_prefetch_receipt import prefetch_receipt_path  # noqa: E402
from omh.plugin_bundle.omh.memory_provider import JOURNAL_LIMIT, OmhMemoryProvider  # noqa: E402
from omh.plugin_bundle.omh.memory_session_digest import (  # noqa: E402
    DIGEST_COUNT_KEYS,
    DIGEST_INSTRUCTION,
    DIGEST_RENDER_BUDGET_CHARS,
    MARK_FUTURE_SKEW_SECONDS,
    SHARED_COUNT_KEYS,
    SHARED_DIGEST_INSTRUCTION,
    clamp_to_reported,
    digest_window_start,
    render_session_digest,
    session_close_path,
    session_digest_counts,
    stamp_epoch,
    write_reported_through,
    write_session_close,
)
from omh.plugin_bundle.omh.tools import memory_tool  # noqa: E402
from omh.plugin_bundle.omh.tools.memory_tool import omh_memory_handler  # noqa: E402
from omh.quality.hermes_state import hermes_epoch  # noqa: E402
from omh.quality.memory_evidence import build_memory_evidence  # noqa: E402
from project_identity_fixture import seed_project_identity  # noqa: E402
from test_memory_provider_hermes_order import hermes_turn  # noqa: E402

DIGEST = re.compile(r"<memory_digest[^>]*>.*?</memory_digest>", re.DOTALL)
NOW = datetime(2026, 10, 12, 9, 0, 0, tzinfo=timezone.utc)
FACT = "The operator deploys from the release branch every Friday afternoon, never from main."
HUGE = 10**400


def stamp(moment: datetime) -> str:
    return moment.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@contextmanager
def provider_clock(moment: datetime):
    """The provider reads ``moment`` as now: a session started at a chosen instant."""

    class _Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    with patch.object(memory_provider, "datetime", _Clock):
        yield


def started_later(seconds: float = 2):
    # The window ends at the session start floored to its second; starting a
    # couple of seconds on keeps writes made just before inside it whatever
    # the clock reads.
    return provider_clock(datetime.now(timezone.utc) + timedelta(seconds=seconds))


def ago(**delta: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(**delta)


def digest_attributes(pack: str) -> dict[str, str] | None:
    """The digest section's attributes, or None when the pack carries none."""
    match = DIGEST.search(pack)
    if match is None:
        return None
    head = match.group(0).split(">", 1)[0]
    return dict(re.findall(r'(\w+)="([^"]*)"', head))


def counts_of(pack: str) -> dict[str, str] | None:
    attributes = digest_attributes(pack)
    return None if attributes is None else {key: value for key, value in attributes.items() if key != "since"}


def write_record(home: Path, record_id: str, *, created_at: object, source: str = "cli", summary: str = "", **fields: object) -> None:
    path = home / "memory" / "records" / f"{record_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "project_memory_record/v2",
        "record_id": record_id,
        "created_at": created_at,
        "source": source,
        "summary": summary or f"fixture {record_id}",
        "source_ref": "",
        "derived_from": [],
        **fields,
    }
    path.write_text(json.dumps(record), encoding="utf-8")


def write_lines(path: Path, *entries: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join((entry if isinstance(entry, str) else json.dumps(entry)) + "\n" for entry in entries), encoding="utf-8")


def append_line(path: Path, entry: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def set_session_marks(home: Path, **marks: datetime) -> None:
    """Set the session marks outright. `write_session_close` only moves a mark
    forward, which is its point; a fixture placing a close in the past sets it."""
    path = session_close_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: moment.isoformat().replace("+00:00", "Z") for name, moment in marks.items()}
    path.write_text(json.dumps({"schema_version": "omh_memory_session_close/v1", **payload}), encoding="utf-8")


def write_receipt(home: Path, *, served_at: object, session_id: str = "earlier", schema: str = "omh_memory_prefetch_receipt/v3") -> None:
    """A receipt as far as the digest reads it; an older release's shape on purpose."""
    path = prefetch_receipt_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": schema, "state": "returned_to_host", "served_at": served_at, "session_id": session_id}), encoding="utf-8")


def copied_record(home: Path, record_id: str, entry: str, *, created_at: str, intact: bool = True) -> None:
    write_record(
        home,
        record_id,
        created_at=created_at,
        source="hermes_demotion",
        summary=entry if intact else entry + " (altered)",
        source_ref=demotion_origin_ref("MEMORY.md", entry),
    )


class FixtureStore:
    """One OMH home whose every row is placed relative to a window [since, until)."""

    def __init__(self, home: Path, *, since: datetime, until: datetime) -> None:
        inside = stamp(since + timedelta(minutes=5))
        before = stamp(since - timedelta(minutes=5))
        at_until = stamp(until)
        # Records: two remembered, one verified copy, one altered copy (not
        # counted), one correction; plus four outside the window or unreadable.
        write_record(home, "rec-a", created_at=inside, source="hermes_model")
        write_record(home, "rec-b", created_at=inside, source="cli")
        copied_record(home, "rec-c", "The CI cache lives on the build-7 host.", created_at=inside)
        copied_record(home, "rec-d", "The staging cluster runs in Seoul.", created_at=inside, intact=False)
        write_record(home, "rec-e", created_at=inside, source="hermes_model", derived_from=["rec-old"])
        write_record(home, "rec-f", created_at=before, source="hermes_model")
        write_record(home, "rec-g", created_at=at_until, source="hermes_model")
        write_record(home, "rec-h", created_at="not a stamp", source="hermes_model")
        write_record(home, "rec-j", created_at=HUGE, source="hermes_model")
        # An operator record's derived_from is a rollup, not a correction.
        write_record(home, "rec-i", created_at=inside, source="cli", derived_from=["rec-a"])
        memory = home / "memory"
        write_lines(
            memory / "write_journal.jsonl",
            {"action": "remove", "removed_entry_demoted": True, "observed_at": inside},
            {"action": "REMOVE", "removed_entry_demoted": True, "observed_at": inside},
            {"action": "remove", "removed_entry_demoted": False, "observed_at": inside},
            {"action": "remove", "observed_at": inside},
            {"action": "add", "removed_entry_demoted": True, "observed_at": inside},
            {"action": "remove", "removed_entry_demoted": True, "observed_at": before},
            {"action": "remove", "removed_entry_demoted": True, "observed_at": HUGE},
            "{torn line",
        )
        write_lines(
            memory / "archive" / "retirements.jsonl",
            {"record_id": "rec-x", "retired_at": inside, "reason": "no_longer_true"},
            {"record_id": "rec-y", "retired_at": inside, "reason": "user_asked"},
            {"record_id": "rec-z", "retired_at": before, "reason": "superseded"},
            {"record_id": "rec-w", "retired_at": at_until, "reason": "superseded"},
            {"record_id": "rec-v", "retired_at": HUGE, "reason": "superseded"},
        )

    # Derived by hand from the rows above, not from the counter.
    EXPECTED = {
        "remembered": 3,  # rec-a, rec-b, rec-i
        "hermes_entries_copied": 1,  # rec-c (rec-d is altered)
        "hermes_entries_freed": 2,  # the two true removes inside the window
        "corrected": 1,  # rec-e
        "retired": 2,  # rec-x, rec-y
    }


def counts(record_homes, home, *, since, until, in_lens=None) -> tuple[dict[str, int], bool]:
    return session_digest_counts(record_homes, home, since=since, until=until, journal_limit=JOURNAL_LIMIT, in_lens=in_lens)


class DigestCountTests(unittest.TestCase):
    def test_counts_are_exact_over_records_and_both_journals(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / ".omh"
            since, until = NOW - timedelta(hours=6), NOW
            FixtureStore(home, since=since, until=until)
            self.assertEqual(counts([home], home, since=since, until=until), (FixtureStore.EXPECTED, False))

    def test_a_record_held_by_both_homes_counts_once(self) -> None:
        with TemporaryDirectory() as tmp:
            project, user = Path(tmp) / "project" / ".omh", Path(tmp) / ".omh"
            inside = stamp(NOW - timedelta(minutes=30))
            write_record(project, "rec-a", created_at=inside)
            write_record(user, "rec-a", created_at=inside)
            write_record(user, "rec-b", created_at=inside)
            write_lines(project / "memory" / "archive" / "retirements.jsonl", {"record_id": "rec-x", "retired_at": inside})
            write_lines(user / "memory" / "archive" / "retirements.jsonl", {"record_id": "rec-x", "retired_at": inside})
            found, _ = counts([project, user], user, since=NOW - timedelta(hours=1), until=NOW)
            self.assertEqual((found["remembered"], found["retired"]), (2, 1))

    def test_an_empty_or_missing_store_counts_zero_everywhere(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / ".omh"
            self.assertEqual(counts([home], home, since=NOW - timedelta(days=7), until=NOW), (dict.fromkeys(DIGEST_COUNT_KEYS, 0), False))

    def test_a_lens_counts_only_what_it_admits_and_no_unscoped_kind(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / ".omh"
            since, until = NOW - timedelta(hours=6), NOW
            FixtureStore(home, since=since, until=until)
            admitted = {"rec-b", "rec-e", "rec-c"}
            found, at_cap = counts([home], home, since=since, until=until, in_lens=lambda record: record["record_id"] in admitted)
            # Copies, frees and retirements carry no scope a lens can judge: not counted at all.
            self.assertEqual(found, {"remembered": 1, "corrected": 1})
            self.assertEqual(tuple(found), SHARED_COUNT_KEYS)
            self.assertFalse(at_cap)

    def test_a_journal_at_its_cap_says_its_frees_may_be_short(self) -> None:
        with TemporaryDirectory() as tmp:
            home = Path(tmp) / ".omh"
            inside = stamp(NOW - timedelta(minutes=5))
            journal = home / "memory" / "write_journal.jsonl"
            write_lines(journal, *[{"action": "add", "observed_at": inside}] * (JOURNAL_LIMIT - 1))
            self.assertFalse(counts([home], home, since=NOW - timedelta(hours=1), until=NOW)[1])
            append_line(journal, {"action": "add", "observed_at": inside})
            self.assertTrue(counts([home], home, since=NOW - timedelta(hours=1), until=NOW)[1])


class DigestWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.home = Path(self.enterContext(TemporaryDirectory())) / ".omh"
        (self.home / "memory").mkdir(parents=True)

    def start(self, **kwargs: str) -> datetime | None:
        return digest_window_start(self.home, until=NOW, **kwargs)

    def test_the_last_recorded_close_starts_the_window_to_the_microsecond(self) -> None:
        closed = NOW - timedelta(hours=3) + timedelta(milliseconds=500)  # exact in a float, not a whole second
        write_session_close(self.home, at=closed)
        write_receipt(self.home, served_at=stamp(NOW - timedelta(hours=1)))
        self.assertEqual(self.start(), closed)

    def test_with_no_close_the_last_served_pack_starts_it_and_with_neither_seven_days(self) -> None:
        self.assertEqual(self.start(), NOW - timedelta(days=7))
        # A receipt an older release wrote still counts: only three fields are read.
        write_receipt(self.home, served_at=stamp(NOW - timedelta(hours=2)), schema="omh_memory_prefetch_receipt/v1")
        self.assertEqual(self.start(), NOW - timedelta(hours=2))

    def test_never_more_than_seven_days_back(self) -> None:
        write_session_close(self.home, at=NOW - timedelta(days=30))
        self.assertEqual(self.start(), NOW - timedelta(days=7))

    def test_the_end_of_the_last_digest_served_raises_the_start(self) -> None:
        write_session_close(self.home, at=NOW - timedelta(hours=3))
        write_reported_through(self.home, until=NOW - timedelta(hours=1))
        self.assertEqual(self.start(), NOW - timedelta(hours=1))
        write_session_close(self.home, at=NOW - timedelta(minutes=30))
        self.assertEqual(self.start(), NOW - timedelta(minutes=30))

    def test_with_no_close_the_mark_still_raises_the_last_served_pack(self) -> None:
        write_receipt(self.home, served_at=stamp(NOW - timedelta(hours=3)))
        write_reported_through(self.home, until=NOW - timedelta(hours=2))
        self.assertEqual(self.start(), NOW - timedelta(hours=2))

    def test_a_mark_too_far_ahead_is_ignored_and_one_within_the_skew_is_kept(self) -> None:
        write_receipt(self.home, served_at=stamp(NOW - timedelta(hours=4)))
        set_session_marks(self.home, last_session_closed_at=NOW + timedelta(days=30), reported_through=NOW + timedelta(days=30))
        # Both ignored: the receipt stands in for the close.
        self.assertEqual(self.start(), NOW - timedelta(hours=4))
        set_session_marks(self.home, last_session_closed_at=NOW - timedelta(hours=3), reported_through=NOW + timedelta(seconds=MARK_FUTURE_SKEW_SECONDS + 1))
        self.assertEqual(self.start(), NOW - timedelta(hours=3))
        set_session_marks(self.home, last_session_closed_at=NOW - timedelta(hours=3), reported_through=NOW + timedelta(minutes=2))
        self.assertEqual(self.start(), NOW + timedelta(minutes=2), "two minutes ahead is clock skew, not a wrong clock")

    def test_a_far_future_mark_is_replaced_not_kept(self) -> None:
        set_session_marks(self.home, last_session_closed_at=datetime.now(timezone.utc) + timedelta(days=30))
        closed = datetime.now(timezone.utc)
        write_session_close(self.home, at=closed)
        marks = json.loads(session_close_path(self.home).read_text(encoding="utf-8"))
        self.assertEqual(stamp_epoch(marks["last_session_closed_at"]), closed.timestamp())

    def test_the_render_clamp_reads_only_the_reported_mark(self) -> None:
        since = NOW - timedelta(hours=3)
        set_session_marks(self.home, last_session_closed_at=NOW - timedelta(hours=1))
        self.assertEqual(clamp_to_reported(self.home, since, now=NOW), since, "a newer close does not hide rows no digest showed")
        write_reported_through(self.home, until=NOW - timedelta(hours=2))
        self.assertEqual(clamp_to_reported(self.home, since, now=NOW), NOW - timedelta(hours=2))
        session_close_path(self.home).write_text("{torn", encoding="utf-8")
        self.assertIsNone(clamp_to_reported(self.home, since, now=NOW))

    def test_marks_only_move_forward_and_keep_each_other(self) -> None:
        # A clock stepped back, or a ~/.omh shared by two machines, must not widen the window.
        write_session_close(self.home, at=NOW - timedelta(hours=1))
        write_session_close(self.home, at=NOW - timedelta(hours=5))
        write_reported_through(self.home, until=NOW - timedelta(hours=2))
        write_reported_through(self.home, until=NOW - timedelta(hours=6))
        marks = json.loads(session_close_path(self.home).read_text(encoding="utf-8"))
        self.assertEqual(stamp_epoch(marks["last_session_closed_at"]), (NOW - timedelta(hours=1)).timestamp())
        self.assertEqual(stamp_epoch(marks["reported_through"]), (NOW - timedelta(hours=2)).timestamp())
        self.assertEqual(self.start(), NOW - timedelta(hours=1))

    def test_an_unreadable_close_or_receipt_means_no_digest_not_a_wider_one(self) -> None:
        cases = {
            "garbage close": (session_close_path(self.home), "{not json"),
            "foreign close": (session_close_path(self.home), json.dumps({"schema_version": "other/v1", "last_session_closed_at": stamp(NOW)})),
            "garbage receipt": (prefetch_receipt_path(self.home), "{not json"),
            "unserved receipt": (prefetch_receipt_path(self.home), json.dumps({"schema_version": "omh_memory_prefetch_receipt/v3", "state": "prepared", "served_at": "", "session_id": "x"})),
            "absurd receipt stamp": (prefetch_receipt_path(self.home), json.dumps({"schema_version": "omh_memory_prefetch_receipt/v3", "state": "returned_to_host", "served_at": "1e300", "session_id": "x"})),
            "huge receipt stamp": (prefetch_receipt_path(self.home), json.dumps({"schema_version": "omh_memory_prefetch_receipt/v3", "state": "returned_to_host", "served_at": HUGE, "session_id": "x"})),
            "absurd close stamp": (session_close_path(self.home), json.dumps({"schema_version": "omh_memory_session_close/v1", "last_session_closed_at": "99999-01-01T00:00:00Z"})),
            "reported mark not a stamp": (session_close_path(self.home), json.dumps({"schema_version": "omh_memory_session_close/v1", "reported_through": 12})),
            "marks file with no mark": (session_close_path(self.home), json.dumps({"schema_version": "omh_memory_session_close/v1"})),
        }
        for name, (path, text) in cases.items():
            for with_close in (False, True):
                with self.subTest(case=name, with_close=with_close):
                    session_close_path(self.home).unlink(missing_ok=True)
                    prefetch_receipt_path(self.home).unlink(missing_ok=True)
                    if with_close:
                        # A readable close does not make an unreadable receipt safe to ignore.
                        write_session_close(self.home, at=NOW - timedelta(hours=1))
                    path.write_text(text, encoding="utf-8")
                    self.assertIsNone(self.start())

    def test_an_initialize_for_the_session_that_was_last_served_gets_none(self) -> None:
        # A gateway rebuilding its agent for a conversation already under way.
        write_receipt(self.home, served_at=stamp(NOW - timedelta(hours=2)), session_id="chat-1")
        self.assertIsNone(self.start(initializing_session="chat-1"))
        self.assertEqual(self.start(initializing_session="chat-2"), NOW - timedelta(hours=2))
        self.assertEqual(self.start(), NOW - timedelta(hours=2))


class DigestRenderTests(unittest.TestCase):
    def test_zero_counts_render_nothing(self) -> None:
        self.assertEqual(render_session_digest(dict.fromkeys(DIGEST_COUNT_KEYS, 0), since=NOW), "")

    def test_only_nonzero_counts_are_named(self) -> None:
        text = render_session_digest({**dict.fromkeys(DIGEST_COUNT_KEYS, 0), "retired": 2}, since=NOW)
        self.assertEqual(digest_attributes(text), {"since": stamp(NOW), "retired": "2"})

    def test_a_journal_at_its_cap_is_named(self) -> None:
        text = render_session_digest({"remembered": 1}, since=NOW, journal_at_cap=True)
        self.assertEqual(digest_attributes(text)["journal_at_cap"], "true")

    def test_the_section_is_counts_and_a_fixed_sentence_under_its_budget(self) -> None:
        text = render_session_digest(dict.fromkeys(DIGEST_COUNT_KEYS, 10**9), since=NOW, journal_at_cap=True)
        self.assertLessEqual(len(text), DIGEST_RENDER_BUDGET_CHARS)
        body = DIGEST.fullmatch(text).group(0).split(">", 1)[1].rsplit("</", 1)[0].strip()
        self.assertEqual(body, DIGEST_INSTRUCTION)
        self.assertEqual(render_session_digest(dict.fromkeys(DIGEST_COUNT_KEYS, HUGE), since=NOW), "")

    def test_a_shared_section_names_no_since_and_says_what_it_counted(self) -> None:
        text = render_session_digest({"remembered": 2, "corrected": 1}, since=NOW, shared=True, journal_at_cap=True)
        self.assertEqual(digest_attributes(text), {"remembered": "2", "corrected": "1"})
        self.assertIn(SHARED_DIGEST_INSTRUCTION, text)

    def test_the_sentences_leave_voice_and_language_to_the_host(self) -> None:
        # SOUL owns voice and reply language; OMH text never directs either.
        for sentence in (DIGEST_INSTRUCTION, SHARED_DIGEST_INSTRUCTION):
            for word in ("language", "voice", "tone", "persona", "korean", "english", "always", "must"):
                self.assertNotIn(word, sentence.lower())


class ReportParityTests(unittest.TestCase):
    """The report and the digest share the parser and the freeing rule; the row rules must agree too."""

    def test_the_report_parses_stamps_with_the_digests_parser(self) -> None:
        corpus = (
            "2026-10-12T09:00:00Z", "2026-10-12T09:00:00+09:00", "2026-10-12T09:00:00", "2026-10-12",
            " 2026-10-12T09:00:00Z ", "1760259600", "1760259600.5", "nan", "inf", "-inf", "1e300", "", "  ",
            "garbage", "99999-01-01T00:00:00Z", None, True, False, 0, 1760259600, 1.5, float("nan"), float("inf"),
            HUGE, -HUGE, "9" * 400, [], {},
        )
        for value in corpus:
            with self.subTest(value=value):
                self.assertEqual(hermes_epoch(value), stamp_epoch(value))
        self.assertIsNone(hermes_epoch(HUGE), "a 400-digit integer is not a stamp; it must not raise")

    def test_records_frees_and_the_cap_agree_with_quality_evidence_memory_in_both_stores(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            user, project = root / ".omh", root / "repo" / ".omh"
            since, until = NOW - timedelta(hours=6), NOW
            FixtureStore(user, since=since, until=until)
            inside = stamp(since + timedelta(minutes=9))
            write_record(project, "proj-a", created_at=inside, source="hermes_model")
            write_record(project, "proj-b", created_at=inside, source="hermes_model", derived_from=["proj-old"])
            copied_record(project, "proj-c", "The release bot posts in the ops channel.", created_at=inside)
            # The report's window has no upper bound; give the digest one past every row.
            found, at_cap = counts([project, user], user, since=since, until=NOW + timedelta(days=1))
            reports = {
                name: build_memory_evidence(OmhPaths(omh_home=home, hermes_home=root / ".hermes"), since=stamp(since), now=(NOW + timedelta(days=1)).timestamp())["omh_store"]
                for name, home in (("user", user), ("project", project))
            }
            altered = 1  # rec-d: a record the report counts and the digest does not name
            self.assertEqual(
                found["remembered"] + found["hermes_entries_copied"] + found["corrected"] + altered,
                sum(report["records"]["created_in_window"] for report in reports.values()),
            )
            self.assertEqual(found["hermes_entries_freed"], reports["user"]["write_journal"]["removed_entry_demoted"]["true"])
            self.assertEqual(at_cap, reports["user"]["write_journal"]["journal_at_cap"])
            for _ in range(JOURNAL_LIMIT):
                append_line(user / "memory" / "write_journal.jsonl", {"action": "add", "observed_at": inside})
            report = build_memory_evidence(OmhPaths(omh_home=user, hermes_home=root / ".hermes"), since=stamp(since), now=NOW.timestamp())
            self.assertTrue(counts([user], user, since=since, until=NOW)[1])
            self.assertTrue(report["omh_store"]["write_journal"]["journal_at_cap"])
            freed = next(row for row in report["headline"] if row["derived_from"].endswith("removed_entry_demoted.true"))
            self.assertIn("earlier removes may be missing", freed["claim"])


def start(home: Path, root: Path, session_id: str, *, platform: str = "cli", agent_context: str | None = None) -> OmhMemoryProvider:
    live = OmhMemoryProvider(home)
    live.initialize(
        session_id,
        hermes_home=str(root / ".hermes"),
        platform=platform,
        agent_context=agent_context or (platform if platform in ("cron", "subagent") else "primary"),
        cwd=str(root),
    )
    return live


class ProviderDigestTests(unittest.TestCase):
    """Hermes' order only: initialize, then on_turn_start and prefetch in the same turn."""

    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        (self.root / ".hermes").mkdir()
        self.home = self.root / ".omh"
        self.retirements = self.home / "memory" / "archive" / "retirements.jsonl"

    def seed_since_last_close(self) -> None:
        """The last session closed three hours ago; a record and a retirement landed since."""
        set_session_marks(self.home, last_session_closed_at=ago(hours=3))
        write_record(self.home, "rec-a", created_at=stamp(ago(hours=2)), source="hermes_model", summary="The user's laptop is a ThinkPad X1.")
        append_line(self.retirements, {"record_id": "rec-x", "retired_at": stamp(ago(hours=2))})

    def test_the_first_prefetch_of_a_session_carries_the_digest_and_later_turns_do_not(self) -> None:
        self.seed_since_last_close()
        live = start(self.home, self.root, "session-a")
        self.assertEqual(counts_of(hermes_turn(live, 1, "what is on my calendar")), {"remembered": "1", "retired": "1"})
        live.queue_prefetch("what is on my calendar")
        self.assertIsNone(digest_attributes(hermes_turn(live, 2, "and tomorrow")))
        self.assertIsNone(digest_attributes(hermes_turn(live, 3, "and the day after")))

    def test_the_digest_carries_no_record_text(self) -> None:
        self.seed_since_last_close()
        section = DIGEST.search(hermes_turn(start(self.home, self.root, "session-a"), 1, "ThinkPad laptop")).group(0)
        for text in ("ThinkPad", "rec-a", "rec-x"):
            self.assertNotIn(text, section)
        self.assertEqual(re.sub(r'="[^"]*"', '=""', section.split(">", 1)[0]), '<memory_digest since="" remembered="" retired=""')

    def test_zero_counts_put_no_section_in_the_pack(self) -> None:
        write_session_close(self.home, at=ago(hours=1))
        write_record(self.home, "rec-a", created_at=stamp(ago(hours=2)), source="hermes_model")
        append_line(self.retirements, {"record_id": "rec-x", "retired_at": stamp(ago(days=9))})
        self.assertNotIn("<memory_digest", hermes_turn(start(self.home, self.root, "session-a"), 1, "anything"))

    def test_a_compression_switch_does_not_restart_it(self) -> None:
        self.seed_since_last_close()
        live = start(self.home, self.root, "session-a")
        self.assertIsNotNone(digest_attributes(hermes_turn(live, 1, "hello there")))
        live.queue_prefetch("hello there")
        # Activity after the reported mark, and a switch two seconds on: a
        # digest wrongly restarted here would have this row to report.
        write_record(self.home, "rec-b", created_at=stamp(datetime.now(timezone.utc)), source="hermes_model")
        with started_later():
            live.on_session_switch("session-a2", parent_session_id="session-a", reset=False)
        self.assertIsNone(digest_attributes(hermes_turn(live, 2, "go on", session_id="session-a2")))

    def test_a_turn_lease_reload_does_not_restart_it(self) -> None:
        # `turn_facade_lease` rotates the id to the compression child of the
        # same conversation and calls no `on_session_switch`; prefetch rebinds
        # the lens. A digest started there would count from the close again.
        self.seed_since_last_close()
        live = start(self.home, self.root, "s1")
        self.assertIsNotNone(digest_attributes(hermes_turn(live, 1, "hello there")))
        live.queue_prefetch("hello there")
        write_record(self.home, "rec-b", created_at=stamp(ago(seconds=1)), source="hermes_model")
        live.prefetch("hello there", session_id="s1-reloaded")
        self.assertIsNone(digest_attributes(hermes_turn(live, 3, "carry on", session_id="s1-reloaded")))

    def test_hermes_new_with_history_counts_none_of_the_old_conversations_writes(self) -> None:
        self.seed_since_last_close()
        live = start(self.home, self.root, "session-a")
        hermes_turn(live, 1, "hello there")
        live.queue_prefetch("hello there")
        # The old conversation writes, then Hermes' worker runs end strictly before switch.
        write_record(self.home, "rec-own", created_at=stamp(datetime.now(timezone.utc)), source="hermes_model")
        live.on_session_end([])
        # Two seconds on, so rec-own is inside the window unless the close excludes it.
        with started_later():
            live.on_session_switch("session-b", parent_session_id="session-a", reset=True)
        self.assertNotIn("<memory_digest", hermes_turn(live, 1, "new topic", session_id="session-b"))

    def test_the_new_race_delivers_the_digest_hermes_reset_starts(self) -> None:
        # Hermes' boundary task lands after the next turn's prefetch already
        # adopted the new id; the explicit reset still starts the digest.
        live = start(self.home, self.root, "session-a")
        hermes_turn(live, 1, "hello there")
        live.queue_prefetch("hello there")
        self.seed_since_last_close()
        closed_before = json.loads(session_close_path(self.home).read_text(encoding="utf-8"))["last_session_closed_at"]
        raced = live.prefetch("new topic", session_id="session-b")
        self.assertNotIn("<memory_digest", raced, "prefetch adopting an id starts nothing")
        live.on_session_switch("session-b", parent_session_id="session-a", reset=True)
        self.assertEqual(counts_of(hermes_turn(live, 2, "new topic", session_id="session-b")), {"remembered": "1", "retired": "1"})
        closed_after = json.loads(session_close_path(self.home).read_text(encoding="utf-8"))["last_session_closed_at"]
        self.assertGreater(stamp_epoch(closed_after), stamp_epoch(closed_before), "the reset records the old session's close")

    def test_a_pack_rendered_before_new_does_not_use_up_the_new_sessions_digest(self) -> None:
        write_session_close(self.home, at=datetime.now(timezone.utc))
        live = start(self.home, self.root, "session-a")  # pending, nothing to say, already rendered
        self.seed_since_last_close()
        live.on_session_switch("session-b", parent_session_id="session-a", reset=True)
        # The same query: prefetch serves the pack `initialize` rendered, unrendered again.
        self.assertNotIn("<memory_digest", live.prefetch("", session_id="session-b"))
        self.assertEqual(counts_of(hermes_turn(live, 1, "new topic", session_id="session-b")), {"remembered": "1", "retired": "1"})

    def test_a_killed_session_reports_nothing_twice_and_keeps_its_own_writes(self) -> None:
        self.seed_since_last_close()
        killed = start(self.home, self.root, "session-a")
        self.assertEqual(counts_of(hermes_turn(killed, 1, "hello")), {"remembered": "1", "retired": "1"})
        self.assertIn("reported_through", json.loads(session_close_path(self.home).read_text(encoding="utf-8")))
        write_record(self.home, "rec-own", created_at=stamp(datetime.now(timezone.utc)), source="hermes_model")
        del killed  # no shutdown, so no close is recorded
        with started_later():
            later = start(self.home, self.root, "session-b")
        # rec-a and the retirement were reported; what the killed session wrote after is not lost.
        self.assertEqual(counts_of(hermes_turn(later, 1, "hello again")), {"remembered": "1"})

    def test_an_overlapping_session_does_not_repeat_what_the_other_reported(self) -> None:
        self.seed_since_last_close()
        first = start(self.home, self.root, "session-a")
        self.assertEqual(counts_of(hermes_turn(first, 1, "hello")), {"remembered": "1", "retired": "1"})
        write_record(self.home, "rec-new", created_at=stamp(datetime.now(timezone.utc)), source="hermes_model")
        with started_later():
            second = start(self.home, self.root, "session-b")  # the first is still open
        self.assertEqual(counts_of(hermes_turn(second, 1, "hello")), {"remembered": "1"})
        hermes_turn(first, 2, "still here")

    def test_a_session_that_serves_after_a_later_one_does_not_repeat_its_rows(self) -> None:
        # The reviewer's probe 8: a closed, then the operator wrote rec-op; x
        # started and idled; y started later and served first; then x served.
        set_session_marks(self.home, last_session_closed_at=ago(hours=3))
        write_record(self.home, "rec-op", created_at=stamp(ago(hours=2)), source="cli")
        x = start(self.home, self.root, "session-x")
        with started_later():
            y = start(self.home, self.root, "session-y")
        self.assertEqual(counts_of(hermes_turn(y, 1, "hello")), {"remembered": "1"})
        self.assertNotIn("<memory_digest", hermes_turn(x, 1, "hello"), "y already reported rec-op")

    def test_an_empty_digest_records_no_mark(self) -> None:
        set_session_marks(self.home, last_session_closed_at=ago(hours=1))
        hermes_turn(start(self.home, self.root, "session-a"), 1, "hello")
        self.assertNotIn("reported_through", json.loads(session_close_path(self.home).read_text(encoding="utf-8")))

    def test_a_far_future_mark_does_not_silence_digests(self) -> None:
        # One wrong clock sharing this home must not hold every digest back until its date.
        set_session_marks(self.home, last_session_closed_at=ago(hours=3), reported_through=ago(days=-30))
        write_record(self.home, "rec-a", created_at=stamp(ago(hours=2)), source="hermes_model")
        self.assertEqual(counts_of(hermes_turn(start(self.home, self.root, "session-a"), 1, "hello")), {"remembered": "1"})
        marks = json.loads(session_close_path(self.home).read_text(encoding="utf-8"))
        self.assertLess(stamp_epoch(marks["reported_through"]), datetime.now(timezone.utc).timestamp() + 60, "the served digest replaced it")

    def test_marks_that_turn_unreadable_mid_session_serve_no_digest(self) -> None:
        self.seed_since_last_close()
        live = start(self.home, self.root, "session-a")
        session_close_path(self.home).write_text("{torn", encoding="utf-8")
        self.assertNotIn("<memory_digest", hermes_turn(live, 1, "hello"))

    def test_this_sessions_first_second_is_outside_its_window(self) -> None:
        # Stamps are whole seconds, floored: a write this session makes in its
        # first second carries that second, which the floored window end excludes.
        write_session_close(self.home, at=ago(hours=1))
        moment = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(milliseconds=900)
        write_record(self.home, "rec-own", created_at=stamp(moment), source="hermes_model")
        with provider_clock(moment):
            live = start(self.home, self.root, "session-a")
        self.assertNotIn("<memory_digest", hermes_turn(live, 1, "hello"))

    def test_a_closed_session_leaves_nothing_to_report_for_the_next(self) -> None:
        self.seed_since_last_close()
        first = start(self.home, self.root, "session-a")
        self.assertIsNotNone(digest_attributes(hermes_turn(first, 1, "hello there")))
        # The session's own write after its digest: only its close keeps it out.
        write_record(self.home, "rec-own", created_at=stamp(datetime.now(timezone.utc)), source="hermes_model")
        first.shutdown()
        with started_later():
            second = start(self.home, self.root, "session-b")
        self.assertNotIn("<memory_digest", hermes_turn(second, 1, "hello again"))

    def test_a_gateway_rebuild_of_the_same_conversation_gets_none(self) -> None:
        write_receipt(self.home, served_at=stamp(ago(hours=3)), session_id="chat-1")
        write_record(self.home, "rec-a", created_at=stamp(ago(hours=2)), source="hermes_model")
        self.assertNotIn("<memory_digest", hermes_turn(start(self.home, self.root, "chat-1"), 1, "next message"))
        write_receipt(self.home, served_at=stamp(ago(hours=3)), session_id="chat-1")
        self.assertEqual(counts_of(hermes_turn(start(self.home, self.root, "chat-2"), 1, "hello")), {"remembered": "1"})

    def test_an_unreadable_receipt_or_close_serves_no_digest(self) -> None:
        write_record(self.home, "rec-a", created_at=stamp(ago(hours=2)), source="hermes_model")
        for path in (prefetch_receipt_path(self.home), session_close_path(self.home)):
            with self.subTest(path=path.name):
                session_close_path(self.home).unlink(missing_ok=True)
                prefetch_receipt_path(self.home).unlink(missing_ok=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{torn", encoding="utf-8")
                self.assertNotIn("<memory_digest", hermes_turn(start(self.home, self.root, "session-a"), 1, "hello"))

    def test_corrupt_stamps_cost_nothing_and_a_store_fault_costs_only_the_digest(self) -> None:
        self.seed_since_last_close()
        write_record(self.home, "rec-huge", created_at=HUGE, source="hermes_model")
        append_line(self.retirements, {"record_id": "rec-y", "retired_at": HUGE})
        append_line(self.home / "memory" / "write_journal.jsonl", {"action": "remove", "removed_entry_demoted": True, "observed_at": HUGE})
        self.assertEqual(counts_of(hermes_turn(start(self.home, self.root, "session-a"), 1, "hello")), {"remembered": "1", "retired": "1"})
        with patch.object(memory_provider, "session_digest_counts", side_effect=RuntimeError("store exploded")):
            # Two seconds on: session-a's mark would leave a same-second window
            # empty, and an empty window never reaches the store.
            with started_later():
                live = start(self.home, self.root, "session-b")
            pack = hermes_turn(live, 1, "hello")
        self.assertNotIn("<memory_digest", pack)
        self.assertIn({"op": "session_digest", "path": "memory_digest", "error": "RuntimeError"}, live._write_failures)
        with patch.object(memory_provider, "digest_window_start", side_effect=OverflowError("absurd")):
            self.assertNotIn("<memory_digest", hermes_turn(start(self.home, self.root, "session-c"), 1, "hello"))

    def test_a_digest_only_pack_does_not_light_the_recall_line(self) -> None:
        write_session_close(self.home, at=ago(hours=3))
        append_line(self.retirements, {"record_id": "rec-x", "retired_at": stamp(ago(hours=2))})
        live = start(self.home, self.root, "session-a")
        pack = hermes_turn(live, 1, "anything at all")
        self.assertEqual(pack, DIGEST.search(pack).group(0))
        self.assertIsNone(live.recall_status())

    def test_cron_and_subagent_contexts_get_none_and_record_no_close(self) -> None:
        self.seed_since_last_close()
        before = session_close_path(self.home).read_bytes()
        for platform, agent_context in (("cron", "cron"), ("cli", "subagent")):
            with self.subTest(agent_context=agent_context):
                live = start(self.home, self.root, f"{agent_context}-1", platform=platform, agent_context=agent_context)
                self.assertNotIn("<memory_digest", hermes_turn(live, 1, "run the nightly job"))
                live.shutdown()
                self.assertEqual(session_close_path(self.home).read_bytes(), before)

    def test_a_trivial_first_turn_that_skips_prefetch_leaves_the_digest_for_the_next(self) -> None:
        self.seed_since_last_close()
        live = start(self.home, self.root, "session-a")
        live.on_turn_start(1, "hi")
        live.queue_prefetch("hi")
        self.assertIsNotNone(digest_attributes(hermes_turn(live, 2, "what is on my calendar")))


class SharedSurfaceDigestTests(unittest.TestCase):
    """The TUI and every gateway are shared surfaces: the digest is confined, not suppressed."""

    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        seed_project_identity(self.root)
        (self.root / ".hermes").mkdir()
        self.home = self.root / ".omh"
        self.enterContext(patch.dict(os.environ, {"OMH_HOME": str(self.home), "HERMES_HOME": str(self.root / ".hermes")}))
        write_session_close(self.home, at=ago(hours=3))
        inside = stamp(ago(hours=2))
        write_record(self.home, "rec-a", created_at=inside, source="hermes_model", scope={"kind": "user-global", "ref": "default"})
        write_record(self.home, "rec-e", created_at=inside, source="hermes_model", derived_from=["rec-old"], scope={"kind": "user-global", "ref": "default"})
        copied_record(self.home, "rec-c", FACT, created_at=inside)
        append_line(self.home / "memory" / "archive" / "retirements.jsonl", {"record_id": "rec-x", "retired_at": inside})
        append_line(self.home / "memory" / "write_journal.jsonl", {"action": "remove", "removed_entry_demoted": True, "observed_at": inside})

    def test_a_shared_surface_counts_only_what_its_lens_admits(self) -> None:
        # Today the shared lens admits no identity-unbound record, so nothing here counts.
        self.assertNotIn("<memory_digest", hermes_turn(start(self.home, self.root, "discord-1", platform="discord"), 1, "hello"))

    def test_an_admitted_record_is_counted_with_no_since_and_no_unscoped_kind(self) -> None:
        live = start(self.home, self.root, "discord-1", platform="discord")
        with patch.object(OmhMemoryProvider, "_recall_lens", return_value=lambda record: True):
            live.on_turn_start(1, "hello")
            pack = live.prefetch("hello")
        self.assertEqual(digest_attributes(pack), {"remembered": "1", "corrected": "1"})
        self.assertIn(SHARED_DIGEST_INSTRUCTION, pack)
        # It counted only its lens, so the home's other rows are not marked reported.
        self.assertNotIn("reported_through", json.loads(session_close_path(self.home).read_text(encoding="utf-8")))

    def test_the_lens_is_the_one_omh_memory_recall_applies(self) -> None:
        identity = seed_project_identity(self.root)
        records = [
            {"schema_version": "project_memory_record/v2", "record_id": f"r{index}", "scope": scope}
            for index, scope in enumerate(
                (
                    {"kind": "user-global", "ref": "default"},
                    {"kind": "project", "ref": identity},
                    {"kind": "project", "ref": "prj:" + "b" * 64},
                    {"kind": "thread", "ref": "other-thread"},
                )
            )
        ]
        # In scope, but seen from another executor's perspective.
        records.append({"schema_version": "project_memory_record/v2", "record_id": "r4", "scope": {"kind": "user-global", "ref": "default"}, "perspective": {"observer": "codex", "observed": "codex"}})
        for platform in ("cli", "tui", "discord"):
            with self.subTest(platform=platform):
                live = start(self.home, self.root, f"{platform}-lens", platform=platform)
                lens = memory_tool._session_lens({"session_id": f"{platform}-lens"})
                gate = live._recall_lens()
                self.assertEqual([gate(record) for record in records], [memory_tool._in_session_scope(record, lens) for record in records])


class RealWritersDigestTests(unittest.TestCase):
    """The next session counts exactly what was written after the last close, through the model's own tools."""

    def setUp(self) -> None:
        self.root = Path(self.enterContext(TemporaryDirectory())).resolve()
        seed_project_identity(self.root)
        self.hermes = self.root / ".hermes"
        (self.hermes / "memories").mkdir(parents=True)
        self.store = self.root / ".omh"
        self.enterContext(patch.dict(os.environ, {"OMH_HOME": str(self.store), "HERMES_HOME": str(self.hermes)}))

    def call(self, session_id: str, action: str, **args: object) -> dict:
        return json.loads(omh_memory_handler({"action": action, **args}, cwd=str(self.root), session_id=session_id))

    def test_capture_correct_retire_demote_and_free_each_count_once(self) -> None:
        (self.hermes / "memories" / "MEMORY.md").write_text(HERMES_MEMORY_DELIMITER.join([FACT]), encoding="utf-8")
        closed = start(self.store, self.root, "session-a")
        hermes_turn(closed, 1, "let us get started")
        closed.shutdown()
        # Placed five minutes back so the writes below fall after it whatever the second.
        set_session_marks(self.store, last_session_closed_at=ago(minutes=5))
        # Another conversation, still open (a gateway chat), writes after that close.
        other = start(self.store, self.root, "session-x")
        kept = self.call("session-x", "capture", summary="The user's staging database is Postgres 16.")
        changed = self.call("session-x", "capture", summary="The user prefers tabs over spaces in Python.")
        dropped = self.call("session-x", "capture", summary="The user is migrating the CI off Jenkins.")
        for result in (kept, changed, dropped):
            self.assertEqual(result["status"], "remembered", result)
        corrected = self.call("session-x", "correct", record_id=changed["record_id"], summary="The user prefers spaces over tabs in Python.")
        self.assertEqual(corrected["status"], "corrected", corrected)
        found = self.call("session-x", "recall", query="Jenkins CI migrating")
        self.assertIn(dropped["record_id"], [row["record_id"] for row in found["records"]], found)
        self.assertEqual(self.call("session-x", "retire", record_id=dropped["record_id"], reason="no_longer_true")["status"], "retired")
        demoted = self.call("session-x", "demote", entry_sha256=hashlib.sha256(FACT.encode("utf-8")).hexdigest())
        self.assertEqual(demoted["status"], "demoted", demoted)
        other.on_memory_write("remove", "memory", "", {"previous_content": FACT})
        with started_later():
            later = start(self.store, self.root, "session-b")
        self.assertEqual(
            counts_of(hermes_turn(later, 1, "what changed while I was away")),
            # kept is remembered; changed became the correction; dropped was retired.
            {"remembered": "1", "hermes_entries_copied": "1", "hermes_entries_freed": "1", "corrected": "1", "retired": "1"},
        )


if __name__ == "__main__":
    unittest.main()
