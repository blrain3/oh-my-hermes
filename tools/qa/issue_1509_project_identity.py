#!/usr/bin/env python3
"""Real local CLI/provider QA for #1509. All fixtures are owned and removed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from datetime import datetime, timezone

from omh.plugin_bundle.omh.memory_provider import OmhMemoryProvider


def repository(root: Path, remote: str) -> Path:
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "config").write_text(f'[remote "origin"]\n url = https://example.invalid/{remote}.git\n', encoding="utf-8")
    return root


def exercise(scratch: Path) -> dict[str, object]:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("OMH_", "HERMES_", "GIT_"))}
    user, hermes = scratch / "user", scratch / "hermes"
    env.update(OMH_HOME=str(user), HERMES_HOME=str(hermes), HOME=str(scratch / "home"), USERPROFILE=str(scratch / "home"))
    calls = 0

    def cli(root: Path, *args: str, home: Path = user) -> dict[str, Any]:
        nonlocal calls
        process = subprocess.run([sys.executable, "-m", "omh.cli", "--omh-home", str(home), "--hermes-home", str(hermes), "memory", *args], cwd=root, env=env, text=True, capture_output=True, timeout=30)
        calls += 1
        if process.returncode:
            raise AssertionError(f"cli_failed:{args[0]}:{process.returncode}")
        return json.loads(process.stdout)

    def approve(root: Path, summary: str, *, home: Path = user, legacy: bool = False) -> str:
        options = ("--scope-ref", "legacy-checkout") if legacy else ()
        captured = cli(root, "capture", summary, "--retention-class", "durable", *options, home=home)
        candidate_id = captured["candidate"]["candidate_id"]
        revision = cli(root, "review", "--candidate", candidate_id, home=home)["cards"][0]["review_revision"]
        return cli(root, "approve", candidate_id, "--candidate-revision", revision, home=home)["record"]["record_id"]

    first = repository(scratch / "first" / "same-name", "first/project")
    second = repository(scratch / "second" / "same-name", "second/project")
    first_id = cli(first, "project-identity", "show")["identity"]
    second_id = cli(second, "project-identity", "show")["identity"]
    assert first_id != second_id
    first_record = approve(first, "first checkout sentinel")
    second_record = approve(second, "second checkout sentinel")
    for root, included, excluded in ((first, first_record, second_record), (second, second_record, first_record)):
        pack = cli(root, "recall")
        assert {row["record_id"] for row in pack["included_records"]} == {included}
        provider = OmhMemoryProvider(user)
        provider.initialize("qa-session", cwd=root, platform="cli", agent_context="primary")
        text = provider.prefetch()
        assert included in text and excluded not in text
        receipt = provider.latest_prefetch_receipt()
        assert receipt is not None
        assert receipt["schema_version"] == "omh_memory_prefetch_receipt/v3"
        assert len([scope for scope in receipt["lens"]["scope_allowlist"] if scope["kind"] == "project"]) == 1
        incident = cli(root, "recall-incident", "--record-id", included, "--session-id", "qa-session", "--observed", "hermes")
        assert incident["evidence_surfaces"]["live_prefetch_receipt"]["status"] == "observed"
        assert incident["reason_code"] == "rendered_delivery_not_observed"
        provider.shutdown()
    approve(first, "foreign checkout store sentinel", home=first / ".omh")
    assert cli(second, "recall", home=first / ".omh")["included_records"] == []
    assert cli(second, "project-identity", "show", home=first / ".omh")["identity"] == second_id
    renamed = first.with_name("renamed")
    first.rename(renamed)
    assert cli(renamed, "project-identity", "show")["identity"] == first_id
    assert cli(renamed, "recall")["included_records"][0]["record_id"] == first_record
    gitdir = renamed / ".git" / "worktrees" / "linked"
    gitdir.mkdir(parents=True)
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    linked = scratch / "linked"
    linked.mkdir()
    (linked / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    assert cli(linked, "project-identity", "show")["identity"] == first_id
    (renamed / ".git" / "config").write_text('[remote "origin"]\n url = "https://user:password@example.invalid/first/project.git" # origin\n', encoding="utf-8")
    assert cli(linked, "project-identity", "show")["identity"] == first_id

    local_user = scratch / "local-user"
    local_repos = [repository(scratch / name / "same-name", "unused") for name in ("local-one", "local-two")]
    for local in local_repos:
        (local.parent / "upstream.git").mkdir()
        (local / ".git" / "config").write_text('[remote "origin"]\n url = ../upstream.git\n', encoding="utf-8")
    assert cli(local_repos[0], "project-identity", "show")["identity"] != cli(local_repos[1], "project-identity", "show")["identity"]
    local_id = approve(local_repos[0], "local endpoint sentinel", home=local_user)
    local_provider = OmhMemoryProvider(local_user)
    local_provider.initialize("local", cwd=local_repos[0], platform="cli", agent_context="primary")
    assert local_id in local_provider.prefetch()
    local_provider.initialize("local", cwd=local_repos[1], platform="cli", agent_context="primary")
    assert local_id not in local_provider.prefetch()
    local_provider.shutdown()
    assert cli(local_repos[1], "recall", home=local_user)["included_records"] == []

    migration_root = repository(scratch / "migration", "migration/project")
    migration_user = scratch / "migration-user"
    legacy_ids = [approve(migration_root, "project legacy sentinel", home=migration_root / ".omh", legacy=True),
                  approve(migration_root, "user legacy sentinel", home=migration_user, legacy=True)]
    before = {p: p.read_bytes() for home in (migration_root / ".omh", migration_user) for p in home.rglob("*") if p.is_file()}
    report = cli(migration_root, "project-identity", "report", home=migration_user)
    assert len(report["entries"]) == 2
    assert {row["store"] for row in report["entries"]} == {"project", "user"}
    assert before == {p: p.read_bytes() for home in (migration_root / ".omh", migration_user) for p in home.rglob("*") if p.is_file()}
    receipt = cli(migration_root, "project-identity", "migrate", "--approve", report["report_digest"], home=migration_user)
    assert len(receipt["successors"]) == 2
    provider = OmhMemoryProvider(migration_user)
    provider.initialize("migration-session", cwd=migration_root, platform="cli", agent_context="primary")
    text = provider.prefetch()
    assert all(record_id in text for record_id in legacy_ids)
    provider.shutdown()
    assert cli(migration_root, "project-identity", "migrate", "--approve", report["report_digest"], home=migration_user)["successors"] == []
    rollback = cli(migration_root, "project-identity", "migrate", "--rollback", receipt["receipt_id"], home=migration_user)
    assert rollback["state"] == "rolled_back"
    for path, original in before.items():
        if path.parent.name in {"records", "reviews"}:
            assert path.read_bytes() == original
    for home in (migration_root / ".omh", migration_user):
        assert cli(migration_root, "recall", home=home)["included_records"] == []
        assert cli(migration_root, "recall", "--scope-kind", "project", "--scope-ref", "legacy-checkout", home=home)["record_count"] == 1
    # Inject an independent reviewed successor at exact synchronous boundaries;
    # the actual report, preflight, transaction and source preservation still run.
    from omh.paths import resolve_paths
    from omh.workflows import memory_project_identity as migration
    from omh.workflows._memory_lifecycle_plans import _approved_record, _review
    from omh.system.local_store import atomic_write_json

    for seam in ("report", "preflight"):
        race_root = repository(scratch / f"race-{seam}", f"race/{seam}")
        race_home = scratch / f"race-{seam}-user"
        record_id = approve(race_root, "approved legacy source", home=race_home, legacy=True)
        paths = resolve_paths(race_home, hermes)
        approved = migration.build_project_identity_migration_report(paths, root=race_root)
        source_path = paths.memory_dir / "records" / f"{record_id}.json"
        source = json.loads(source_path.read_bytes())
        replacement = _approved_record({**source, "summary": "independently reviewed successor"}, record_id, source["revision"] + 1, "qa-reviewer", datetime.now(timezone.utc))
        admission = replacement["admission"]
        assert isinstance(admission, dict)
        review_id = admission["review_id"]
        review = _review(replacement, review_id, "qa-reviewer")
        original_operation = migration.build_project_identity_migration_report if seam == "report" else migration.execute_memory_lifecycle

        def race(*args, **kwargs):
            result = original_operation(*args, **kwargs) if seam == "report" else None
            atomic_write_json(paths.memory_dir / "reviews" / f"{review_id}.json", review, private=True)
            atomic_write_json(source_path, replacement, private=True)
            return result if seam == "report" else original_operation(*args, **kwargs)

        target = "build_project_identity_migration_report" if seam == "report" else "execute_memory_lifecycle"
        with patch.object(migration, target, side_effect=race):
            refused = migration.migrate_project_identity(paths, root=race_root, approve=approved["report_digest"])
        assert refused["successors"] == []
        assert refused["skipped"] == [{"record_id": record_id, "store": "user", "reason_code": "source_changed_since_report"}]
        assert json.loads(source_path.read_bytes()) == replacement

    return {"same_basename_isolation": True, "capture_recall": True, "rename_stability": True,
            "local_remote_isolation": True, "quoted_credentials_normalized": True,
            "foreign_store_isolation": True, "external_store_receipt_accepted": True,
            "approval_race_report_skipped": True, "approval_race_preflight_skipped": True,
            "worktree_sharing": True, "migration_stores": 2, "migrated_successors": 2,
            "report_read_only": True, "idempotent": True, "rollback": True, "cli_calls": calls}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="omh-1509-qa-", dir=Path.cwd()) as temporary:
        scratch = Path(temporary)
        summary = exercise(scratch)
    summary.update(schema_version="issue_1509_project_identity_qa/v1", passed=True, cleanup_verified=not scratch.exists())
    encoded = json.dumps(summary, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
