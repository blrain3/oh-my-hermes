"""Agent Skills host-discovery evidence layer; no third-party host execution.

Covers the descriptor table, fail-closed receipt parsing, stage
non-promotion, the summary matrix, and the isolated probe runner against
fixture stand-ins only. Real-host probes are opt-in and run outside CI.
"""
from __future__ import annotations

import json
import unittest

from omh.quality.agent_skills_discovery import (
    PROBE_DESCRIPTORS,
    SCHEMA,
    STAGES,
    DiscoveryEvidenceError,
    MatrixRow,
    ProbeDescriptor,
    ProbeHost,
    ProbeMechanism,
    ProbeReceipt,
    ProbeScope,
    StageOutcome,
    descriptors_for,
    parse_receipt,
    run_probe,
    source_digest,
    summarize_receipts,
)

FILES = {
    "omh-plan/SKILL.md": "# plan\n",
    "omh-ask/SKILL.md": "# ask\n",
}
REVISION = "a" * 64
DIGEST = "b" * 64


def receipt_payload(**overrides):
    payload = {
        "schema": SCHEMA,
        "host": "codex",
        "host_version": "0.158.0",
        "os": "linux",
        "scope": "repo",
        "install_relpath": ".agents/skills",
        "catalog_revision": REVISION,
        "source_digest": DIGEST,
        "argv": ["codex", "--version"],
        "mechanism": "model_mediated",
        "credential_required": True,
        "started_at": "2026-10-13T03:00:00Z",
        "ended_at": "2026-10-13T03:00:05Z",
        "exit_status": 0,
        "stages": [
            {"stage": "installed", "outcome": "observed", "detail": "2 files at .agents/skills"},
            {"stage": "discovered", "outcome": "observed", "detail": "omh-plan named by model_mediated surface"},
            {"stage": "nominated", "outcome": "not_observed", "detail": "not probed by this layer"},
            {"stage": "loaded", "outcome": "not_observed", "detail": "not probed by this layer"},
            {"stage": "executed", "outcome": "not_observed", "detail": "not probed by this layer"},
        ],
        "cleanup": "removed",
    }
    payload.update(overrides)
    return payload


class DescriptorTests(unittest.TestCase):
    def test_every_contract_host_has_a_descriptor(self):
        hosts = {d.host for d in PROBE_DESCRIPTORS}
        self.assertEqual(hosts, set(ProbeHost))

    def test_every_descriptor_is_either_probeable_or_explains_why_not(self):
        for d in PROBE_DESCRIPTORS:
            with self.subTest(host=d.host):
            # either a discovery surface, or an explicit unsupported reason
                self.assertTrue(d.discovery_args is not None or d.probe_unsupported)

    def test_user_scope_relocates_install_path_only(self):
        for repo in PROBE_DESCRIPTORS:
            user = [d for d in descriptors_for(ProbeScope.USER) if d.host is repo.host][0]
            with self.subTest(host=repo.host):
                self.assertEqual(user.scope, ProbeScope.USER)
                self.assertTrue(user.install_relpath.startswith("~/"))
                self.assertEqual(user.executable, repo.executable)
                self.assertEqual(user.discovery_args, repo.discovery_args)
                self.assertEqual(user.probe_unsupported, repo.probe_unsupported)

    def test_descriptor_rejects_contradictory_shapes(self):
        base = PROBE_DESCRIPTORS[0]
        with self.assertRaises(DiscoveryEvidenceError):
            ProbeDescriptor(
                host=base.host,
                scope=base.scope,
                executable=base.executable,
                version_args=base.version_args,
                discovery_args=None,
                mechanism=base.mechanism,
                credential_required=True,
                timeout_seconds=base.timeout_seconds,
                expected_skill=base.expected_skill,
                install_relpath=base.install_relpath,
                probe_unsupported=None,
            )


class ReceiptParsingTests(unittest.TestCase):
    def test_round_trip(self):
        receipt = parse_receipt(receipt_payload())
        self.assertIsInstance(receipt, ProbeReceipt)
        self.assertEqual(receipt.highest_observed_stage(), "discovered")

    def test_wrong_schema_fails_closed(self):
        with self.assertRaises(DiscoveryEvidenceError):
            parse_receipt(receipt_payload(schema="other/v1"))

    def test_secret_shaped_detail_fails_closed(self):
        payload = receipt_payload()
        payload["stages"][0]["detail"] = "token sk-abcdef123456 leaked"
        with self.assertRaises(DiscoveryEvidenceError):
            parse_receipt(payload)

    def test_invalid_digest_fails_closed(self):
        with self.assertRaises(DiscoveryEvidenceError):
            parse_receipt(receipt_payload(catalog_revision="not-a-digest"))

    def test_duplicate_stage_fails_closed(self):
        payload = receipt_payload()
        payload["stages"].append(dict(payload["stages"][0]))
        with self.assertRaises(DiscoveryEvidenceError):
            parse_receipt(payload)

    def test_unknown_stage_fails_closed(self):
        payload = receipt_payload()
        payload["stages"][0]["stage"] = "teleported"
        with self.assertRaises(DiscoveryEvidenceError):
            parse_receipt(payload)

    def test_stage_order_is_fixed(self):
        self.assertEqual(STAGES, ("installed", "discovered", "nominated", "loaded", "executed"))


class MatrixTests(unittest.TestCase):
    def test_model_mediated_discovery_is_never_filed_as_host_command(self):
        receipt = parse_receipt(receipt_payload())
        rows = summarize_receipts([receipt], current_catalog_revision=REVISION)
        self.assertEqual(rows[0].evidence_class, "discovered_model_mediated")

    def test_installed_only_never_promotes(self):
        payload = receipt_payload()
        payload["stages"][1]["outcome"] = "not_observed"
        receipt = parse_receipt(payload)
        rows = summarize_receipts([receipt], current_catalog_revision=REVISION)
        self.assertEqual(rows[0].evidence_class, "installed_only")
        self.assertEqual(receipt.highest_observed_stage(), "installed")

    def test_stale_digest_is_explicit(self):
        receipt = parse_receipt(receipt_payload())
        rows = summarize_receipts([receipt], current_catalog_revision="c" * 64)
        self.assertEqual(rows[0].freshness, "stale")

    def test_unsupported_host_row(self):
        payload = receipt_payload()
        payload["host"] = "cursor"
        payload["probe_unsupported"] = "no documented credential-free discovery command"
        payload["stages"][1]["outcome"] = "not_observed"
        receipt = parse_receipt(payload)
        rows = summarize_receipts([receipt], current_catalog_revision=REVISION)
        self.assertEqual(rows[0].evidence_class, "probe_unsupported")

    def test_matrix_is_deterministic(self):
        receipts = tuple(
            parse_receipt(receipt_payload(host=host)) for host in ("codex", "claude", "opencode")
        )
        first = summarize_receipts(receipts, current_catalog_revision=REVISION)
        second = summarize_receipts(reversed(receipts), current_catalog_revision=REVISION)
        self.assertEqual(first, second)
        self.assertEqual([r.host for r in first], ["claude", "codex", "opencode"])
        self.assertTrue(all(isinstance(r, MatrixRow) for r in first))


class ProbeRunnerTests(unittest.TestCase):
    def test_unsupported_host_records_reason_without_executing(self):
        descriptor = [d for d in PROBE_DESCRIPTORS if d.probe_unsupported][0]
        receipt = run_probe(descriptor, files=FILES, catalog_revision=REVISION)
        self.assertEqual(receipt.stage("installed").outcome, StageOutcome.OBSERVED)
        self.assertEqual(receipt.stage("discovered").outcome, StageOutcome.NOT_OBSERVED)
        self.assertIn(descriptor.probe_unsupported, receipt.stage("discovered").detail)
        self.assertEqual(receipt.cleanup, "removed")
        self.assertEqual(receipt.exit_status, -1)

    def test_missing_binary_holds_discovery(self):
        descriptor = ProbeDescriptor(
            host=ProbeHost.CODEX,
            scope=ProbeScope.REPO,
            executable="omh-nonexistent-host-binary",
            version_args=("--version",),
            discovery_args=("exec",),
            mechanism=ProbeMechanism.MODEL_MEDIATED,
            credential_required=True,
            timeout_seconds=30,
            expected_skill="omh-plan",
            install_relpath=".agents/skills",
        )
        receipt = run_probe(descriptor, files=FILES, catalog_revision=REVISION)
        self.assertEqual(receipt.stage("discovered").outcome, StageOutcome.HOLD)
        self.assertIn("not found", receipt.stage("discovered").detail)

    def test_model_mediated_lane_stays_opt_in(self):
        descriptor = [d for d in PROBE_DESCRIPTORS if d.mechanism is ProbeMechanism.MODEL_MEDIATED][0]
        receipt = run_probe(descriptor, files=FILES, catalog_revision=REVISION)
        self.assertEqual(receipt.stage("discovered").outcome, StageOutcome.NOT_OBSERVED)
        self.assertIn("opt-in", receipt.stage("discovered").detail)
        self.assertEqual(receipt.argv, ())

    def test_receipt_binds_digests_and_cleans_up(self):
        descriptor = [d for d in PROBE_DESCRIPTORS if d.probe_unsupported][0]
        receipt = run_probe(descriptor, files=FILES, catalog_revision=REVISION)
        self.assertEqual(receipt.source_digest, source_digest(FILES))
        self.assertEqual(receipt.catalog_revision, REVISION)
        self.assertEqual(receipt.cleanup, "removed")
        parsed = parse_receipt(json.loads(json.dumps(receipt.to_json())))
        self.assertEqual(parsed.host, receipt.host)


if __name__ == "__main__":
    unittest.main()
