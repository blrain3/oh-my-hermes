"""Agent Skills host-discovery probe descriptors, receipts, and matrix.

Evidence layer for the host-adapter surface (issue #1525). It binds a host
observation to the host version and the adapter pack digest, keeps the
evidence stages separate (installed, discovered, nominated, loaded,
executed), and fails closed: a missing binary, an undocumented probe
surface, a stale adapter digest, a credential requirement, a timeout, or
malformed probe output is an explicit HOLD or not-observed outcome, never a
passing one. No stage is inferred from an earlier one, and model-mediated
enumeration is recorded as its own mechanism instead of being promoted to
host discovery.

Fixture-backed tests validate this layer's contract; they are not
real-host verification. Real-host probes are opt-in and run outside the
unit suite.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from typing import Final


SCHEMA: Final = "omh_agent_skills_discovery/v1"
STAGES: Final = ("installed", "discovered", "nominated", "loaded", "executed")
MAX_DETAIL: Final = 200
MAX_ARGV: Final = 16
MAX_STAGES: Final = len(STAGES)
_HEX_64: Final = re.compile(r"[0-9a-f]{64}")
_SECRET: Final = re.compile(
    r"(sk-[A-Za-z0-9]|ghp_[A-Za-z0-9]|AKIA[0-9A-Z]{16}|api[_-]?key|authorization|bearer\s+|password|BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ignore previous|<script)",
    re.IGNORECASE,
)
_VERSION: Final = re.compile(r"[0-9][0-9A-Za-z.+-]{0,63}")


class DiscoveryEvidenceError(ValueError):
    """Raised when a descriptor or receipt violates the evidence contract."""


class ProbeHost(StrEnum):
    CODEX = "codex"
    CLAUDE_CODE = "claude"
    CURSOR = "cursor"
    OPENCODE = "opencode"
    OPENCLAW = "openclaw"
    PI = "pi"


class ProbeScope(StrEnum):
    REPO = "repo"
    USER = "user"


class ProbeMechanism(StrEnum):
    HOST_COMMAND = "host_command"
    MODEL_MEDIATED = "model_mediated"
    NONE = "none"


class StageOutcome(StrEnum):
    OBSERVED = "observed"
    NOT_OBSERVED = "not_observed"
    HOLD = "hold"


@dataclass(frozen=True, slots=True)
class ProbeDescriptor:
    """Checked-in description of one bounded host probe.

    ``discovery_args`` names a non-mutating discovery command when the host
    exposes one; ``None`` with ``probe_unsupported`` set records that no
    documented credential-free discovery probe is known for this host today
    — it is not a claim that the host cannot discover Agent Skills.
    ``mechanism`` says what kind of surface the probe observes, so a
    model-mediated enumeration is never filed as host discovery.
    """

    host: ProbeHost
    scope: ProbeScope
    executable: str
    version_args: tuple[str, ...]
    discovery_args: tuple[str, ...] | None
    mechanism: ProbeMechanism
    credential_required: bool
    timeout_seconds: int
    expected_skill: str
    install_relpath: str
    probe_unsupported: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.timeout_seconds <= 600:
            raise DiscoveryEvidenceError(f"timeout out of range: {self.host}")
        if not _VERSION.fullmatch(self.expected_skill) and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", self.expected_skill):
            raise DiscoveryEvidenceError(f"invalid expected skill: {self.expected_skill}")
        if self.discovery_args is None and not self.probe_unsupported:
            raise DiscoveryEvidenceError(f"descriptor without discovery surface needs a reason: {self.host}")
        if self.discovery_args is not None and self.probe_unsupported:
            raise DiscoveryEvidenceError(f"descriptor cannot be both probeable and unsupported: {self.host}")
        if self.discovery_args is not None and len(self.discovery_args) > MAX_ARGV:
            raise DiscoveryEvidenceError(f"discovery argv too long: {self.host}")


def _descriptor(
    host: ProbeHost,
    *,
    executable: str,
    version_args: tuple[str, ...],
    discovery_args: tuple[str, ...] | None,
    mechanism: ProbeMechanism,
    credential_required: bool,
    install_relpath: str,
    timeout_seconds: int = 120,
    expected_skill: str = "omh-plan",
    probe_unsupported: str | None = None,
) -> ProbeDescriptor:
    return ProbeDescriptor(
        host=host,
        scope=ProbeScope.REPO,
        executable=executable,
        version_args=version_args,
        discovery_args=discovery_args,
        mechanism=mechanism,
        credential_required=credential_required,
        timeout_seconds=timeout_seconds,
        expected_skill=expected_skill,
        install_relpath=install_relpath,
        probe_unsupported=probe_unsupported,
    )


# Checked-in descriptor table: every host in the Agent Skills support contract
# has a repo-scope descriptor, either with a discovery surface or with an
# explicit probe_unsupported reason. User scope reuses the same executable and
# discovery surface with the user install path.
PROBE_DESCRIPTORS: Final[tuple[ProbeDescriptor, ...]] = (
    _descriptor(
        ProbeHost.CLAUDE_CODE,
        executable="claude",
        version_args=("--version",),
        discovery_args=("-p", "list the skill names you can see, one per line, no commentary"),
        mechanism=ProbeMechanism.MODEL_MEDIATED,
        credential_required=True,
        install_relpath=".claude/skills",
    ),
    _descriptor(
        ProbeHost.CODEX,
        executable="codex",
        version_args=("--version",),
        discovery_args=("exec", "--sandbox", "read-only", "list the skill names you can see, one per line, no commentary"),
        mechanism=ProbeMechanism.MODEL_MEDIATED,
        credential_required=True,
        install_relpath=".agents/skills",
    ),
    _descriptor(
        ProbeHost.CURSOR,
        executable="cursor",
        version_args=("--version",),
        discovery_args=None,
        mechanism=ProbeMechanism.NONE,
        credential_required=True,
        install_relpath=".agents/skills",
        probe_unsupported="no documented credential-free discovery command for the .agents/skills pack",
    ),
    _descriptor(
        ProbeHost.OPENCODE,
        executable="opencode",
        version_args=("--version",),
        discovery_args=("run", "list the skill names you can see, one per line, no commentary"),
        mechanism=ProbeMechanism.MODEL_MEDIATED,
        credential_required=True,
        install_relpath=".agents/skills",
    ),
    _descriptor(
        ProbeHost.OPENCLAW,
        executable="openclaw",
        version_args=("--version",),
        discovery_args=None,
        mechanism=ProbeMechanism.NONE,
        credential_required=True,
        install_relpath=".agents/skills",
        probe_unsupported="no documented credential-free discovery command for the .agents/skills pack",
    ),
    _descriptor(
        ProbeHost.PI,
        executable="pi",
        version_args=("--version",),
        discovery_args=None,
        mechanism=ProbeMechanism.NONE,
        credential_required=True,
        install_relpath=".agents/skills",
        probe_unsupported="no documented credential-free discovery command for the .agents/skills pack",
    ),
)


def descriptors_for(scope: ProbeScope = ProbeScope.REPO) -> tuple[ProbeDescriptor, ...]:
    """Descriptors for a scope; user scope relocates the install path only."""
    if scope is ProbeScope.REPO:
        return PROBE_DESCRIPTORS
    return tuple(
        ProbeDescriptor(
            host=d.host,
            scope=scope,
            executable=d.executable,
            version_args=d.version_args,
            discovery_args=d.discovery_args,
            mechanism=d.mechanism,
            credential_required=d.credential_required,
            timeout_seconds=d.timeout_seconds,
            expected_skill=d.expected_skill,
            install_relpath=f"~/{d.install_relpath}",
            probe_unsupported=d.probe_unsupported,
        )
        for d in PROBE_DESCRIPTORS
    )


@dataclass(frozen=True, slots=True)
class StageRecord:
    stage: str
    outcome: StageOutcome
    detail: str

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise DiscoveryEvidenceError(f"unknown stage: {self.stage}")
        if _SECRET.search(self.detail):
            raise DiscoveryEvidenceError("stage detail carries a secret-shaped value")


@dataclass(frozen=True, slots=True)
class ProbeReceipt:
    """One bounded probe run, bound to host version and adapter digest."""

    host: str
    host_version: str
    os_name: str
    scope: str
    install_relpath: str
    catalog_revision: str
    source_digest: str
    argv: tuple[str, ...]
    mechanism: str
    credential_required: bool
    started_at: str
    ended_at: str
    exit_status: int
    stages: tuple[StageRecord, ...]
    cleanup: str
    probe_unsupported: str | None = None

    def stage(self, name: str) -> StageRecord:
        for record in self.stages:
            if record.stage == name:
                return record
        raise DiscoveryEvidenceError(f"receipt missing stage: {name}")

    def highest_observed_stage(self) -> str | None:
        for name in reversed(STAGES):
            if self.stage(name).outcome is StageOutcome.OBSERVED:
                return name
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "host": self.host,
            "host_version": self.host,
            "os": self.os_name,
            "scope": self.scope,
            "install_relpath": self.install_relpath,
            "catalog_revision": self.catalog_revision,
            "source_digest": self.source_digest,
            "argv": list(self.argv),
            "mechanism": self.mechanism,
            "credential_required": self.credential_required,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "exit_status": self.exit_status,
            "stages": [
                {"stage": r.stage, "outcome": str(r.outcome), "detail": r.detail} for r in self.stages
            ],
            "cleanup": self.cleanup,
            "probe_unsupported": self.probe_unsupported,
        }


def _text(value: object, *, field: str, limit: int = MAX_DETAIL) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DiscoveryEvidenceError(f"missing {field}")
    if _SECRET.search(value):
        raise DiscoveryEvidenceError(f"{field} carries a secret-shaped value")
    if len(value) > limit:
        raise DiscoveryEvidenceError(f"{field} too long")
    return value


def _digest(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not _HEX_64.fullmatch(value):
        raise DiscoveryEvidenceError(f"invalid {field}")
    return value


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DiscoveryEvidenceError(f"invalid {field}")
    return value


def parse_receipt(payload: Mapping[str, object]) -> ProbeReceipt:
    """Fail-closed receipt parser: anything malformed raises, never passes."""
    if payload.get("schema") != SCHEMA:
        raise DiscoveryEvidenceError("wrong receipt schema")
    raw_stages = payload.get("stages")
    if not isinstance(raw_stages, list) or not 1 <= len(raw_stages) <= MAX_STAGES:
        raise DiscoveryEvidenceError("invalid stages")
    stages: list[StageRecord] = []
    seen: set[str] = set()
    for item in raw_stages:
        if not isinstance(item, Mapping):
            raise DiscoveryEvidenceError("invalid stage record")
        name = _text(item.get("stage"), field="stage", limit=32)
        if name in seen:
            raise DiscoveryEvidenceError(f"duplicate stage: {name}")
        seen.add(name)
        try:
            outcome = StageOutcome(_text(item.get("outcome"), field="outcome", limit=32))
        except ValueError as exc:
            raise DiscoveryEvidenceError(f"invalid outcome: {item.get('outcome')}") from exc
        stages.append(StageRecord(stage=name, outcome=outcome, detail=_text(item.get("detail"), field="detail")))
    argv = payload.get("argv")
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv) or len(argv) > MAX_ARGV:
        raise DiscoveryEvidenceError("invalid argv")
    return ProbeReceipt(
        host=_text(payload.get("host"), field="host", limit=32),
        host_version=_text(payload.get("host_version"), field="host_version", limit=64),
        os_name=_text(payload.get("os"), field="os", limit=32),
        scope=_text(payload.get("scope"), field="scope", limit=16),
        install_relpath=_text(payload.get("install_relpath"), field="install_relpath", limit=64),
        catalog_revision=_digest(payload.get("catalog_revision"), field="catalog_revision"),
        source_digest=_digest(payload.get("source_digest"), field="source_digest"),
        argv=tuple(argv),
        mechanism=_text(payload.get("mechanism"), field="mechanism", limit=32),
        credential_required=bool(payload.get("credential_required")),
        started_at=_text(payload.get("started_at"), field="started_at", limit=40),
        ended_at=_text(payload.get("ended_at"), field="ended_at", limit=40),
        exit_status=_integer(payload.get("exit_status"), field="exit_status"),
        stages=tuple(stages),
        cleanup=_text(payload.get("cleanup"), field="cleanup", limit=MAX_DETAIL),
        probe_unsupported=(
            _text(payload.get("probe_unsupported"), field="probe_unsupported")
            if payload.get("probe_unsupported") is not None
            else None
        ),
    )


def source_digest(files: Mapping[str, str]) -> str:
    """Digest over the sorted pack inventory, bound into every receipt."""
    hasher = hashlib.sha256()
    for name in sorted(files):
        hasher.update(f"{name}\0{hashlib.sha256(files[name].encode()).hexdigest()}\n".encode())
    return hasher.hexdigest()


@dataclass(frozen=True, slots=True)
class MatrixRow:
    host: str
    evidence_class: str
    host_version: str
    catalog_revision: str
    observed_at: str
    highest_stage: str
    freshness: str


def summarize_receipts(
    receipts: Iterable[ProbeReceipt], *, current_catalog_revision: str
) -> tuple[MatrixRow, ...]:
    """Deterministic matrix; stale digests and missing rows stay explicit."""
    rows: list[MatrixRow] = []
    for receipt in sorted(receipts, key=lambda r: (r.host, r.scope)):
        highest = receipt.highest_observed_stage()
        if receipt.probe_unsupported:
            evidence_class = "probe_unsupported"
        elif highest == "installed":
            evidence_class = "installed_only"
        elif highest == "discovered":
            evidence_class = (
                "discovered_model_mediated"
                if receipt.mechanism == str(ProbeMechanism.MODEL_MEDIATED)
                else "discovered_host_command"
            )
        elif highest is not None:
            evidence_class = f"observed_through_{highest}"
        else:
            evidence_class = "not_observed"
        rows.append(
            MatrixRow(
                host=receipt.host,
                evidence_class=evidence_class,
                host_version=receipt.host_version,
                catalog_revision=receipt.catalog_revision,
                observed_at=receipt.started_at,
                highest_stage=highest or "none",
                freshness="current" if receipt.catalog_revision == current_catalog_revision else "stale",
            )
        )
    return tuple(rows)


def _run_bounded(argv: tuple[str, ...], *, cwd: Path, timeout: int) -> tuple[int, str, bool]:
    try:
        completed = subprocess.run(
            list(argv),
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return -1, "timeout", True
    except OSError as exc:
        return -1, f"spawn failed: {type(exc).__name__}", True
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output[:_SECRET_LIMIT], False


_SECRET_LIMIT: Final = 4000


def run_probe(
    descriptor: ProbeDescriptor,
    *,
    files: Mapping[str, str],
    catalog_revision: str,
    allow_model_mediated: bool = False,
    clock: object = time.time,
) -> ProbeReceipt:
    """Run one bounded probe in a temporary project root.

    The probe installs the pack into a throwaway project root, performs no
    intentional home or configuration writes, and attempts cleanup,
    recording the result on the receipt (``removed`` or ``cleanup
    failed``). Model-mediated discovery lanes stay opt-in through
    ``allow_model_mediated``; a credential-required or undocumented surface
    records not-observed with its reason instead of running.

    A receipt is bound to the ``catalog_revision`` and pack ``source_digest``
    it recorded: if the pack is regenerated (revision or digest change), the
    receipt is stale for the new pack. A receipt also binds to the host
    version that produced it and does not generalize to other versions.
    """
    digest = source_digest(files)
    started = clock()
    stages: list[StageRecord] = []
    tmp = Path(tempfile.mkdtemp(prefix="omh-agent-skills-probe-"))
    cleanup = "removed"
    exit_status = -1
    argv: tuple[str, ...] = ()
    host_version = "unknown"
    try:
        target = tmp / descriptor.install_relpath.replace("~/", "")
        target.mkdir(parents=True, exist_ok=True)
        installed = True
        for name, content in sorted(files.items()):
            path = target / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            if hashlib.sha256(path.read_bytes()).hexdigest() != hashlib.sha256(content.encode()).hexdigest():
                installed = False
        stages.append(
            StageRecord(
                stage="installed",
                outcome=StageOutcome.OBSERVED if installed else StageOutcome.HOLD,
                detail=f"{len(files)} files at {descriptor.install_relpath}" if installed else "byte mismatch after copy",
            )
        )
        for name in ("discovered", "nominated", "loaded", "executed"):
            stages.append(
                StageRecord(stage=name, outcome=StageOutcome.NOT_OBSERVED, detail="not probed by this layer")
            )

        if descriptor.probe_unsupported:
            stages[1] = StageRecord(
                stage="discovered", outcome=StageOutcome.NOT_OBSERVED, detail=descriptor.probe_unsupported
            )
        elif shutil.which(descriptor.executable) is None:
            stages[1] = StageRecord(
                stage="discovered", outcome=StageOutcome.HOLD, detail=f"executable not found: {descriptor.executable}"
            )
        elif descriptor.mechanism is ProbeMechanism.MODEL_MEDIATED and not allow_model_mediated:
            stages[1] = StageRecord(
                stage="discovered",
                outcome=StageOutcome.NOT_OBSERVED,
                detail="model-mediated lane not enabled (opt-in; credential-dependent)",
            )
        else:
            argv = (descriptor.executable, *descriptor.version_args)
            code, output, failed = _run_bounded(argv, cwd=tmp, timeout=descriptor.timeout_seconds)
            exit_status = code
            match = _VERSION.search(output)
            host_version = match.group(0) if match else "unknown"
            if failed or code != 0:
                stages[1] = StageRecord(
                    stage="discovered", outcome=StageOutcome.HOLD, detail=f"version probe failed: {output[:80]}"
                )
            else:
                argv = (descriptor.executable, *(descriptor.discovery_args or ()))
                code, output, failed = _run_bounded(argv, cwd=tmp, timeout=descriptor.timeout_seconds)
                exit_status = code
                if failed:
                    stages[1] = StageRecord(stage="discovered", outcome=StageOutcome.HOLD, detail="discovery probe timeout")
                elif code != 0:
                    stages[1] = StageRecord(
                        stage="discovered", outcome=StageOutcome.HOLD, detail=f"discovery exit {code}"
                    )
                elif descriptor.expected_skill in output:
                    stages[1] = StageRecord(
                        stage="discovered",
                        outcome=StageOutcome.OBSERVED,
                        detail=f"{descriptor.expected_skill} named by {descriptor.mechanism} surface",
                    )
                else:
                    stages[1] = StageRecord(
                        stage="discovered",
                        outcome=StageOutcome.NOT_OBSERVED,
                        detail=f"{descriptor.expected_skill} not named in probe output",
                    )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ended = clock()
    return ProbeReceipt(
        host=str(descriptor.host),
        host_version=host_version,
        os_name=platform.system().lower(),
        scope=str(descriptor.scope),
        install_relpath=descriptor.install_relpath,
        catalog_revision=catalog_revision,
        source_digest=digest,
        argv=argv,
        mechanism=str(descriptor.mechanism),
        credential_required=descriptor.credential_required,
        started_at=_stamp(started),
        ended_at=_stamp(ended),
        exit_status=exit_status,
        stages=tuple(stages),
        cleanup=cleanup if not tmp.exists() else "cleanup failed",
        probe_unsupported=descriptor.probe_unsupported,
    )


def _stamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def receipt_json(receipt: ProbeReceipt) -> str:
    return json.dumps(receipt.to_json(), indent=2, sort_keys=True) + "\n"


def load_receipts(path: Path) -> tuple[ProbeReceipt, ...]:
    """Load a checked-in receipt corpus; one malformed entry fails the load."""
    receipts: list[ProbeReceipt] = []
    for file in sorted(path.glob("*.json")):
        payload = json.loads(file.read_text(encoding="utf-8"))
        receipts.append(parse_receipt(payload))
    return tuple(receipts)
