from __future__ import annotations

from pathlib import Path

import pytest

from . import _operation_audit_helpers

audit = _operation_audit_helpers.audit
runtime = _operation_audit_helpers.runtime
seal = _operation_audit_helpers.seal


def test_valid_audit_control_is_not_acceptance(audit, runtime):
    audit.validate_provenance(runtime)
    with pytest.raises(ValueError, match="purpose"):
        audit.route.validate_acceptance(runtime)


def test_relabeling_diagnostic_build_does_not_authorize_acceptance(audit, runtime):
    runtime["purpose"] = "acceptance"
    runtime["manifest"]["purpose"] = "acceptance"
    seal(audit, runtime)
    with pytest.raises(ValueError, match="non-Cargo"):
        audit.route.validate_acceptance(runtime)


@pytest.mark.parametrize("target", ["descriptor", "manifest", "embedded"])
def test_audit_purpose_mismatch(audit, runtime, target):
    if target == "descriptor":
        runtime["purpose"] = "acceptance"
    elif target == "manifest":
        runtime["manifest"]["purpose"] = "acceptance"
        seal(audit, runtime)
    else:
        audit.route.save(Path(runtime["package"]) / "_route_evidence_manifest.json", {})
    with pytest.raises(ValueError, match="purpose|Embedded"):
        audit.validate_provenance(runtime)


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("patch", "patch/revision"),
        ("patch-hash", "patch/revision"),
        ("patch-location", "No such file"),
        ("overlay-revision", "patch/revision"),
        ("revision", "revision"),
        ("changed-before", "changed-file"),
        ("changed-after", "changed-file"),
        ("unknown-overlay", "Unknown"),
        ("undeclared-overlay", "Unknown"),
        ("source", "resulting sources"),
        ("declared-source", "Undeclared audit source"),
        ("original-source", "Original source hashes"),
        ("archive", "Original source archive"),
        ("build", "build log"),
        ("wheel", "wheel hash"),
        ("extension", "installed extension"),
    ],
)
def test_reject_tampered_provenance(audit, runtime, mutation, match):
    manifest = runtime["manifest"]
    receipt = manifest["observer_overlay"]
    source = Path(manifest["source_root"])
    if mutation == "patch":
        Path(receipt["path"]).write_bytes(b"tampered patch")
    elif mutation == "patch-hash":
        receipt["sha256"] = "0" * 64
    elif mutation == "patch-location":
        receipt["path"] += ".missing"
    elif mutation == "overlay-revision":
        receipt["revision"] = "b" * 40
    elif mutation == "revision":
        manifest["sha"] = "b" * 40
    elif mutation.startswith("changed-"):
        next(iter(receipt["changed_files"].values()))[mutation.split("-")[1]] = "0" * 64
    elif mutation == "unknown-overlay":
        manifest["snapshot_overlays"]["extra.rs"] = {}
    elif mutation == "undeclared-overlay":
        del manifest["snapshot_overlays"]["operation-observers"]
    elif mutation in ("source", "declared-source"):
        (source / "Cargo.toml").write_text("malicious source\n")
        if mutation == "declared-source":
            manifest["sources"] = audit.route.files_manifest(source)
    elif mutation == "original-source":
        manifest["git_sources"]["Cargo.toml"] = "0" * 64
    elif mutation == "archive":
        (source.parent / "source.tar").write_bytes(b"tampered")
    elif mutation == "build":
        (source.parent / "build.log").write_text("tampered")
    # Seal intentional manifest edits; wheel/extension tampering happens afterwards.
    seal(audit, runtime)
    if mutation in ("wheel", "extension"):
        Path(runtime[mutation]).write_bytes(b"tampered")
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        audit.validate_provenance(runtime)


def test_git_revision_independently_bound(audit, runtime, monkeypatch):
    monkeypatch.setattr(audit, "git_archive", lambda repo, rev: b"another revision")
    with pytest.raises(ValueError, match="Git revision/archive"):
        audit.validate_provenance(runtime)


@pytest.mark.parametrize("kind", ["cold", "warm", "rss", "scaling", "evidence"])
def test_audit_worker_refuses_acceptance_modes(audit, kind, monkeypatch):
    calls = []
    monkeypatch.setattr(audit, "validate_provenance", lambda *a: calls.append(a))
    with pytest.raises(ValueError, match="only accepts"):
        audit.audit_worker({"purpose": audit.PURPOSE, "kind": kind})
    assert calls == []


def test_unmatched_operation_edit_is_fatal(audit):
    with pytest.raises(ValueError, match="Unmatched"):
        audit.overlay({name: b"changed" for name, _, _, _ in audit.EDITS}, audit.route.BASELINE)


def test_missing_positive_controls_prevent_synthetic_zeros(audit):
    empty = dict.fromkeys(audit.EVENTS, 0)
    for baseline, sort, native in [
        (True, True, False),
        (False, True, False),
        (False, False, False),
    ]:
        with pytest.raises(ValueError):
            audit.check_events(empty, baseline, sort, native)
    candidate = dict(empty, sort_first=1)
    audit.check_events(candidate, False, True, False)
    baseline = dict(empty, hash_vectors=1, hash_maps=7, output_sort=1)
    audit.check_events(baseline, True, True, False)
    for native in (False, True):
        unsorted = dict(empty, hash_vectors=1, hash_maps=7)
        unsorted["firstseen_u64" if native else "firstseen_u32"] = 1
        audit.check_events(unsorted, False, False, native)
        unsorted["output_sort"] = 1
        with pytest.raises(ValueError, match="Unsorted"):
            audit.check_events(unsorted, False, False, native)
