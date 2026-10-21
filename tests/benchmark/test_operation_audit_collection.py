from __future__ import annotations

import json
import types
from pathlib import Path

import pytest

from . import test_operation_audit

audit = test_operation_audit.audit
runtime = test_operation_audit.runtime


@pytest.fixture
def collector(audit, tmp_path, monkeypatch):
    """Replace process/identity boundaries, retaining the real event checker."""
    descriptors = {}
    for role, revision in [("baseline", audit.route.BASELINE), ("candidate", "a" * 40)]:
        descriptor = {
            "purpose": audit.PURPOSE,
            "python": str(tmp_path / role / "bin/python"),
            "manifest": {
                "sha": revision,
                "source_root": str(tmp_path / role / "source"),
                "lock_sha256": "paired-lock",
                "cargo_lock_sha256": "paired-cargo-lock",
            },
        }
        path = tmp_path / (role + ".json")
        path.write_text(json.dumps(descriptor))
        descriptors[role] = str(path)
    args = types.SimpleNamespace(
        **descriptors, output=str(tmp_path / "report.json"), rows=4096, threads=2
    )
    monkeypatch.setattr(audit, "validate_provenance", lambda runtime: None)
    monkeypatch.setattr(audit.route, "validate_comparison", lambda runtimes: None)
    requests = []

    def install(fault=None):
        def process(command, **kwargs):
            request = json.loads(Path(command[-2]).read_text())
            requests.append(request)
            assert command[0] == request["runtime"]["python"]
            assert command[1] == "-I"
            assert kwargs["env"] == audit.route.clean_env(args.threads)
            assert Path(kwargs["cwd"]) == Path(args.output).with_suffix(".artifacts")
            baseline = request["runtime"]["manifest"]["sha"] == audit.route.BASELINE
            if request["sort"]:
                events = (
                    ["hash_vectors", "hash_maps", "output_sort"] if baseline else ["sort_first"]
                )
            else:
                events = [
                    "hash_vectors",
                    "hash_maps",
                    "firstseen_u64" if request["native_u64"] else "firstseen_u32",
                ]
            returncode = 0
            # Corrupt one observation; subsequent controls must still be collected.
            if len(requests) == 1:
                if fault == "process":
                    returncode = 7
                elif fault == "missing":
                    events = []
                elif fault == "unknown":
                    events.append("unrecognized_physical_operation")
            if returncode == 0:
                Path(command[-1]).write_text(
                    json.dumps({"purpose": audit.PURPOSE, "parity": "exact"})
                )
            stderr = "worker diagnostic\n" + "".join(
                audit.PREFIX + event + "\n" for event in events
            )
            return types.SimpleNamespace(
                returncode=returncode, stdout="worker stdout\n", stderr=stderr
            )

        monkeypatch.setattr(audit.subprocess, "run", process)

    return args, requests, install


def assert_collector_matrix(requests):
    expected = {
        (role, dtype, agg, sort, native)
        for role in ("baseline", "candidate")
        for dtype in ("int64", "float64")
        for agg, sort, native in [
            ("count", True, False),
            ("count", False, False),
            ("max", False, False),
            ("count", False, True),
            ("max", False, True),
        ]
    }
    actual = [
        (
            Path(request["runtime"]["python"]).parent.parent.name,
            request["dtype"],
            request["agg"],
            request["sort"],
            request["native_u64"],
        )
        for request in requests
    ]
    assert len(actual) == len(expected) == 20
    assert set(actual) == expected
    assert all(
        request["rows"] == 4096
        and request["threads"] == 2
        and request["purpose"] == "audit-only"
        and request["kind"] == "operations"
        for request in requests
    )


def test_collector_positive_controls_are_diagnostic_only(audit, collector):
    args, requests, install = collector
    install()
    audit.run(args)
    assert_collector_matrix(requests)
    report = json.loads(Path(args.output).read_text())
    assert report["status"] == "AUDIT_PASS"
    assert report["failures"] == []
    assert report["acceptance_usable"] is False
    assert report["purpose"] == audit.PURPOSE
    assert report["allocator_bound"] == "unknown"
    assert len(report["observations"]) == 20
    for observation in report["observations"]:
        assert observation["returncode"] == 0
        assert observation["result"]["parity"] == "exact"
        assert "error" not in observation
        assert sum(observation["events"].values()) > 0


@pytest.mark.parametrize(
    "fault,match",
    [
        ("process", "Audit worker failed"),
        ("missing", "positive physical control failed"),
        ("unknown", "Unknown observer event"),
    ],
)
def test_collector_retains_failed_physical_evidence(audit, collector, fault, match):
    args, requests, install = collector
    install(fault)
    with pytest.raises(SystemExit) as error:
        audit.run(args)
    assert error.value.code == 1
    assert_collector_matrix(requests)
    report = json.loads(Path(args.output).read_text())
    assert report["status"] == "FAILED"
    assert report["acceptance_usable"] is False
    assert len(report["observations"]) == 20
    assert len(report["failures"]) == 1
    failed = report["observations"][0]
    assert match in failed["error"]
    assert match in report["failures"][0]
    assert failed["returncode"] == (7 if fault == "process" else 0)
    assert Path(failed["request"]).is_file()
    assert Path(failed["response"]).is_file() is (fault != "process")
    artifacts = Path(args.output).with_suffix(".artifacts")
    assert (artifacts / (failed["name"] + "-stdout.log")).read_text() == "worker stdout\n"
    stderr = (artifacts / (failed["name"] + "-stderr.log")).read_text()
    assert "worker diagnostic" in stderr
    if fault == "missing":
        assert failed["events"] == dict.fromkeys(audit.EVENTS, 0)
        assert audit.PREFIX not in stderr
    elif fault == "unknown":
        assert audit.PREFIX + "unrecognized_physical_operation" in stderr
    else:
        assert audit.PREFIX + "hash_vectors" in stderr
    assert all("error" not in observation for observation in report["observations"][1:])
