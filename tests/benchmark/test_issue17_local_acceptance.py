"""Validate retrospective local accounting without executing benchmark workers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
RECEIPT = ROOT / "benchmarks/evidence/issue17-local-acceptance.json"


@pytest.fixture(scope="module")
def evidence():
    receipt = json.loads(RECEIPT.read_text())
    for pin in receipt["provenance"].values():
        if isinstance(pin, dict):
            assert hashlib.sha256((ROOT / pin["path"]).read_bytes()).hexdigest() == pin["sha256"]
    summary = json.loads((ROOT / receipt["provenance"]["recovery_summary"]["path"]).read_text())
    expected = {
        "recovery/" + item.get("file", item.get("report")): item["sha256"]
        for item in summary["raw_reports"] + summary["independent_operation_audits"]
    }
    assert set(receipt["provenance"]["validated_recovery_members"]) == set(expected)
    reports = {}
    with tarfile.open(ROOT / receipt["provenance"]["recovery_archive"]["path"]) as archive:
        members = archive.getmembers()
        for name, digest in expected.items():
            matches = [member for member in members if member.name == name]
            assert len(matches) == 1, name
            stream = archive.extractfile(matches[0])
            assert stream is not None
            payload = stream.read()
            assert hashlib.sha256(payload).hexdigest() == digest, name
            reports[name.removeprefix("recovery/")] = json.loads(payload)
    return receipt, summary, reports


def primary_max_evidence(report, pins):
    rows = [
        row for row in report["raw"] if row["kind"] == "evidence" and row["case"]["agg"] == "max"
    ]
    assert len(rows) == 2
    assert {row["role"] for row in rows} == {"baseline", "candidate"}
    for row in rows:
        assert row["threads"] == 4
        assert row["case"] == {
            "name": "primary",
            "n": 5000000,
            "widths": [500, 500, 500],
            "dtype": "float64",
            "sort": True,
            "agg": "max",
            "seed": 42,
        }
        result = row["result"]
        expected_sha = pins[
            "baseline_sha" if row["role"] == "baseline" else "measured_candidate_sha"
        ]
        assert result["identity"]["sha"] == expected_sha
        assert result["status"] == "passed" and result["parity"] is True
        assert result["groups"] == 4901309
        for field in (
            "prepare_inputs_s",
            "python_normalize_s",
            "pandas_index_construction_s",
            "profile_pipeline_s",
            "public_pipeline_s",
        ):
            assert math.isfinite(result[field]) and result[field] > 0
        profile = result["profile"]
        for field in (
            "hash_build_s",
            "partition_scatter_s",
            "partition_aggregation_s",
            "flatten_s",
            "sort_key_construction_s",
            "radix_sort_s",
            "sorted_materialization_s",
            "conversion_s",
        ):
            assert math.isfinite(profile[field]) and profile[field] >= 0
        if row["role"] == "candidate":
            assert profile["route"] == "sort_first"
            assert profile["hash_first_aggregation_count"] == 0
            assert profile["post_aggregation_sort_count"] == 0
            # Inclusive permutation timing is intentionally excluded from this sum.
            leaves = sum(
                profile[field]
                for field in (
                    "routing_s",
                    "sort_key_construction_s",
                    "radix_sort_s",
                    "sort_first_boundary_s",
                    "sort_first_setup_s",
                    "sort_first_segment_scan_s",
                )
            )
            assert 0 <= profile["rust_elapsed_s"] - leaves < 0.01
        else:
            assert profile["route"] == "hash_first"


def test_original_criteria_have_archived_max_phase_and_route_evidence(evidence):
    receipt, summary, reports = evidence
    pins = receipt["source_equivalence"]
    assert pins["baseline_sha"] == summary["baseline_sha"]
    assert pins["measured_candidate_sha"] == summary["native_candidate_sha"]
    assert pins["reviewed_pr20_sha"] == receipt["pr20"]["head_sha"]
    primary_max_evidence(reports["sampler-primary-t4.json"], pins)
    profiles = [p for p in summary["candidate_profiles"] if p["case"] == "primary"]
    assert len(profiles) == 6
    assert {(p["agg"], p["threads"]) for p in profiles} == {
        (agg, threads) for agg in ("max", "count") for threads in (1, 2, 4)
    }
    for row in profiles:
        assert row["profile"]["route"] == "sort_first"
        assert row["profile"]["hash_first_aggregation_count"] == 0
        assert row["profile"]["post_aggregation_sort_count"] == 0
    assert [item["id"] for item in receipt["original_acceptance"]] == [
        "phase_attribution",
        "sort_first_route",
        "pandas_semantics",
        "plateau_improvement",
    ]
    assert all(item["status"] == "supported_locally" for item in receipt["original_acceptance"])


def test_acceptance_runtime_pins_and_recorded_public_parity(evidence):
    _, summary, reports = evidence
    profiles = []
    for spec in summary["raw_reports"]:
        report = reports[spec["file"]]
        assert report["status"] == "DRAFT" and report["certified"] is False
        assert report["unmet_gates"] == spec["unmet_gates_unchanged"]
        for role, runtime in report["runtimes"].items():
            expected = summary["runtimes"][role]
            assert runtime["purpose"] == runtime["manifest"]["purpose"] == "acceptance"
            assert runtime["manifest"]["sha"] == expected["source_sha"]
            assert runtime["wheel_sha256"] == expected["wheel_sha256"]
        for row in report["raw"]:
            assert (
                row["result"]["identity"]["sha"] == summary["runtimes"][row["role"]]["source_sha"]
            )
            if row["kind"] == "evidence" and row["role"] == "candidate" and row["case"]["sort"]:
                assert row["result"]["parity"] is True
                profiles.append(row)
    assert len(profiles) == len(summary["candidate_profiles"]) == 26
    assert all(row["public_parity"] is True for row in summary["candidate_profiles"])


def test_independent_audits_are_operation_not_timing_certificates(evidence):
    _, summary, reports = evidence
    for spec in summary["independent_operation_audits"]:
        report = reports[spec["report"]]
        assert report["purpose"] == "audit-only" and report["acceptance_usable"] is False
        assert report["allocator_bound"] == "unknown" and report["status"] == "AUDIT_PASS"
        assert len(report["observations"]) == len(spec["observations"]) == 20
        for row, recorded in zip(report["observations"], spec["observations"]):
            assert row["name"] == recorded["name"]
            assert row["events"] == recorded["events"]
            assert row["result"]["parity"] == "exact"
        for dtype in ("float64", "int64"):
            row = next(
                r
                for r in spec["observations"]
                if r["name"] == f"candidate-count-sorted-public-{dtype}"
            )
            assert row["events"] == {
                "firstseen_u32": 0,
                "firstseen_u64": 0,
                "hash_maps": 0,
                "hash_vectors": 0,
                "output_sort": 0,
                "sort_first": 1,
            }
            baseline = next(
                r
                for r in spec["observations"]
                if r["name"] == f"baseline-count-sorted-public-{dtype}"
            )
            assert baseline["events"]["hash_maps"] > 0
            assert baseline["events"]["output_sort"] == 1


def test_t4_improvement_arithmetic_and_unchanged_thresholds(evidence):
    receipt, summary, reports = evidence
    observations = receipt["primary_t4_performance"]
    assert {(row["agg"], row["mode"]) for row in observations} == {
        (agg, mode) for agg in ("max", "count") for mode in ("cold", "warm")
    }
    for row in observations:
        recorded = next(
            p
            for p in summary["performance"]
            if (p["case"], p["threads"], p["agg"], p["mode"])
            == ("primary", 4, row["agg"], row["mode"])
        )
        raw = next(
            p
            for p in reports["sampler-primary-t4.json"]["summary"]
            if (p["case"]["agg"], p["mode"]) == (row["agg"], row["mode"])
        )
        for field, value in row.items():
            assert value == recorded[field]
        for field in ("pairs", "ratio", "u95"):
            assert row[field] == raw[field]
        assert row["pairs"] == 40
        assert row["ratio"] == pytest.approx(row["candidate_median_s"] / row["baseline_median_s"])
        assert row["u95"] <= (1.05 if row["mode"] == "cold" else 1.0)
        if row["mode"] == "warm":
            assert row["ratio"] <= 0.90 and row["u95"] < 1.00
    assert summary["method"]["primary_T4"] == "warm ratio<=0.90 AND U95<1.00; cold U95<=1.05"
    assert summary["method"]["control_domain_T4"] == "cold/warm U95<=1.05"
    binding = [p for p in summary["performance"] if p["threads"] == 4]
    assert all(row["passed"] is True for row in binding)
    assert all(p["passed"] is None for p in summary["performance"] if p["threads"] != 4)


def test_unknown_unsorted_envelope_and_historical_failures_are_not_promoted(evidence):
    receipt, summary, _ = evidence
    goals = receipt["expanded_goals_not_original_requirements"]
    model = goals["memory"]
    unsorted = [m for m in summary["memory"] if m["case"] == "unsorted"]
    sorted_cells = [m for m in summary["memory"] if m["case"] != "unsorted"]
    assert len(sorted_cells) == model["sorted_full_envelope_cells_passed"] == 26
    assert all(m["full_envelope_pass"] is True for m in sorted_cells)
    assert len(unsorted) == model["unsorted_measured_regression_cells_passed"] == 4
    assert all(m["full_envelope_pass"] is None for m in unsorted)
    assert all(m["measured_regression_branch_pass"] is True for m in unsorted)
    assert all(
        len(m["candidate_workers"]) == model["workers_per_candidate_cell"] == 10
        for m in summary["memory"]
    )
    assert model["unsorted_allocator_live_payload_model"] == "UNKNOWN"
    assert model["unsorted_full_envelope_pass"] is None
    assert model["unsorted_analytical_limit_bytes"] is None
    for key in ("certified", "merge_authorized", "issue_closure_authorized"):
        assert receipt[key] is False
    historical = json.loads(
        (ROOT / receipt["provenance"]["historical_summary"]["path"]).read_text()
    )
    assert any("Primary T4 warm gate FAILED" in text for text in historical["outstanding"])
    assert any("20 total" in text and "FAILED" in text for text in historical["outstanding"])


@pytest.mark.parametrize(
    "mutation", ["sha", "role", "case", "parity", "phase", "route", "duplicate"]
)
def test_primary_evidence_rejects_wrong_missing_or_ambiguous_observations(evidence, mutation):
    receipt, _, reports = evidence
    report = copy.deepcopy(reports["sampler-primary-t4.json"])
    row = next(
        r
        for r in report["raw"]
        if r["kind"] == "evidence" and r["case"]["agg"] == "max" and r["role"] == "candidate"
    )
    if mutation == "sha":
        row["result"]["identity"]["sha"] = receipt["source_equivalence"]["baseline_sha"]
    elif mutation == "role":
        row["role"] = "baseline"
    elif mutation == "case":
        row["case"]["sort"] = False
    elif mutation == "parity":
        row["result"]["parity"] = False
    elif mutation == "phase":
        del row["result"]["pandas_index_construction_s"]
    elif mutation == "route":
        row["result"]["profile"]["hash_first_aggregation_count"] = 1
    else:
        report["raw"].append(copy.deepcopy(row))
    with pytest.raises((AssertionError, KeyError)):
        primary_max_evidence(report, receipt["source_equivalence"])
