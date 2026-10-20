from __future__ import annotations

import pytest

from . import test_route_evidence

driver = test_route_evidence.driver


def test_bootstrap_uses_matched_pairs_and_ratio_of_medians(driver):
    pairs = [{"baseline": float(i), "candidate": i * 0.8} for i in range(1, 41)]
    result = driver.paired_summary(pairs)
    assert result["ratio"] == pytest.approx(0.8)
    assert result["u95"] == pytest.approx(0.8)
    assert result["pairs"] == 40
    assert result["bootstrap_draws"] == 10000
    assert result["bootstrap_seed"] == 42
    assert result == driver.paired_summary(pairs)
    # Ratio of medians is not the median of individual pair ratios.
    unequal = [
        {"baseline": 1.0, "candidate": 3.0},
        {"baseline": 2.0, "candidate": 100.0},
        {"baseline": 100.0, "candidate": 1.0},
    ]
    assert driver.paired_summary(unequal, 100)["ratio"] == 1.5


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_bootstrap_rejects_invalid_durations(driver, value):
    with pytest.raises(ValueError, match="finite and positive"):
        driver.paired_summary([{"baseline": 1.0, "candidate": value}], 10)


def test_bootstrap_rejects_empty_evidence(driver):
    with pytest.raises(ValueError):
        driver.paired_summary([])


def test_primary_matrix_and_domain_bounds(driver):
    cases = driver.case_specs()
    assert len(cases) == 14
    primary = [c for c in cases if c["name"] == "primary"]
    assert {c["agg"] for c in primary} == {"max", "count"}
    assert all(c["n"] == 5000000 and c["seed"] == 42 and c["widths"] == [500] * 3 for c in primary)
    assert {len(c["widths"]) for c in cases if c["dtype"] == "int64"} == {2, 10}
    assert all(c["n"] == 1000000 for c in cases if c["name"].startswith("packed"))
    assert all(c["n"] == 100003 for c in driver.case_specs(True))


def test_profile_is_fail_closed_for_missing_and_wrong_route(driver):
    gaps = driver.profile_gaps({"route": "hash_first"}, candidate=True, admitted=True)
    assert "route mismatch: expected sort_first" in gaps
    assert "missing candidate diagnostic: estimated_peak_bytes" in gaps
    assert "admitted route lacks zero hash-engine call evidence" in gaps
    assert driver.profile_gaps({"route": "hash_first"}, candidate=False, admitted=True) == []


def test_phase_accounting_excludes_inclusive_permutation(driver):
    profile = dict.fromkeys(driver.LEAVES + driver.FALLBACK_LEAVES, 1.0)
    profile.update(
        rust_total_s=11.0, rust_elapsed_s=11.1, conversion_s=0.5, sort_first_permutation_s=1000.0
    )
    assert driver.phase_gaps(profile, 11.6) == []
    profile["rust_total_s"] += 1000
    assert "Rust total is not the sum of disjoint leaves" in driver.phase_gaps(profile, 11.6)
    profile["rust_total_s"] = 11.0
    assert "leaf sum plus conversion exceeds independent profile outer" in driver.phase_gaps(
        profile, 10.0
    )
    del profile["routing_s"]
    assert "missing timing: routing_s" in driver.phase_gaps(profile, 11.6)


def test_nested_failure_strings_are_not_accepted_as_metadata(driver):
    assert driver.failure_strings({"profile": {"route": "fail: unavailable"}}) == [
        "fail: unavailable"
    ]
    assert driver.failure_strings({"route": "hash_first", "samples": [1, 2]}) == []


def test_memory_gate_checks_each_worker_and_analytical_cap(driver):
    mib = driver.MIB
    worker = {"pre_input": {"VmRSS": 10 * mib}, "rss": {"VmHWM": 100 * mib}}
    assert driver.memory_gaps(worker, 100 * mib, 20 * mib, 30 * mib, 4) == []
    worker["rss"]["VmHWM"] = 143 * mib
    assert driver.memory_gaps(worker, 100 * mib, 20 * mib, 30 * mib, 4) == ["RSS envelope exceeded"]
    assert driver.memory_gaps(worker, 100 * mib, 0, 513 * mib, 1) == [
        "analytical route peak outside 512 MiB cap"
    ]
    worker["rss"]["VmHWM"] = 100 * mib
    assert driver.memory_gaps(worker, 100 * mib, 0, 513 * mib, 1, admitted=False) == []
