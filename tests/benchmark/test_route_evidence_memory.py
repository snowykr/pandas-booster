from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest

from . import test_route_evidence_failures

driver = test_route_evidence_failures.driver
procfs = test_route_evidence_failures.procfs
worker_case = test_route_evidence_failures.worker_case


@pytest.mark.parametrize("kind", ["cold", "warm", "rss", "scaling", "evidence"])
@pytest.mark.parametrize("fail_on", [{1}, {2}, {1, 2, 3}])
def test_correct_work_rejects_missing_required_memory_samples(
    driver, worker_case, procfs, tmp_path, kind, fail_on
):
    request, _, expected = worker_case
    request["kind"] = kind
    procfs.fail_on = fail_on
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    driver.save(request_path, request)
    with pytest.raises(ValueError, match="Required Linux RSS evidence unavailable"):
        driver.worker(request_path, response_path)
    result = json.loads(response_path.read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "memory_validation"
    assert result["error"]["type"] == "ValueError"
    assert ("pre_input" in result) == (1 not in fail_on)
    assert ("rss" in result) == (2 not in fail_on)
    assert len(result["memory_errors"]) == len(fail_on & {1, 2})
    assert all(error["error"]["type"] == "FileNotFoundError" for error in result["memory_errors"])
    if kind == "evidence":
        assert result["parity"] is True
        assert result["public_pipeline_s"] > 0
    else:
        assert result["output_groups"] == len(expected)
        assert result["elapsed_s"] > 0


@pytest.mark.parametrize("fail_on", [set(), {1}, {3}])
def test_profile_memory_samples_never_hide_earlier_errors_or_reuse_stale_rss(
    driver, worker_case, procfs, tmp_path, fail_on
):
    request, rust, expected = worker_case
    profile = {"route": "sort_first", "routing_s": 0.0123}
    rust.profile_groupby_multi_max_i64_sorted = lambda *args: (
        [expected.index.get_level_values(i).to_numpy() for i in range(2)],
        expected.to_numpy(),
        profile,
    )
    procfs.fail_on = fail_on
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    driver.save(request_path, request)
    if fail_on:
        with pytest.raises(ValueError, match="Required Linux RSS evidence unavailable"):
            driver.worker(request_path, response_path)
    else:
        driver.worker(request_path, response_path)
    result = json.loads(response_path.read_text())
    assert result["status"] == ("failed" if fail_on else "passed")
    assert result["stage"] == ("memory_validation" if fail_on else "complete")
    assert result["profile"] == profile
    assert result["profile_outer_s"] >= 0
    assert result["profile_pipeline_s"] >= 0
    assert ("rss" in result) == (3 not in fail_on)
    if fail_on:
        assert len(result["memory_errors"]) == 1
        assert result["memory_errors"][0]["sample"] == ("pre_input" if 1 in fail_on else "profile")
    else:
        assert "memory_errors" not in result


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_otherwise_correct_unsupported_worker_is_not_accepted(
    driver, worker_case, procfs, tmp_path, monkeypatch, platform
):
    request, _, _ = worker_case
    request["kind"] = "cold"
    monkeypatch.setattr(driver.sys, "platform", platform)
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    driver.save(request_path, request)
    with pytest.raises(ValueError, match="Required Linux RSS evidence unavailable"):
        driver.worker(request_path, response_path)
    result = json.loads(response_path.read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "memory_validation"
    assert result["elapsed_s"] > 0
    assert "rss" not in result and "pre_input" not in result
    assert all(
        "requires Linux procfs" in error["error"]["message"] for error in result["memory_errors"]
    )
    assert procfs.reads == 0


def test_rss_parser_records_exact_linux_bytes(driver, procfs):
    assert driver.rss() == {"VmRSS": 100 * 1024, "VmHWM": 200 * 1024}


def test_procfs_double_recognizes_windows_path_spelling(procfs):
    assert Path.read_text(PureWindowsPath("/proc/self/status")) == procfs.text
    assert procfs.reads == 1


@pytest.mark.parametrize(
    "text",
    [
        "",
        "VmRSS: 100 kB\n",
        "VmHWM: 200 kB\n",
        "VmRSS:\nVmHWM: 200 kB\n",
        "VmRSS: invalid kB\nVmHWM: 200 kB\n",
        "VmRSS: 100 MB\nVmHWM: 200 kB\n",
        "VmRSS: 0 kB\nVmHWM: 200 kB\n",
        "VmRSS: 100 kB\nVmHWM: -1 kB\n",
    ],
)
def test_rss_parser_rejects_incomplete_or_invalid_evidence(driver, procfs, text):
    procfs.text = text
    with pytest.raises(ValueError):
        driver.rss()


def test_memory_recorder_preserves_unreadable_procfs_error(driver, monkeypatch):
    def unreadable():
        raise PermissionError("procfs denied")

    monkeypatch.setattr(driver, "rss", unreadable)
    result = {}
    driver.record_rss(result, "pre_input", "pre_input")
    assert result == {
        "memory_errors": [
            {
                "sample": "pre_input",
                "field": "pre_input",
                "error": {"type": "PermissionError", "message": "procfs denied"},
            }
        ]
    }
