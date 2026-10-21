from __future__ import annotations

import json
import sys
from pathlib import Path, PureWindowsPath
from types import ModuleType, SimpleNamespace

import pytest

from . import _route_evidence_helpers, test_route_evidence

driver = test_route_evidence.driver
procfs = _route_evidence_helpers.procfs


@pytest.fixture
def memory_mode(request, driver, procfs, monkeypatch):
    mode = request.param
    if mode == "missing":
        procfs.fail_on = {1, 2, 3}
    else:
        monkeypatch.setattr(driver.sys, "platform", mode)
    return mode


@pytest.fixture
def worker_case(driver, tmp_path, monkeypatch, procfs):
    import numpy as np
    import pandas as pd

    case = {
        "name": "primary",
        "n": 8,
        "widths": [2, 2],
        "dtype": "int64",
        "sort": True,
        "agg": "max",
        "seed": 42,
    }
    directory = tmp_path / "dataset"
    manifest = driver.create_dataset(case, directory)
    arrays = {Path(name).stem: np.load(directory / name) for name in manifest["files"]}
    frame = pd.DataFrame(arrays)
    expected = frame.groupby(["k0", "k1"])["value"].max()
    package = ModuleType("pandas_booster")
    package.__path__ = []
    rust = ModuleType("pandas_booster._rust")
    package._rust = rust
    monkeypatch.setitem(sys.modules, "pandas_booster", package)
    monkeypatch.setitem(sys.modules, "pandas_booster._rust", rust)
    monkeypatch.setattr(driver, "validate_acceptance", lambda *args: None)
    monkeypatch.setattr(driver, "validate_runtime", lambda *args: {"validated": True})
    monkeypatch.setattr(
        pd.DataFrame,
        "booster",
        property(lambda self: SimpleNamespace(groupby=lambda *a, **kw: expected)),
        raising=False,
    )
    request = {
        "purpose": "acceptance",
        "runtime": {},
        "threads": 1,
        "dataset": str(directory),
        "dataset_manifest": manifest,
        "case": case,
        "kind": "evidence",
        "candidate": True,
    }
    return request, rust, expected


@pytest.mark.parametrize("memory_mode", ["linux", "missing", "win32", "darwin"], indirect=True)
def test_profile_parity_failure_preserves_raw_profile_and_measurements(
    driver, worker_case, tmp_path, memory_mode
):
    request, rust, expected = worker_case
    profile = {"route": "sort_first", "routing_s": 0.0123}
    rust.profile_groupby_multi_max_i64_sorted = lambda *args: (
        [expected.index.get_level_values(i).to_numpy() for i in range(2)],
        expected.to_numpy() + 1,
        profile,
    )
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    driver.save(request_path, request)
    with pytest.raises(AssertionError):
        driver.worker(request_path, response_path)
    result = json.loads(response_path.read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "profile_parity"
    assert result["error"]["type"] == "AssertionError"
    assert result["profile"] == profile
    for field in ("profile_outer_s", "profile_cpu_s", "public_pipeline_s", "public_cpu_s"):
        assert result[field] >= 0
    if memory_mode == "linux":
        assert result["rss"] == {"VmRSS": 100 * 1024, "VmHWM": 200 * 1024}
        assert "memory_errors" not in result
    else:
        assert "rss" not in result and "pre_input" not in result
        assert [error["sample"] for error in result["memory_errors"]] == [
            "pre_input",
            "public",
            "profile",
        ]
    assert not {"reference", "public_output", "profiled_output", "out_keys"} & result.keys()


@pytest.mark.parametrize("kind", ["cold", "warm", "rss", "scaling"])
@pytest.mark.parametrize("memory_mode", ["linux", "missing", "win32", "darwin"], indirect=True)
def test_count_failure_preserves_ordinary_measurements(
    driver, worker_case, tmp_path, monkeypatch, kind, memory_mode
):
    import pandas as pd

    request, _, expected = worker_case
    request["kind"] = kind
    monkeypatch.setattr(
        pd.DataFrame,
        "booster",
        property(lambda self: SimpleNamespace(groupby=lambda *a, **kw: expected.iloc[:0])),
    )
    request_path, response_path = tmp_path / "request.json", tmp_path / "response.json"
    driver.save(request_path, request)
    with pytest.raises(ValueError, match="output group count"):
        driver.worker(request_path, response_path)
    result = json.loads(response_path.read_text())
    assert result["status"] == "failed"
    assert result["stage"] == "ordinary_output_validation"
    assert result["output_groups"] == 0
    assert result["elapsed_s"] > 0
    assert result["cpu_s"] >= 0
    if memory_mode == "linux":
        assert result["rss"] == {"VmRSS": 100 * 1024, "VmHWM": 200 * 1024}
        assert "memory_errors" not in result
    else:
        assert "rss" not in result and "pre_input" not in result
        assert [error["sample"] for error in result["memory_errors"]] == ["pre_input", "ordinary"]
    assert "reference" not in result


@pytest.mark.parametrize("nonzero_exit", [False, True])
@pytest.mark.parametrize("bounded", ["full", "cases", "memory"])
@pytest.mark.parametrize("failure_stage", ["ordinary_output_validation", "memory_validation"])
def test_failed_partial_responses_never_enter_accepted_pairs(
    driver, tmp_path, monkeypatch, nonzero_exit, bounded, failure_stage
):
    case = {
        "name": "primary",
        "n": 8,
        "widths": [2, 2],
        "dtype": "int64",
        "sort": True,
        "agg": "count",
        "seed": 42,
    }
    for role in ("baseline", "candidate"):
        driver.save(
            tmp_path / (role + ".json"),
            {
                "purpose": "acceptance",
                "python": str(tmp_path / role / "python"),
                "dependencies": [],
                "wheel": role + ".whl",
                "manifest": {
                    "purpose": "acceptance",
                    "snapshot_overlays": {"Cargo.lock": {}},
                    "sha": driver.BASELINE,
                    "source_root": str(tmp_path / role),
                    "lock_sha256": "lock",
                    "cargo_lock_sha256": "cargo-lock",
                },
            },
        )
    monkeypatch.setattr(driver, "validate_acceptance", lambda *a: None)
    monkeypatch.setattr(driver, "validate_comparison", lambda *a: None)
    monkeypatch.setattr(driver.os, "uname", lambda: ("Linux", "test", "", "", ""), raising=False)
    monkeypatch.setattr(driver.os, "sched_getaffinity", lambda pid: {0}, raising=False)
    read_text = Path.read_text

    def read_cpuinfo(path, *args, **kwargs):
        if path.as_posix() == "/proc/cpuinfo":
            return "model name: Intel N100\n"
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_cpuinfo)
    assert Path.read_text(PureWindowsPath("/proc/cpuinfo")) == "model name: Intel N100\n"
    other_case = dict(case, name="low")
    monkeypatch.setattr(driver, "case_specs", lambda *a: [case, other_case])

    def fail_worker(args, cwd, env):
        result = {
            "status": "failed",
            "stage": failure_stage,
            "error": {"type": "ValueError", "message": "required evidence failed"},
            "elapsed_s": 1.0,
            "cpu_s": 0.5,
            "profile": {"routing_s": 0.1},
        }
        if failure_stage == "memory_validation":
            result["memory_errors"] = [
                {
                    "sample": "pre_input",
                    "field": "pre_input",
                    "error": {"type": "FileNotFoundError", "message": "no procfs"},
                }
            ]
        else:
            result["rss"] = {"VmHWM": 123}
        driver.save(Path(args[-1]), result)
        if nonzero_exit:
            raise RuntimeError("worker exited with required evidence failure")
        return ""

    monkeypatch.setattr(driver, "command", fail_worker)
    output = tmp_path / "report.json"
    driver.run(
        SimpleNamespace(
            baseline=str(tmp_path / "baseline.json"),
            candidate=str(tmp_path / "candidate.json"),
            output=str(output),
            host="n100",
            quick=bounded == "full",
            threads=[1],
            cases=None if bounded == "full" else ["primary"],
            memory_only=bounded == "memory",
        )
    )
    report = json.loads(output.read_text())
    assert report["raw"]
    assert all(observation["error"] for observation in report["raw"])
    assert all(observation["result"]["elapsed_s"] == 1 for observation in report["raw"])
    assert all(
        observation["result"]["profile"] == {"routing_s": 0.1} for observation in report["raw"]
    )
    if failure_stage == "memory_validation":
        assert all(observation["result"]["memory_errors"] for observation in report["raw"])
        assert all("rss" not in observation["result"] for observation in report["raw"])
    assert report["summary"] == []
    if bounded == "memory":
        assert {row["kind"] for row in report["raw"]} == {"evidence", "rss"}
        assert sum(row["kind"] == "rss" for row in report["raw"]) == 20
        assert any("Memory-only collection" in gap for gap in report["unmet_gates"])
    else:
        assert "Incomplete paired samples" in report["unmet_gates"]
        assert {row["kind"] for row in report["raw"]} == {"evidence", "rss", "cold", "warm"}
    if bounded != "full":
        assert {row["case"]["name"] for row in report["raw"]} == {"primary"}
        assert "Explicit case subset: incomplete certification matrix" in report["unmet_gates"]
        assert report["status"] == "DRAFT"
        if bounded == "cases":
            for mode in ("cold", "warm"):
                rows = [row for row in report["raw"] if row["kind"] == mode]
                assert len(rows) == 80
                assert {row["pair"] for row in rows} == set(range(40))
    else:
        assert {row["case"]["name"] for row in report["raw"]} == {"primary", "low"}
        assert report["status"] == "UNCERTIFIED"
    assert report["certified"] is False


def test_build_transcript_captures_stderr_without_polluting_stdout(driver, tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="inventory stdout\n", stderr="build stderr\n"
        ),
    )
    transcript = tmp_path / "build.log"
    assert driver.command(["build"], tmp_path, {}, transcript=transcript) == "inventory stdout\n"
    assert transcript.read_text() == "inventory stdout\nbuild stderr\n"
    assert driver.command(["inventory"], tmp_path, {}) == "inventory stdout\n"
