from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import pytest
from conftest import _BENCHMARK_PATH


def _run_worker_payload(benchmark_module, monkeypatch, payload):
    def fake_worker(**kwargs):
        return {"ok": True, "kwargs": kwargs}

    monkeypatch.setattr(
        sys,
        "argv",
        [str(_BENCHMARK_PATH), "--worker", json.dumps(payload)],
    )
    return benchmark_module._cli_main(benchmark_worker_func=fake_worker)


def test_worker_rejects_non_object_json(benchmark_module, monkeypatch):
    monkeypatch.setattr(sys, "argv", [str(_BENCHMARK_PATH), "--worker", "[]"])

    with pytest.raises(ValueError, match="object"):
        benchmark_module._cli_main(benchmark_worker_func=lambda **kwargs: {"ok": True})


def test_worker_rejects_output_file_outside_temp_root(benchmark_module, monkeypatch):
    output_path = (
        Path(__file__).resolve().parents[2]
        / f"missing-worker-output-{uuid.uuid4().hex}"
        / "blocked-worker-output.json"
    )

    with pytest.raises(ValueError, match="output_file"):
        _run_worker_payload(
            benchmark_module,
            monkeypatch,
            {"preset": "1key", "output_file": str(output_path)},
        )


def test_worker_rejects_symlink_output_file(benchmark_module, monkeypatch):
    temp_root = Path(tempfile.gettempdir())
    target = temp_root / f"pandas-booster-worker-target-{uuid.uuid4().hex}.json"
    output_path = temp_root / f"pandas-booster-worker-output-{uuid.uuid4().hex}.json"
    target.write_text("{}", encoding="utf-8")
    output_path.symlink_to(target)

    try:
        with pytest.raises(ValueError, match="output_file"):
            _run_worker_payload(
                benchmark_module,
                monkeypatch,
                {"preset": "1key", "output_file": str(output_path)},
            )
    finally:
        output_path.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


def test_worker_rejects_nested_temp_output_file(benchmark_module, monkeypatch, tmp_path):
    output_path = tmp_path / "worker-output.json"

    with pytest.raises(ValueError, match="output_file"):
        _run_worker_payload(
            benchmark_module,
            monkeypatch,
            {"preset": "1key", "output_file": str(output_path)},
        )


def test_worker_writes_valid_temp_output_file(benchmark_module, monkeypatch):
    output_path = (
        Path(tempfile.gettempdir())
        / f"pandas-booster-worker-output-{uuid.uuid4().hex}.json"
    )
    output_path.unlink(missing_ok=True)

    try:
        result = _run_worker_payload(
            benchmark_module,
            monkeypatch,
            {"preset": "1key", "output_file": str(output_path)},
        )

        assert result is None
        assert json.loads(output_path.read_text(encoding="utf-8")) == {
            "ok": True,
            "kwargs": {"preset": "1key"},
        }
    finally:
        output_path.unlink(missing_ok=True)


def test_run_worker_process_retries_failed_subprocess(benchmark_module, monkeypatch):
    bench_utils_globals = benchmark_module.run_cold_warm_benchmark.__globals__
    run_worker_process = bench_utils_globals["run_worker_process"]
    calls: list[int] = []

    def fake_run(cmd, **_kwargs):
        calls.append(1)
        payload = json.loads(cmd[3])
        output_file = Path(payload["output_file"])

        if len(calls) == 1:
            raise subprocess.CalledProcessError(
                returncode=-11,
                cmd=cmd,
                output="",
                stderr="",
            )

        output_file.write_text(
            json.dumps({"warm_time_s": 0.12, "correctness": "pass"}),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(bench_utils_globals["subprocess"], "run", fake_run)

    result = run_worker_process(
        _BENCHMARK_PATH,
        {"preset_name": "1key", "mode": "warm"},
        max_attempts=2,
    )

    assert result == {"warm_time_s": 0.12, "correctness": "pass"}
    assert len(calls) == 2


def test_run_worker_process_raises_after_retry_exhaustion(
    benchmark_module, monkeypatch
):
    bench_utils_globals = benchmark_module.run_cold_warm_benchmark.__globals__
    run_worker_process = bench_utils_globals["run_worker_process"]
    calls: list[int] = []

    def fake_run(cmd, **_kwargs):
        calls.append(1)
        raise subprocess.CalledProcessError(
            returncode=-11,
            cmd=cmd,
            output="",
            stderr="",
        )

    monkeypatch.setattr(bench_utils_globals["subprocess"], "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_worker_process(
            _BENCHMARK_PATH,
            {"preset_name": "1key", "mode": "warm"},
            max_attempts=2,
        )

    assert len(calls) == 2
