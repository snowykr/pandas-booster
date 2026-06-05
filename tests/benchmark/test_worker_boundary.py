from __future__ import annotations

import json
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
