from __future__ import annotations

import pytest


def test_main_rejects_zero_samples(benchmark_module, monkeypatch):
    def fail_run_benchmarks(**kwargs):
        _ = kwargs
        raise AssertionError("run_benchmarks should not run for invalid sample count")

    monkeypatch.setattr(benchmark_module, "run_benchmarks", fail_run_benchmarks)
    monkeypatch.setattr(
        benchmark_module.sys,
        "argv",
        [
            "benchmark.py",
            "--samples",
            "0",
        ],
    )

    with pytest.raises(SystemExit):
        benchmark_module.main()
