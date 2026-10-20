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


@pytest.mark.parametrize("mode", ["cold", "warm"])
@pytest.mark.parametrize("sort", [True, False])
@pytest.mark.parametrize("all_nan", [False, True])
@pytest.mark.parametrize("backend", ["booster", "polars"])
def test_worker_validates_nan_median_with_retained_groups(
    benchmark_module, monkeypatch, mode, sort, all_nan, backend
):
    import runner_worker

    if backend == "polars":
        pytest.importorskip("polars")
    nan = float("nan")
    df = benchmark_module.pd.DataFrame(
        {
            "key": [2, 1, 2, 3, 1, 3],
            "value": [nan, nan, nan if all_nan else 4.0, nan, nan if all_nan else 8.0, nan],
        }
    )
    monkeypatch.setattr(runner_worker, "generate_multi_key_dataset", lambda **kwargs: df)
    monkeypatch.setitem(runner_worker.PRESETS, "test_nan_median", {"key_configs": [("key", 3)]})

    result = runner_worker.benchmark_worker(
        "test_nan_median",
        backend,
        agg="median",
        sort=sort,
        verify_correctness=True,
        mode=mode,
    )

    assert result["correctness"] == "pass"


@pytest.mark.parametrize("mode", ["cold", "warm"])
def test_worker_nan_median_validation_detects_dropped_all_nan_group(
    benchmark_module, monkeypatch, mode
):
    import runner_worker
    from pandas_booster.accessor import BoosterAccessor

    df = benchmark_module.pd.DataFrame(
        {
            "key": [1, 1, 2, 2],
            "value": [2.0, float("nan"), float("nan"), float("nan")],
        }
    )
    monkeypatch.setattr(runner_worker, "generate_multi_key_dataset", lambda **kwargs: df)
    monkeypatch.setitem(runner_worker.PRESETS, "test_nan_median", {"key_configs": [("key", 2)]})
    incomplete = df.groupby("key")["value"].median().dropna()
    monkeypatch.setattr(BoosterAccessor, "groupby", lambda *args, **kwargs: incomplete)

    result = runner_worker.benchmark_worker(
        "test_nan_median",
        "booster",
        agg="median",
        verify_correctness=True,
        mode=mode,
    )

    assert result["correctness"].startswith("fail:")


@pytest.mark.parametrize("agg", ["sum", "mean", "prod", "std", "var", "min", "max", "count"])
def test_worker_keeps_nonmedian_nan_value_guard(benchmark_module, monkeypatch, agg):
    import runner_worker

    df = benchmark_module.pd.DataFrame({"key": [1], "value": [float("nan")]})
    monkeypatch.setattr(runner_worker, "generate_multi_key_dataset", lambda **kwargs: df)
    monkeypatch.setitem(runner_worker.PRESETS, "test_nan_guard", {"key_configs": [("key", 1)]})

    with pytest.raises(ValueError, match="contains NaNs"):
        runner_worker.benchmark_worker(
            "test_nan_guard",
            "booster",
            agg=agg,
            verify_correctness=True,
        )


@pytest.mark.parametrize("keys", [[1.0, float("nan")], ["a", "b"]])
def test_worker_median_preserves_integer_key_guard(benchmark_module, monkeypatch, keys):
    import runner_worker

    df = benchmark_module.pd.DataFrame({"key": keys, "value": [float("nan"), 1.0]})
    monkeypatch.setattr(runner_worker, "generate_multi_key_dataset", lambda **kwargs: df)
    monkeypatch.setitem(runner_worker.PRESETS, "test_key_guard", {"key_configs": [("key", 2)]})

    with pytest.raises(ValueError, match="must be integer dtype"):
        runner_worker.benchmark_worker(
            "test_key_guard",
            "booster",
            agg="median",
            verify_correctness=True,
        )


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
        Path(tempfile.gettempdir()) / f"pandas-booster-worker-output-{uuid.uuid4().hex}.json"
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
