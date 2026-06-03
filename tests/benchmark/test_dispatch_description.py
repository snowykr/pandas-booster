"""Benchmark dispatch description tests."""

from __future__ import annotations

import pytest
from conftest import _loaded_benchmark_module


@pytest.fixture(scope="module")
def benchmark_module():
    with _loaded_benchmark_module() as module:
        yield module


def test_describe_booster_execution_uses_pandas_label_for_large_single_float_sum_mean_rollback(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

    def fail_select(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError(
            "Rust kernel selection should not run when float sum/mean rollback forces pandas"
        )

    monkeypatch.setattr(groupby_accel, "select_rust_groupby_func", fail_select)

    df = benchmark_module.pd.DataFrame(
        {
            "key": benchmark_module.np.tile([1, 2], 50_001),
            "value": benchmark_module.np.linspace(
                0.0, 1.0, 100_002, dtype=benchmark_module.np.float64
            ),
        }
    )

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "sum", True) == (
        "booster->pandas.groupby.sum"
    )
    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "mean", False) == (
        "booster->pandas.groupby.mean"
    )


def test_describe_booster_execution_reports_benchmark_abi_skew_for_missing_kernel(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust

    monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)
    monkeypatch.delattr(rust, "groupby_std_f64", raising=False)
    monkeypatch.delattr(rust, "groupby_std_f64_sorted", raising=False)

    df = benchmark_module.pd.DataFrame({"key": [1, 1, 2, 2], "value": [1.0, 3.0, 10.0, 14.0]})

    with (
        pytest.warns(abi.PandasBoosterAbiSkewWarning, match=r"ABI skew \(benchmark\)"),
        pytest.raises(
            abi.PandasBoosterKeyShapeSkewError,
            match=r"pandas-booster ABI skew \(benchmark\).*groupby_std_f64",
        ),
    ):
        benchmark_module.describe_booster_execution(df, ["key"], "value", "std", True)


def test_describe_booster_execution_treats_small_supported_median_as_rust_first(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    class FakeFunc:
        __name__ = "groupby_median_f64_sorted"

    df = benchmark_module.pd.DataFrame(
        {
            "key": [1, 1, 2, 2],
            "value": benchmark_module.np.array(
                [1.0, 3.0, 2.0, 8.0], dtype=benchmark_module.np.float64
            ),
        }
    )

    monkeypatch.setattr(
        groupby_accel,
        "classify_groupby_compatibility",
        lambda **_kwargs: groupby_accel.GroupByCompatibility(True, False),
    )
    monkeypatch.setattr(groupby_accel, "has_rust_groupby_func", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        groupby_accel,
        "select_rust_groupby_func",
        lambda *args, **kwargs: (FakeFunc(), False),
    )

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "median", True) == (
        "booster->rust.groupby_median_f64_sorted"
    )


def test_describe_booster_execution_uses_pandas_label_when_exact_median_kernel_is_missing(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    df = benchmark_module.pd.DataFrame(
        {
            "key": benchmark_module.np.tile([1, 2], 50_000),
            "value": benchmark_module.np.linspace(
                0.0, 1.0, 100_000, dtype=benchmark_module.np.float64
            ),
        }
    )

    monkeypatch.setattr(
        groupby_accel,
        "classify_groupby_compatibility",
        lambda **_kwargs: groupby_accel.GroupByCompatibility(True, False),
    )
    monkeypatch.setattr(groupby_accel, "has_rust_groupby_func", lambda *args, **kwargs: False)

    def fail_select(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("exact median kernel lookup should gate benchmark dispatch")

    monkeypatch.setattr(groupby_accel, "select_rust_groupby_func", fail_select)

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "median", False) == (
        "booster->pandas.groupby.median"
    )


def test_describe_booster_execution_skips_median_kernel_probe_for_unsupported_input(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    df = benchmark_module.pd.DataFrame({"key": [1, 2], "value": ["a", "b"]})

    def fail_has_rust_groupby_func(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("unsupported benchmark inputs should not probe median kernels")

    monkeypatch.setattr(groupby_accel, "has_rust_groupby_func", fail_has_rust_groupby_func)

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "median", True) == (
        "booster->pandas.groupby.median"
    )


def test_describe_booster_execution_skips_median_kernel_probe_when_force_pandas_enabled(
    benchmark_module,
    monkeypatch,
):
    import pandas_booster._groupby_accel as groupby_accel

    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")
    df = benchmark_module.pd.DataFrame({"key": [1, 2], "value": [1.0, 2.0]})

    def fail_has_rust_groupby_func(*args, **kwargs):
        _ = (args, kwargs)
        raise AssertionError("forced pandas benchmark inputs should not probe median kernels")

    monkeypatch.setattr(groupby_accel, "has_rust_groupby_func", fail_has_rust_groupby_func)

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "median", False) == (
        "booster->pandas.groupby.median"
    )


def test_build_polars_agg_expr_supports_prod(benchmark_module):
    if benchmark_module.pl is None:
        pytest.skip("Polars is not installed")

    expr = benchmark_module.build_polars_agg_expr("value", "prod")
    assert "value" in repr(expr)


def test_build_polars_agg_expr_supports_median(benchmark_module):
    if benchmark_module.pl is None:
        pytest.skip("Polars is not installed")

    expr = benchmark_module.build_polars_agg_expr("value", "median")
    assert "value" in repr(expr)


def test_describe_booster_execution_reports_single_key_float_prod_rust_dispatch(
    benchmark_module, monkeypatch
):
    import pandas_booster._rust as rust

    threshold = rust.get_fallback_threshold()
    df = benchmark_module.pd.DataFrame(
        {
            "key": benchmark_module.np.resize(
                benchmark_module.np.array([1, 2], dtype=benchmark_module.np.int64), threshold
            ),
            "value": benchmark_module.np.linspace(
                1.001, 1.01, threshold, dtype=benchmark_module.np.float64
            ),
        }
    )

    def _sentinel_prod(*_args, **_kwargs):
        raise AssertionError("single-key float prod label should resolve this Rust kernel")

    monkeypatch.setattr(rust, "groupby_prod_f64_sorted", _sentinel_prod, raising=False)

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "prod", True) == (
        "booster->rust._sentinel_prod"
    )


def test_describe_booster_execution_uses_pandas_label_when_ordered_prod_abi_marker_is_missing(
    benchmark_module, monkeypatch
):
    import pandas_booster._rust as rust

    threshold = rust.get_fallback_threshold()
    df = benchmark_module.pd.DataFrame(
        {
            "key": benchmark_module.np.resize(
                benchmark_module.np.array([1, 2], dtype=benchmark_module.np.int64), threshold
            ),
            "value": benchmark_module.np.linspace(
                1.001, 1.01, threshold, dtype=benchmark_module.np.float64
            ),
        }
    )

    monkeypatch.delattr(rust, "has_ordered_single_key_float_prod_abi", raising=False)

    def _stale_prod(*_args, **_kwargs):
        raise AssertionError("stale single-key float prod benchmark kernel must not be selected")

    monkeypatch.setattr(rust, "groupby_prod_f64_sorted", _stale_prod, raising=False)

    assert benchmark_module.describe_booster_execution(df, ["key"], "value", "prod", True) == (
        "booster->pandas.groupby.prod"
    )
