"""Edge case tests for pandas-booster.

Tests covering numerical extremes, empty data, and boundary conditions
that are critical for ensuring library robustness.
"""

import warnings

import numpy as np
import pandas as pd
import pytest


class TestEmptyData:
    """Tests for empty DataFrame and edge-size scenarios."""

    def test_empty_dataframe_uses_fallback(self):
        """Empty DataFrame should use fallback and return empty Series."""
        import pandas_booster  # noqa: F401

        df = pd.DataFrame(
            {"key": pd.array([], dtype="int64"), "val": pd.array([], dtype="float64")}
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(result, expected)
        assert len(result) == 0

    def test_single_row_uses_fallback(self):
        """Single row should use fallback (below threshold)."""
        import pandas_booster  # noqa: F401

        df = pd.DataFrame({"key": [1], "val": [42.0]})

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(result, expected)

    def test_exactly_at_threshold_boundary(self):
        """Test behavior exactly at the fallback threshold boundary."""
        import pandas_booster  # noqa: F401

        # Just below threshold (should use fallback)
        n = 99_999
        df_below = pd.DataFrame(
            {"key": np.random.randint(0, 100, size=n), "val": np.random.random(size=n)}
        )
        result_below = df_below.booster.groupby("key", "val", "sum")
        expected_below = df_below.groupby("key")["val"].sum()
        pd.testing.assert_series_equal(
            result_below.sort_index(), expected_below.sort_index(), check_exact=False, rtol=1e-10
        )

        # At threshold (should use Rust)
        n = 100_000
        df_at = pd.DataFrame(
            {"key": np.random.randint(0, 100, size=n), "val": np.random.random(size=n)}
        )
        result_at = df_at.booster.groupby("key", "val", "sum")
        expected_at = df_at.groupby("key")["val"].sum()
        pd.testing.assert_series_equal(
            result_at.sort_index(), expected_at.sort_index(), check_exact=False, rtol=1e-10
        )


class TestInfiniteValues:
    """Tests for inf and -inf handling in value columns."""

    @pytest.fixture
    def df_with_inf(self):
        """Create DataFrame with infinite values."""
        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 100
        # Insert some infinities
        values[100] = np.inf
        values[200] = -np.inf
        values[300] = np.inf
        values[400] = -np.inf
        return pd.DataFrame({"key": np.random.randint(0, 100, size=n), "val": values})

    def test_sum_with_inf(self, df_with_inf):
        """Sum with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "sum")
        expected = df_with_inf.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_mean_with_inf(self, df_with_inf):
        """Mean with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "mean")
        expected = df_with_inf.groupby("key")["val"].mean()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_min_with_inf(self, df_with_inf):
        """Min with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "min")
        expected = df_with_inf.groupby("key")["val"].min()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_max_with_inf(self, df_with_inf):
        """Max with inf should match Pandas behavior."""
        import pandas_booster  # noqa: F401

        result = df_with_inf.booster.groupby("key", "val", "max")
        expected = df_with_inf.groupby("key")["val"].max()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_all_inf_group(self):
        """Group with only inf values."""
        import pandas_booster  # noqa: F401

        n = 200_000
        # Group 1 has only +inf, Group 2 has only -inf, Group 3 has normal values
        keys = np.concatenate(
            [
                np.ones(n // 3, dtype=int),
                np.full(n // 3, 2, dtype=int),
                np.full(n - 2 * (n // 3), 3, dtype=int),
            ]
        )
        values = np.concatenate(
            [
                np.full(n // 3, np.inf),
                np.full(n // 3, -np.inf),
                np.random.random(n - 2 * (n // 3)),
            ]
        )
        df = pd.DataFrame({"key": keys, "val": values})

        for agg in ["sum", "mean", "min", "max"]:
            result = df.booster.groupby("key", "val", agg)
            expected = getattr(df.groupby("key")["val"], agg)()
            pd.testing.assert_series_equal(
                result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
            )


class TestIntegerOverflow:
    """Tests for integer overflow behavior parity with pandas."""

    def test_large_i64_sum_wraps_like_pandas(self):
        """Large i64 sums should match pandas overflow/wrap semantics."""
        import pandas_booster  # noqa: F401

        n = 200_000
        # Large values that would overflow i64 when summed
        large_val = 2**62  # Close to i64::MAX / 2
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": np.full(n, large_val, dtype=np.int64),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=True,
        )

    def test_mixed_sign_large_values(self):
        """Large positive and negative values should cancel correctly."""
        import pandas_booster  # noqa: F401

        n = 200_000
        large_val = 2**62
        # Alternating large positive and negative values
        values = np.where(np.arange(n) % 2 == 0, large_val, -large_val).astype(np.int64)
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": values,
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=True,
        )


class TestExtremeCardinality:
    """Tests for extreme group cardinality scenarios."""

    def test_single_group_large_dataset(self):
        """All rows in a single group."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        assert len(result) == 1
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_every_row_unique_key(self):
        """Every row is its own group (maximum cardinality)."""
        import pandas_booster  # noqa: F401

        n = 150_000  # Just above threshold
        df = pd.DataFrame(
            {
                "key": np.arange(n, dtype=np.int64),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        assert len(result) == n
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_two_groups_only(self):
        """Only two groups in entire dataset."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.where(np.arange(n) < n // 2, 0, 1).astype(np.int64),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "mean")
        expected = df.groupby("key")["val"].mean()

        assert len(result) == 2
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )


class TestNegativeKeys:
    """Tests for negative integer keys."""

    def test_negative_keys(self):
        """Negative integer keys should work correctly."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(-1000, 1000, size=n),
                "val": np.random.random(n),
            }
        )

        for agg in ["sum", "mean", "min", "max"]:
            result = df.booster.groupby("key", "val", agg)
            expected = getattr(df.groupby("key")["val"], agg)()
            pd.testing.assert_series_equal(
                result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
            )

    def test_extreme_key_values(self):
        """Keys near i64 min/max bounds."""
        import pandas_booster  # noqa: F401

        n = 200_000
        # Use keys near the boundaries of i64
        extreme_keys = np.array([-(2**62), -(2**62) + 1, 0, 2**62 - 1, 2**62], dtype=np.int64)
        keys = np.tile(extreme_keys, n // 5)
        df = pd.DataFrame(
            {
                "key": keys[:n],
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )


class TestMixedNaNAndInf:
    """Tests combining NaN and inf values."""

    def test_nan_and_inf_mixed(self):
        """Mix of NaN and inf values in same column."""
        import pandas_booster  # noqa: F401

        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 100
        # Scatter NaN, +inf, -inf
        nan_mask = np.random.random(n) < 0.05
        inf_mask = np.random.random(n) < 0.02
        ninf_mask = np.random.random(n) < 0.02

        values = np.where(nan_mask, np.nan, values)
        values = np.where(inf_mask, np.inf, values)
        values = np.where(ninf_mask, -np.inf, values)

        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 100, size=n),
                "val": values,
            }
        )

        for agg in ["sum", "mean", "min", "max"]:
            result = df.booster.groupby("key", "val", agg)
            expected = getattr(df.groupby("key")["val"], agg)()
            pd.testing.assert_series_equal(
                result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
            )


class TestDtypeFallback:
    """Tests for dtype-based fallback scenarios."""

    def test_string_key_uses_fallback(self):
        """String key column should use Pandas fallback."""
        import pandas_booster  # noqa: F401

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.array([f"key_{i % 100}" for i in range(n)]),
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_float_key_uses_fallback(self):
        """Float key column should use Pandas fallback."""
        import pandas_booster  # noqa: F401

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.random.random(n) * 100,  # Float keys
                "val": np.random.random(n),
            }
        )

        result = df.booster.groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_string_value_uses_fallback(self):
        """String value column should use Pandas fallback."""
        import pandas_booster  # noqa: F401

        n = 200_000
        df = pd.DataFrame(
            {
                "key": np.random.randint(0, 100, size=n),
                "val": np.array([f"val_{i}" for i in range(n)]),
            }
        )

        # String columns can't use sum/mean/min/max meaningfully, but we test fallback behavior
        # The accessor should fall back to Pandas which will handle it appropriately
        result = df.booster.groupby("key", "val", "min")
        expected = df.groupby("key")["val"].min()

        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())


class TestAllOperationsConsistency:
    """Ensure all operations work consistently across edge cases."""

    @pytest.fixture
    def stress_df(self):
        """Create a stress-test DataFrame with various edge values."""
        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 1000 - 500  # Range: -500 to 500

        # Inject edge cases
        values[::100] = np.nan  # 1% NaN
        values[::500] = np.inf  # 0.2% +inf
        values[::700] = -np.inf  # ~0.14% -inf
        values[::1000] = 0.0  # 0.1% zeros

        return pd.DataFrame(
            {
                "key": np.random.randint(-50, 50, size=n),
                "val": values,
            }
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
    def test_operation_consistency(self, stress_df, agg):
        """All operations should match Pandas on stress DataFrame."""
        import pandas_booster  # noqa: F401

        result = stress_df.booster.groupby("key", "val", agg)
        expected = getattr(stress_df.groupby("key")["val"], agg)()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )


def _accessor_groupby_result(
    df: pd.DataFrame,
    by: str | list[str],
    target: str,
    agg: str,
    *,
    sort: bool = True,
) -> pd.Series:
    import pandas_booster  # noqa: F401

    return df.booster.groupby(by, target, agg, sort=sort)


def _proxy_groupby_result(
    df: pd.DataFrame,
    by: str | list[str],
    target: str,
    agg: str,
    *,
    sort: bool = True,
) -> pd.Series:
    import pandas_booster

    pandas_booster.activate()
    try:
        return getattr(df.groupby(by, sort=sort)[target], agg)()
    finally:
        pandas_booster.deactivate()


def _patch_single_std_var_kernel(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: str,
    *,
    kernel: str,
    result_dtype: np.dtype,
) -> None:
    def fake_groupby(_keys_arr, _values_arr):
        return (
            np.asarray(expected.index.to_numpy(), dtype=np.int64),
            np.asarray(expected.to_numpy(), dtype=result_dtype),
        )

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}{suffix}", fake_groupby, raising=False)


def _patch_multi_std_var_kernel(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: str,
    *,
    kernel: str,
    result_dtype: np.dtype,
) -> None:
    def fake_groupby(_key_arrays, _values_arr):
        keys_cols = [
            np.asarray(expected.index.get_level_values(i), dtype=np.int64)
            for i in range(expected.index.nlevels)
        ]
        return keys_cols, np.asarray(expected.to_numpy(), dtype=result_dtype)

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(
            rust,
            f"groupby_multi_{agg}_{kernel}{suffix}",
            fake_groupby,
            raising=False,
        )


def _patch_all_std_var_kernels_to_raise(
    monkeypatch: pytest.MonkeyPatch, rust: object, message: str
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError(message)

    for agg in ("std", "var"):
        for kernel in ("f64", "i64"):
            for prefix in ("groupby", "groupby_multi"):
                for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                    monkeypatch.setattr(
                        rust,
                        f"{prefix}_{agg}_{kernel}{suffix}",
                        _boom,
                        raising=False,
                    )


def _patch_all_i64_kernels_for_agg_to_raise(
    monkeypatch: pytest.MonkeyPatch, rust: object, agg: str, message: str
) -> None:
    def _boom(*_args, **_kwargs):
        raise AssertionError(message)

    for prefix in ("groupby", "groupby_multi"):
        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"{prefix}_{agg}_i64{suffix}", _boom, raising=False)


def _delete_std_var_kernel_symbols(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    agg: str,
    *,
    kernel: str,
    multi: bool,
) -> None:
    prefix = "groupby_multi" if multi else "groupby"
    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.delattr(rust, f"{prefix}_{agg}_{kernel}{suffix}", raising=False)


def _patch_pandas_series_groupby_agg_to_raise(
    monkeypatch: pytest.MonkeyPatch, agg: str, message: str
) -> None:
    from pandas.core.groupby.generic import SeriesGroupBy

    def _boom(self, *args, **kwargs):
        raise AssertionError(message)

    monkeypatch.setattr(SeriesGroupBy, agg, _boom, raising=True)


def _patch_single_std_var_firstseen_only_kernel(
    monkeypatch: pytest.MonkeyPatch,
    rust: object,
    expected: pd.Series,
    agg: str,
    *,
    kernel: str,
    result_dtype: np.dtype,
    calls: list[str],
) -> None:
    def fake_groupby(_keys_arr, _values_arr):
        calls.append("firstseen")
        return (
            np.asarray(expected.index.to_numpy(), dtype=np.int64),
            np.asarray(expected.to_numpy(), dtype=result_dtype),
        )

    def _boom(*_args, **_kwargs):
        raise AssertionError(f"sort=False {agg} should route through first-seen kernels")

    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}", _boom, raising=False)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_sorted", _boom, raising=False)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_firstseen_u32", fake_groupby, raising=False)
    monkeypatch.setattr(rust, f"groupby_{agg}_{kernel}_firstseen_u64", fake_groupby, raising=False)


class TestStdVarContracts:
    @pytest.mark.parametrize(
        ("kernel", "values", "expected_order"),
        [
            (
                "f64",
                np.array([1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan]),
                [5, 2, 4, 7, 8, 1],
            ),
            (
                "i64",
                np.array([2, 6, 4, 10, 14, 12], dtype=np.int64),
                [8, 3, 5, 4],
            ),
        ],
    )
    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_std_var_supported_calls_route_to_firstseen_kernels(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agg: str,
        kernel: str,
        values: np.ndarray,
        expected_order: list[int],
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        key_values = [8, 3, 8, 5, 3, 4] if kernel == "i64" else [5, 2, 5, 4, 2, 7, 8, 8, 1]
        df = pd.DataFrame({"key": key_values, "val": values})
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()
        calls: list[str] = []

        _patch_single_std_var_firstseen_only_kernel(
            monkeypatch,
            rust,
            expected,
            agg,
            kernel=kernel,
            result_dtype=np.dtype(np.float64),
            calls=calls,
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "supported single-key unsorted std/var should not fall back to pandas",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg, sort=False)

        assert calls == ["firstseen", "firstseen"]
        assert accessor_result.index.tolist() == expected_order
        assert proxy_result.index.tolist() == expected_order
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_small_single_key_float_std_var_uses_rust_without_env_rollback(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2, 3, 3],
                "val": [1.5, 2.5, 10.0, 14.0, 5.5, 8.5],
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="f64", result_dtype=np.float64
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            (
                "supported single-key float std/var should not fall back to pandas "
                "when rollback is off"
            ),
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_float_std_var_env_rollback_forces_fallback(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2, 3], 4),
                "val": np.array([1.0, 2.0, 4.0, 8.0, 2.5, 3.5, 6.5, 7.5, 5.0, 6.0, 9.0, 10.0]),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _patch_all_std_var_kernels_to_raise(
            monkeypatch,
            rust,
            "single-key float std/var should use pandas fallback when rollback env is enabled",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_float_std_var_env_rollback_forces_fallback(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": [5, 2, 5, 4, 2, 7, 8, 8, 1],
                "val": [1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan],
            }
        )
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        _patch_all_std_var_kernels_to_raise(
            monkeypatch,
            rust,
            (
                "single-key unsorted float std/var should use pandas fallback "
                "when rollback env is enabled"
            ),
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg, sort=False)

        assert accessor_result.index.tolist() == [5, 2, 4, 7, 8, 1]
        assert proxy_result.index.tolist() == [5, 2, 4, 7, 8, 1]
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_float_std_var_preserves_first_seen_order_and_nan_semantics(
        self, agg: str
    ):
        df = pd.DataFrame(
            {
                "key": [5, 2, 5, 4, 2, 7, 8, 8, 1],
                "val": [1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan],
            }
        )

        result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert result.index.tolist() == [5, 2, 4, 7, 8, 1]
        assert result.dtype == np.dtype(np.float64)
        assert result.loc[8] == 0.0
        assert np.isnan(result.loc[2])
        assert np.isnan(result.loc[4])
        assert np.isnan(result.loc[7])
        assert np.isnan(result.loc[1])
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_unsorted_int_backed_std_var_preserves_first_seen_order_and_float64_dtype(
        self, agg: str
    ):
        df = pd.DataFrame(
            {
                "key": [8, 3, 8, 5, 3, 4],
                "val": np.array([2, 6, 4, 10, 14, 12], dtype=np.int64),
            }
        )

        result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert result.index.tolist() == [8, 3, 5, 4]
        assert result.dtype == np.dtype(np.float64)
        assert np.isnan(result.loc[5])
        assert np.isnan(result.loc[4])
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_single_key_high_cardinality_partitioned_std_var_preserves_first_seen_order(
        self, agg: str
    ):
        group_count = 10_000
        expected_order = np.concatenate(
            (
                np.arange(5_000, group_count, dtype=np.int64),
                np.arange(0, 5_000, dtype=np.int64),
            )
        )
        keys = np.repeat(expected_order, 2)
        base = np.arange(group_count, dtype=np.float64)
        values = np.empty(group_count * 2, dtype=np.float64)
        values[0::2] = base
        values[1::2] = base + 0.5

        df = pd.DataFrame({"key": keys, "val": values})

        accessor_result = _accessor_groupby_result(df, "key", "val", agg, sort=False)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg, sort=False)
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        assert accessor_result.index.tolist() == expected_order.tolist()
        assert proxy_result.index.tolist() == expected_order.tolist()
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_multi_key_unsorted_std_var_preserves_existing_first_seen_order(
        self, agg: str
    ):
        df = pd.DataFrame(
            {
                "k1": [2, 1, 2, 1, 2, 3, 1],
                "k2": [9, 8, 9, 7, 8, 7, 8],
                "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            }
        )

        result = _accessor_groupby_result(df, ["k1", "k2"], "val", agg, sort=False)
        expected = getattr(df.groupby(["k1", "k2"], sort=False)["val"], agg)()

        assert result.index.tolist() == [(2, 9), (1, 8), (1, 7), (2, 8), (3, 7)]
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_float_env_rollback_scope_does_not_broaden_to_multi_key_std_var(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "k1": [1, 1, 1, 2, 2, 2],
                "k2": [10, 10, 20, 10, 10, 20],
                "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
            }
        )
        expected = getattr(df.groupby(["k1", "k2"], sort=True)["val"], agg)()

        _patch_multi_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="f64", result_dtype=np.float64
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "float rollback env must not broaden to multi-key std/var pandas fallback",
        )

        accessor_result = _accessor_groupby_result(df, ["k1", "k2"], "val", agg)
        proxy_result = _proxy_groupby_result(df, ["k1", "k2"], "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_float_env_rollback_scope_does_not_broaden_to_int_backed_std_var(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2, 3, 3],
                "val": np.array([1, 3, 10, 14, 5, 9], dtype=np.int64),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="i64", result_dtype=np.float64
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "float rollback env must not broaden to int-backed std/var pandas fallback",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_accessor_and_proxy_normalize_float_result_abi_to_float64(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2, 3, 3],
                "val": np.array([4, 8, 10, 18, 20, 28], dtype=np.int64),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)().astype(np.float64)

        _patch_single_std_var_kernel(
            monkeypatch, rust, expected, agg, kernel="i64", result_dtype=np.float32
        )
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            agg,
            "supported std/var float-result ABI should not fall back to pandas",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        assert accessor_result.dtype == np.dtype(np.float64)
        assert proxy_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-6)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-6)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_single_key_std_var_symbols_fall_back_as_abi_skew(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust
        from pandas_booster.accessor import BoosterAccessor

        monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2, 3], 4),
                "val": np.array([1.0, 2.0, 4.0, 8.0, 2.5, 3.5, 6.5, 7.5, 5.0, 6.0, 9.0, 10.0]),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        _delete_std_var_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        booster = df.booster
        assert isinstance(booster, BoosterAccessor)
        fallback_called = {"n": 0}
        orig_fallback = booster._pandas_fallback

        def wrapped_fallback(by_cols, target, wrapped_agg, *, sort: bool):
            fallback_called["n"] += 1
            return orig_fallback(by_cols, target, wrapped_agg, sort=sort)

        monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            accessor_result = _accessor_groupby_result(df, "key", "val", agg)

            pandas_booster.activate()
            try:
                proxy_result = getattr(df.groupby("key", sort=True)["val"], agg)()
            finally:
                pandas_booster.deactivate()

        assert fallback_called["n"] == 1
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

        abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
        abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
        assert len(abi_warnings) == 1
        assert f"missing Rust kernel symbol 'groupby_{agg}_f64'" in str(abi_warnings[0].message)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_multi_key_std_var_symbols_fall_back_as_abi_skew(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust
        from pandas_booster.accessor import BoosterAccessor

        monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "k1": [1, 1, 1, 2, 2, 2],
                "k2": [10, 10, 20, 10, 10, 20],
                "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
            }
        )
        by_cols = ["k1", "k2"]
        expected = getattr(df.groupby(by_cols, sort=True)["val"], agg)()

        _delete_std_var_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=True)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        booster = df.booster
        assert isinstance(booster, BoosterAccessor)
        fallback_called = {"n": 0}
        orig_fallback = booster._pandas_fallback

        def wrapped_fallback(wrapped_by_cols, target, wrapped_agg, *, sort: bool):
            fallback_called["n"] += 1
            return orig_fallback(wrapped_by_cols, target, wrapped_agg, sort=sort)

        monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            accessor_result = _accessor_groupby_result(df, by_cols, "val", agg)

            pandas_booster.activate()
            try:
                proxy_result = getattr(df.groupby(by_cols, sort=True)["val"], agg)()
            finally:
                pandas_booster.deactivate()

        assert fallback_called["n"] == 1
        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

        abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
        abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
        assert len(abi_warnings) == 1
        assert (
            f"missing Rust kernel symbol 'groupby_multi_{agg}_f64'"
            in str(abi_warnings[0].message)
        )

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_single_key_std_var_symbols_hard_fail_in_strict_abi(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "key": np.repeat([1, 2, 3], 4),
                "val": np.array([1.0, 2.0, 4.0, 8.0, 2.5, 3.5, 6.5, 7.5, 5.0, 6.0, 9.0, 10.0]),
            }
        )

        _delete_std_var_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=False)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError,
            match=f"missing Rust kernel symbol 'groupby_{agg}_f64'",
        ):
            _ = _accessor_groupby_result(df, "key", "val", agg)

        abi._WARNED_ABI_SKEW = False
        pandas_booster.activate()
        try:
            with pytest.raises(
                abi.PandasBoosterKeyShapeSkewError,
                match=f"missing Rust kernel symbol 'groupby_{agg}_f64'",
            ):
                _ = getattr(df.groupby("key", sort=True)["val"], agg)()
        finally:
            pandas_booster.deactivate()

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_missing_multi_key_std_var_symbols_hard_fail_in_strict_abi(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster
        import pandas_booster._abi_compat as abi
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
        monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)
        monkeypatch.delenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", raising=False)

        df = pd.DataFrame(
            {
                "k1": [1, 1, 1, 2, 2, 2],
                "k2": [10, 10, 20, 10, 10, 20],
                "val": [1.0, 5.0, 9.0, 2.0, 6.0, 10.0],
            }
        )
        by_cols = ["k1", "k2"]

        _delete_std_var_kernel_symbols(monkeypatch, rust, agg, kernel="f64", multi=True)
        monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

        with pytest.raises(
            abi.PandasBoosterKeyShapeSkewError,
            match=f"missing Rust kernel symbol 'groupby_multi_{agg}_f64'",
        ):
            _ = _accessor_groupby_result(df, by_cols, "val", agg)

        abi._WARNED_ABI_SKEW = False
        pandas_booster.activate()
        try:
            with pytest.raises(
                abi.PandasBoosterKeyShapeSkewError,
                match=f"missing Rust kernel symbol 'groupby_multi_{agg}_f64'",
            ):
                _ = getattr(df.groupby(by_cols, sort=True)["val"], agg)()
        finally:
            pandas_booster.deactivate()

    @pytest.mark.parametrize("agg", ["std", "var"])
    @pytest.mark.parametrize(
        ("case_name", "df"),
        [
            (
                "extension_dtype_value",
                pd.DataFrame(
                    {
                        "key": np.repeat([1, 2, 3], 4),
                        "val": pd.array([1.0, 2.0, 4.0, 8.0] * 3, dtype="Float64"),
                    }
                ),
            ),
            (
                "nullable_pd_na_value",
                pd.DataFrame(
                    {
                        "key": np.repeat([1, 2, 3], 4),
                        "val": pd.array([1.0, pd.NA, 4.0, 8.0] * 3, dtype="Float64"),
                    }
                ),
            ),
            (
                "non_integer_key",
                pd.DataFrame(
                    {
                        "key": ["a", "a", "b", "b", "c", "c"],
                        "val": [1.0, 2.0, 10.0, 14.0, 5.0, 9.0],
                    }
                ),
            ),
            (
                "uint64_value",
                pd.DataFrame(
                    {
                        "key": np.repeat([1, 2, 3], 4),
                        "val": np.array([1, 3, 7, 15] * 3, dtype=np.uint64),
                    }
                ),
            ),
        ],
    )
    def test_std_var_unsupported_inputs_fall_back_for_accessor_and_proxy(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agg: str,
        case_name: str,
        df: pd.DataFrame,
    ):
        import pandas_booster._rust as rust

        _patch_all_std_var_kernels_to_raise(
            monkeypatch,
            rust,
            f"{case_name} should stay on pandas fallback for std/var",
        )

        expected = getattr(df.groupby("key", sort=True)["val"], agg)()

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        pd.testing.assert_series_equal(accessor_result, expected, check_exact=False, rtol=1e-12)
        pd.testing.assert_series_equal(proxy_result, expected, check_exact=False, rtol=1e-12)

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "std", "var"])
    def test_uint64_values_fall_back_for_accessor_and_proxy(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2, 3], dtype=np.int64), 100_002),
                "val": np.resize(
                    np.array(
                        [0, int(np.iinfo(np.int64).max) + 1, np.iinfo(np.uint64).max],
                        dtype=np.uint64,
                    ),
                    100_002,
                ),
            }
        )
        expected = getattr(df.groupby("key", sort=True)["val"], agg)()
        _patch_all_i64_kernels_for_agg_to_raise(
            monkeypatch,
            rust,
            agg,
            "uint64 value inputs should stay on pandas fallback to avoid int64 wrapping",
        )

        accessor_result = _accessor_groupby_result(df, "key", "val", agg)
        proxy_result = _proxy_groupby_result(df, "key", "val", agg)

        pd.testing.assert_series_equal(accessor_result, expected)
        pd.testing.assert_series_equal(proxy_result, expected)

    @pytest.mark.parametrize("agg", ["std", "var"])
    def test_std_var_unsupported_object_coercions_match_pandas_error(
        self, monkeypatch: pytest.MonkeyPatch, agg: str
    ):
        import pandas_booster._rust as rust

        df = pd.DataFrame(
            {
                "key": [1, 1, 2, 2],
                "val": np.array(["a", "b", "c", "d"], dtype=object),
            }
        )

        _patch_all_std_var_kernels_to_raise(
            monkeypatch,
            rust,
            "object-backed std/var inputs should not be coerced onto Rust kernels",
        )

        with pytest.raises(Exception) as pandas_exc:
            _ = getattr(df.groupby("key", sort=True)["val"], agg)()

        with pytest.raises(type(pandas_exc.value)):
            _ = _accessor_groupby_result(df, "key", "val", agg)

        with pytest.raises(type(pandas_exc.value)):
            _ = _proxy_groupby_result(df, "key", "val", agg)
