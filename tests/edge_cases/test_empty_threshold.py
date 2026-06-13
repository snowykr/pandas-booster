"""Empty input, fallback threshold, and basic median edge behavior tests."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster.accessor import BoosterAccessor

AggFunc = Literal["sum", "mean", "min", "max", "count"]


def _clear_dispatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PANDAS_BOOSTER_FORCE_PANDAS_SORT",
        "PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY",
        "PANDAS_BOOSTER_STRICT_ABI",
        "PANDAS_BOOSTER_ABI_SKEW_NOTICE",
    ):
        monkeypatch.delenv(name, raising=False)


class TestEmptyData:
    """Tests for empty DataFrame and edge-size scenarios."""

    def test_empty_dataframe_uses_fallback(self):
        """Empty DataFrame should use fallback and return empty Series."""
        import pandas_booster  # noqa: F401

        df = pd.DataFrame(
            {"key": pd.array([], dtype="int64"), "val": pd.array([], dtype="float64")}
        )

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
        expected = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(result, expected)
        assert len(result) == 0

    def test_single_row_uses_fallback(self):
        """Single row should use fallback (below threshold)."""
        import pandas_booster  # noqa: F401

        df = pd.DataFrame({"key": [1], "val": [42.0]})

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "sum")
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
        result_below = cast(BoosterAccessor, df_below.booster).groupby("key", "val", "sum")
        expected_below = df_below.groupby("key")["val"].sum()
        pd.testing.assert_series_equal(
            result_below.sort_index(), expected_below.sort_index(), check_exact=False, rtol=1e-10
        )

        # At threshold (should use Rust)
        n = 100_000
        df_at = pd.DataFrame(
            {"key": np.random.randint(0, 100, size=n), "val": np.random.random(size=n)}
        )
        result_at = cast(BoosterAccessor, df_at.booster).groupby("key", "val", "sum")
        expected_at = df_at.groupby("key")["val"].sum()
        pd.testing.assert_series_equal(
            result_at.sort_index(), expected_at.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_median_matches_pandas_on_supported_input(self):
        import pandas_booster  # noqa: F401

        df = pd.DataFrame(
            {
                "key": [1, 2, 1, 2, 1, 2],
                "val": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            }
        )

        result = cast(BoosterAccessor, df.booster).groupby("key", "val", "median")
        expected = df.groupby("key")["val"].median()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_median_kwargs_delegate_to_pandas(self, monkeypatch: pytest.MonkeyPatch):
        """Median kwargs should be forwarded to pandas unchanged."""
        import pandas_booster
        from pandas.core.groupby.generic import SeriesGroupBy
        from pandas_booster.proxy import BoosterSeriesGroupBy

        df = pd.DataFrame({"key": [1, 1, 2, 2], "val": [1.0, 3.0, 2.0, 6.0]})
        expected = df.groupby("key")["val"].median(numeric_only=False)
        calls: list[dict[str, object]] = []

        original_median = SeriesGroupBy.median

        def wrapped(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_median(self, *args, **kwargs)

        def _boom(*_args, **_kwargs):
            raise AssertionError("median with kwargs must delegate to pandas")

        monkeypatch.setattr(SeriesGroupBy, "median", wrapped, raising=True)
        monkeypatch.setattr(BoosterSeriesGroupBy, "_try_accelerate", _boom, raising=True)

        pandas_booster.activate()
        try:
            result = df.groupby("key")["val"].median(numeric_only=False)
        finally:
            pandas_booster.deactivate()

        assert calls == [{"args": (), "kwargs": {"numeric_only": False}}]
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
    def test_below_threshold_float_targets_fallback_for_accessor_and_proxy(
        self, monkeypatch: pytest.MonkeyPatch, agg: AggFunc
    ) -> None:
        import pandas_booster
        import pandas_booster._rust as rust

        _clear_dispatch_env(monkeypatch)
        n_rows = max(rust.get_fallback_threshold() - 1, 1)
        keys = np.resize(np.array([5, 2, 4, 1], dtype=np.int64), n_rows)
        values = (np.arange(n_rows, dtype=np.float64) % 19.0) / 4.0
        values[np.arange(n_rows) % 31 == 0] = np.nan
        df = pd.DataFrame({"key": keys, "val": values})
        expected = getattr(df.groupby("key", sort=False)["val"], agg)()

        def _boom(*_args, **_kwargs):
            raise AssertionError("below-threshold public call should use pandas fallback")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_{agg}_f64{suffix}", _boom, raising=False)

        accessor_result = cast(BoosterAccessor, df.booster).groupby("key", "val", agg, sort=False)

        pandas_booster.activate()
        try:
            proxy_result = getattr(df.groupby("key", sort=False)["val"], agg)()
        finally:
            pandas_booster.deactivate()

        pd.testing.assert_series_equal(accessor_result, expected)
        pd.testing.assert_series_equal(proxy_result, expected)
