"""Median numerical extreme behavior tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _patch_pandas_series_groupby_agg_to_raise,
)


class TestMedianNumericalExtremes:
    """Tests for accelerated median arithmetic at f64 boundaries."""

    @pytest.mark.parametrize("sort", [True, False])
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(
                [np.finfo(np.float64).max, np.finfo(np.float64).max],
                np.inf,
                id="positive-max-equal-overflows-like-pandas",
            ),
            pytest.param(
                [np.finfo(np.float64).max / 2.0, np.finfo(np.float64).max],
                np.inf,
                id="positive-mixed-large-overflows-like-pandas",
            ),
            pytest.param(
                [-np.finfo(np.float64).max, -np.finfo(np.float64).max],
                -np.inf,
                id="negative-max-equal-overflows-like-pandas",
            ),
            pytest.param(
                [-np.finfo(np.float64).max, -np.finfo(np.float64).max / 2.0],
                -np.inf,
                id="negative-mixed-large-overflows-like-pandas",
            ),
            pytest.param(
                [-np.finfo(np.float64).max, np.finfo(np.float64).max],
                0.0,
                id="opposite-sign-max-stays-zero-like-pandas",
            ),
            pytest.param(
                [np.finfo(np.float64).max / 2.0, np.finfo(np.float64).max / 2.0],
                np.finfo(np.float64).max / 2.0,
                id="half-max-equal-stays-finite",
            ),
        ],
    )
    def test_single_key_float_median_large_even_middle_values_match_pandas(
        self, values: list[float], expected: float, sort: bool, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster  # noqa: F401

        df = pd.DataFrame(
            {
                "key": np.array([2, 1, 1, 2], dtype=np.int64),
                "val": np.array([0.0, values[0], values[1], 0.0], dtype=np.float64),
            }
        )

        pandas_expected = df.groupby("key", sort=sort)["val"].median()
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            "median",
            "float median overflow parity must be proven by the Rust accelerator",
        )

        actual = df.booster.groupby("key", "val", "median", sort=sort)

        pd.testing.assert_series_equal(actual, pandas_expected, check_exact=True)
        assert actual.loc[1] == expected

    @pytest.mark.parametrize("sort", [True, False])
    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            pytest.param(
                [np.finfo(np.float64).max, np.finfo(np.float64).max],
                np.inf,
                id="positive-max-equal-overflows-like-pandas",
            ),
            pytest.param(
                [np.finfo(np.float64).max / 2.0, np.finfo(np.float64).max],
                np.inf,
                id="positive-mixed-large-overflows-like-pandas",
            ),
            pytest.param(
                [-np.finfo(np.float64).max, -np.finfo(np.float64).max],
                -np.inf,
                id="negative-max-equal-overflows-like-pandas",
            ),
            pytest.param(
                [-np.finfo(np.float64).max, -np.finfo(np.float64).max / 2.0],
                -np.inf,
                id="negative-mixed-large-overflows-like-pandas",
            ),
            pytest.param(
                [-np.finfo(np.float64).max, np.finfo(np.float64).max],
                0.0,
                id="opposite-sign-max-stays-zero-like-pandas",
            ),
            pytest.param(
                [np.finfo(np.float64).max / 2.0, np.finfo(np.float64).max / 2.0],
                np.finfo(np.float64).max / 2.0,
                id="half-max-equal-stays-finite",
            ),
        ],
    )
    def test_multi_key_float_median_large_even_middle_values_match_pandas(
        self, values: list[float], expected: float, sort: bool, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster  # noqa: F401

        df = pd.DataFrame(
            {
                "k1": np.array([2, 1, 1, 2], dtype=np.int64),
                "k2": np.array([20, 10, 10, 20], dtype=np.int64),
                "val": np.array([0.0, values[0], values[1], 0.0], dtype=np.float64),
            }
        )

        pandas_expected = df.groupby(["k1", "k2"], sort=sort)["val"].median()
        _patch_pandas_series_groupby_agg_to_raise(
            monkeypatch,
            "median",
            "multi-key float median overflow parity must be proven by the Rust accelerator",
        )

        actual = df.booster.groupby(["k1", "k2"], "val", "median", sort=sort)

        pd.testing.assert_series_equal(actual, pandas_expected, check_exact=True)
        assert actual.loc[(1, 10)] == expected
