"""Std/var unsupported-input fallback contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _accessor_groupby_result,
    _patch_all_i64_kernels_for_agg_to_raise,
    _patch_all_std_var_kernels_to_raise,
    _proxy_groupby_result,
)


class TestStdVarFallbackContracts:
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
