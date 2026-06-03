"""Product dtype fallback contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ._helpers import (
    _proxy_groupby_result,
)


class TestProdFallbackContracts:
    @pytest.mark.parametrize("dtype", ["Int64", "boolean"])
    def test_nullable_and_bool_prod_fallback_avoids_rust(
        self, monkeypatch: pytest.MonkeyPatch, dtype: str
    ):
        import pandas_booster._rust as rust

        n = 120_000
        if dtype == "Int64":
            val = pd.array(np.resize(np.array([1, 2, pd.NA], dtype=object), n), dtype="Int64")
        else:
            val = pd.array(np.resize(np.array([True, False], dtype=object), n), dtype="boolean")
        df = pd.DataFrame({"key": np.resize(np.array([1, 2], dtype=np.int64), n), "val": val})

        def _boom(*_args, **_kwargs):
            raise AssertionError("unsupported prod dtype should use pandas fallback")

        for kernel in ("f64", "i64"):
            for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                monkeypatch.setattr(rust, f"groupby_prod_{kernel}{suffix}", _boom, raising=False)

        result = df.booster.groupby("key", "val", "prod")
        expected = df.groupby("key")["val"].prod()
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

    def test_uint64_value_prod_fallback_avoids_unsafe_cast(self, monkeypatch: pytest.MonkeyPatch):
        import pandas_booster._rust as rust

        n = 120_000
        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2], dtype=np.int64), n),
                "val": np.resize(np.array([2**63, 3], dtype=np.uint64), n),
            }
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("uint64 prod should not enter Rust acceleration")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_i64{suffix}", _boom, raising=False)

        result = df.booster.groupby("key", "val", "prod")
        expected = df.groupby("key")["val"].prod()
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index())

    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
    def test_small_unsigned_prod_is_pandas_compatible(self, dtype):
        import pandas_booster  # noqa: F401

        n = 120_000
        df = pd.DataFrame(
            {
                "key": np.resize(np.array([1, 2, 3], dtype=np.int64), n),
                "val": np.resize(np.array([2, 3, 4], dtype=dtype), n),
            }
        )
        result = df.booster.groupby("key", "val", "prod")
        expected = df.groupby("key")["val"].prod()
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_dtype=True)
        assert result.dtype == expected.dtype == np.dtype("uint64")

    @pytest.mark.parametrize(
        ("dtype", "factor", "expected_prod"),
        [(np.int8, 20, 400), (np.int16, 200, 40_000), (np.int32, 50_000, 2_500_000_000)],
    )
    @pytest.mark.parametrize("sort", [True, False])
    def test_signed_narrow_integer_prod_falls_back_to_pandas_single_key(
        self, monkeypatch: pytest.MonkeyPatch, dtype, factor: int, expected_prod: int, sort: bool
    ):
        import pandas_booster._rust as rust

        n = 120_000
        values = np.ones(n, dtype=dtype)
        values[0] = factor
        values[1] = factor
        values[n // 2] = 3
        values[n // 2 + 1] = 4
        df = pd.DataFrame(
            {
                "key": np.repeat(np.array([1, 2], dtype=np.int64), n // 2),
                "val": values,
            }
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("narrow signed prod should use pandas fallback")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_i64{suffix}", _boom, raising=False)

        result = df.booster.groupby("key", "val", "prod", sort=sort)
        expected = df.groupby("key", sort=sort)["val"].prod()
        pd.testing.assert_series_equal(result, expected, check_exact=True, check_dtype=True)
        assert expected.loc[1] == expected_prod

    @pytest.mark.parametrize(
        ("dtype", "factor", "expected_prod"),
        [(np.int8, 20, 400), (np.int16, 200, 40_000), (np.int32, 50_000, 2_500_000_000)],
    )
    @pytest.mark.parametrize("sort", [True, False])
    def test_signed_narrow_integer_prod_falls_back_to_pandas_multi_key(
        self, monkeypatch: pytest.MonkeyPatch, dtype, factor: int, expected_prod: int, sort: bool
    ):
        import pandas_booster._rust as rust

        n = 120_000
        keys = np.repeat(np.array([1, 2], dtype=np.int64), n // 2)
        values = np.ones(n, dtype=dtype)
        values[0] = factor
        values[1] = factor
        values[n // 2] = 3
        values[n // 2 + 1] = 4
        df = pd.DataFrame(
            {
                "k1": keys,
                "k2": np.where(keys == 1, 10, 20).astype(np.int64),
                "val": values,
            }
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("narrow signed multi-key prod should use pandas fallback")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_multi_prod_i64{suffix}", _boom, raising=False)

        result = df.booster.groupby(["k1", "k2"], "val", "prod", sort=sort)
        expected = df.groupby(["k1", "k2"], sort=sort)["val"].prod()
        pd.testing.assert_series_equal(result, expected, check_exact=True, check_dtype=True)
        assert expected.loc[(1, 10)] == expected_prod

    def test_multi_key_proxy_unsigned_prod_falls_back_to_pandas(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster._rust as rust

        n = 120_000
        df = pd.DataFrame(
            {
                "k1": np.resize(np.array([1, 2, 3], dtype=np.int64), n),
                "k2": np.resize(np.array([10, 20], dtype=np.int64), n),
                "val": np.resize(np.array([2, 3, 4], dtype=np.uint32), n),
            }
        )

        def _boom(*_args, **_kwargs):
            raise AssertionError("unsigned multi-key proxy prod should use pandas fallback")

        for kernel in ("f64", "i64"):
            for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
                monkeypatch.setattr(
                    rust, f"groupby_multi_prod_{kernel}{suffix}", _boom, raising=False
                )

        result = _proxy_groupby_result(df, ["k1", "k2"], "val", "prod", sort=False)
        expected = df.groupby(["k1", "k2"], sort=False)["val"].prod()
        pd.testing.assert_series_equal(result, expected, check_dtype=True)
        assert result.dtype == expected.dtype == np.dtype("uint64")
