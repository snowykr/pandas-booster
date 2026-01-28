from typing import cast

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def cleanup_activation():
    import pandas_booster

    yield
    pandas_booster.deactivate()


def _make_df_int32_keys(n: int = 200_000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "k1": rng.integers(0, 100, size=n, dtype=np.int32),
            "k2": rng.integers(0, 50, size=n, dtype=np.int64),
            "val": rng.random(size=n),
        }
    )


def test_accessor_single_key_preserves_index_dtype_int32():
    import pandas_booster  # noqa: F401

    df = _make_df_int32_keys()
    booster_result = df.booster.groupby("k1", "val", "sum")
    pandas_result = df.groupby("k1")["val"].sum()

    assert booster_result.index.dtype == pandas_result.index.dtype
    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=False,
        rtol=1e-10,
    )


def test_accessor_multi_key_preserves_level_dtypes_mixed_int32_int64():
    import pandas_booster  # noqa: F401

    df = _make_df_int32_keys()
    booster_result = df.booster.groupby(["k1", "k2"], "val", "sum")
    pandas_result = df.groupby(["k1", "k2"])["val"].sum()

    assert isinstance(booster_result.index, pd.MultiIndex)
    assert isinstance(pandas_result.index, pd.MultiIndex)
    lvl0_booster = cast(pd.Index, booster_result.index.get_level_values(0))
    lvl0_pandas = cast(pd.Index, pandas_result.index.get_level_values(0))
    lvl1_booster = cast(pd.Index, booster_result.index.get_level_values(1))
    lvl1_pandas = cast(pd.Index, pandas_result.index.get_level_values(1))
    assert lvl0_booster.dtype == lvl0_pandas.dtype
    assert lvl1_booster.dtype == lvl1_pandas.dtype

    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=False,
        rtol=1e-10,
    )


def test_proxy_single_key_preserves_index_dtype_int32():
    import pandas_booster

    df = _make_df_int32_keys()

    pandas_booster.activate()
    booster_result = df.groupby("k1")["val"].sum()
    pandas_booster.deactivate()
    pandas_result = df.groupby("k1")["val"].sum()

    assert booster_result.index.dtype == pandas_result.index.dtype
    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=False,
        rtol=1e-10,
    )


def test_proxy_multi_key_preserves_level_dtypes_mixed_int32_int64():
    import pandas_booster

    df = _make_df_int32_keys()

    pandas_booster.activate()
    booster_result = df.groupby(["k1", "k2"])["val"].sum()
    pandas_booster.deactivate()
    pandas_result = df.groupby(["k1", "k2"])["val"].sum()

    assert isinstance(booster_result.index, pd.MultiIndex)
    assert isinstance(pandas_result.index, pd.MultiIndex)
    lvl0_booster = cast(pd.Index, booster_result.index.get_level_values(0))
    lvl0_pandas = cast(pd.Index, pandas_result.index.get_level_values(0))
    lvl1_booster = cast(pd.Index, booster_result.index.get_level_values(1))
    lvl1_pandas = cast(pd.Index, pandas_result.index.get_level_values(1))
    assert lvl0_booster.dtype == lvl0_pandas.dtype
    assert lvl1_booster.dtype == lvl1_pandas.dtype

    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=False,
        rtol=1e-10,
    )


def test_accessor_single_key_count_preserves_index_dtype_and_value_dtype():
    import pandas_booster  # noqa: F401

    df = _make_df_int32_keys()
    booster_result = df.booster.groupby("k1", "val", "count")
    pandas_result = df.groupby("k1")["val"].count()

    assert booster_result.index.dtype == pandas_result.index.dtype
    assert booster_result.dtype == pandas_result.dtype
    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=True,
    )


def test_proxy_single_key_count_preserves_index_dtype_and_value_dtype():
    import pandas_booster

    df = _make_df_int32_keys()

    pandas_booster.activate()
    booster_result = df.groupby("k1")["val"].count()
    pandas_booster.deactivate()
    pandas_result = df.groupby("k1")["val"].count()

    assert booster_result.index.dtype == pandas_result.index.dtype
    assert booster_result.dtype == pandas_result.dtype
    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=True,
    )


def test_accessor_single_key_empty_result_preserves_index_dtype(monkeypatch):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_sum_f64(keys, values):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(rust, "groupby_sum_f64", fake_groupby_sum_f64)
    monkeypatch.setattr(rust, "groupby_sum_f64_sorted", fake_groupby_sum_f64)

    result = df.booster.groupby("k1", "val", "sum")
    assert len(result) == 0
    assert result.index.dtype == np.dtype(np.int32)


def test_accessor_multi_key_empty_result_preserves_level_dtypes(monkeypatch):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_multi_sum_f64(key_arrays, values):
        empty_keys = [np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)]
        return empty_keys, np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", fake_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", fake_groupby_multi_sum_f64)

    result = df.booster.groupby(["k1", "k2"], "val", "sum")
    assert len(result) == 0
    assert isinstance(result.index, pd.MultiIndex)
    lvl0 = cast(pd.Index, result.index.get_level_values(0))
    lvl1 = cast(pd.Index, result.index.get_level_values(1))
    assert lvl0.dtype == np.dtype(np.int32)
    assert lvl1.dtype == np.dtype(np.int64)


def test_accessor_multi_key_empty_count_has_int64_values_and_preserves_index_dtypes(monkeypatch):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_multi_count_f64(key_arrays, values):
        empty_keys = [np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)]
        return empty_keys, np.empty((0,), dtype=np.int64)

    monkeypatch.setattr(rust, "groupby_multi_count_f64", fake_groupby_multi_count_f64)
    monkeypatch.setattr(rust, "groupby_multi_count_f64_sorted", fake_groupby_multi_count_f64)

    result = df.booster.groupby(["k1", "k2"], "val", "count")
    assert len(result) == 0
    assert result.dtype == np.dtype(np.int64)
    assert isinstance(result.index, pd.MultiIndex)
    lvl0 = cast(pd.Index, result.index.get_level_values(0))
    lvl1 = cast(pd.Index, result.index.get_level_values(1))
    assert lvl0.dtype == np.dtype(np.int32)
    assert lvl1.dtype == np.dtype(np.int64)
