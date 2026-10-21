from typing import cast

import numpy as np
import pandas as pd
import pytest
from pandas_booster._groupby_result import build_series_from_multi_result
from pandas_booster.accessor import BoosterAccessor


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
    booster_result = cast(BoosterAccessor, df.booster).groupby("k1", "val", "sum")
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
    booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
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
    booster_result = cast(BoosterAccessor, df.booster).groupby("k1", "val", "count")
    pandas_result = df.groupby("k1")["val"].count()

    assert booster_result.index.dtype == pandas_result.index.dtype
    assert booster_result.dtype == pandas_result.dtype
    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=True,
    )


def test_accessor_single_key_median_preserves_index_dtype_and_float64_value_dtype():
    import pandas_booster  # noqa: F401

    df = _make_df_int32_keys()
    df["val_int"] = (np.arange(len(df), dtype=np.int64) % 13) - 6

    booster_result = cast(BoosterAccessor, df.booster).groupby("k1", "val_int", "median")
    pandas_result = df.groupby("k1")["val_int"].median()

    assert booster_result.index.dtype == pandas_result.index.dtype
    assert booster_result.dtype == pandas_result.dtype == np.dtype(np.float64)
    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=False,
        rtol=1e-10,
    )


def test_accessor_multi_key_median_preserves_level_dtypes_and_float64_value_dtype():
    import pandas_booster  # noqa: F401

    df = _make_df_int32_keys()
    df["val_int"] = (np.arange(len(df), dtype=np.int64) % 13) - 6

    booster_result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val_int", "median")
    pandas_result = df.groupby(["k1", "k2"])["val_int"].median()

    assert isinstance(booster_result.index, pd.MultiIndex)
    assert isinstance(pandas_result.index, pd.MultiIndex)
    lvl0_booster = cast(pd.Index, booster_result.index.get_level_values(0))
    lvl0_pandas = cast(pd.Index, pandas_result.index.get_level_values(0))
    lvl1_booster = cast(pd.Index, booster_result.index.get_level_values(1))
    lvl1_pandas = cast(pd.Index, pandas_result.index.get_level_values(1))
    assert lvl0_booster.dtype == lvl0_pandas.dtype
    assert lvl1_booster.dtype == lvl1_pandas.dtype
    assert booster_result.dtype == pandas_result.dtype == np.dtype(np.float64)

    pd.testing.assert_series_equal(
        booster_result.sort_index(),
        pandas_result.sort_index(),
        check_exact=False,
        rtol=1e-10,
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

    result = cast(BoosterAccessor, df.booster).groupby("k1", "val", "sum")
    assert len(result) == 0
    expected = df.iloc[:0].groupby("k1")["val"].sum()
    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_accessor_multi_key_empty_result_preserves_level_dtypes(monkeypatch):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_multi_sum_f64(key_arrays, values):
        empty_keys = [np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)]
        return empty_keys, np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", fake_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", fake_groupby_multi_sum_f64)

    result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "sum")
    assert len(result) == 0
    assert isinstance(result.index, pd.MultiIndex)
    expected = df.iloc[:0].groupby(["k1", "k2"])["val"].sum()
    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_accessor_multi_key_empty_count_has_int64_values_and_preserves_index_dtypes(monkeypatch):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_multi_count_f64(key_arrays, values):
        empty_keys = [np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)]
        return empty_keys, np.empty((0,), dtype=np.int64)

    monkeypatch.setattr(rust, "groupby_multi_count_f64", fake_groupby_multi_count_f64)
    monkeypatch.setattr(rust, "groupby_multi_count_f64_sorted", fake_groupby_multi_count_f64)

    result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "count")
    assert len(result) == 0
    assert result.dtype == np.dtype(np.int64)
    assert isinstance(result.index, pd.MultiIndex)
    expected = df.iloc[:0].groupby(["k1", "k2"])["val"].count()
    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_accessor_single_key_empty_median_has_float64_values_and_preserves_index_dtype(
    monkeypatch,
):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_median_f64(keys, values):
        return np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(rust, "groupby_median_f64", fake_groupby_median_f64, raising=False)
    monkeypatch.setattr(rust, "groupby_median_f64_sorted", fake_groupby_median_f64, raising=False)

    result = cast(BoosterAccessor, df.booster).groupby("k1", "val", "median")
    assert len(result) == 0
    assert result.dtype == np.dtype(np.float64)
    expected = df.iloc[:0].groupby("k1")["val"].median()
    pd.testing.assert_series_equal(result, expected, check_exact=True)


def test_accessor_multi_key_empty_median_has_float64_values_and_preserves_level_dtypes(
    monkeypatch,
):
    import pandas_booster._rust as rust

    df = _make_df_int32_keys(n=100_000)

    def fake_groupby_multi_median_f64(key_arrays, values):
        empty_keys = [np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)]
        return empty_keys, np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(
        rust, "groupby_multi_median_f64", fake_groupby_multi_median_f64, raising=False
    )
    monkeypatch.setattr(
        rust, "groupby_multi_median_f64_sorted", fake_groupby_multi_median_f64, raising=False
    )

    result = cast(BoosterAccessor, df.booster).groupby(["k1", "k2"], "val", "median")
    assert len(result) == 0
    assert result.dtype == np.dtype(np.float64)
    assert isinstance(result.index, pd.MultiIndex)
    expected = df.iloc[:0].groupby(["k1", "k2"])["val"].median()
    pd.testing.assert_series_equal(result, expected, check_exact=True)


def _build_multi(keys, *, dtypes=None, sort=False, needs_python_sort=False, agg="max"):
    n = len(keys[0])
    values = np.arange(n, dtype=np.int64) - 3
    return build_series_from_multi_result(
        keys,
        values,
        by_cols=[f"key{i}" for i in range(len(keys))],
        key_dtypes=dtypes if dtypes is not None else [key.dtype for key in keys],
        name="measurement",
        value_dtype=np.dtype("int16"),
        agg=agg,
        is_val_int=True,
        sort=sort,
        needs_python_sort=needs_python_sort,
    )


def _assert_multi_reference(keys, *, dtypes=None, sort=False, needs_python_sort=False, agg="max"):
    actual = _build_multi(
        keys, dtypes=dtypes, sort=sort, needs_python_sort=needs_python_sort, agg=agg
    )
    arrays = keys if dtypes is None else [key.astype(dtype) for key, dtype in zip(keys, dtypes)]
    index = pd.MultiIndex.from_arrays(arrays, names=[f"key{i}" for i in range(len(keys))])
    dtype = np.int64 if agg == "count" else np.float64 if agg == "mean" else np.int16
    expected = pd.Series(
        np.arange(len(keys[0]), dtype=np.int64) - 3,
        index=index,
        dtype=dtype,
        name="measurement",
    )
    if sort and needs_python_sort:
        expected = expected.sort_index()
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.index.sortorder == expected.index.sortorder
    for level, reference in zip(actual.index.levels, expected.index.levels):
        pd.testing.assert_index_equal(level, reference, exact=True)
    for codes, reference in zip(actual.index.codes, expected.index.codes):
        np.testing.assert_array_equal(codes, reference)
        assert codes.dtype == reference.dtype
    return actual


@pytest.mark.parametrize(
    "dtype", ["int8", "uint8", "int16", "uint16", "int32", "uint32", "int64", "uint64"]
)
@pytest.mark.parametrize("size", [0, 1, 4])
def test_multi_builder_integer_widths_and_extrema(dtype, size):
    dtype = np.dtype(dtype)
    info = np.iinfo(dtype)
    first = np.array([info.min, 0 if dtype.kind == "i" else 1, 2, info.max], dtype=dtype)
    keys = [first[:size], np.array([2, 0, 2, 1], dtype=np.uint8)[:size]]
    _assert_multi_reference(keys)


@pytest.mark.parametrize("sort", [False, True])
@pytest.mark.parametrize("needs_python_sort", [False, True])
@pytest.mark.parametrize("agg", ["max", "count", "mean"])
@pytest.mark.parametrize(
    "first",
    [[-3, 0, 2, 9], [0, 0, 2, 9], [2, -3, 9, 0], [9, 2, 0, -3]],
    ids=["increasing", "duplicate", "shuffled", "descending"],
)
def test_multi_builder_proof_and_sort_contract(first, sort, needs_python_sort, agg):
    _assert_multi_reference(
        [
            np.array(first, dtype=np.int64),
            np.array([9, -2, 9, 1], dtype=np.int16),
            np.array([2, 1, 0, 2], dtype=np.uint32),
        ],
        sort=sort,
        needs_python_sort=needs_python_sort,
        agg=agg,
    )


@pytest.mark.parametrize(
    "keys",
    [
        [np.array([np.nan]), np.array([1])],
        [np.array([1]), np.array([np.nan])],
        [np.array([0.0, np.nan, 2.0]), np.array([2, 1, 0])],
        [np.array([0, 1, 2]), np.array(["z", "a", "z"])],
        [np.array([0, 1, 2]), np.array([False, True, False])],
    ],
)
def test_multi_builder_noninteger_and_nan_fallback(keys):
    _assert_multi_reference(keys)


@pytest.mark.parametrize("first", [[126, 127, 128], [0, 256, 512], [256, 1, 2]])
@pytest.mark.parametrize("sort", [False, True])
def test_multi_builder_proves_only_final_cast_order(first, sort):
    _assert_multi_reference(
        [np.array(first, dtype=np.int64), np.array([2, 0, 1], dtype=np.int64)],
        dtypes=[np.dtype("int8"), np.dtype("uint8")],
        sort=sort,
        needs_python_sort=True,
    )


@pytest.mark.parametrize("size", [127, 128, 32767, 32768])
def test_multi_builder_code_width_transitions(size):
    _assert_multi_reference(
        [np.arange(size, dtype=np.int64), np.arange(size - 1, -1, -1, dtype=np.int32)]
    )


@pytest.mark.parametrize("dtype", ["int8", "uint8", "int32", "uint32", "int64", "uint64"])
def test_multi_builder_reuses_first_level_where_index_supports_it(dtype):
    first = np.arange(8, dtype=dtype)
    result = _assert_multi_reference([first, np.arange(8, dtype=np.int16) % 3])
    reference = pd.Index(first, copy=False).to_numpy(copy=False)
    level = result.index.levels[0].to_numpy(copy=False)
    if np.shares_memory(reference, first):
        assert np.shares_memory(level, first)
        assert level.__array_interface__["data"][0] == first.__array_interface__["data"][0]


@pytest.mark.parametrize(
    ("first", "eligible"),
    [
        (np.array([0, 1, 2], dtype=np.int64), True),
        (np.array([1], dtype=np.int32), True),
        (np.array([1, 0, 2], dtype=np.int64), False),
        (np.array([0, 0, 2], dtype=np.int64), False),
        (np.array([], dtype=np.int64), False),
        (np.array([np.nan]), False),
    ],
)
def test_multi_builder_factorizes_only_remaining_columns(monkeypatch, first, eligible):
    second = (np.arange(len(first)) % 2).astype(np.int16)
    from_arrays = pd.MultiIndex.from_arrays
    categorical = pd.Categorical
    factorized = []
    fallback_calls = []

    def record_categorical(values, *args, **kwargs):
        factorized.append(values)
        return categorical(values, *args, **kwargs)

    def record_from_arrays(arrays, *args, **kwargs):
        fallback_calls.append(arrays)
        if eligible:
            raise AssertionError("eligible first key must not be factorized by from_arrays")
        return from_arrays(arrays, *args, **kwargs)

    monkeypatch.setattr(pd, "Categorical", record_categorical)
    monkeypatch.setattr(pd.MultiIndex, "from_arrays", record_from_arrays)
    _build_multi([first, second])
    if eligible:
        assert not fallback_calls
        assert len(factorized) == 1
        assert factorized[0] is second
    else:
        assert len(fallback_calls) == 1
        if len(first):
            assert fallback_calls[0][0] is first
        else:
            np.testing.assert_array_equal(fallback_calls[0][0], first)
            assert fallback_calls[0][0].dtype == first.dtype


@pytest.mark.parametrize(
    ("keys", "values", "names", "dtypes", "message"),
    [
        ([np.array([0])], np.array([1]), ["a"], [], "key_dtypes length mismatch"),
        ([], np.array([1]), ["a"], [np.dtype("int64")], "keys_cols length mismatch"),
        (
            [np.array([0])],
            np.array([[1]]),
            ["a"],
            [np.dtype("int64")],
            "result_values must be 1D",
        ),
        (
            [np.array([[0]])],
            np.array([1]),
            ["a"],
            [np.dtype("int64")],
            r"keys_cols\[0\] must be 1D",
        ),
        (
            [np.array([0, 1])],
            np.array([1]),
            ["a"],
            [np.dtype("int64")],
            r"keys_cols\[0\] length",
        ),
        (
            [np.array([0]), np.array([[1]])],
            np.array([1]),
            ["a", "b"],
            [np.dtype("int64"), np.dtype("int64")],
            r"keys_cols\[1\] must be 1D",
        ),
        (
            [np.array([0]), np.array([1, 2])],
            np.array([1]),
            ["a", "b"],
            [np.dtype("int64"), np.dtype("int64")],
            r"keys_cols\[1\] length",
        ),
    ],
)
def test_multi_builder_validates_before_constructing(keys, values, names, dtypes, message):
    with pytest.raises(ValueError, match=message):
        build_series_from_multi_result(
            keys,
            values,
            by_cols=names,
            key_dtypes=dtypes,
            name="v",
            value_dtype=np.dtype("int64"),
            agg="max",
            is_val_int=True,
            sort=True,
        )
