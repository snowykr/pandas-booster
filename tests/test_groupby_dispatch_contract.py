from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest


def _clear_dispatch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "PANDAS_BOOSTER_FORCE_PANDAS_SORT",
        "PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY",
        "PANDAS_BOOSTER_STRICT_ABI",
        "PANDAS_BOOSTER_ABI_SKEW_NOTICE",
    ):
        monkeypatch.delenv(name, raising=False)


def _accessor_groupby_result(
    df: pd.DataFrame,
    by: str | list[str],
    target: str,
    agg: str,
    *,
    sort: bool,
) -> pd.Series:
    import pandas_booster

    _ = pandas_booster
    return df.booster.groupby(by, target, agg, sort=sort)


def _proxy_groupby_result(
    df: pd.DataFrame,
    by: str | list[str],
    target: str,
    agg: str,
    *,
    sort: bool,
) -> pd.Series:
    import pandas_booster

    pandas_booster.activate()
    try:
        grouped = df.groupby(by, sort=sort)[target]
        return getattr(grouped, agg)()
    finally:
        pandas_booster.deactivate()


def _assert_prod_series_matches_pandas(
    actual: pd.Series,
    expected: pd.Series,
) -> None:
    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=False,
        check_dtype=True,
        rtol=1e-10,
        atol=0.0,
    )
    zero_mask = (expected.to_numpy() == 0.0) & ~pd.isna(expected.to_numpy())
    if zero_mask.any():
        np.testing.assert_array_equal(
            np.signbit(actual.to_numpy()[zero_mask]),
            np.signbit(expected.to_numpy()[zero_mask]),
        )


@pytest.mark.parametrize("sort", [True, False], ids=["sorted", "firstseen"])
def test_discussion_r3196780027_narrow_integer_prod_fallback_matches_accessor_and_proxy(
    monkeypatch: pytest.MonkeyPatch, sort: bool
) -> None:
    import pandas_booster._rust as rust

    _clear_dispatch_env(monkeypatch)
    n_rows = max(rust.get_fallback_threshold(), 120_000)
    values = np.ones(n_rows, dtype=np.int32)
    values[0] = 50_000
    values[1] = 50_000
    values[n_rows // 2] = 3
    values[n_rows // 2 + 1] = 4
    df = pd.DataFrame(
        {"key": np.repeat(np.array([1, 2], dtype=np.int64), n_rows // 2), "val": values}
    )
    expected = df.groupby("key", sort=sort)["val"].prod()

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("narrow signed prod should stay on pandas fallback")

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(rust, f"groupby_prod_i64{suffix}", _boom, raising=False)

    accessor_result = _accessor_groupby_result(df, "key", "val", "prod", sort=sort)
    proxy_result = _proxy_groupby_result(df, "key", "val", "prod", sort=sort)

    pd.testing.assert_series_equal(accessor_result, expected, check_exact=True, check_dtype=True)
    pd.testing.assert_series_equal(proxy_result, expected, check_exact=True, check_dtype=True)


@pytest.mark.parametrize("sort", [True, False], ids=["sorted", "firstseen"])
def test_discussion_r3197179732_single_key_float_prod_matches_accessor_and_proxy(
    monkeypatch: pytest.MonkeyPatch, sort: bool
) -> None:
    import pandas_booster._rust as rust
    from pandas.core.groupby.generic import SeriesGroupBy

    _clear_dispatch_env(monkeypatch)
    n_rows = max(rust.get_fallback_threshold(), 100_000)
    df = pd.DataFrame(
        {
            "key": np.resize(np.array([1, 2], dtype=np.int64), n_rows),
            "val": np.resize(np.array([1e308, 1e308, 1e-308, 1e-308], dtype=np.float64), n_rows),
        }
    )
    expected = df.groupby("key", sort=sort)["val"].prod()

    def _boom(_self: SeriesGroupBy, *_args: object, **_kwargs: object) -> pd.Series:
        raise AssertionError("float prod regression coverage must stay on the Rust path")

    monkeypatch.setattr(SeriesGroupBy, "prod", _boom, raising=True)

    accessor_result = _accessor_groupby_result(df, "key", "val", "prod", sort=sort)
    proxy_result = _proxy_groupby_result(df, "key", "val", "prod", sort=sort)

    _assert_prod_series_matches_pandas(accessor_result, expected)
    _assert_prod_series_matches_pandas(proxy_result, expected)


@pytest.mark.parametrize("sort", [True, False], ids=["sorted", "firstseen"])
def test_discussion_r3197709485_float_median_overflow_matches_accessor_and_proxy(
    monkeypatch: pytest.MonkeyPatch, sort: bool
) -> None:
    from pandas.core.groupby.generic import SeriesGroupBy

    _clear_dispatch_env(monkeypatch)
    df = pd.DataFrame(
        {
            "key": np.array([2, 1, 1, 2], dtype=np.int64),
            "val": np.array(
                [0.0, np.finfo(np.float64).max, np.finfo(np.float64).max / 2.0, 0.0],
                dtype=np.float64,
            ),
        }
    )
    expected = df.groupby("key", sort=sort)["val"].median()

    def _boom(_self: SeriesGroupBy, *_args: object, **_kwargs: object) -> pd.Series:
        raise AssertionError("float median overflow parity must stay on the Rust path")

    monkeypatch.setattr(SeriesGroupBy, "median", _boom, raising=True)

    accessor_result = _accessor_groupby_result(df, "key", "val", "median", sort=sort)
    proxy_result = _proxy_groupby_result(df, "key", "val", "median", sort=sort)

    pd.testing.assert_series_equal(accessor_result, expected, check_exact=True)
    pd.testing.assert_series_equal(proxy_result, expected, check_exact=True)
    assert accessor_result.loc[1] == np.inf
    assert proxy_result.loc[1] == np.inf


def test_missing_ordered_float_prod_abi_marker_falls_back_when_not_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust

    _clear_dispatch_env(monkeypatch)
    n_rows = max(rust.get_fallback_threshold(), 100_000)
    df = pd.DataFrame(
        {
            "key": np.resize(np.array([1, 2], dtype=np.int64), n_rows),
            "val": np.resize(np.array([1e308, 1e308, 1e-308, 1e-308], dtype=np.float64), n_rows),
        }
    )
    expected = df.groupby("key", sort=True)["val"].prod()

    monkeypatch.delattr(rust, "has_ordered_single_key_float_prod_abi", raising=False)
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("missing ordered-prod ABI marker should force pandas fallback")

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        accessor_result = _accessor_groupby_result(df, "key", "val", "prod", sort=True)
        proxy_result = _proxy_groupby_result(df, "key", "val", "prod", sort=True)

    _assert_prod_series_matches_pandas(accessor_result, expected)
    _assert_prod_series_matches_pandas(proxy_result, expected)
    assert any("has_ordered_single_key_float_prod_abi" in str(w.message) for w in recorded)


def test_missing_ordered_float_prod_abi_marker_raises_when_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust

    n_rows = max(rust.get_fallback_threshold(), 100_000)
    df = pd.DataFrame(
        {
            "key": np.resize(np.array([1, 2], dtype=np.int64), n_rows),
            "val": np.resize(np.array([1e308, 1e308, 1e-308, 1e-308], dtype=np.float64), n_rows),
        }
    )

    _clear_dispatch_env(monkeypatch)
    monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
    monkeypatch.delattr(rust, "has_ordered_single_key_float_prod_abi", raising=False)
    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    with pytest.raises(
        abi.PandasBoosterKeyShapeSkewError,
        match="has_ordered_single_key_float_prod_abi",
    ):
        _accessor_groupby_result(df, "key", "val", "prod", sort=True)

    abi._WARNED_ABI_SKEW = False
    with pytest.raises(
        abi.PandasBoosterKeyShapeSkewError,
        match="has_ordered_single_key_float_prod_abi",
    ):
        _proxy_groupby_result(df, "key", "val", "prod", sort=True)


def test_force_pandas_float_groupby_forces_single_key_float_prod_fallback_for_both_surfaces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pandas_booster._rust as rust

    n_rows = max(rust.get_fallback_threshold(), 120_000)
    df = pd.DataFrame(
        {
            "key": np.repeat(np.array([1, 2], dtype=np.int64), n_rows // 2),
            "val": np.linspace(1.001, 1.01, n_rows, dtype=np.float64),
        }
    )
    expected = df.groupby("key", sort=False)["val"].prod()

    _clear_dispatch_env(monkeypatch)
    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY should force fallback")

    for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
        monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

    accessor_result = _accessor_groupby_result(df, "key", "val", "prod", sort=False)
    proxy_result = _proxy_groupby_result(df, "key", "val", "prod", sort=False)

    _assert_prod_series_matches_pandas(accessor_result, expected)
    _assert_prod_series_matches_pandas(proxy_result, expected)


@pytest.mark.parametrize(
    ("sort", "expected_call"),
    [(True, "sorted"), (False, "firstseen_u32")],
    ids=["sorted-kernel", "firstseen-kernel"],
)
def test_sort_flag_selects_matching_single_key_float_prod_kernel_for_accessor_and_proxy(
    monkeypatch: pytest.MonkeyPatch, sort: bool, expected_call: str
) -> None:
    import pandas_booster._rust as rust
    from pandas.core.groupby.generic import SeriesGroupBy

    _clear_dispatch_env(monkeypatch)
    monkeypatch.setattr(rust, "has_ordered_single_key_float_prod_abi", True, raising=False)

    n_rows = max(rust.get_fallback_threshold(), 100_000)
    df = pd.DataFrame(
        {
            "key": np.resize(np.array([5, 2, 4, 7, 8, 1], dtype=np.int64), n_rows),
            "val": np.resize(np.array([1.5, 0.5, 4.0, 2.0, 8.0, 16.0], dtype=np.float64), n_rows),
        }
    )
    expected = df.groupby("key", sort=sort)["val"].prod()
    calls: list[str] = []

    def _payload() -> tuple[np.ndarray, np.ndarray]:
        return (
            np.asarray(expected.index.to_numpy(), dtype=np.int64),
            np.asarray(expected.to_numpy(), dtype=np.float64),
        )

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("prod should route through the selected sort kernel")

    def _sorted(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        calls.append("sorted")
        return _payload()

    def _firstseen_u32(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        calls.append("firstseen_u32")
        return _payload()

    monkeypatch.setattr(rust, "groupby_prod_f64", _boom, raising=False)
    monkeypatch.setattr(rust, "groupby_prod_f64_sorted", _sorted if sort else _boom, raising=False)
    monkeypatch.setattr(
        rust,
        "groupby_prod_f64_firstseen_u32",
        _firstseen_u32 if not sort else _boom,
        raising=False,
    )
    monkeypatch.setattr(rust, "groupby_prod_f64_firstseen_u64", _boom, raising=False)

    def _pandas_prod(_self: SeriesGroupBy, *_args: object, **_kwargs: object) -> pd.Series:
        raise AssertionError("sort-kernel contract must not use pandas fallback")

    monkeypatch.setattr(SeriesGroupBy, "prod", _pandas_prod, raising=True)

    accessor_result = _accessor_groupby_result(df, "key", "val", "prod", sort=sort)
    proxy_result = _proxy_groupby_result(df, "key", "val", "prod", sort=sort)

    _assert_prod_series_matches_pandas(accessor_result, expected)
    _assert_prod_series_matches_pandas(proxy_result, expected)
    assert calls == [expected_call, expected_call]
