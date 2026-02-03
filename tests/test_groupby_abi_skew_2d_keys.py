from __future__ import annotations

import inspect
import warnings
from typing import cast

import numpy as np
import pandas as pd
import pytest


def _make_df(n: int = 120_000) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "k1": rng.integers(0, 100, size=n, dtype=np.int32),
            "k2": rng.integers(0, 50, size=n, dtype=np.int64),
            "val": rng.random(size=n).astype(np.float64, copy=False),
        }
    )


def test_legacy_2d_keys_exact_shape_normalizes_and_accelerates(monkeypatch):
    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)

    import pandas_booster
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    pandas_expected = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())

    called = {"n": 0}

    def legacy_groupby_multi_sum_f64(key_arrays, values):
        called["n"] += 1
        # Produce a correct payload in sorted group order, but with legacy 2D keys.
        keys0 = np.asarray(pandas_expected.index.get_level_values(0), dtype=np.int64)
        keys1 = np.asarray(pandas_expected.index.get_level_values(1), dtype=np.int64)
        result_values = np.asarray(pandas_expected.to_numpy(), dtype=np.float64)
        keys_2d = np.column_stack([keys0, keys1])
        assert keys_2d.shape == (result_values.shape[0], len(by_cols))
        return keys_2d, result_values

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", legacy_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", legacy_groupby_multi_sum_f64)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")

        # Accessor path
        out1 = cast(BoosterAccessor, df.booster).groupby(by_cols, "val", "sum", sort=True)

        # Proxy path
        pandas_booster.activate()
        try:
            out2 = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())
        finally:
            pandas_booster.deactivate()

    assert called["n"] >= 1
    pd.testing.assert_series_equal(out1.sort_index(), pandas_expected.sort_index(), rtol=1e-10)
    pd.testing.assert_series_equal(out2.sort_index(), pandas_expected.sort_index(), rtol=1e-10)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 1


def test_legacy_2d_keys_shape_mismatch_falls_back(monkeypatch):
    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)

    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    pandas_expected = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())

    called = {"n": 0}
    fallback_called = {"n": 0}

    def bad_2d_groupby_multi_sum_f64(key_arrays, values):
        called["n"] += 1
        keys0 = np.asarray(pandas_expected.index.get_level_values(0), dtype=np.int64)
        keys1 = np.asarray(pandas_expected.index.get_level_values(1), dtype=np.int64)
        result_values = np.asarray(pandas_expected.to_numpy(), dtype=np.float64)
        keys_2d = np.column_stack([keys0, keys1, keys0])  # wrong n_keys
        assert keys_2d.shape == (result_values.shape[0], len(by_cols) + 1)
        return keys_2d, result_values

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", bad_2d_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", bad_2d_groupby_multi_sum_f64)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    # Confirm we take the accessor fallback boundary.
    booster = cast(BoosterAccessor, df.booster)
    orig = booster._pandas_fallback

    def wrapped_fallback(by_cols, target, agg, *, sort: bool):
        fallback_called["n"] += 1
        return orig(by_cols, target, agg, sort=sort)

    monkeypatch.setattr(booster, "_pandas_fallback", wrapped_fallback)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = booster.groupby(by_cols, "val", "sum", sort=True)

    assert called["n"] >= 1
    assert fallback_called["n"] == 1
    pd.testing.assert_series_equal(out.sort_index(), pandas_expected.sort_index(), rtol=1e-10)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 1


def test_abi_skew_warning_as_error_does_not_break_fallback(monkeypatch):
    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    pandas_expected = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())

    def bad_2d_groupby_multi_sum_f64(key_arrays, values):
        keys0 = np.asarray(pandas_expected.index.get_level_values(0), dtype=np.int64)
        keys1 = np.asarray(pandas_expected.index.get_level_values(1), dtype=np.int64)
        result_values = np.asarray(pandas_expected.to_numpy(), dtype=np.float64)
        keys_2d = np.column_stack([keys0, keys1, keys0])  # wrong n_keys
        return keys_2d, result_values

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", bad_2d_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", bad_2d_groupby_multi_sum_f64)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)

    with warnings.catch_warnings():
        warnings.simplefilter("error", abi.PandasBoosterAbiSkewWarning)
        out = booster.groupby(by_cols, "val", "sum", sort=True)

    pd.testing.assert_series_equal(out.sort_index(), pandas_expected.sort_index(), rtol=1e-10)


def test_unrelated_rust_exception_propagates(monkeypatch):
    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    def boom(key_arrays, values):
        raise RuntimeError("boom")

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", boom)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", boom)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="boom"):
            _ = cast(BoosterAccessor, df.booster).groupby(by_cols, "val", "sum", sort=True)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 0


@pytest.mark.parametrize("env_value", ["1", "true", "yes", "on", " ON "])
def test_strict_abi_legacy_2d_keys_hard_fails(monkeypatch, env_value):
    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", env_value)

    df = _make_df()
    by_cols = ["k1", "k2"]

    pandas_expected = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())

    def legacy_groupby_multi_sum_f64(key_arrays, values):
        keys0 = np.asarray(pandas_expected.index.get_level_values(0), dtype=np.int64)
        keys1 = np.asarray(pandas_expected.index.get_level_values(1), dtype=np.int64)
        result_values = np.asarray(pandas_expected.to_numpy(), dtype=np.float64)
        keys_2d = np.column_stack([keys0, keys1])
        return keys_2d, result_values

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", legacy_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", legacy_groupby_multi_sum_f64)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    with pytest.raises(abi.PandasBoosterKeyShapeSkewError, match=abi.ABI_SKEW_PREFIX):
        _ = booster.groupby(by_cols, "val", "sum", sort=True)


@pytest.mark.parametrize("env_value", ["0", "false", "no", "off", " 0 "])
def test_abi_skew_notice_can_be_disabled(monkeypatch, env_value):
    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
    monkeypatch.setenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", env_value)

    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    pandas_expected = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())

    def bad_2d_groupby_multi_sum_f64(key_arrays, values):
        keys0 = np.asarray(pandas_expected.index.get_level_values(0), dtype=np.int64)
        keys1 = np.asarray(pandas_expected.index.get_level_values(1), dtype=np.int64)
        result_values = np.asarray(pandas_expected.to_numpy(), dtype=np.float64)
        keys_2d = np.column_stack([keys0, keys1, keys0])  # wrong n_keys
        return keys_2d, result_values

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", bad_2d_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", bad_2d_groupby_multi_sum_f64)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = booster.groupby(by_cols, "val", "sum", sort=True)

    pd.testing.assert_series_equal(out.sort_index(), pandas_expected.sort_index(), rtol=1e-10)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 0


def test_abi_skew_warning_points_to_callsite(monkeypatch):
    monkeypatch.delenv("PANDAS_BOOSTER_STRICT_ABI", raising=False)
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    pandas_expected = cast(pd.Series, df.groupby(by_cols, sort=True)["val"].sum())

    def legacy_groupby_multi_sum_f64(key_arrays, values):
        keys0 = np.asarray(pandas_expected.index.get_level_values(0), dtype=np.int64)
        keys1 = np.asarray(pandas_expected.index.get_level_values(1), dtype=np.int64)
        result_values = np.asarray(pandas_expected.to_numpy(), dtype=np.float64)
        keys_2d = np.column_stack([keys0, keys1])
        assert keys_2d.shape == (result_values.shape[0], len(by_cols))
        return keys_2d, result_values

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", legacy_groupby_multi_sum_f64)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", legacy_groupby_multi_sum_f64)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        frame = inspect.currentframe()
        assert frame is not None
        expected_lineno = frame.f_lineno + 1
        out = booster.groupby(by_cols, "val", "sum", sort=True)

    pd.testing.assert_series_equal(out.sort_index(), pandas_expected.sort_index(), rtol=1e-10)

    abi_warnings = [w for w in rec if issubclass(w.category, abi.PandasBoosterAbiSkewWarning)]
    abi_warnings = [w for w in abi_warnings if abi.ABI_SKEW_PREFIX in str(w.message)]
    assert len(abi_warnings) == 1
    assert abi_warnings[0].filename.endswith("test_groupby_abi_skew_2d_keys.py")
    assert abi_warnings[0].lineno == expected_lineno


def test_strict_abi_non_sequence_keys_hard_fails_without_fallback_wording(monkeypatch):
    monkeypatch.setenv("PANDAS_BOOSTER_STRICT_ABI", "1")
    monkeypatch.delenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE", raising=False)

    import pandas_booster._abi_compat as abi
    import pandas_booster._rust as rust
    from pandas_booster.accessor import BoosterAccessor

    df = _make_df()
    by_cols = ["k1", "k2"]

    def bad_keys_container(key_arrays, values):
        # Non-sequence keys payload (recognized ABI skew).
        return 123, np.asarray([1.0], dtype=np.float64)

    monkeypatch.setattr(rust, "groupby_multi_sum_f64", bad_keys_container)
    monkeypatch.setattr(rust, "groupby_multi_sum_f64_sorted", bad_keys_container)

    monkeypatch.setattr(abi, "_WARNED_ABI_SKEW", False)

    booster = cast(BoosterAccessor, df.booster)
    with pytest.raises(abi.PandasBoosterKeyShapeSkewError) as excinfo:
        _ = booster.groupby(by_cols, "val", "sum", sort=True)

    msg = str(excinfo.value)
    assert abi.ABI_SKEW_PREFIX in msg
    assert "Falling back to pandas" not in msg
