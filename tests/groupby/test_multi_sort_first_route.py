"""Actual-extension admission, stable-segment semantics, and public-call parity."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas_booster
import pandas_booster._rust as rust
import pytest

KEY_NAMES = ["primary", "secondary", "tertiary"]


def _frame(n, fused, kernel, order):
    row = np.arange(n, dtype=np.int64)
    keys = [row[::-1].copy(), (row % 11).astype(np.int32), (row % 3).astype(np.int16)]
    if fused:
        keys[1] = keys[1].astype(np.int64)
        keys[0][:2] = [np.iinfo(np.int64).min, np.iinfo(np.int64).max]
        keys[1][:2] = [np.iinfo(np.int64).max, np.iinfo(np.int64).min]
    values = row.copy() if kernel == "i64" else row.astype(np.float64) - 100.0
    pairs = [
        (-0.0, 0.0),
        (0.0, -0.0),
        (np.nan, 9.0),
        (np.nan, np.nan),
        (-np.inf, np.inf),
        (-np.inf, -np.inf),
        (np.inf, np.nan),
    ]
    for j, pair in enumerate(pairs):
        a, b = n // 4 - 1 + j, 3 * n // 4 + j
        for column in keys:
            column[b] = column[a]
        values[[a, b]] = (
            [np.iinfo(np.int64).min, np.iinfo(np.int64).max] if kernel == "i64" else pair
        )
    df = pd.DataFrame(dict(zip(KEY_NAMES, keys)))
    df["measurement"] = values
    if order == "reversed":
        df = df.iloc[::-1]
    elif order == "shuffled":
        df = df.iloc[np.random.default_rng(42).permutation(n)]
    elif order == "preordered":
        df = df.sort_values(KEY_NAMES, kind="stable")
    return df.reset_index(drop=True)


def _assert_exact(actual, expected):
    pd.testing.assert_series_equal(actual, expected, check_exact=True)
    pd.testing.assert_index_equal(actual.index, expected.index, exact=True)
    assert actual.name == expected.name == "measurement"
    assert actual.index.names == KEY_NAMES
    assert actual.index.is_monotonic_increasing
    assert actual.dtype == expected.dtype
    assert [level.dtype for level in actual.index.levels] == [
        level.dtype for level in expected.index.levels
    ]
    if actual.dtype.kind == "f":
        zeros = expected.to_numpy() == 0
        np.testing.assert_array_equal(
            np.signbit(actual.to_numpy()[zeros]), np.signbit(expected.to_numpy()[zeros])
        )


def _from_raw(keys, values, df):
    assert len(keys) == len(KEY_NAMES)
    assert all(column.dtype == np.dtype("int64") for column in keys)
    arrays = [column.astype(df[name].dtype) for name, column in zip(KEY_NAMES, keys)]
    return pd.Series(
        values,
        index=pd.MultiIndex.from_arrays(arrays, names=KEY_NAMES),
        name="measurement",
    )


def _assert_profile(profile, n, groups, threads):
    assert profile["route"] == "sort_first"
    assert profile["route_reason"] == "cardinality_lower_bound_admitted"
    assert profile["hash_first_aggregation_count"] == 0
    assert profile["post_aggregation_sort_count"] == 0
    assert 9 * n <= 10 * profile["distinct_lower_bound"] <= 10 * groups
    assert 0 < profile["estimated_peak_bytes"] <= 512 * 1024 * 1024
    assert profile["rayon_threads"] == threads
    assert profile["final_group_count"] == groups
    assert profile["sort_first_segment_scan_count"] == 1


def _exercise_actual_extension(threads):
    assert rust.get_thread_count() == threads
    assert os.environ["PANDAS_BOOSTER_STRICT_ABI"] == "1"
    n = max(100_003, int(rust.get_fallback_threshold()) + 1)
    pandas_booster.deactivate()
    for fused in [False, True]:
        for order in ["original", "reversed", "shuffled", "preordered"]:
            for kernel in ["f64", "i64"]:
                df = _frame(n, fused, kernel, order)
                for agg in ["max", "count"]:
                    expected = getattr(df.groupby(KEY_NAMES, sort=True)["measurement"], agg)()
                    # Seven duplicate rows, not a tiny tiled seed pretending to be high-ratio.
                    assert len(expected) == n - 7
                    keys = [np.ascontiguousarray(df[name], dtype=np.int64) for name in KEY_NAMES]
                    values = np.ascontiguousarray(df["measurement"])
                    symbol = f"groupby_multi_{agg}_{kernel}_sorted"
                    normal_keys, normal_values = getattr(rust, symbol)(keys, values)
                    profile_keys, profile_values, profile = getattr(rust, f"profile_{symbol}")(
                        keys, values
                    )
                    _assert_profile(profile, n, len(expected), threads)
                    assert profile["selected_sort_strategy"] == (
                        "fused_multi_key_radix" if fused else "packed_u64"
                    )
                    _assert_exact(_from_raw(normal_keys, normal_values, df), expected)
                    _assert_exact(_from_raw(profile_keys, profile_values, df), expected)
                    assert normal_values.dtype == profile_values.dtype == expected.dtype
                    for normal_column, profile_column in zip(normal_keys, profile_keys):
                        np.testing.assert_array_equal(normal_column, profile_column)
                    np.testing.assert_array_equal(normal_values, profile_values)
                    if kernel == "f64":
                        np.testing.assert_array_equal(
                            np.signbit(normal_values), np.signbit(profile_values)
                        )

                    # Both public surfaces must call the real sorted export exactly once;
                    # a Python fallback or postsort cannot silently satisfy parity.
                    for surface in ["accessor", "proxy"]:
                        pandas_booster.deactivate()
                        if surface == "proxy":
                            pandas_booster.activate()
                        try:
                            with (
                                patch.object(rust, symbol, wraps=getattr(rust, symbol)) as spy,
                                patch.object(
                                    pd.Series,
                                    "sort_index",
                                    side_effect=AssertionError("unexpected Python output sort"),
                                ),
                            ):
                                if surface == "accessor":
                                    actual = df.booster.groupby(
                                        KEY_NAMES, "measurement", agg, sort=True
                                    )
                                else:
                                    actual = getattr(
                                        df.groupby(KEY_NAMES, sort=True)["measurement"], agg
                                    )()
                            spy.assert_called_once()
                            assert len(spy.call_args.args[1]) == n
                            _assert_exact(actual, expected)
                        finally:
                            pandas_booster.deactivate()

    # Entire all-NaN payload is still admitted: cardinality concerns keys, not values.
    df = _frame(n, True, "f64", "original")
    df["measurement"] = np.nan
    keys = [np.ascontiguousarray(df[name], dtype=np.int64) for name in KEY_NAMES]
    for agg in ["max", "count"]:
        expected = getattr(df.groupby(KEY_NAMES)["measurement"], agg)()
        symbol = f"groupby_multi_{agg}_f64_sorted"
        normal_keys, normal_values = getattr(rust, symbol)(keys, df["measurement"].to_numpy())
        profile_keys, profile_values, profile = getattr(rust, f"profile_{symbol}")(
            keys, df["measurement"].to_numpy()
        )
        _assert_profile(profile, n, len(expected), threads)
        _assert_exact(_from_raw(normal_keys, normal_values, df), expected)
        _assert_exact(_from_raw(profile_keys, profile_values, df), expected)


@pytest.mark.parametrize("threads", [1, 2, 4])
def test_real_sort_first_exports_and_public_surfaces_in_actual_rayon_pools(threads):
    # Rayon initializes once per process; changing its environment in-process is not a test.
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(threads)
    env["PANDAS_BOOSTER_STRICT_ABI"] = "1"
    env.pop("PANDAS_BOOSTER_FORCE_PANDAS_SORT", None)
    env.pop("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", None)
    script = (
        "import runpy, sys; "
        "ns = runpy.run_path(sys.argv[1]); "
        "ns['_exercise_actual_extension'](int(sys.argv[2]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(threads)],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
