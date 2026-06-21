from __future__ import annotations

import json

import pytest
from test_sort_false_determinism import (
    FAST_REPEATS,
    FAST_SINGLE_ROWS,
    STRESS_REPEATS,
    STRESS_SINGLE_ROWS,
    _run_script_once,
)

_SINGLE_KEY_PARTITIONED_SCRIPT_TEMPLATE = r"""
import hashlib
import json

import numpy as np
import pandas as pd
import pandas_booster._rust as _rust
from pandas_booster.accessor import BoosterAccessor  # noqa: F401


def series_fingerprint(series: pd.Series) -> str:
    h = hashlib.sha256()
    arr_idx = np.asarray(series.index)
    h.update(arr_idx.dtype.str.encode("ascii"))
    h.update(arr_idx.tobytes())

    vals = np.asarray(series.to_numpy())
    h.update(vals.dtype.str.encode("ascii"))
    h.update(vals.tobytes())
    return h.hexdigest()


threshold = _rust.get_fallback_threshold()
n = max(__N_ROWS__, threshold + 1, 20_000)
if n % 2:
    n += 1

group_count = n // 2
perm = np.arange(group_count, dtype=np.int64)
perm = np.concatenate((perm[group_count // 2 :], perm[: group_count // 2]))
k = np.repeat(perm, 2)

base = np.arange(group_count, dtype=np.float64)
v = np.empty(n, dtype=np.float64)
v[0::2] = base
v[1::2] = base + 0.5

df = pd.DataFrame({"k": k, "v": v})

res = {}
for agg in ("sum", "mean", "min", "max", "count"):
    res[f"{agg}_firstseen"] = series_fingerprint(df.booster.groupby("k", "v", agg, sort=False))

res.update({
    "std_sorted": series_fingerprint(df.booster.groupby("k", "v", "std", sort=True)),
    "std_firstseen": series_fingerprint(df.booster.groupby("k", "v", "std", sort=False)),
    "var_sorted": series_fingerprint(df.booster.groupby("k", "v", "var", sort=True)),
    "var_firstseen": series_fingerprint(df.booster.groupby("k", "v", "var", sort=False)),
    "median_sorted": series_fingerprint(df.booster.groupby("k", "v", "median", sort=True)),
    "median_firstseen": series_fingerprint(df.booster.groupby("k", "v", "median", sort=False)),
    "prod_sorted": series_fingerprint(df.booster.groupby("k", "v", "prod", sort=True)),
    "prod_firstseen": series_fingerprint(df.booster.groupby("k", "v", "prod", sort=False)),
})

print(json.dumps(res, sort_keys=True))
"""


def _run_single_key_partitioned_once(ray_threads: int, n_rows: int) -> str:
    script = _SINGLE_KEY_PARTITIONED_SCRIPT_TEMPLATE.replace("__N_ROWS__", str(n_rows))
    return _run_script_once(ray_threads, script)


def test_single_key_partitioned_template_uses_threshold_relative_row_count() -> None:
    payload = json.loads(_run_single_key_partitioned_once(1, 1))

    assert set(payload) == {
        "sum_firstseen",
        "mean_firstseen",
        "min_firstseen",
        "max_firstseen",
        "count_firstseen",
        "std_sorted",
        "std_firstseen",
        "var_sorted",
        "var_firstseen",
        "median_sorted",
        "median_firstseen",
        "prod_sorted",
        "prod_firstseen",
    }
    assert all(isinstance(value, str) and value for value in payload.values())


def test_single_key_partitioned_targets_bitwise_deterministic_across_threads() -> None:
    baseline = _run_single_key_partitioned_once(1, FAST_SINGLE_ROWS)

    assert _run_single_key_partitioned_once(1, FAST_SINGLE_ROWS) == baseline

    for _ in range(FAST_REPEATS):
        assert _run_single_key_partitioned_once(8, FAST_SINGLE_ROWS) == baseline


@pytest.mark.stress
def test_single_key_partitioned_std_var_bitwise_deterministic_stress_across_threads() -> None:
    baseline = _run_single_key_partitioned_once(1, STRESS_SINGLE_ROWS)

    assert _run_single_key_partitioned_once(1, STRESS_SINGLE_ROWS) == baseline

    for _ in range(STRESS_REPEATS):
        assert _run_single_key_partitioned_once(8, STRESS_SINGLE_ROWS) == baseline
