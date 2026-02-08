import os
import subprocess
import sys
from pathlib import Path

import pytest

FAST_REPEATS = 2
STRESS_REPEATS = 4
FAST_MULTI_ROWS = 120_000
STRESS_MULTI_ROWS = 160_000
FAST_SINGLE_ROWS = 120_000
STRESS_SINGLE_ROWS = 180_000

_MULTI_KEY_SCRIPT_TEMPLATE = r"""
import hashlib
import json

import numpy as np
import pandas as pd
import pandas_booster._rust as _rust
from pandas_booster.accessor import BoosterAccessor  # noqa: F401


def series_fingerprint(series: pd.Series) -> str:
    h = hashlib.sha256()
    idx = series.index
    if isinstance(idx, pd.MultiIndex):
        for level in range(idx.nlevels):
            arr = np.asarray(idx.get_level_values(level))
            h.update(arr.dtype.str.encode("ascii"))
            h.update(arr.tobytes())
    else:
        arr = np.asarray(idx)
        h.update(arr.dtype.str.encode("ascii"))
        h.update(arr.tobytes())

    vals = np.asarray(series.to_numpy())
    h.update(vals.dtype.str.encode("ascii"))
    h.update(vals.tobytes())
    return h.hexdigest()


rng = np.random.default_rng(20260207)
n = __N_ROWS__
if n < _rust.get_fallback_threshold():
    raise RuntimeError("determinism test must run above rust fallback threshold")
df = pd.DataFrame(
    {
        "k1": rng.integers(0, 40_000, size=n, dtype=np.int64),
        "k2": rng.integers(0, 20_000, size=n, dtype=np.int64),
        "v_float": rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64),
        "v_int": rng.integers(-50, 50, size=n, dtype=np.int64),
    }
)

res_float = df.booster.groupby(["k1", "k2"], "v_float", "sum", sort=False)
res_int = df.booster.groupby(["k1", "k2"], "v_int", "sum", sort=False)

print(
    json.dumps(
        {
            "float_sum": series_fingerprint(res_float),
            "int_sum": series_fingerprint(res_int),
        },
        sort_keys=True,
    )
)
"""

_SINGLE_KEY_SCRIPT_TEMPLATE = r"""
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


n = __N_ROWS__
if n < _rust.get_fallback_threshold():
    raise RuntimeError("determinism test must run above rust fallback threshold")
idx = np.arange(n, dtype=np.int64)
k = idx % 257

v = np.where(
    idx % 4 == 0,
    1e16,
    np.where(idx % 4 == 1, 1.0, np.where(idx % 4 == 2, -1e16, 0.25)),
).astype(np.float64)
v[idx % 997 == 0] = np.nan
v[idx % 991 == 0] = -0.0

df = pd.DataFrame({"k": k, "v": v})

res = {
    "sum_sorted": series_fingerprint(df.booster.groupby("k", "v", "sum", sort=True)),
    "sum_firstseen": series_fingerprint(df.booster.groupby("k", "v", "sum", sort=False)),
    "mean_sorted": series_fingerprint(df.booster.groupby("k", "v", "mean", sort=True)),
    "mean_firstseen": series_fingerprint(df.booster.groupby("k", "v", "mean", sort=False)),
}

print(json.dumps(res, sort_keys=True))
"""


def _run_script_once(ray_threads: int, script: str) -> str:
    env = os.environ.copy()
    env["RAYON_NUM_THREADS"] = str(ray_threads)
    env["PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY"] = "0"
    env["PANDAS_BOOSTER_FORCE_PANDAS_SORT"] = "0"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def _run_multi_key_once(ray_threads: int, n_rows: int) -> str:
    script = _MULTI_KEY_SCRIPT_TEMPLATE.replace("__N_ROWS__", str(n_rows))
    return _run_script_once(ray_threads, script)


def _run_single_key_once(ray_threads: int, n_rows: int) -> str:
    script = _SINGLE_KEY_SCRIPT_TEMPLATE.replace("__N_ROWS__", str(n_rows))
    return _run_script_once(ray_threads, script)


def test_sort_false_fingerprint_deterministic_across_threads_and_repeats() -> None:
    baseline = _run_multi_key_once(1, FAST_MULTI_ROWS)

    assert _run_multi_key_once(1, FAST_MULTI_ROWS) == baseline

    for _ in range(FAST_REPEATS):
        assert _run_multi_key_once(8, FAST_MULTI_ROWS) == baseline


def test_single_key_float_sum_mean_bitwise_deterministic_across_threads() -> None:
    baseline = _run_single_key_once(1, FAST_SINGLE_ROWS)

    assert _run_single_key_once(1, FAST_SINGLE_ROWS) == baseline

    for _ in range(FAST_REPEATS):
        assert _run_single_key_once(8, FAST_SINGLE_ROWS) == baseline


@pytest.mark.stress
def test_sort_false_fingerprint_deterministic_stress_across_threads_and_repeats() -> None:
    baseline = _run_multi_key_once(1, STRESS_MULTI_ROWS)

    assert _run_multi_key_once(1, STRESS_MULTI_ROWS) == baseline

    for _ in range(STRESS_REPEATS):
        assert _run_multi_key_once(8, STRESS_MULTI_ROWS) == baseline


@pytest.mark.stress
def test_single_key_float_sum_mean_bitwise_deterministic_stress_across_threads() -> None:
    baseline = _run_single_key_once(1, STRESS_SINGLE_ROWS)

    assert _run_single_key_once(1, STRESS_SINGLE_ROWS) == baseline

    for _ in range(STRESS_REPEATS):
        assert _run_single_key_once(8, STRESS_SINGLE_ROWS) == baseline
