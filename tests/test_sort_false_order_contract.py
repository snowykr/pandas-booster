from __future__ import annotations

import json

import pytest
from test_sort_false_determinism import STRESS_REPEATS, _run_script_once

_ORDER_CONTRACT_SCRIPT_TEMPLATE = r"""
import json

import numpy as np
import pandas as pd
from pandas_booster.accessor import BoosterAccessor  # noqa: F401


def encode_scalar(value: float) -> str:
    scalar = np.float64(value)
    if np.isnan(scalar):
        return "nan"
    return scalar.hex()


def encode_series(series: pd.Series) -> dict[str, list[object]]:
    index = series.index
    if isinstance(index, pd.MultiIndex):
        encoded_index = [[int(part) for part in item] for item in index.tolist()]
    else:
        encoded_index = [int(item) for item in index.tolist()]

    return {
        "index": encoded_index,
        "values": [encode_scalar(value) for value in series.to_numpy()],
    }


float_df = pd.DataFrame(
    {
        "key": [5, 2, 5, 4, 2, 7, 8, 8, 1],
        "val": [1.0, np.nan, 3.0, np.nan, 5.0, 11.0, 2.0, 2.0, np.nan],
    }
)
int_df = pd.DataFrame(
    {
        "key": [8, 3, 8, 5, 3, 4],
        "val": np.array([2, 6, 4, 10, 14, 12], dtype=np.int64),
    }
)
multi_df = pd.DataFrame(
    {
        "k1": [2, 1, 2, 1, 2, 3, 1],
        "k2": [9, 8, 9, 7, 8, 7, 8],
        "val": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    }
)

payload = {}
for agg in ("std", "var", "median"):
    payload[f"float_{agg}"] = encode_series(float_df.booster.groupby("key", "val", agg, sort=False))
    payload[f"int_{agg}"] = encode_series(int_df.booster.groupby("key", "val", agg, sort=False))
    payload[f"multi_{agg}"] = encode_series(
        multi_df.booster.groupby(["k1", "k2"], "val", agg, sort=False)
    )

print(json.dumps(payload, sort_keys=True))
"""

EXPECTED_ORDER_CONTRACT_PAYLOAD = {
    "float_std": {
        "index": [5, 2, 4, 7, 8, 1],
        "values": ["0x1.6a09e667f3bcdp+0", "nan", "nan", "nan", "0x0.0p+0", "nan"],
    },
    "float_var": {
        "index": [5, 2, 4, 7, 8, 1],
        "values": ["0x1.0000000000000p+1", "nan", "nan", "nan", "0x0.0p+0", "nan"],
    },
    "int_std": {
        "index": [8, 3, 5, 4],
        "values": [
            "0x1.6a09e667f3bcdp+0",
            "0x1.6a09e667f3bcdp+2",
            "nan",
            "nan",
        ],
    },
    "int_var": {
        "index": [8, 3, 5, 4],
        "values": ["0x1.0000000000000p+1", "0x1.0000000000000p+5", "nan", "nan"],
    },
    "multi_std": {
        "index": [[2, 9], [1, 8], [1, 7], [2, 8], [3, 7]],
        "values": ["0x1.6a09e667f3bcdp+0", "0x1.c48c6001f0ac0p+1", "nan", "nan", "nan"],
    },
    "multi_var": {
        "index": [[2, 9], [1, 8], [1, 7], [2, 8], [3, 7]],
        "values": ["0x1.0000000000000p+1", "0x1.9000000000000p+3", "nan", "nan", "nan"],
    },
    "float_median": {
        "index": [5, 2, 4, 7, 8, 1],
        "values": [
            "0x1.0000000000000p+1",
            "0x1.4000000000000p+2",
            "nan",
            "0x1.6000000000000p+3",
            "0x1.0000000000000p+1",
            "nan",
        ],
    },
    "int_median": {
        "index": [8, 3, 5, 4],
        "values": [
            "0x1.8000000000000p+1",
            "0x1.4000000000000p+3",
            "0x1.4000000000000p+3",
            "0x1.8000000000000p+3",
        ],
    },
    "multi_median": {
        "index": [[2, 9], [1, 8], [1, 7], [2, 8], [3, 7]],
        "values": [
            "0x1.0000000000000p+1",
            "0x1.2000000000000p+2",
            "0x1.0000000000000p+2",
            "0x1.4000000000000p+2",
            "0x1.8000000000000p+2",
        ],
    },
}


def _run_order_contract_once(ray_threads: int) -> str:
    return _run_script_once(ray_threads, _ORDER_CONTRACT_SCRIPT_TEMPLATE)


@pytest.mark.stress
def test_sort_false_std_var_order_contract_stable_across_threads() -> None:
    baseline = json.loads(_run_order_contract_once(1))

    assert baseline == EXPECTED_ORDER_CONTRACT_PAYLOAD
    assert json.loads(_run_order_contract_once(1)) == EXPECTED_ORDER_CONTRACT_PAYLOAD

    for _ in range(STRESS_REPEATS):
        assert json.loads(_run_order_contract_once(8)) == EXPECTED_ORDER_CONTRACT_PAYLOAD
