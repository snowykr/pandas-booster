"""Product overflow and special-value numerical contract tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


class TestProdNumericContracts:
    def _assert_float_prod_equal(self, result: pd.Series, expected: pd.Series) -> None:
        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=False,
            rtol=1e-10,
            atol=0.0,
        )
        result_sorted = result.sort_index()
        expected_sorted = expected.sort_index()
        zero_mask = (expected_sorted.to_numpy() == 0.0) & ~pd.isna(expected_sorted.to_numpy())
        if zero_mask.any():
            np.testing.assert_array_equal(
                np.signbit(result_sorted.to_numpy()[zero_mask]),
                np.signbit(expected_sorted.to_numpy()[zero_mask]),
            )

    def test_int64_prod_overflow_matches_pandas(self):
        import pandas_booster  # noqa: F401

        n = 120_000
        df = pd.DataFrame(
            {
                "key": np.ones(n, dtype=np.int64),
                "val": np.resize(np.array([2**62, 4, -3], dtype=np.int64), n),
            }
        )
        result = df.booster.groupby("key", "val", "prod")
        expected = df.groupby("key")["val"].prod()
        pd.testing.assert_series_equal(result.sort_index(), expected.sort_index(), check_exact=True)

    def test_float_prod_special_values_match_pandas(self):
        import pandas_booster  # noqa: F401

        pairs = [
            (1, np.nan),
            (1, np.nan),
            (2, np.nan),
            (2, 2.0),
            (3, np.inf),
            (3, 0.0),
            (4, -np.inf),
            (4, 0.0),
            (5, -0.0),
            (5, 2.0),
            (6, 0.0),
            (6, -2.0),
            (7, -0.0),
            (7, -2.0),
            (8, 1e308),
            (8, 1e308),
            (9, 1e-308),
            (9, 1e-308),
        ]
        repeats = 12_500
        df = pd.DataFrame(
            {
                "key": np.array([k for k, _ in pairs] * repeats, dtype=np.int64),
                "val": np.array([v for _, v in pairs] * repeats, dtype=np.float64),
            }
        )
        result = df.booster.groupby("key", "val", "prod")
        expected = df.groupby("key")["val"].prod()
        self._assert_float_prod_equal(result, expected)
        assert expected.loc[1] == 1.0
        assert np.isnan(expected.loc[3]) and np.isnan(result.loc[3])
        assert np.isnan(expected.loc[4]) and np.isnan(result.loc[4])
