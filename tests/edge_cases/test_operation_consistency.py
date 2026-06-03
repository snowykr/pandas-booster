"""Cross-operation consistency tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestAllOperationsConsistency:
    """Ensure all operations work consistently across edge cases."""

    @pytest.fixture
    def stress_df(self):
        """Create a stress-test DataFrame with various edge values."""
        np.random.seed(42)
        n = 200_000
        values = np.random.random(n) * 1000 - 500  # Range: -500 to 500

        # Inject edge cases
        values[::100] = np.nan  # 1% NaN
        values[::500] = np.inf  # 0.2% +inf
        values[::700] = -np.inf  # ~0.14% -inf
        values[::1000] = 0.0  # 0.1% zeros

        return pd.DataFrame(
            {
                "key": np.random.randint(-50, 50, size=n),
                "val": values,
            }
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max"])
    def test_operation_consistency(self, stress_df, agg):
        """All operations should match Pandas on stress DataFrame."""
        import pandas_booster  # noqa: F401

        result = stress_df.booster.groupby("key", "val", agg)
        expected = getattr(stress_df.groupby("key")["val"], agg)()

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )
