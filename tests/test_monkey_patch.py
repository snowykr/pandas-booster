import warnings

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def large_df():
    np.random.seed(42)
    n = 200_000
    return pd.DataFrame(
        {
            "key": np.random.randint(0, 100, size=n),
            "key2": np.random.randint(0, 50, size=n),
            "val_int": np.random.randint(0, 100, size=n),
            "val_float": np.random.random(size=n) * 100,
            "val_str": np.random.choice(["a", "b", "c"], size=n),
        }
    )


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "key": [1, 2, 1, 2, 1],
            "val_int": [10, 20, 30, 40, 50],
            "val_float": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


@pytest.fixture(autouse=True)
def cleanup_activation():
    import pandas_booster

    yield
    pandas_booster.deactivate()


class TestActivateDeactivate:
    def test_activate_sets_flag(self):
        import pandas_booster

        assert not pandas_booster.is_active()
        pandas_booster.activate()
        assert pandas_booster.is_active()

    def test_deactivate_clears_flag(self):
        import pandas_booster

        pandas_booster.activate()
        assert pandas_booster.is_active()
        pandas_booster.deactivate()
        assert not pandas_booster.is_active()

    def test_double_activate_is_safe(self):
        import pandas_booster

        pandas_booster.activate()
        pandas_booster.activate()
        assert pandas_booster.is_active()

    def test_double_deactivate_is_safe(self):
        import pandas_booster

        pandas_booster.activate()
        pandas_booster.deactivate()
        pandas_booster.deactivate()
        assert not pandas_booster.is_active()


class TestProxyReturnsCorrectType:
    def test_groupby_returns_proxy(self, large_df):
        import pandas_booster
        from pandas_booster.proxy import BoosterDataFrameGroupBy

        pandas_booster.activate()
        gb = large_df.groupby("key")
        assert isinstance(gb, BoosterDataFrameGroupBy)

    def test_getitem_returns_series_proxy(self, large_df):
        import pandas_booster
        from pandas_booster.proxy import BoosterSeriesGroupBy

        pandas_booster.activate()
        sgb = large_df.groupby("key")["val_float"]
        assert isinstance(sgb, BoosterSeriesGroupBy)

    def test_multikey_groupby_returns_proxy(self, large_df):
        import pandas_booster
        from pandas_booster.proxy import BoosterDataFrameGroupBy

        pandas_booster.activate()
        gb = large_df.groupby(["key", "key2"])
        assert isinstance(gb, BoosterDataFrameGroupBy)


class TestGroupbyPositionalArgsCompatibility:
    def test_positional_axis_0_returns_proxy(self, large_df):
        import pandas_booster
        from pandas_booster.proxy import BoosterDataFrameGroupBy

        pandas_booster.activate()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            gb = large_df.groupby("key", 0)
        assert isinstance(gb, BoosterDataFrameGroupBy)

    def test_positional_axis_1_returns_pandas_groupby(self, large_df):
        import pandas_booster
        from pandas.core.groupby import DataFrameGroupBy

        pandas_booster.activate()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            gb = large_df.groupby("key", 1)
        assert isinstance(gb, DataFrameGroupBy)

    def test_positional_axis_with_kwargs_matches_pandas(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            booster_result = large_df.groupby("key", 0, sort=False)["val_float"].sum()
        pandas_booster.deactivate()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            pandas_result = large_df.groupby("key", 0, sort=False)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestAcceleratedAggregations:
    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
    def test_single_key_agg_matches_pandas(self, large_df, agg):
        import pandas_booster

        pandas_booster.activate()
        booster_result = getattr(large_df.groupby("key")["val_float"], agg)()
        pandas_booster.deactivate()
        pandas_result = getattr(large_df.groupby("key")["val_float"], agg)()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
    def test_multi_key_agg_matches_pandas(self, large_df, agg):
        import pandas_booster

        pandas_booster.activate()
        booster_result = getattr(large_df.groupby(["key", "key2"])["val_float"], agg)()
        pandas_booster.deactivate()
        pandas_result = getattr(large_df.groupby(["key", "key2"])["val_float"], agg)()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count"])
    def test_integer_values_match_pandas(self, large_df, agg):
        import pandas_booster

        pandas_booster.activate()
        booster_result = getattr(large_df.groupby("key")["val_int"], agg)()
        pandas_booster.deactivate()
        pandas_result = getattr(large_df.groupby("key")["val_int"], agg)()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        if agg in ("sum", "min", "max"):
            pandas_sorted = pandas_sorted.astype(float)

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=False,
            rtol=1e-10,
        )


class TestFallbackBehavior:
    def test_small_df_falls_back(self, small_df):
        import pandas_booster

        pandas_booster.activate()
        booster_result = small_df.groupby("key")["val_float"].sum()
        pandas_booster.deactivate()
        pandas_result = small_df.groupby("key")["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
        )

    def test_non_integer_key_falls_back(self, large_df):
        import pandas_booster

        large_df = large_df.copy()
        large_df["str_key"] = large_df["val_str"]

        pandas_booster.activate()
        booster_result = large_df.groupby("str_key")["val_float"].sum()
        pandas_booster.deactivate()
        pandas_result = large_df.groupby("str_key")["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
        )

    def test_non_numeric_value_falls_back(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        booster_result = large_df.groupby("key")["val_str"].count()
        pandas_booster.deactivate()
        pandas_result = large_df.groupby("key")["val_str"].count()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
        )

    def test_dropna_false_falls_back(self, large_df):
        import pandas_booster
        from pandas.core.groupby import DataFrameGroupBy

        pandas_booster.activate()
        gb = large_df.groupby("key", dropna=False)
        assert isinstance(gb, DataFrameGroupBy)

    def test_as_index_false_falls_back(self, large_df):
        import pandas_booster
        from pandas.core.groupby import DataFrameGroupBy

        pandas_booster.activate()
        gb = large_df.groupby("key", as_index=False)
        assert isinstance(gb, DataFrameGroupBy)


class TestNonAcceleratedMethodsFallback:
    def test_apply_works(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        result = large_df.groupby("key")["val_float"].apply(lambda x: x.mean())
        pandas_booster.deactivate()
        expected = large_df.groupby("key")["val_float"].apply(lambda x: x.mean())

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=False,
        )

    def test_transform_works(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        result = large_df.groupby("key")["val_float"].transform("mean")
        pandas_booster.deactivate()
        expected = large_df.groupby("key")["val_float"].transform("mean")

        pd.testing.assert_series_equal(result, expected, check_exact=False)

    def test_std_works(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        result = large_df.groupby("key")["val_float"].std()
        pandas_booster.deactivate()
        expected = large_df.groupby("key")["val_float"].std()

        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=False,
        )


class TestSortParameter:
    def test_sort_true_matches_pandas(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        booster_result = large_df.groupby("key", sort=True)["val_float"].sum()
        pandas_booster.deactivate()
        pandas_result = large_df.groupby("key", sort=True)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result,
            pandas_result,
            check_exact=False,
            rtol=1e-10,
        )

    def test_sort_false_same_values(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        booster_result = large_df.groupby("key", sort=False)["val_float"].sum()
        pandas_booster.deactivate()
        pandas_result = large_df.groupby("key", sort=False)["val_float"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )


class TestNaNHandling:
    def test_sum_with_nan_matches_pandas(self):
        import pandas_booster

        np.random.seed(123)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.1
        df = pd.DataFrame(
            {
                "key": np.repeat(np.arange(100), n // 100),
                "val": np.where(mask, np.nan, values),
            }
        )

        pandas_booster.activate()
        booster_result = df.groupby("key")["val"].sum()
        pandas_booster.deactivate()
        pandas_result = df.groupby("key")["val"].sum()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            rtol=1e-10,
        )

    def test_count_with_nan_matches_pandas(self):
        import pandas_booster

        np.random.seed(456)
        n = 200_000
        values = np.random.random(n)
        mask = np.random.random(n) < 0.2
        df = pd.DataFrame(
            {
                "key": np.repeat(np.arange(100), n // 100),
                "val": np.where(mask, np.nan, values),
            }
        )

        pandas_booster.activate()
        booster_result = df.groupby("key")["val"].count()
        pandas_booster.deactivate()
        pandas_result = df.groupby("key")["val"].count()

        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
        )


class TestEdgeCases:
    def test_multicolumn_selection_not_proxied(self, large_df):
        import pandas_booster
        from pandas.core.groupby import DataFrameGroupBy

        pandas_booster.activate()
        gb = large_df.groupby("key")[["val_float", "val_int"]]
        assert isinstance(gb, DataFrameGroupBy)

    def test_iteration_works(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        gb = large_df.groupby("key")
        groups = list(gb)
        assert len(groups) == large_df["key"].nunique()

    def test_ngroups_works(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        gb = large_df.groupby("key")
        assert gb.ngroups == large_df["key"].nunique()

    def test_groups_works(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        gb = large_df.groupby("key")
        assert len(gb.groups) == large_df["key"].nunique()
