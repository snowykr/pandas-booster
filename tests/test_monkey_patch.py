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
    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
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

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
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

    @pytest.mark.parametrize("agg", ["sum", "mean", "min", "max", "count", "prod"])
    def test_integer_values_match_pandas(self, large_df, agg):
        import pandas_booster

        pandas_booster.activate()
        booster_result = getattr(large_df.groupby("key")["val_int"], agg)()
        pandas_booster.deactivate()
        pandas_result = getattr(large_df.groupby("key")["val_int"], agg)()

        booster_sorted = booster_result.sort_index()
        pandas_sorted = pandas_result.sort_index()

        pd.testing.assert_series_equal(
            booster_sorted,
            pandas_sorted,
            check_exact=(agg != "mean"),
            check_dtype=True,
            rtol=(1e-10 if agg == "mean" else 0.0),
        )

    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_single_key_median_matches_pandas(self, large_df, target):
        import pandas_booster

        pandas_booster.activate()
        booster_result = large_df.groupby("key")[target].median()
        pandas_booster.deactivate()
        pandas_result = large_df.groupby("key")[target].median()

        assert booster_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            check_dtype=True,
            rtol=1e-10,
        )

    @pytest.mark.parametrize("target", ["val_float", "val_int"])
    def test_multi_key_median_matches_pandas(self, large_df, target):
        import pandas_booster

        pandas_booster.activate()
        booster_result = large_df.groupby(["key", "key2"])[target].median()
        pandas_booster.deactivate()
        pandas_result = large_df.groupby(["key", "key2"])[target].median()

        assert booster_result.dtype == np.dtype(np.float64)
        pd.testing.assert_series_equal(
            booster_result.sort_index(),
            pandas_result.sort_index(),
            check_exact=False,
            check_dtype=True,
            rtol=1e-10,
        )


class TestProxyMedianCallParity:
    def test_median_kwargs_delegate_to_pandas_without_acceleration(
        self, large_df, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        from pandas.core.groupby.generic import SeriesGroupBy
        from pandas_booster.proxy import BoosterSeriesGroupBy

        expected = large_df.groupby("key")["val_float"].median(numeric_only=False)
        calls: list[dict[str, object]] = []
        original_median = SeriesGroupBy.median

        def wrapped(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": dict(kwargs)})
            return original_median(self, *args, **kwargs)

        def _boom(*_args, **_kwargs):
            raise AssertionError("median kwargs must pass through to pandas")

        monkeypatch.setattr(SeriesGroupBy, "median", wrapped, raising=True)
        monkeypatch.setattr(BoosterSeriesGroupBy, "_try_accelerate", _boom, raising=True)

        pandas_booster.activate()
        try:
            result = large_df.groupby("key")["val_float"].median(numeric_only=False)
        finally:
            pandas_booster.deactivate()

        assert calls == [{"args": (), "kwargs": {"numeric_only": False}}]
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
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

    def test_median_works(self, large_df, monkeypatch: pytest.MonkeyPatch):
        import pandas_booster
        from pandas_booster.proxy import BoosterSeriesGroupBy

        expected = large_df.groupby("key")["val_float"].median()
        calls: list[str] = []

        def fake_try_accelerate(self, agg):
            calls.append(agg)
            return expected

        monkeypatch.setattr(
            BoosterSeriesGroupBy, "_try_accelerate", fake_try_accelerate, raising=True
        )

        pandas_booster.activate()
        try:
            result = large_df.groupby("key")["val_float"].median()
        finally:
            pandas_booster.deactivate()

        assert calls == ["median"]
        pd.testing.assert_series_equal(
            result.sort_index(),
            expected.sort_index(),
            check_exact=False,
            rtol=1e-10,
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
    def test_force_pandas_float_groupby_env_applies_to_proxy_single_key_sum(
        self, large_df, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        import pandas_booster._rust as rust

        monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY", "1")

        def _boom(*_args, **_kwargs):
            raise AssertionError("Rust float sum kernel should not be called when env is enabled")

        monkeypatch.setattr(rust, "groupby_sum_f64", _boom)
        monkeypatch.setattr(rust, "groupby_sum_f64_sorted", _boom)

        pandas_result = large_df.groupby("key", sort=True)["val_float"].sum()

        pandas_booster.activate()
        proxy_result = large_df.groupby("key", sort=True)["val_float"].sum()

        pd.testing.assert_series_equal(proxy_result.sort_index(), pandas_result.sort_index())

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


class TestProdProxy:
    def test_prod_no_arg_single_and_multi_key_match_pandas(self, large_df):
        import pandas_booster

        pandas_booster.activate()
        single_result = large_df.groupby("key")["val_float"].prod()
        multi_result = large_df.groupby(["key", "key2"], sort=False)["val_int"].prod()
        pandas_booster.deactivate()

        single_expected = large_df.groupby("key")["val_float"].prod()
        multi_expected = large_df.groupby(["key", "key2"], sort=False)["val_int"].prod()

        pd.testing.assert_series_equal(
            single_result.sort_index(), single_expected.sort_index(), check_exact=False, rtol=1e-10
        )
        pd.testing.assert_series_equal(multi_result, multi_expected, check_exact=True)

    def test_prod_no_arg_uses_accelerated_proxy_when_eligible(
        self, large_df, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        import pandas_booster._rust as rust

        expected = large_df.groupby("key", sort=True)["val_float"].prod()
        calls: list[str] = []

        def fake_sorted(_keys, _values):
            calls.append("prod")
            return expected.index.to_numpy(dtype=np.int64), expected.to_numpy(dtype=np.float64)

        monkeypatch.setattr(rust, "groupby_prod_f64_sorted", fake_sorted, raising=False)

        pandas_booster.activate()
        try:
            result = large_df.groupby("key", sort=True)["val_float"].prod()
        finally:
            pandas_booster.deactivate()

        assert calls == ["prod"]
        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_prod_no_arg_uses_accelerated_proxy_for_multi_key_int_sort_true(
        self, large_df, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        import pandas_booster._rust as rust

        expected = large_df.groupby(["key", "key2"], sort=True)["val_int"].prod()
        calls: list[str] = []

        def fake_sorted(_key_arrays, _values):
            calls.append("prod")
            return [
                expected.index.get_level_values(0).to_numpy(dtype=np.int64),
                expected.index.get_level_values(1).to_numpy(dtype=np.int64),
            ], expected.to_numpy(dtype=np.int64)

        monkeypatch.setattr(rust, "groupby_multi_prod_i64_sorted", fake_sorted, raising=False)

        pandas_booster.activate()
        try:
            result = large_df.groupby(["key", "key2"], sort=True)["val_int"].prod()
        finally:
            pandas_booster.deactivate()

        assert calls == ["prod"]
        pd.testing.assert_series_equal(result, expected, check_exact=True)

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            pytest.param({"min_count": 2}, id="min_count"),
            pytest.param({"numeric_only": False}, id="numeric_only"),
        ],
    )
    def test_prod_kwargs_delegate_to_pandas(
        self, large_df, monkeypatch: pytest.MonkeyPatch, kwargs
    ):
        import pandas_booster
        import pandas_booster._rust as rust

        def _boom(*_args, **_kwargs):
            raise AssertionError("prod with kwargs must delegate to pandas")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

        pandas_booster.activate()
        try:
            result = large_df.groupby("key")["val_float"].prod(**kwargs)
        finally:
            pandas_booster.deactivate()
        expected = large_df.groupby("key")["val_float"].prod(**kwargs)

        pd.testing.assert_series_equal(
            result.sort_index(), expected.sort_index(), check_exact=False, rtol=1e-10
        )

    def test_prod_invalid_positional_arg_delegates_and_preserves_type_error(
        self, large_df, monkeypatch: pytest.MonkeyPatch
    ):
        import pandas_booster
        import pandas_booster._rust as rust

        def _boom(*_args, **_kwargs):
            raise AssertionError("prod with positional args must delegate to pandas")

        for suffix in ("", "_sorted", "_firstseen_u32", "_firstseen_u64"):
            monkeypatch.setattr(rust, f"groupby_prod_f64{suffix}", _boom, raising=False)

        pandas_booster.activate()
        try:
            with pytest.raises(TypeError):
                large_df.groupby("key")["val_float"].prod(1, 2, 3)
        finally:
            pandas_booster.deactivate()
