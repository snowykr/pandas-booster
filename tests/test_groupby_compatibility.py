import pandas as pd
import pytest
from pandas_booster._groupby_accel import classify_groupby_compatibility
from pandas_booster._groupby_policy import should_fallback_groupby


def test_groupby_compatibility_result_supports_named_and_tuple_access() -> None:
    df = pd.DataFrame({"key": [1, 2], "val": [1.0, 2.0]})

    compatibility = classify_groupby_compatibility(
        key_cols=[df["key"]],
        val_col=df["val"],
        agg="std",
        force_pandas_float_groupby=True,
    )

    assert compatibility.supported is True
    assert compatibility.force_pandas is True
    assert compatibility[0] is True
    assert compatibility[1] is True
    supported, force_pandas = compatibility
    assert supported is True
    assert force_pandas is True


def test_single_key_float_prod_is_rust_eligible_by_default() -> None:
    df = pd.DataFrame({"key": [1, 1], "val": [0.5, 2.0]})

    compatibility = classify_groupby_compatibility(
        key_cols=[df["key"]],
        val_col=df["val"],
        agg="prod",
        force_pandas_float_groupby=False,
    )

    assert compatibility.supported is True
    assert compatibility.force_pandas is False


def test_single_key_float_prod_force_pandas_escape_hatch_still_applies() -> None:
    df = pd.DataFrame({"key": [1, 1], "val": [0.5, 2.0]})

    compatibility = classify_groupby_compatibility(
        key_cols=[df["key"]],
        val_col=df["val"],
        agg="prod",
        force_pandas_float_groupby=True,
    )

    assert compatibility.supported is True
    assert compatibility.force_pandas is True


def test_multi_key_float_prod_remains_rust_eligible() -> None:
    df = pd.DataFrame({"k1": [1, 1], "k2": [1, 2], "val": [0.5, 2.0]})

    compatibility = classify_groupby_compatibility(
        key_cols=[df["k1"], df["k2"]],
        val_col=df["val"],
        agg="prod",
        force_pandas_float_groupby=False,
    )

    assert compatibility.supported is True
    assert compatibility.force_pandas is False


def test_multi_key_over_rust_limit_is_not_rust_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame({f"k{i}": [1, 2, 1, 2] for i in range(11)})
    df["val"] = [0.5, 2.0, 4.0, 8.0]

    monkeypatch.setattr(
        "pandas_booster._groupby_policy._has_median_kernel",
        lambda *args, **kwargs: True,
    )

    assert (
        should_fallback_groupby(
            df,
            [df[f"k{i}"] for i in range(11)],
            df["val"],
            "median",
            context="proxy",
            multi=True,
            sort=True,
        )
        is True
    )


def test_multi_key_at_rust_limit_remains_rust_eligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    df = pd.DataFrame({f"k{i}": [1, 2, 1, 2] for i in range(10)})
    df["val"] = [0.5, 2.0, 4.0, 8.0]

    monkeypatch.setattr(
        "pandas_booster._groupby_policy._has_median_kernel",
        lambda *args, **kwargs: True,
    )

    assert (
        should_fallback_groupby(
            df,
            [df[f"k{i}"] for i in range(10)],
            df["val"],
            "median",
            context="proxy",
            multi=True,
            sort=True,
        )
        is False
    )
