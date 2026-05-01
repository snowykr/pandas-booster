import pandas as pd
from pandas_booster._groupby_accel import classify_groupby_compatibility


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
