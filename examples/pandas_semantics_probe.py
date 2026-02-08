import numpy as np
import pandas as pd


def main() -> None:
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")

    print("\n--- Standard int64 (numpy-backed) ---")
    df = pd.DataFrame(
        {
            "g": ["a", "a", "b", "b"],
            "v": [2**62, 2**62, 1, 2],
            "v_small": [1, 2, 3, 4],
        }
    )

    print("Data types:")
    print(df.dtypes)

    print("\n[SUM]")
    g = df.groupby("g")
    res_sum = g["v"].sum()
    print("Result:\n", res_sum)
    print("Dtype:", res_sum.dtype)
    print("Value 'a':", res_sum["a"])
    print("Expected 2^63:", 2**63)
    if res_sum["a"] < 0:
        print("-> OVERFLOW DETECTED (Wrapped around)")
    elif res_sum["a"] == 2**63:
        print("-> EXACT MATCH (No overflow? Object?)")
    else:
        print(f"-> OTHER: {res_sum['a']}")

    print("\n[MEAN]")
    res_mean = g["v_small"].mean()
    print("Dtype:", res_mean.dtype)

    print("\n[COUNT]")
    res_count = g["v_small"].count()
    print("Dtype:", res_count.dtype)

    print("\n[MIN/MAX]")
    res_min = g["v_small"].min()
    print("Min Dtype:", res_min.dtype)
    res_max = g["v_small"].max()
    print("Max Dtype:", res_max.dtype)

    print("\n--- Nullable Int64 (pandas-backed) ---")
    df_null = pd.DataFrame(
        {
            "g": ["a", "a"],
            "v": pd.Series([2**62, 2**62], dtype="Int64"),
        }
    )
    print("Data types:")
    print(df_null.dtypes)

    res_sum_null = df_null.groupby("g")["v"].sum()
    print("\n[SUM Int64]")
    print("Result:\n", res_sum_null)
    print("Dtype:", res_sum_null.dtype)
    print("Value 'a':", res_sum_null["a"])

    print("\n--- Min Count Behavior ---")
    df_mc = pd.DataFrame({"g": ["a"], "v": [1]})
    res_mc = df_mc.groupby("g")["v"].sum(min_count=2)
    print("Result with min_count=2 (should be NaN):")
    print(res_mc)
    print("Dtype:", res_mc.dtype)

    print("\n--- NumPy Direct ---")
    arr = np.array([2**62, 2**62], dtype="int64")
    print(f"Array: {arr}")
    print(f"Sum: {arr.sum()}")
    print(f"Sum dtype: {arr.sum().dtype}")


if __name__ == "__main__":
    main()
