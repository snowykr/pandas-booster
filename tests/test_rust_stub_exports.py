import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUST_LIB_PATH = _REPO_ROOT / "src" / "lib.rs"
_RUST_STUB_PATH = _REPO_ROOT / "python" / "pandas_booster" / "_rust.pyi"


def test_rust_pyi_exports_match_pyo3_module_registrations():
    registered_exports = set(
        re.findall(
            r"m\.add_function\(wrap_pyfunction!\((\w+), m\)\?\)\?;",
            _RUST_LIB_PATH.read_text(),
        )
    )
    stubbed_exports = set(re.findall(r"^def (\w+)\(", _RUST_STUB_PATH.read_text(), re.MULTILINE))

    assert registered_exports == stubbed_exports


def _prod_expected_symbols() -> set[str]:
    prefixes = ("groupby", "groupby_multi")
    aggs = ("prod",)
    kernels = ("f64", "i64")
    suffixes = ("", "_sorted", "_firstseen_u32", "_firstseen_u64")
    return {
        f"{prefix}_{agg}_{kernel}{suffix}"
        for prefix in prefixes
        for agg in aggs
        for kernel in kernels
        for suffix in suffixes
    }


def test_prod_symbols_are_registered_and_stubbed():
    registered_exports = set(
        re.findall(
            r"m\.add_function\(wrap_pyfunction!\((\w+), m\)\?\)\?;",
            _RUST_LIB_PATH.read_text(),
        )
    )
    stubbed_exports = set(re.findall(r"^def (\w+)\(", _RUST_STUB_PATH.read_text(), re.MULTILINE))

    expected = _prod_expected_symbols()
    assert expected <= registered_exports
    assert expected <= stubbed_exports


def test_prod_surface_mentions_stay_in_sync():
    repo = _REPO_ROOT
    surface_expectations = {
        repo / "python" / "pandas_booster" / "_groupby_accel.py": [
            '"prod"',
            'and agg in {"sum", "mean", "prod", "std", "var", "median"}',
            'agg in {"sum", "prod", "min", "max"}',
        ],
        repo / "python" / "pandas_booster" / "accessor.py": [
            '"prod"',
            "_SUPPORTED_AGGS",
            "sum/mean/prod/std/var",
        ],
        repo / "python" / "pandas_booster" / "proxy.py": [
            '"prod"',
            "_ACCELERATED_AGGS",
            "def prod(",
        ],
        repo / "README.md": ["prod", "PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY"],
        repo / "benches" / "benchmark.py": [
            '"prod"',
            "build_polars_agg_expr",
            "agg: Literal[",
        ],
    }

    missing: list[str] = []
    for path, needles in surface_expectations.items():
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                missing.append(f"{path.relative_to(repo)} missing {needle!r}")
    assert not missing


def test_median_surface_mentions_stay_in_sync():
    repo = _REPO_ROOT
    surface_expectations = {
        repo / "README.md": [
            "| `median` | Median of values in each group |",
            "single-key float `sum`/`mean`/`prod`/`std`/`var`/`median`",
            "mean/std/var/median",
        ],
        repo / "benches" / "benchmark.py": [
            '"median"',
            "pl.col(value_col).median().alias(value_col)",
            "SUPPORTED_AGGS",
        ],
        repo / "python" / "pandas_booster" / "_groupby_accel.py": [
            'and agg in {"sum", "mean", "prod", "std", "var", "median"}',
        ],
    }

    missing: list[str] = []
    for path, needles in surface_expectations.items():
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                missing.append(f"{path.relative_to(repo)} missing {needle!r}")
    assert not missing


def test_prod_exact_dispatch_mapping_selects_expected_symbols():
    from pandas_booster._groupby_accel import select_rust_groupby_func

    class FakeRust:
        pass

    rust = FakeRust()
    expected = _prod_expected_symbols()
    calls: list[str] = []
    for symbol in expected:

        def _make(name: str):
            def _fn(*_args, **_kwargs):
                calls.append(name)

            _fn.__name__ = name
            return _fn

        setattr(rust, symbol, _make(symbol))

    cases = [
        ("groupby_prod_i64", True, False, "groupby_prod_i64_sorted", False),
        ("groupby_prod_i64", True, True, "groupby_prod_i64", True),
        ("groupby_prod_i64", False, False, "groupby_prod_i64_firstseen_u32", False),
        ("groupby_prod_i64", False, False, "groupby_prod_i64_firstseen_u64", False, 1 << 32),
        ("groupby_prod_f64", True, False, "groupby_prod_f64_sorted", False),
        ("groupby_multi_prod_i64", True, False, "groupby_multi_prod_i64_sorted", False),
        ("groupby_multi_prod_f64", False, False, "groupby_multi_prod_f64_firstseen_u32", False),
        (
            "groupby_multi_prod_f64",
            False,
            False,
            "groupby_multi_prod_f64_firstseen_u64",
            False,
            1 << 32,
        ),
    ]

    for case in cases:
        func_base, sort, force_pandas_sort, expected_name, expected_python_sort, *maybe_n = case
        n_rows = maybe_n[0] if maybe_n else 100_000
        func, needs_python_sort = select_rust_groupby_func(
            rust,
            func_base,
            sort=sort,
            n_rows=n_rows,
            force_pandas_sort=force_pandas_sort,
        )
        assert func.__name__ == expected_name
        assert needs_python_sort is expected_python_sort
        assert "sum" not in func.__name__


def _median_expected_symbols() -> set[str]:
    prefixes = ("groupby", "groupby_multi")
    agg = "median"
    kernels = ("f64", "i64")
    suffixes = ("", "_sorted", "_firstseen_u32", "_firstseen_u64")
    return {
        f"{prefix}_{agg}_{kernel}{suffix}"
        for prefix in prefixes
        for kernel in kernels
        for suffix in suffixes
    }


def test_median_symbols_are_registered_and_stubbed():
    registered_exports = set(
        re.findall(
            r"m\.add_function\(wrap_pyfunction!\((\w+), m\)\?\)\?;",
            _RUST_LIB_PATH.read_text(),
        )
    )
    stubbed_exports = set(re.findall(r"^def (\w+)\(", _RUST_STUB_PATH.read_text(), re.MULTILINE))

    expected = _median_expected_symbols()
    assert expected <= registered_exports
    assert expected <= stubbed_exports


def test_median_exact_dispatch_mapping_selects_expected_symbols():
    from pandas_booster._groupby_accel import select_rust_groupby_func

    class FakeRust:
        pass

    rust = FakeRust()
    for symbol in _median_expected_symbols():

        def _make(name: str):
            def _fn(*_args, **_kwargs):
                return name

            _fn.__name__ = name
            return _fn

        setattr(rust, symbol, _make(symbol))

    cases = [
        ("groupby_median_f64", True, False, "groupby_median_f64_sorted", False),
        ("groupby_median_f64", True, True, "groupby_median_f64", True),
        ("groupby_median_f64", False, False, "groupby_median_f64_firstseen_u32", False),
        ("groupby_median_f64", False, False, "groupby_median_f64_firstseen_u64", False, 1 << 32),
        ("groupby_median_i64", True, False, "groupby_median_i64_sorted", False),
        ("groupby_median_i64", True, True, "groupby_median_i64", True),
        ("groupby_median_i64", False, False, "groupby_median_i64_firstseen_u32", False),
        ("groupby_median_i64", False, False, "groupby_median_i64_firstseen_u64", False, 1 << 32),
        ("groupby_multi_median_f64", True, False, "groupby_multi_median_f64_sorted", False),
        ("groupby_multi_median_f64", True, True, "groupby_multi_median_f64", True),
        ("groupby_multi_median_f64", False, False, "groupby_multi_median_f64_firstseen_u32", False),
        (
            "groupby_multi_median_f64",
            False,
            False,
            "groupby_multi_median_f64_firstseen_u64",
            False,
            1 << 32,
        ),
        ("groupby_multi_median_i64", True, False, "groupby_multi_median_i64_sorted", False),
        ("groupby_multi_median_i64", True, True, "groupby_multi_median_i64", True),
        ("groupby_multi_median_i64", False, False, "groupby_multi_median_i64_firstseen_u32", False),
        (
            "groupby_multi_median_i64",
            False,
            False,
            "groupby_multi_median_i64_firstseen_u64",
            False,
            1 << 32,
        ),
    ]

    for case in cases:
        func_base, sort, force_pandas_sort, expected_name, expected_python_sort, *maybe_n = case
        n_rows = maybe_n[0] if maybe_n else 100_000
        func, needs_python_sort = select_rust_groupby_func(
            rust,
            func_base,
            sort=sort,
            n_rows=n_rows,
            force_pandas_sort=force_pandas_sort,
        )
        assert func.__name__ == expected_name
        assert needs_python_sort is expected_python_sort
