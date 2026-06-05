from ._helpers import _expected_groupby_export_matrix


def test_prod_exact_dispatch_mapping_selects_expected_symbols():
    from pandas_booster._groupby_accel import select_rust_groupby_func

    class FakeRust:
        @staticmethod
        def has_ordered_single_key_float_prod_abi() -> bool:
            return True

    rust = FakeRust()
    expected = _expected_groupby_export_matrix(("prod",))
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


def test_single_key_float_prod_requires_ordered_abi_marker():
    from pandas_booster._groupby_accel import has_rust_groupby_func, select_rust_groupby_func

    class FakeRust:
        @staticmethod
        def groupby_prod_f64_sorted(*_args, **_kwargs):
            raise AssertionError("stale groupby_prod_f64 symbol must not be selected")

        @staticmethod
        def groupby_prod_f64_firstseen_u32(*_args, **_kwargs):
            raise AssertionError("stale groupby_prod_f64 symbol must not be selected")

    rust = FakeRust()

    assert (
        has_rust_groupby_func(
            rust,
            "groupby_prod_f64",
            sort=True,
            n_rows=100_000,
            force_pandas_sort=False,
        )
        is False
    )

    try:
        select_rust_groupby_func(
            rust,
            "groupby_prod_f64",
            sort=True,
            n_rows=100_000,
            force_pandas_sort=False,
        )
    except AttributeError as exc:
        assert exc.args == ("has_ordered_single_key_float_prod_abi",)
    else:
        raise AssertionError("missing ordered prod ABI marker should block symbol resolution")


def test_median_exact_dispatch_mapping_selects_expected_symbols():
    from pandas_booster._groupby_accel import select_rust_groupby_func

    class FakeRust:
        pass

    rust = FakeRust()
    for symbol in _expected_groupby_export_matrix(("median",)):

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
