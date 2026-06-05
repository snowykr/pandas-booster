from ._helpers import (
    _BENCHMARK_DISPATCH_PATH,
    _BENCHMARK_REPORTING_CONSTANTS_PATH,
    _GROUPBY_ACCEL_PATH,
    _REPO_ROOT,
    _RUST_SINGLE_WRAPPERS_PATH,
    _RUST_STUB_PATH,
    _assigned_strings,
    _class_assigned_strings,
    _dict_keys_assigned_in_function,
    _function_names,
    _function_references_name,
    _literal_string_collections,
    _readme_force_float_groupby_aggs,
    _readme_operation_names,
    _registered_exports,
    re,
)


def test_ordered_single_key_float_prod_abi_source_contract():
    marker = "has_ordered_single_key_float_prod_abi"
    rust_source = _RUST_SINGLE_WRAPPERS_PATH.read_text()

    assert marker in _registered_exports()
    assert re.search(rf"(?m)^def {marker}\(\) -> bool: \.\.\.$", _RUST_STUB_PATH.read_text())
    assert re.search(rf"(?ms)^pub\(crate\) fn {marker}\(\) -> bool \{{\s*true\s*\}}", rust_source)
    assert _assigned_strings(_GROUPBY_ACCEL_PATH, "ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER") == {
        marker
    }
    assert _function_references_name(
        _GROUPBY_ACCEL_PATH,
        "select_rust_groupby_func",
        "ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER",
    )
    assert _function_references_name(
        _GROUPBY_ACCEL_PATH,
        "has_rust_groupby_func",
        "ORDERED_SINGLE_KEY_FLOAT_PROD_ABI_MARKER",
    )


def test_prod_surface_mentions_stay_in_sync():
    repo = _REPO_ROOT
    groupby_accel = _GROUPBY_ACCEL_PATH
    accessor = repo / "python" / "pandas_booster" / "accessor.py"
    proxy = repo / "python" / "pandas_booster" / "proxy.py"
    benchmark = _BENCHMARK_DISPATCH_PATH
    reporting_constants = _BENCHMARK_REPORTING_CONSTANTS_PATH

    assert "prod" in _assigned_strings(groupby_accel, "AggFunc")
    assert any(
        {"sum", "mean", "std", "var", "median"} <= strings
        for strings in _literal_string_collections(groupby_accel)
    )
    assert any(
        {"sum", "prod", "min", "max"} <= strings
        for strings in _literal_string_collections(groupby_accel)
    )
    assert "prod" in _class_assigned_strings(accessor, "BoosterAccessor", "_SUPPORTED_AGGS")
    assert "prod" in _assigned_strings(proxy, "_ACCELERATED_AGGS")
    assert "prod" in _function_names(proxy)
    assert "prod" in _assigned_strings(reporting_constants, "SUPPORTED_AGGS")
    assert "prod" in _dict_keys_assigned_in_function(benchmark, "build_polars_agg_expr", "agg_map")
    assert "prod" in _readme_operation_names()
    assert "prod" in _readme_force_float_groupby_aggs()


def test_median_surface_mentions_stay_in_sync():
    groupby_accel = _GROUPBY_ACCEL_PATH
    benchmark = _BENCHMARK_DISPATCH_PATH
    reporting_constants = _BENCHMARK_REPORTING_CONSTANTS_PATH

    assert "median" in _assigned_strings(groupby_accel, "AggFunc")
    assert any(
        {"sum", "mean", "prod", "std", "var", "median"} <= strings
        for strings in _literal_string_collections(groupby_accel)
    )
    assert "median" in _assigned_strings(reporting_constants, "SUPPORTED_AGGS")
    assert "median" in _dict_keys_assigned_in_function(
        benchmark, "build_polars_agg_expr", "agg_map"
    )
    assert "median" in _readme_operation_names()
    assert "median" in _readme_force_float_groupby_aggs()


def test_rust_module_docs_mention_supported_groupby_aggs():
    expected_terms = {
        "sum": ("sum",),
        "prod": ("prod", "product"),
        "mean": ("mean",),
        "median": ("median",),
        "std": ("std", "standard deviation"),
        "var": ("var", "variance"),
        "min": ("min",),
        "max": ("max",),
        "count": ("count",),
    }

    for path in (_REPO_ROOT / "src" / "lib.rs", _REPO_ROOT / "src" / "aggregation.rs"):
        text = path.read_text().lower()
        for token, terms in expected_terms.items():
            assert any(term in text for term in terms), f"{path} missing {token!r}"
