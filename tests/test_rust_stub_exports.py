import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUST_LIB_PATH = _REPO_ROOT / "src" / "lib.rs"
_RUST_STUB_PATH = _REPO_ROOT / "python" / "pandas_booster" / "_rust.pyi"
_GROUPBY_ACCEL_PATH = _REPO_ROOT / "python" / "pandas_booster" / "_groupby_accel.py"
_BENCHMARK_REPORTING_PATH = _REPO_ROOT / "benchmarks" / "reporting.py"
_BENCHMARK_REPORTING_CONSTANTS_PATH = _REPO_ROOT / "benchmarks" / "reporting_constants.py"
_ALL_GROUPBY_AGGS = ("sum", "prod", "mean", "median", "var", "std", "min", "max", "count")


def _python_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _literal_strings(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Subscript):
        return _literal_strings(node.slice)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {
            "frozenset",
            "set",
            "tuple",
            "list",
        }
    ):
        return set().union(*(_literal_strings(arg) for arg in node.args))
    if isinstance(node, ast.Dict):
        return set().union(*(_literal_strings(key) for key in node.keys if key is not None))
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return set().union(*(_literal_strings(elt) for elt in node.elts))
    return set()


def _assigned_strings(path: Path, name: str) -> set[str]:
    for node in ast.walk(_python_module(path)):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return _literal_strings(node.value)
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return _literal_strings(node.value) if node.value is not None else set()
    return set()


def _class_assigned_strings(path: Path, class_name: str, attr_name: str) -> set[str]:
    for node in ast.walk(_python_module(path)):
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == attr_name:
                        return _literal_strings(stmt.value)
            if (
                isinstance(stmt, ast.AnnAssign)
                and isinstance(stmt.target, ast.Name)
                and stmt.target.id == attr_name
            ):
                return _literal_strings(stmt.value) if stmt.value is not None else set()
    return set()


def _function_names(path: Path) -> set[str]:
    return {
        node.name for node in ast.walk(_python_module(path)) if isinstance(node, ast.FunctionDef)
    }


def _dict_keys_assigned_in_function(path: Path, function_name: str, dict_name: str) -> set[str]:
    for node in ast.walk(_python_module(path)):
        if not isinstance(node, ast.FunctionDef) or node.name != function_name:
            continue
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == dict_name:
                        return _literal_strings(stmt.value)
    return set()


def _literal_string_collections(path: Path) -> list[set[str]]:
    collections = []
    for node in ast.walk(_python_module(path)):
        if isinstance(node, (ast.List, ast.Tuple, ast.Set, ast.Dict)):
            strings = _literal_strings(node)
            if strings:
                collections.append(strings)
    return collections


def _readme_operation_names() -> set[str]:
    return set(re.findall(r"(?m)^\|\s*`([^`]+)`\s*\|", (_REPO_ROOT / "README.md").read_text()))


def _readme_force_float_groupby_aggs() -> set[str]:
    text = (_REPO_ROOT / "README.md").read_text()
    match = re.search(
        r"(?ms)^- `PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY`.*?(?=^\s*$)",
        text,
    )
    assert match is not None
    inline_code = re.findall(r"`([^`]+)`", match.group(0))
    return {
        token
        for phrase in inline_code
        for token in phrase.split("/")
        if token in {"sum", "mean", "prod", "std", "var", "median"}
    }


def _strip_rust_comments(source: str) -> str:
    without_block_comments = re.sub(r"(?s)/\*.*?\*/", "", source)
    return re.sub(r"(?m)//.*$", "", without_block_comments)


def _registered_exports_from_source(source: str) -> set[str]:
    registered_exports: set[str] = set()
    stripped_source = _strip_rust_comments(source)
    for match in re.finditer(
        r"add_pyfunctions!\(\s*m\s*,(?P<body>.*?)\)\s*;", stripped_source, re.DOTALL
    ):
        body = match.group("body")
        registered_exports.update(re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", body))
    return registered_exports


def _registered_exports() -> set[str]:
    return _registered_exports_from_source(_RUST_LIB_PATH.read_text())


def _stubbed_exports() -> set[str]:
    return set(re.findall(r"^def (\w+)\(", _RUST_STUB_PATH.read_text(), re.MULTILINE))


def _expected_groupby_export_matrix(aggs: tuple[str, ...] = _ALL_GROUPBY_AGGS) -> set[str]:
    prefixes = ("groupby", "groupby_multi")
    kernels = ("f64", "i64")
    suffixes = ("", "_sorted", "_firstseen_u32", "_firstseen_u64")
    return {
        f"{prefix}_{agg}_{kernel}{suffix}"
        for prefix in prefixes
        for agg in aggs
        for kernel in kernels
        for suffix in suffixes
    }


def _expected_profile_exports() -> set[str]:
    return {
        f"profile_groupby_{agg}_f64{suffix}"
        for agg in ("var", "std")
        for suffix in ("_sorted", "_firstseen_u32", "_firstseen_u64")
    }


def _expected_support_exports() -> set[str]:
    return {
        "get_fallback_threshold",
        "get_thread_count",
        "has_ordered_single_key_float_prod_abi",
    }


def _expected_exports() -> set[str]:
    return (
        _expected_groupby_export_matrix()
        | _expected_profile_exports()
        | _expected_support_exports()
    )


def _function_references_name(path: Path, function_name: str, name: str) -> bool:
    for node in ast.walk(_python_module(path)):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))
    return False


def _assert_expected_groupby_exports_are_registered_and_stubbed(aggs: tuple[str, ...]) -> None:
    expected = _expected_groupby_export_matrix(aggs)
    assert expected <= _registered_exports()
    assert expected <= _stubbed_exports()


def _missing_exports(
    expected: set[str], registered: set[str], stubbed: set[str]
) -> tuple[set[str], set[str]]:
    return expected - registered, expected - stubbed


def test_rust_pyi_exports_match_expected_pyo3_surface():
    expected_exports = _expected_exports()
    registered_exports = _registered_exports()
    stubbed_exports = _stubbed_exports()
    missing_registered, missing_stubbed = _missing_exports(
        expected_exports,
        registered_exports,
        stubbed_exports,
    )

    assert registered_exports, "src/lib.rs must register exports through add_pyfunctions!(m, ...)"
    assert not missing_registered
    assert not missing_stubbed
    assert registered_exports == expected_exports
    assert stubbed_exports == expected_exports


def test_export_matrix_detection_flags_missing_macro_registration():
    expected_exports = _expected_exports()
    registered_exports = _registered_exports_from_source(
        "add_pyfunctions!(m, groupby_sum_f64, groupby_sum_i64);"
    )
    stubbed_exports = expected_exports

    missing_registered, missing_stubbed = _missing_exports(
        expected_exports,
        registered_exports,
        stubbed_exports,
    )

    assert "groupby_prod_f64" in missing_registered
    assert not missing_stubbed


def test_export_matrix_detection_flags_missing_stub_entry():
    expected_exports = _expected_exports()
    registered_exports = expected_exports
    stubbed_exports = expected_exports - {"groupby_prod_f64"}

    missing_registered, missing_stubbed = _missing_exports(
        expected_exports,
        registered_exports,
        stubbed_exports,
    )

    assert not missing_registered
    assert missing_stubbed == {"groupby_prod_f64"}


def test_prod_symbols_are_registered_and_stubbed():
    _assert_expected_groupby_exports_are_registered_and_stubbed(("prod",))


def test_ordered_single_key_float_prod_abi_source_contract():
    marker = "has_ordered_single_key_float_prod_abi"
    rust_source = _RUST_LIB_PATH.read_text()

    assert marker in _registered_exports()
    assert re.search(rf"(?m)^def {marker}\(\) -> bool: \.\.\.$", _RUST_STUB_PATH.read_text())
    assert re.search(rf"(?ms)^fn {marker}\(\) -> bool \{{\s*true\s*\}}", rust_source)
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
    benchmark = repo / "benchmarks" / "benchmark.py"
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
    repo = _REPO_ROOT
    groupby_accel = _GROUPBY_ACCEL_PATH
    benchmark = repo / "benchmarks" / "benchmark.py"
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


def test_median_symbols_are_registered_and_stubbed():
    _assert_expected_groupby_exports_are_registered_and_stubbed(("median",))


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


def test_pr12_regression_anchor_targets_still_exist():
    anchor_targets = {
        "discussion_r3196780027": (
            _REPO_ROOT / "tests" / "test_rust_stub_exports.py",
            "test_prod_exact_dispatch_mapping_selects_expected_symbols",
        ),
        "discussion_r3196782200": (
            _REPO_ROOT / "tests" / "test_rust_stub_exports.py",
            "test_rust_pyi_exports_match_expected_pyo3_surface",
        ),
        "discussion_r3197179732": (
            _REPO_ROOT / "src" / "groupby.rs",
            "test_groupby_prod_f64_preserves_row_order_ieee_semantics",
        ),
        "discussion_r3197709485": (
            _REPO_ROOT / "tests" / "test_edge_cases.py",
            "test_single_key_float_median_large_even_middle_values_match_pandas",
        ),
    }

    for anchor, (path, needle) in anchor_targets.items():
        assert needle in path.read_text(), (
            f"{anchor} lost its characterization target {needle!r} in {path}"
        )


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
