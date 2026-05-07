import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_RUST_LIB_PATH = _REPO_ROOT / "src" / "lib.rs"
_RUST_STUB_PATH = _REPO_ROOT / "python" / "pandas_booster" / "_rust.pyi"


def _python_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _literal_strings(node: ast.AST) -> set[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.Subscript):
        return _literal_strings(node.slice)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {
        "frozenset",
        "set",
        "tuple",
        "list",
    }:
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


def test_ordered_single_key_float_prod_abi_marker_is_registered_and_stubbed():
    registered_exports = set(
        re.findall(
            r"m\.add_function\(wrap_pyfunction!\((\w+), m\)\?\)\?;",
            _RUST_LIB_PATH.read_text(),
        )
    )
    stubbed_exports = set(re.findall(r"^def (\w+)\(", _RUST_STUB_PATH.read_text(), re.MULTILINE))

    marker = "has_ordered_single_key_float_prod_abi"
    assert marker in registered_exports
    assert marker in stubbed_exports


def test_prod_surface_mentions_stay_in_sync():
    repo = _REPO_ROOT
    groupby_accel = repo / "python" / "pandas_booster" / "_groupby_accel.py"
    accessor = repo / "python" / "pandas_booster" / "accessor.py"
    proxy = repo / "python" / "pandas_booster" / "proxy.py"
    benchmark = repo / "benches" / "benchmark.py"

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
    assert "prod" in _assigned_strings(benchmark, "SUPPORTED_AGGS")
    assert "prod" in _dict_keys_assigned_in_function(benchmark, "build_polars_agg_expr", "agg_map")
    assert "prod" in _readme_operation_names()
    assert "prod" in _readme_force_float_groupby_aggs()


def test_median_surface_mentions_stay_in_sync():
    repo = _REPO_ROOT
    groupby_accel = repo / "python" / "pandas_booster" / "_groupby_accel.py"
    benchmark = repo / "benches" / "benchmark.py"

    assert "median" in _assigned_strings(groupby_accel, "AggFunc")
    assert any(
        {"sum", "mean", "prod", "std", "var", "median"} <= strings
        for strings in _literal_string_collections(groupby_accel)
    )
    assert "median" in _assigned_strings(benchmark, "SUPPORTED_AGGS")
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
