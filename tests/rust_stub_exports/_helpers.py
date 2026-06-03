import ast
import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUST_LIB_PATH = _REPO_ROOT / "src" / "lib.rs"
_RUST_REGISTRATION_PATH = _REPO_ROOT / "src" / "python_wrappers" / "register.rs"
_RUST_SINGLE_WRAPPERS_PATH = _REPO_ROOT / "src" / "python_wrappers" / "single" / "abi.rs"
_RUST_STUB_PATH = _REPO_ROOT / "python" / "pandas_booster" / "_rust.pyi"
_GROUPBY_ACCEL_PATH = _REPO_ROOT / "python" / "pandas_booster" / "_groupby_accel.py"
_BENCHMARK_DISPATCH_PATH = _REPO_ROOT / "benchmarks" / "dispatch.py"
_BENCHMARK_REPORTING_PATH = _REPO_ROOT / "benchmarks" / "reporting.py"
_BENCHMARK_REPORTING_CONSTANTS_PATH = _REPO_ROOT / "benchmarks" / "reporting_constants.py"
_ALL_GROUPBY_AGGS = ("sum", "prod", "mean", "median", "var", "std", "min", "max", "count")

__all__ = [
    "_ALL_GROUPBY_AGGS",
    "_BENCHMARK_DISPATCH_PATH",
    "_BENCHMARK_REPORTING_CONSTANTS_PATH",
    "_BENCHMARK_REPORTING_PATH",
    "_GROUPBY_ACCEL_PATH",
    "_REPO_ROOT",
    "_RUST_LIB_PATH",
    "_RUST_REGISTRATION_PATH",
    "_RUST_SINGLE_WRAPPERS_PATH",
    "_RUST_STUB_PATH",
    "_assert_expected_groupby_exports_are_registered_and_stubbed",
    "_assigned_strings",
    "_class_assigned_strings",
    "_dict_keys_assigned_in_function",
    "_expected_exports",
    "_expected_groupby_export_matrix",
    "_expected_profile_exports",
    "_expected_support_exports",
    "_function_names",
    "_function_references_name",
    "_literal_string_collections",
    "_literal_strings",
    "_missing_exports",
    "_python_module",
    "_readme_force_float_groupby_aggs",
    "_readme_operation_names",
    "_registered_exports",
    "_registered_exports_from_source",
    "_strip_rust_comments",
    "_stubbed_exports",
    "re",
]


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
    return _registered_exports_from_source(_RUST_REGISTRATION_PATH.read_text())


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
