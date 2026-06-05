from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("python", "tests", "scripts", "benchmarks")
_TYPE_ALIAS_BUILTINS = {
    "bool",
    "bytes",
    "dict",
    "float",
    "frozenset",
    "int",
    "list",
    "None",
    "set",
    "str",
    "tuple",
}


def _has_future_annotations(module: ast.Module) -> bool:
    for node in module.body:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
            continue
        break
    return False


def _annotation_nodes(module: ast.Module):
    for node in ast.walk(module):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if arg.annotation is not None:
                    yield arg.annotation
            if node.args.vararg is not None and node.args.vararg.annotation is not None:
                yield node.args.vararg.annotation
            if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.annotation
            if node.returns is not None:
                yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _contains_pep604_union(annotation: ast.AST) -> bool:
    return any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr)
        for node in ast.walk(annotation)
    )


def _is_type_alias_name(name: str) -> bool:
    return name[:1].isupper() and not name.isupper()


def _is_type_union_operand(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant):
        return node.value is None
    if isinstance(node, ast.Name):
        return node.id in _TYPE_ALIAS_BUILTINS or _is_type_alias_name(node.id)
    if isinstance(node, ast.Subscript):
        return _is_type_union_operand(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _is_runtime_type_union(node)
    return False


def _is_runtime_type_union(node: ast.AST) -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.BitOr):
        return False
    return _is_type_union_operand(node.left) and _is_type_union_operand(node.right)


def _runtime_type_alias_name(stmt: ast.stmt) -> str | None:
    if isinstance(stmt, ast.Assign) and _is_runtime_type_union(stmt.value):
        for target in stmt.targets:
            if isinstance(target, ast.Name) and _is_type_alias_name(target.id):
                return target.id
    if (
        isinstance(stmt, ast.AnnAssign)
        and isinstance(stmt.target, ast.Name)
        and stmt.value is not None
        and _is_type_alias_name(stmt.target.id)
        and _is_runtime_type_union(stmt.value)
    ):
        return stmt.target.id
    return None


def test_pep604_annotations_are_postponed_for_python39_runtime() -> None:
    """Python 3.9 evaluates annotations unless modules opt into postponed annotations."""
    offenders: list[str] = []

    for dirname in _SCANNED_DIRS:
        for path in sorted((_REPO_ROOT / dirname).rglob("*.py")):
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if _has_future_annotations(module):
                continue
            if any(_contains_pep604_union(annotation) for annotation in _annotation_nodes(module)):
                offenders.append(str(path.relative_to(_REPO_ROOT)))

    assert offenders == []


def test_pep604_runtime_detection_ignores_bitwise_values() -> None:
    module = ast.parse(
        "\n".join(
            [
                "Flags = os.O_WRONLY | os.O_CREAT",
                "JsonScalar = None | bool | int",
                "JsonValue: TypeAlias = JsonScalar | list['JsonValue']",
            ]
        )
    )

    offenders = [
        type_alias_name
        for stmt in module.body
        if (type_alias_name := _runtime_type_alias_name(stmt)) is not None
    ]

    assert offenders == ["JsonScalar", "JsonValue"]


def test_pep604_runtime_expressions_are_not_used_for_python39() -> None:
    offenders: list[str] = []

    for dirname in _SCANNED_DIRS:
        for path in sorted((_REPO_ROOT / dirname).rglob("*.py")):
            module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(_runtime_type_alias_name(stmt) is not None for stmt in module.body):
                offenders.append(str(path.relative_to(_REPO_ROOT)))

    assert offenders == []
