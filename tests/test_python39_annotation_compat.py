from __future__ import annotations

import ast
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCANNED_DIRS = ("python", "tests", "scripts", "benches")


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
