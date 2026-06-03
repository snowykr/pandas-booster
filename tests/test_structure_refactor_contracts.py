from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _pure_loc(path: Path) -> int:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "//")):
            count += 1
    return count


def test_split_refactor_files_stay_under_pure_loc_budget():
    candidates = [
        *_REPO_ROOT.joinpath("src", "groupby").glob("*.rs"),
        *_REPO_ROOT.joinpath("tests", "benchmark").glob("*.py"),
        *_REPO_ROOT.joinpath("tests", "edge_cases").glob("*.py"),
    ]

    over_budget = {
        str(path.relative_to(_REPO_ROOT)): _pure_loc(path)
        for path in candidates
        if _pure_loc(path) >= 250
    }

    assert over_budget == {}


def test_ruff_docstring_ignore_covers_nested_tests():
    pyproject = _REPO_ROOT.joinpath("pyproject.toml").read_text(encoding="utf-8")

    assert '"tests/**/*.py" = ["D"]' in pyproject
    assert '"tests/*.py" = ["D"]' not in pyproject


def test_unsafe_pointer_transport_keeps_typed_provenance_and_safety_notes():
    convert = _REPO_ROOT.joinpath("src", "python_wrappers", "convert.rs").read_text(
        encoding="utf-8"
    )
    partition = _REPO_ROOT.joinpath("src", "radix_groupby", "partition.rs").read_text(
        encoding="utf-8"
    )

    assert "struct WritePtr<T>(NonNull<T>)" in convert
    assert "Vec<usize>" not in convert
    assert " as usize" not in convert
    assert " as *mut" not in convert
    assert convert.count("SAFETY:") >= 4

    assert "struct PermPtr(NonNull<usize>)" in partition
    assert "unsafe impl Send for PermPtr" in partition
    assert "unsafe impl Sync for PermPtr" in partition
    assert "PermPtr(*mut usize)" not in partition
    assert partition.count("SAFETY:") >= 4
