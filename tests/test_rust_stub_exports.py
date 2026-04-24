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
