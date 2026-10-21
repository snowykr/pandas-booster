from __future__ import annotations

import sys
from pathlib import Path
from typing import Final, NamedTuple

import pytest

from ._helpers import _job_block

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_CARGO_PATH: Final = _REPO_ROOT / "Cargo.toml"
_PYPROJECT_PATH: Final = _REPO_ROOT / "pyproject.toml"
_CI_WORKFLOW_PATH: Final = _REPO_ROOT / ".github" / "workflows" / "ci.yml"
_PUBLISH_WORKFLOW_PATH: Final = _REPO_ROOT / ".github" / "workflows" / "publish.yml"
_RUST_CI_EXPORT: Final = (
    "uv export --locked --no-dev --no-emit-project --format requirements-txt "
    '--output-file "$RUNNER_TEMP/rust-requirements.txt"'
)
_RUST_CI_INSTALL: Final = (
    "uv pip install --system --python \"$(python -c 'import sys; print(sys.executable)')\" "
    '--require-hashes -r "$RUNNER_TEMP/rust-requirements.txt"'
)
_CI_WHEEL_BUILD_JOBS: Final = (
    "build-wheel-smoke",
    "release-matrix",
    "build-and-test-quick",
    "stress-tests",
)
_CI_WHEEL_TEST_JOBS: Final = (
    (
        "release-matrix",
        'uv run --no-sync pytest tests/ -v --strict-markers -m "not stress" -o pythonpath=',
    ),
    (
        "build-and-test-quick",
        'uv run --no-sync pytest tests/ -v --strict-markers -m "not stress" -o pythonpath=',
    ),
    (
        "stress-tests",
        "uv run --no-sync pytest tests/test_sort_false_determinism.py "
        "-v --strict-markers -m stress -o pythonpath=",
    ),
)
_PUBLISH_RELEASE_ARGS: Final = (
    "args: --release --out dist --compatibility pypi --interpreter ${{ matrix.python-version }}"
)


class CargoBuildContract(NamedTuple):
    crate_types: tuple[str, ...]
    release_opt_level: int
    release_lto: str
    release_codegen_units: int
    release_panic: str


class MaturinContract(NamedTuple):
    module_name: str
    features: tuple[str, ...]
    includes: tuple[str, ...]


def _cargo_build_contract_from_text(text: str) -> CargoBuildContract:
    parsed = tomllib.loads(text)
    lib_section = parsed["lib"]
    release_profile = parsed["profile"]["release"]
    crate_types = tuple(lib_section["crate-type"])
    return CargoBuildContract(
        crate_types=crate_types,
        release_opt_level=release_profile["opt-level"],
        release_lto=release_profile["lto"],
        release_codegen_units=release_profile["codegen-units"],
        release_panic=release_profile["panic"],
    )


def _maturin_contract_from_text(text: str) -> MaturinContract:
    parsed = tomllib.loads(text)
    maturin_section = parsed["tool"]["maturin"]
    return MaturinContract(
        module_name=maturin_section["module-name"],
        features=tuple(maturin_section["features"]),
        includes=tuple(maturin_section["include"]),
    )


def _mutated_file_text(path: Path, old: str, new: str) -> str:
    text = path.read_text(encoding="utf-8")
    assert old in text, f"expected to find {old!r} in {path}"
    return text.replace(old, new, 1)


def _assert_cargo_build_contract(contract: CargoBuildContract) -> None:
    assert contract.crate_types == ("cdylib",), "Cargo lib crate-type must remain cdylib"
    assert contract.release_opt_level == 3, "Cargo release opt-level must remain 3"
    assert contract.release_lto == "fat", 'Cargo release lto must remain "fat"'
    assert contract.release_codegen_units == 1, "Cargo release codegen-units must remain 1"
    assert contract.release_panic == "abort", 'Cargo release panic strategy must remain "abort"'


def _assert_maturin_contract(contract: MaturinContract) -> None:
    assert contract.module_name == "pandas_booster._rust"
    assert contract.features == ("extension-module",)
    assert "python/pandas_booster/py.typed" in contract.includes
    assert "python/pandas_booster/_rust.pyi" in contract.includes


def _assert_ci_wheel_job_uses_release_build(workflow_text: str, job_name: str) -> None:
    job_block = _job_block(workflow_text, job_name)

    assert "maturin build --release --out dist" in job_block, (
        f"{job_name} must build wheels with --release"
    )
    assert "--profile dev-fast" not in job_block, f"{job_name} must not use --profile dev-fast"


def _assert_rust_ci_installs_numpy(workflow_text: str) -> None:
    job_block = _job_block(workflow_text, "rust-checks")
    assert _RUST_CI_EXPORT in job_block, "rust-checks must export locked dependency hashes"
    assert _RUST_CI_INSTALL in job_block, (
        "rust-checks must require hashes and install into the selected standalone Python"
    )
    assert "python -m pip install" not in job_block, "must not bypass locked provisioning"
    assert 'python -c "import sys, os;' in job_block, "PYO3_PYTHON must use the selected Python"
    assert "PYO3_PYTHON={sys.executable}" in job_block
    setup_position = job_block.index("uses: actions/setup-python@v5")
    uv_position = job_block.index("uses: astral-sh/setup-uv@v5")
    export_position = job_block.index(_RUST_CI_EXPORT)
    install_position = job_block.index(_RUST_CI_INSTALL)
    interpreter_position = job_block.index('python -c "import sys, os;')
    assert setup_position < uv_position < export_position < install_position, (
        "NumPy installation must follow Python and uv setup and locked export"
    )
    for command in (
        "cargo fmt --all -- --check",
        "cargo clippy --all-targets -- -D warnings",
        "cargo test",
    ):
        gate = f"run: {command}"
        assert gate in job_block, f"rust-checks must retain {command}"
        assert setup_position < install_position < job_block.index(gate), (
            "NumPy installation must follow Python setup and precede every Rust gate"
        )
        assert interpreter_position < job_block.index(gate), "Select PYO3_PYTHON before Rust gates"


def _assert_ci_credential_contract(workflow_text: str) -> None:
    header = workflow_text.split("jobs:", 1)[0]
    assert "permissions:\n  contents: read\n" in header, "CI must restrict token permissions"
    checkout = "uses: actions/checkout@v4"
    assert checkout in workflow_text
    for step in workflow_text.split(checkout)[1:]:
        assert step.startswith("\n        with:\n          persist-credentials: false\n"), (
            "CI checkouts must not persist credentials"
        )


def _assert_publish_wheel_job_uses_release_build(workflow_text: str) -> None:
    build_job = _job_block(workflow_text, "build-wheels")
    args_lines = [
        line.strip() for line in build_job.splitlines() if line.strip().startswith("args: ")
    ]

    assert build_job.count("uses: PyO3/maturin-action@v1") == 3
    assert args_lines == [_PUBLISH_RELEASE_ARGS, _PUBLISH_RELEASE_ARGS, _PUBLISH_RELEASE_ARGS]
    assert "target: universal2-apple-darwin" in build_job
    assert "--profile dev-fast" not in build_job, "publish wheel builds must not use dev-fast"


def _assert_wheel_smoke_install_contract(workflow_text: str) -> None:
    smoke_job = _job_block(workflow_text, "test-wheel-smoke")

    assert "uv sync --locked --extra dev --no-install-project" in smoke_job, (
        "wheel smoke must keep --no-install-project"
    )
    assert 'if [ "${#wheels[@]}" -ne 1 ]; then' in smoke_job, (
        "wheel smoke must require exactly 1 wheel"
    )
    assert "uv pip install -e ." not in smoke_job, "wheel smoke must not use editable install"
    assert 'uv pip install "${wheels[0]}"' in smoke_job, (
        "wheel smoke must install the built wheel artifact"
    )
    assert "site-packages" in smoke_job
    assert "uv run --no-sync pytest" in smoke_job
    assert "-o pythonpath=" in smoke_job


def _assert_ci_wheel_test_job_uses_installed_wheel(
    workflow_text: str,
    job_name: str,
    expected_pytest_command: str,
) -> None:
    job_block = _job_block(workflow_text, job_name)

    assert expected_pytest_command in job_block, (
        f"{job_name} must run pytest with uv --no-sync and clear pytest pythonpath"
    )


def test_cargo_toml_keeps_certified_release_build_settings() -> None:
    contract = _cargo_build_contract_from_text(_CARGO_PATH.read_text(encoding="utf-8"))

    _assert_cargo_build_contract(contract)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ('crate-type = ["cdylib"]', 'crate-type = ["rlib"]', "cdylib"),
        ('lto = "fat"', 'lto = "thin"', "fat"),
        ("codegen-units = 1", "codegen-units = 16", "1"),
    ),
)
def test_cargo_toml_guardrails_reject_unsafe_release_mutations(
    old: str,
    new: str,
    message: str,
) -> None:
    mutated_text = _mutated_file_text(_CARGO_PATH, old, new)

    with pytest.raises(AssertionError, match=message):
        _assert_cargo_build_contract(_cargo_build_contract_from_text(mutated_text))


def test_pyproject_keeps_extension_module_packaging_contract() -> None:
    contract = _maturin_contract_from_text(_PYPROJECT_PATH.read_text(encoding="utf-8"))

    _assert_maturin_contract(contract)


def test_pyproject_guardrails_reject_removed_extension_module_feature() -> None:
    mutated_text = _mutated_file_text(
        _PYPROJECT_PATH,
        'features = ["extension-module"]',
        "features = []",
    )

    with pytest.raises(AssertionError, match="extension-module"):
        _assert_maturin_contract(_maturin_contract_from_text(mutated_text))


def test_rust_ci_installs_numpy_before_all_gates() -> None:
    _assert_rust_ci_installs_numpy(_CI_WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("placement", ("missing", "before-setup", "after-gate", "wrong-job"))
def test_rust_ci_numpy_guard_rejects_missing_or_misordered_runtime(placement: str) -> None:
    workflow = _CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    job_block = _job_block(workflow, "rust-checks")
    install_step = (
        "      - name: Install locked dependencies for embedded converter tests\n"
        "        run: |\n"
        f"          {_RUST_CI_EXPORT}\n"
        f"          {_RUST_CI_INSTALL}\n\n"
    )
    assert install_step in job_block
    mutated_job = job_block.replace(install_step, "", 1)
    if placement == "before-setup":
        mutated_job = mutated_job.replace(
            "      - name: Install uv", install_step + "      - name: Install uv", 1
        )
    elif placement == "after-gate":
        mutated_job = mutated_job.replace(
            "        run: cargo fmt --all -- --check\n",
            "        run: cargo fmt --all -- --check\n\n" + install_step,
            1,
        )
    mutated = workflow.replace(job_block, mutated_job, 1)
    if placement == "wrong-job":
        other_job = _job_block(mutated, "python-quality")
        mutated = mutated.replace(
            other_job, other_job.replace("    steps:\n", "    steps:\n" + install_step, 1), 1
        )
    with pytest.raises(AssertionError, match="locked dependency|installation must follow"):
        _assert_rust_ci_installs_numpy(mutated)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        ("uv export --locked", "uv export --frozen", "locked dependency"),
        ("--no-emit-project ", "", "locked dependency"),
        ("--require-hashes ", "", "require hashes"),
        ("--system ", "", "standalone Python"),
        (
            _RUST_CI_INSTALL,
            'python -m pip install "numpy>=1.21.0,<3.0.0"',
            "require hashes",
        ),
        (
            "\"$(python -c 'import sys; print(sys.executable)')\"",
            "python3.12",
            "selected standalone Python",
        ),
    ),
)
def test_rust_ci_numpy_guard_rejects_unlocked_or_mismatched_runtime(
    old: str,
    new: str,
    message: str,
) -> None:
    mutated = _mutated_file_text(_CI_WORKFLOW_PATH, old, new)
    with pytest.raises(AssertionError, match=message):
        _assert_rust_ci_installs_numpy(mutated)


def test_ci_restricts_checkout_credentials_and_token_permissions() -> None:
    _assert_ci_credential_contract(_CI_WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("permissions:\n  contents: read\n", ""),
        ("contents: read", "contents: write"),
        ("          persist-credentials: false\n", ""),
        ("persist-credentials: false", "persist-credentials: true"),
    ),
)
def test_ci_credential_guard_rejects_unsafe_mutations(old: str, new: str) -> None:
    mutated = _mutated_file_text(
        _CI_WORKFLOW_PATH,
        old,
        new,
    )
    with pytest.raises(AssertionError, match="permissions|credentials"):
        _assert_ci_credential_contract(mutated)


def test_ci_wheel_build_jobs_keep_release_wheel_commands() -> None:
    workflow_text = _CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    for job_name in _CI_WHEEL_BUILD_JOBS:
        _assert_ci_wheel_job_uses_release_build(workflow_text, job_name)


def test_ci_wheel_build_guardrail_rejects_dev_fast_wheel_job() -> None:
    mutated_text = _mutated_file_text(
        _CI_WORKFLOW_PATH,
        "maturin build --release --out dist",
        "maturin build --profile dev-fast --out dist",
    )

    with pytest.raises(AssertionError, match="maturin build --release --out dist"):
        _assert_ci_wheel_job_uses_release_build(mutated_text, "build-wheel-smoke")


def test_publish_workflow_keeps_release_wheel_build_args() -> None:
    workflow_text = _PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")

    _assert_publish_wheel_job_uses_release_build(workflow_text)


def test_publish_workflow_guardrail_rejects_dev_fast_release_build() -> None:
    mutated_text = _mutated_file_text(
        _PUBLISH_WORKFLOW_PATH,
        "--release --out dist --compatibility pypi",
        "--profile dev-fast --out dist --compatibility pypi",
    )

    with pytest.raises(AssertionError, match="dev-fast"):
        _assert_publish_wheel_job_uses_release_build(mutated_text)


def test_ci_wheel_smoke_keeps_no_editable_install_contract() -> None:
    workflow_text = _CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    _assert_wheel_smoke_install_contract(workflow_text)


@pytest.mark.parametrize(("job_name", "expected_pytest_command"), _CI_WHEEL_TEST_JOBS)
def test_ci_wheel_test_jobs_run_pytest_against_installed_wheel(
    job_name: str,
    expected_pytest_command: str,
) -> None:
    workflow_text = _CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    _assert_ci_wheel_test_job_uses_installed_wheel(
        workflow_text,
        job_name,
        expected_pytest_command,
    )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    (
        (
            "uv sync --locked --extra dev --no-install-project",
            "uv sync --locked --extra dev",
            "--no-install-project",
        ),
        ('uv pip install "${wheels[0]}"', "uv pip install -e .", "editable"),
        ('if [ "${#wheels[@]}" -ne 1 ]; then', 'if [ "${#wheels[@]}" -lt 1 ]; then', "exactly 1"),
    ),
)
def test_ci_wheel_smoke_guardrails_reject_unsafe_install_mutations(
    old: str,
    new: str,
    message: str,
) -> None:
    mutated_text = _mutated_file_text(_CI_WORKFLOW_PATH, old, new)

    with pytest.raises(AssertionError, match=message):
        _assert_wheel_smoke_install_contract(mutated_text)
