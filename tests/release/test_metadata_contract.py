"""Release contract tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from conftest import _load_release_contract_module

from ._helpers import _write_package_init


def test_validate_metadata_requires_release_readme_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# pandas-booster\n", encoding="utf-8")
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="README.md must contain"):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_requires_release_readme_preconditions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
Trusted Publisher
publish.yml
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError) as exc_info:
        contract.validate_metadata(argparse.Namespace())

    message = str(exc_info.value)
    assert "README.md must contain 'PyPI project exists.'" in message
    assert "README.md must contain 'GitHub environment `pypi` is configured'" in message


def test_validate_metadata_requires_supported_python_classifier_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
PyPI project exists.
Trusted Publisher
publish.yml
GitHub environment `pypi` is configured
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="must declare Python classifiers"):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_allows_uv_documentation_in_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
uv add pandas-booster
uvx ruff check .
PyPI project exists.
Trusted Publisher
publish.yml
GitHub environment `pypi` is configured
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    assert contract.validate_metadata(argparse.Namespace()) == 0


def test_validate_metadata_rejects_placeholder_org_in_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.0\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.0\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
PyPI project exists.
Trusted Publisher
publish.yml
GitHub environment `pypi` is configured

https://github.com/your-org/pandas-booster
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_package_init(tmp_path, "0.1.0")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="README.md must not contain 'your-org'"):
        contract.validate_metadata(argparse.Namespace())


def test_validate_metadata_accepts_current_repo_release_readme():
    contract = _load_release_contract_module()
    assert contract.validate_metadata(argparse.Namespace()) == 0


def test_validate_metadata_requires_package_dunder_version_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    contract = _load_release_contract_module()

    (tmp_path / "scripts").mkdir()
    (tmp_path / "pyproject.toml").write_text(
        """
[build-system]
requires = [\"maturin>=1.13,<2.0\"]

[project]
name = \"pandas-booster\"
version = \"0.1.2\"
readme = \"README.md\"
requires-python = \">=3.9\"
classifiers = [
    \"Programming Language :: Python :: 3.9\",
    \"Programming Language :: Python :: 3.10\",
    \"Programming Language :: Python :: 3.11\",
    \"Programming Language :: Python :: 3.12\",
]

[project.urls]
Homepage = \"https://github.com/snowykr/pandas-booster\"
Repository = \"https://github.com/snowykr/pandas-booster\"
Issues = \"https://github.com/snowykr/pandas-booster/issues\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "Cargo.toml").write_text(
        """
[package]
name = \"pandas_booster\"
version = \"0.1.2\"
repository = \"https://github.com/snowykr/pandas-booster\"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        """
# pandas-booster

https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml
pip install pandas-booster
PyPI project exists.
Trusted Publisher
publish.yml
GitHub environment `pypi` is configured
""".strip()
        + "\n",
        encoding="utf-8",
    )
    _write_package_init(tmp_path, "0.1.1")

    monkeypatch.setattr(contract, "project_root", lambda: tmp_path)

    with pytest.raises(contract.ContractError, match="__version__ must exactly match"):
        contract.validate_metadata(argparse.Namespace())
