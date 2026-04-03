from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tomllib

REPO_URL = "https://github.com/snowykr/pandas-booster"
README_REQUIRED_TOKENS = (
    "# pandas-booster",
    "https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml",
    'pip install "maturin>=1.4,<2.0"',
    "PyPI project exists.",
    "Trusted Publisher",
    "publish.yml",
    "GitHub environment `pypi` is configured",
)
README_FORBIDDEN_TOKENS = ("uv ", "uvx", "your-org")
WORKFLOW_REQUIRED_TOKENS = (
    "workflow_dispatch:",
    "publish:",
    "default: false",
    "tags:",
    "v*.*.*",
    "build-wheels",
    "build-sdist",
    "PyO3/maturin-action",
    "manylinux: 2014",
    "universal2-apple-darwin",
    "maturin sdist --out dist",
    "uses: actions/upload-artifact@v4",
    "uses: actions/download-artifact@v4",
    "contents: read",
    "id-token: write",
    "uses: pypa/gh-action-pypi-publish@release/v1",
    "name: pypi",
    "https://pypi.org/p/pandas-booster",
    "inputs.publish",
    'python scripts/check_release_contract.py tag --tag "$GITHUB_REF_NAME"',
    (
        "python scripts/check_release_contract.py artifacts --dist dist "
        "--expected-wheel-count 12 --require-sdist"
    ),
)


class ContractError(Exception):
    pass


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_toml(path: Path) -> dict:
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except FileNotFoundError as exc:
        raise ContractError(f"Required file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ContractError(f"Failed to parse TOML file {path}: {exc}") from exc


def load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(f"Required file not found: {path}") from exc


def fail(errors: list[str]) -> None:
    message = "Release contract check failed:\n"
    message += "\n".join(f"- {error}" for error in errors)
    raise ContractError(message)


def validate_metadata(_: argparse.Namespace) -> int:
    root = project_root()
    pyproject = load_toml(root / "pyproject.toml")
    cargo = load_toml(root / "Cargo.toml")
    readme_text = load_text(root / "README.md")

    errors: list[str] = []

    build_system = pyproject.get("build-system", {})
    project = pyproject.get("project", {})
    project_urls = project.get("urls", {})
    classifiers = project.get("classifiers", [])

    build_requires = build_system.get("requires", [])
    if "maturin>=1.4,<2.0" not in build_requires:
        errors.append("pyproject.toml build-system.requires must include 'maturin>=1.4,<2.0'")

    if project.get("readme") != "README.md":
        errors.append("pyproject.toml project.readme must equal 'README.md'")

    if project.get("requires-python") != ">=3.9":
        errors.append("pyproject.toml project.requires-python must equal '>=3.9'")

    required_classifiers = {
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    }
    missing_classifiers = sorted(required_classifiers.difference(classifiers))
    if missing_classifiers:
        errors.append(
            "pyproject.toml must declare Python classifiers for 3.9, 3.10, 3.11, and 3.12; "
            f"missing: {', '.join(missing_classifiers)}"
        )

    for key, expected in {
        "Homepage": REPO_URL,
        "Repository": REPO_URL,
        "Issues": f"{REPO_URL}/issues",
    }.items():
        if project_urls.get(key) != expected:
            errors.append(f"pyproject.toml project.urls.{key} must equal '{expected}'")

    if "Programming Language :: Python :: 3.13" in classifiers:
        errors.append("pyproject.toml must not declare a Python 3.13 classifier yet")

    package = cargo.get("package", {})
    if package.get("repository") != REPO_URL:
        errors.append(f"Cargo.toml package.repository must equal '{REPO_URL}'")

    project_version = project.get("version")
    cargo_version = package.get("version")
    if cargo_version != project_version:
        errors.append(
            "Cargo.toml package.version must exactly match pyproject.toml project.version"
        )

    for token in README_REQUIRED_TOKENS:
        if token not in readme_text:
            errors.append(f"README.md must contain '{token}'")

    for token in README_FORBIDDEN_TOKENS:
        if token in readme_text:
            errors.append(f"README.md must not contain '{token}'")

    if errors:
        fail(errors)

    print("metadata: release contract checks passed")
    return 0


def validate_tag(args: argparse.Namespace) -> int:
    tag = args.tag
    if not tag.startswith("v"):
        raise ContractError(
            f"Release tag '{tag}' is invalid: tags must start with 'v' (example: v0.1.0)"
        )

    project = load_toml(project_root() / "pyproject.toml").get("project", {})
    version = project.get("version")
    expected_tag = f"v{version}"
    if tag != expected_tag:
        raise ContractError(
            "Release tag/version mismatch: "
            f"got '{tag}', but pyproject.toml project.version is '{version}', "
            f"so the tag must be '{expected_tag}'"
        )

    print(f"tag: '{tag}' matches project.version '{version}'")
    return 0


def validate_workflow(args: argparse.Namespace) -> int:
    workflow_path = Path(args.file)
    if not workflow_path.is_absolute():
        workflow_path = project_root() / workflow_path

    workflow_text = load_text(workflow_path)
    missing_tokens = [token for token in WORKFLOW_REQUIRED_TOKENS if token not in workflow_text]
    if missing_tokens:
        fail([f"{workflow_path} is missing required token '{token}'" for token in missing_tokens])

    print(f"workflow: required release guardrails found in {workflow_path}")
    return 0


def validate_artifacts(args: argparse.Namespace) -> int:
    dist_path = Path(args.dist)
    if not dist_path.is_absolute():
        dist_path = project_root() / dist_path

    if not dist_path.exists():
        raise ContractError(f"Distribution directory does not exist: {dist_path}")
    if not dist_path.is_dir():
        raise ContractError(f"Distribution path is not a directory: {dist_path}")

    wheel_paths = sorted(dist_path.glob("*.whl"))
    sdist_paths = sorted(dist_path.glob("*.tar.gz")) + sorted(dist_path.glob("*.zip"))

    errors: list[str] = []
    if len(wheel_paths) != args.expected_wheel_count:
        errors.append(
            "Expected exactly "
            f"{args.expected_wheel_count} wheel file(s) in {dist_path}, "
            f"found {len(wheel_paths)}"
        )

    if args.require_sdist and len(sdist_paths) != 1:
        errors.append(f"Expected exactly 1 sdist file in {dist_path}, found {len(sdist_paths)}")

    if errors:
        fail(errors)

    print(
        "artifacts: "
        f"found {len(wheel_paths)} wheel(s)"
        + (f" and {len(sdist_paths)} sdist file(s)" if args.require_sdist else "")
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the repository release contract")
    subparsers = parser.add_subparsers(dest="command", required=True)

    metadata_parser = subparsers.add_parser(
        "metadata",
        help="Validate pyproject, Cargo, and README release metadata",
    )
    metadata_parser.set_defaults(func=validate_metadata)

    tag_parser = subparsers.add_parser(
        "tag",
        help="Validate a release tag against project.version",
    )
    tag_parser.add_argument("--tag", required=True, help="Release tag to validate, e.g. v0.1.0")
    tag_parser.set_defaults(func=validate_tag)

    workflow_parser = subparsers.add_parser(
        "workflow",
        help="Validate required publish workflow guardrail tokens",
    )
    workflow_parser.add_argument("--file", required=True, help="Workflow file to inspect")
    workflow_parser.set_defaults(func=validate_workflow)

    artifacts_parser = subparsers.add_parser(
        "artifacts",
        help="Validate built wheel and sdist artifact counts",
    )
    artifacts_parser.add_argument("--dist", required=True, help="Distribution directory to inspect")
    artifacts_parser.add_argument(
        "--expected-wheel-count",
        type=int,
        default=12,
        help="Exact number of wheels expected in the dist directory",
    )
    artifacts_parser.add_argument(
        "--require-sdist",
        action="store_true",
        help="Require exactly one source distribution file",
    )
    artifacts_parser.set_defaults(func=validate_artifacts)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except ContractError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
