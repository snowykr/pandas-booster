from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_URL = "https://github.com/snowykr/pandas-booster"
README_REQUIRED_TOKENS = (
    "# pandas-booster",
    "https://github.com/snowykr/pandas-booster/actions/workflows/ci.yml",
    "pip install pandas-booster",
    "PyPI project exists.",
    "Trusted Publisher",
    "publish.yml",
    "GitHub environment `pypi` is configured",
)
README_FORBIDDEN_TOKENS = ("your-org",)
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
    "startsWith(github.ref, 'refs/tags/v') && (github.event_name == 'push' || inputs.publish)",
    'python scripts/check_release_contract.py tag --tag "$GITHUB_REF_NAME"',
    (
        "python scripts/check_release_contract.py artifacts --dist dist "
        "--expected-wheel-count 12 --require-sdist"
    ),
)
SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS = (
    "name: Supply Chain Risk Guard",
    "pull_request:",
    "types: [opened, synchronize, reopened]",
    "pull-requests: write",
    "contents: read",
    "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
    "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    "Select audit scripts",
    'git show "$BASE_SHA:scripts/supply_chain_audit.py"',
    'git show "$BASE_SHA:scripts/supply_chain_comment.py"',
    "Base branch does not contain supply-chain risk guard scripts.",
    "AUDIT_SCRIPT=",
    "COMMENT_SCRIPT=",
    'python "$AUDIT_SCRIPT" scan',
    'python "$COMMENT_SCRIPT" sync',
    "scripts/supply_chain_audit.py",
    "scripts/supply_chain_comment.py",
    "BASE_SHA:",
    "HEAD_SHA:",
    "PR_NUMBER:",
    "GH_TOKEN:",
    "$RUNNER_TEMP/findings.md",
    "pandas-booster:supply-chain-risk-guard",
    "Sync risk guard comment",
    "Fail on risk findings",
    "steps.scan.outputs.found == 'true'",
)
LEGACY_SUPPLY_CHAIN_SCRIPT = "supply_chain" + "_audit.sh"
SUPPLY_CHAIN_WORKFLOW_FORBIDDEN_TOKENS = (
    f'cat > "$RUNNER_TEMP/{LEGACY_SUPPLY_CHAIN_SCRIPT}"',
    f"$RUNNER_TEMP/{LEGACY_SUPPLY_CHAIN_SCRIPT}",
    f"bash scripts/{LEGACY_SUPPLY_CHAIN_SCRIPT}",
    LEGACY_SUPPLY_CHAIN_SCRIPT,
    "pull_request_target",
    "paths:",
    "paths-ignore:",
    "credential-" + "stealing",
    "lite" + "llm",
    "Manage supply-chain " + "audit comment",
    "<!-- pandas-booster:supply-chain-" + "audit -->",
    "actions/checkout@v4",
    "actions/setup-python@v5",
    "cp scripts/supply_chain_audit.py",
    "cp scripts/supply_chain_comment.py",
)
SUPPLY_CHAIN_CONCURRENCY_GROUP = (
    "group: supply-chain-risk-guard-${{ github.event.pull_request.number }}"
)
SUPPLY_CHAIN_CONCURRENCY_CANCEL = "cancel-in-progress: true"
SUPPLY_CHAIN_FORBIDDEN_LINE_MESSAGES = {
    "push:": "must not use push triggers",
    "workflow_dispatch:": "must not use workflow_dispatch triggers",
    "schedule:": "must not use schedule triggers",
    "repository_dispatch:": "must not use repository_dispatch triggers",
    "workflow_run:": "must not use workflow_run triggers",
    "issue_comment:": "must not use issue_comment triggers",
    "paths-ignore:": "must not use workflow-level paths filters",
}
WRITE_PERMISSION_RE = re.compile(r"^([a-z-]+): write$")
ALLOWED_WRITE_PERMISSIONS = frozenset({"pull-requests"})
PYTHON_COMMAND_RE = re.compile(r"\bpython(?:3(?:\.\d+)?)?\b")
SUPPLY_CHAIN_SCRIPT_PATH_RE = re.compile(
    r"(?:^|[\s'\"])(?:[\w${}./_-]+/)*"
    r"scripts/supply_chain_(?:audit|comment)\.py(?:$|[\s'\"])"
)
SUPPLY_CHAIN_MODULE_RE = re.compile(
    r"(?:^|\s)-m\s+scripts\.supply_chain_(?:audit|comment)(?:$|\s)"
)
SUPPLY_CHAIN_DOTTED_MODULE_RE = re.compile(
    r"\bscripts\.supply_chain_(?:audit|comment)\b"
)
SUPPLY_CHAIN_FROM_IMPORT_RE = re.compile(
    r"\bfrom\s+scripts\s+import\s+supply_chain_(?:audit|comment)\b"
)
SUPPLY_CHAIN_SCRIPT_ENV_KEY_RE = re.compile(
    r"(?:^|[{,]\s*)(AUDIT_SCRIPT|COMMENT_SCRIPT)\s*:"
)
SUPPLY_CHAIN_GITHUB_ENV_SCRIPT_ASSIGNMENT_RE = re.compile(
    r"\b(AUDIT_SCRIPT|COMMENT_SCRIPT)=([^\s\"']+)"
)
PERMISSION_LINE_RE = re.compile(r"^([a-z-]+)\s*:\s*(read|write|none)$")
ALLOWED_SUPPLY_CHAIN_PERMISSIONS = {
    "contents": "read",
    "pull-requests": "write",
}


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


def _check_supply_chain_run_blocks(workflow_text: str, workflow_path: Path) -> list[str]:
    lines = workflow_text.splitlines()
    errors: list[str] = []
    in_run_block = False
    run_key_indent = 0
    run_in_select_audit = False
    in_select_audit_step = False
    run_block_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if in_run_block and stripped and indent <= run_key_indent:
            _check_supply_chain_run_content(
                join_run_block_lines(run_block_lines),
                workflow_path,
                errors,
            )
            run_block_lines.clear()
            in_run_block = False
            run_in_select_audit = False

        if in_run_block:
            run_block_lines.append(line)
            _check_supply_chain_run_content(line, workflow_path, errors)
            if run_in_select_audit and (
                "cp scripts/supply_chain_audit.py" in line
                or "cp scripts/supply_chain_comment.py" in line
            ):
                errors.append(
                    f"{workflow_path} must not use PR-controlled fallback copies in "
                    "Select audit scripts"
                )
            continue

        if stripped.startswith("- "):
            in_select_audit_step = stripped == "- name: Select audit scripts"

        if stripped.startswith("run: ") and not (
            stripped.startswith("run: |") or stripped.startswith("run: >")
        ):
            _check_supply_chain_run_content(stripped, workflow_path, errors)

        if stripped.startswith("run: |") or stripped.startswith("run: >"):
            in_run_block = True
            run_key_indent = indent
            run_in_select_audit = in_select_audit_step

    if in_run_block:
        _check_supply_chain_run_content(
            join_run_block_lines(run_block_lines),
            workflow_path,
            errors,
        )

    return errors


def _check_supply_chain_script_env_overrides(
    workflow_lines: list[str],
    workflow_path: Path,
) -> list[str]:
    errors: list[str] = []

    for line in workflow_lines:
        stripped = line.partition("#")[0].strip()
        match = SUPPLY_CHAIN_SCRIPT_ENV_KEY_RE.search(stripped)
        if match is None:
            continue
        errors.append(f"{workflow_path} must not set {match.group(1)} from YAML env")

    return errors


def join_run_block_lines(lines: list[str]) -> str:
    return " ".join(line.strip().rstrip("\\") for line in lines)


def _check_supply_chain_run_content(
    line: str,
    workflow_path: Path,
    errors: list[str],
) -> None:
    if "${{ github.event" in line:
        errors.append(f"{workflow_path} must not reference github.event inside run blocks")
    script_env_assignment = SUPPLY_CHAIN_GITHUB_ENV_SCRIPT_ASSIGNMENT_RE.search(line)
    if script_env_assignment is not None and "$GITHUB_ENV" in line:
        script_name = script_env_assignment.group(1)
        script_value = script_env_assignment.group(2)
        if is_pr_controlled_supply_chain_execution(script_value):
            errors.append(
                f"{workflow_path} must not write PR-controlled {script_name} to GITHUB_ENV"
            )
            return
    if is_pr_controlled_supply_chain_execution(line):
        errors.append(f"{workflow_path} must not execute PR-controlled supply-chain scripts")


def is_pr_controlled_supply_chain_execution(line: str) -> bool:
    normalized_line = normalize_run_content(line)
    if SUPPLY_CHAIN_SCRIPT_PATH_RE.search(normalized_line) is not None:
        return True
    return (
        SUPPLY_CHAIN_MODULE_RE.search(normalized_line) is not None
        or SUPPLY_CHAIN_DOTTED_MODULE_RE.search(normalized_line) is not None
        or SUPPLY_CHAIN_FROM_IMPORT_RE.search(normalized_line) is not None
    )


def normalize_run_content(line: str) -> str:
    unquoted = line.replace('"', "").replace("'", "")
    normalized_workspace = re.sub(
        r"\${{\s*github\.workspace\s*}}",
        "GITHUB_WORKSPACE",
        unquoted,
    )
    normalized_segments = normalized_workspace.replace("/./", "/")
    return re.sub(r"(?<!:)//+", "/", normalized_segments)


def _check_supply_chain_concurrency(workflow_lines: list[str], workflow_path: Path) -> list[str]:
    errors: list[str] = []
    concurrency_children = _block_direct_children(workflow_lines, "concurrency:")

    if not concurrency_children:
        errors.append(f"{workflow_path} must define workflow concurrency")
        return errors

    has_group = False
    has_cancel = False
    for child_stripped in concurrency_children:
        if child_stripped == SUPPLY_CHAIN_CONCURRENCY_GROUP:
            has_group = True
        elif child_stripped == SUPPLY_CHAIN_CONCURRENCY_CANCEL:
            has_cancel = True
        elif child_stripped.split(":", 1)[0] in {"group", "cancel-in-progress"}:
            errors.append(f"{workflow_path} has malformed concurrency key '{child_stripped}'")

    if not has_group:
        errors.append(
            f"{workflow_path} is missing required concurrency group "
            f"'{SUPPLY_CHAIN_CONCURRENCY_GROUP}'"
        )
    if not has_cancel:
        errors.append(
            f"{workflow_path} is missing required concurrency setting "
            f"'{SUPPLY_CHAIN_CONCURRENCY_CANCEL}'"
        )

    return errors


def _check_supply_chain_on_block(
    workflow_lines: list[str],
    workflow_path: Path,
) -> list[str]:
    errors: list[str] = []
    on_index = next(
        (
            index
            for index, line in enumerate(workflow_lines)
            if line.strip().startswith("on:") and len(line) == len(line.lstrip())
        ),
        None,
    )
    on_line = workflow_lines[on_index].strip() if on_index is not None else ""
    if on_line and on_line != "on:":
        errors.append(f"{workflow_path} must use a pull_request-only on block")
        return errors
    if on_index is None:
        errors.append(f"{workflow_path} must define an on block")
        return errors

    on_children = _block_direct_children_from_index(workflow_lines, on_index)
    if on_children != ["pull_request:"]:
        errors.append(f"{workflow_path} must define only pull_request in the on block")
    for child in on_children:
        if child.endswith(":") and child != "pull_request:":
            errors.append(f"{workflow_path} must not use {child.removesuffix(':')} triggers")

    pull_request_index = _find_direct_child_index(
        workflow_lines,
        parent_index=on_index,
        child="pull_request:",
    )
    if pull_request_index is None:
        return errors

    pull_request_children = _block_direct_children_from_index(
        workflow_lines,
        pull_request_index,
    )
    required_types = "types: [opened, synchronize, reopened]"
    if required_types not in pull_request_children:
        errors.append(f"{workflow_path} must set pull_request {required_types}")

    return errors


def _check_supply_chain_permissions_block(
    workflow_lines: list[str],
    workflow_path: Path,
) -> list[str]:
    errors: list[str] = []
    top_level_permission_indices: list[int] = []
    for index, line in enumerate(workflow_lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if not stripped.startswith("permissions:"):
            continue
        if stripped == "permissions:":
            if indent != 0:
                errors.append(f"{workflow_path} must not define job-level permissions")
            else:
                top_level_permission_indices.append(index)
            continue
        errors.append(f"{workflow_path} must not use inline or job-level permissions")

    if len(top_level_permission_indices) > 1:
        errors.append(f"{workflow_path} must define exactly one top-level permissions block")

    seen_permissions: dict[str, str] = {}
    for permissions_index in top_level_permission_indices:
        for child in _block_direct_children_from_index(workflow_lines, permissions_index):
            permission_child = strip_yaml_inline_comment(child)
            match = PERMISSION_LINE_RE.fullmatch(permission_child)
            if match is None:
                continue

            permission = match.group(1)
            value = match.group(2)
            seen_permissions[permission] = value
            expected_value = ALLOWED_SUPPLY_CHAIN_PERMISSIONS.get(permission)
            if expected_value is None:
                errors.append(f"{workflow_path} must not request {permission_child}")
            elif value != expected_value:
                errors.append(
                    f"{workflow_path} must request {permission}: {expected_value}, not {value}"
                )

    for permission, expected_value in ALLOWED_SUPPLY_CHAIN_PERMISSIONS.items():
        if seen_permissions.get(permission) != expected_value:
            errors.append(f"{workflow_path} must request {permission}: {expected_value}")

    return errors


def strip_yaml_inline_comment(line: str) -> str:
    return line.split("#", 1)[0].rstrip()


def _block_direct_children(workflow_lines: list[str], header: str) -> list[str]:
    header_index: int | None = next(
        (
            index
            for index, line in enumerate(workflow_lines)
            if line.strip() == header
        ),
        None,
    )
    if header_index is None:
        return []

    return _block_direct_children_from_index(workflow_lines, header_index)


def _block_direct_children_from_index(
    workflow_lines: list[str],
    header_index: int,
) -> list[str]:
    header_indent = len(workflow_lines[header_index]) - len(
        workflow_lines[header_index].lstrip()
    )
    child_indent: int | None = None
    children: list[str] = []

    for line in workflow_lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())
        if indent <= header_indent:
            break

        if child_indent is None or indent < child_indent:
            child_indent = indent
            children.clear()

        if indent == child_indent:
            children.append(stripped)

    return children


def _find_direct_child_index(
    workflow_lines: list[str],
    *,
    parent_index: int,
    child: str,
) -> int | None:
    parent_indent = len(workflow_lines[parent_index]) - len(
        workflow_lines[parent_index].lstrip()
    )
    child_indent: int | None = None

    for index, line in enumerate(workflow_lines[parent_index + 1 :], start=parent_index + 1):
        stripped = line.strip()
        if not stripped:
            continue

        indent = len(line) - len(line.lstrip())
        if indent <= parent_indent:
            return None

        if child_indent is None or indent < child_indent:
            child_indent = indent

        if indent == child_indent and stripped == child:
            return index

    return None


def _check_supply_chain_forbidden_lines(
    workflow_lines: list[str],
    workflow_path: Path,
) -> list[str]:
    errors: list[str] = []
    for line in workflow_lines:
        stripped = line.strip()
        message = SUPPLY_CHAIN_FORBIDDEN_LINE_MESSAGES.get(stripped)
        if message is not None:
            errors.append(f"{workflow_path} {message}")
        permission_match = WRITE_PERMISSION_RE.fullmatch(stripped)
        if (
            permission_match is not None
            and permission_match.group(1) not in ALLOWED_WRITE_PERMISSIONS
        ):
            errors.append(f"{workflow_path} must not request {stripped}")
    return errors


def fail(errors: list[str]) -> None:
    message = "Release contract check failed:\n"
    message += "\n".join(f"- {error}" for error in errors)
    raise ContractError(message)


def validate_metadata(_: argparse.Namespace) -> int:
    root = project_root()
    pyproject = load_toml(root / "pyproject.toml")
    cargo = load_toml(root / "Cargo.toml")
    readme_text = load_text(root / "README.md")
    package_init_text = load_text(root / "python" / "pandas_booster" / "__init__.py")

    errors: list[str] = []

    build_system = pyproject.get("build-system", {})
    project = pyproject.get("project", {})
    project_urls = project.get("urls", {})
    classifiers = project.get("classifiers", [])

    build_requires = build_system.get("requires", [])
    if "maturin>=1.13,<2.0" not in build_requires:
        errors.append("pyproject.toml build-system.requires must include 'maturin>=1.13,<2.0'")

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

    expected_dunder_version = f'__version__ = "{project_version}"'
    if expected_dunder_version not in package_init_text:
        errors.append(
            "python/pandas_booster/__init__.py __version__ must exactly match "
            "pyproject.toml project.version"
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


def validate_supply_chain_workflow(args: argparse.Namespace) -> int:
    workflow_path = Path(args.file)
    if not workflow_path.is_absolute():
        workflow_path = project_root() / workflow_path

    workflow_text = load_text(workflow_path)
    workflow_lines = workflow_text.splitlines()
    errors = [
        f"{workflow_path} is missing required token '{token}'"
        for token in SUPPLY_CHAIN_WORKFLOW_REQUIRED_TOKENS
        if token not in workflow_text
    ]

    for token in SUPPLY_CHAIN_WORKFLOW_FORBIDDEN_TOKENS:
        if token in workflow_text:
            if token == "pull_request_target":
                errors.append(f"{workflow_path} must not use pull_request_target")
            elif token in {"paths:", "paths-ignore:"}:
                errors.append(f"{workflow_path} must not use workflow-level paths filters")
            else:
                errors.append(f"{workflow_path} must not contain forbidden token '{token}'")

    errors.extend(_check_supply_chain_run_blocks(workflow_text, workflow_path))
    errors.extend(_check_supply_chain_script_env_overrides(workflow_lines, workflow_path))
    errors.extend(_check_supply_chain_concurrency(workflow_lines, workflow_path))
    errors.extend(_check_supply_chain_on_block(workflow_lines, workflow_path))
    errors.extend(_check_supply_chain_permissions_block(workflow_lines, workflow_path))
    errors.extend(_check_supply_chain_forbidden_lines(workflow_lines, workflow_path))

    if errors:
        fail(errors)

    print(f"supply-chain-workflow: required guardrails found in {workflow_path}")
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

    supply_chain_parser = subparsers.add_parser(
        "supply-chain-workflow",
        help="Validate required supply-chain audit workflow guardrail tokens",
    )
    supply_chain_parser.add_argument("--file", required=True, help="Workflow file to inspect")
    supply_chain_parser.set_defaults(func=validate_supply_chain_workflow)

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
