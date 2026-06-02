from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

LOCKFILE_NAMES: Final[frozenset[str]] = frozenset(
    {"uv.lock", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"}
)
PYTHON_STARTUP_FILES: Final[frozenset[str]] = frozenset(
    {"sitecustomize.py", "usercustomize.py", "__init__.pth"}
)
INSTALL_HOOK_FILES: Final[frozenset[str]] = frozenset({"setup.py", "setup.cfg"})
ENCODED_TEXT_RE: Final[re.Pattern[str]] = re.compile(
    r"base64\.(?:b64decode|decodebytes|urlsafe_b64decode)|\\x[0-9a-f]{2}|chr\(",
    re.IGNORECASE,
)
DYNAMIC_EXEC_RE: Final[re.Pattern[str]] = re.compile(r"\b(?:exec|eval)\s*\(")
SUBPROCESS_RE: Final[re.Pattern[str]] = re.compile(r"\bsubprocess\.(?:Popen|call|run)\s*\(")
SUBPROCESS_PLAIN_BASE64_RE: Final[re.Pattern[str]] = re.compile(r"\bbase64\b", re.IGNORECASE)


@dataclass(frozen=True)
class Finding:
    title: str
    details: tuple[str, ...]

    def to_markdown(self) -> str:
        body = [f"### {self.title}", "", "```"]
        body.extend(self.details)
        body.extend(["```", ""])
        return "\n".join(body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan PR diffs for pandas-booster supply risks")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan_parser = subparsers.add_parser("scan", help="Scan a git range")
    scan_parser.add_argument("--base", required=True, help="Base commit SHA")
    scan_parser.add_argument("--head", required=True, help="Head commit SHA")
    scan_parser.add_argument("--findings", required=True, type=Path, help="Markdown output path")
    scan_parser.set_defaults(func=scan_command)
    return parser


def scan_command(args: argparse.Namespace) -> int:
    result = scan_range(base=args.base, head=args.head)
    if isinstance(result, ScanError):
        args.findings.unlink(missing_ok=True)
        print(result.message, file=sys.stderr)
        return 2

    findings = result
    if findings:
        findings_text = "\n".join(finding.to_markdown() for finding in findings)
        args.findings.write_text(findings_text, encoding="utf-8")
        print("found=true")
        return 1

    args.findings.unlink(missing_ok=True)
    print("found=false")
    return 0


@dataclass(frozen=True)
class ScanError:
    message: str


def scan_range(base: str, head: str) -> tuple[Finding, ...] | ScanError:
    changed_files = git_lines("diff", "--name-only", "--diff-filter=AMR", f"{base}..{head}")
    if changed_files.returncode != 0:
        return ScanError(f"could not diff revisions: {base}..{head}")

    diff = git_lines(
        "diff",
        f"{base}..{head}",
        "--",
        ".",
        ":(exclude)uv.lock",
        ":(exclude)*.lock",
        ":(exclude)package-lock.json",
        ":(exclude)yarn.lock",
        ":(exclude)pnpm-lock.yaml",
    )
    if diff.returncode != 0:
        return ScanError(f"could not diff revisions: {base}..{head}")

    files = tuple(line for line in changed_files.stdout.splitlines() if line)
    added_lines = tuple(extract_added_lines(diff.stdout))
    findings: list[Finding] = []

    startup_files = tuple(path for path in files if is_python_startup_file(path))
    if startup_files:
        findings.append(Finding("Python startup hook", startup_files))

    package_hooks = tuple(path for path in files if is_package_hook_file(path))
    if package_hooks:
        findings.append(Finding("Package install hook", package_hooks))

    dynamic_exec_hits = tuple(
        line
        for line in added_lines
        if ENCODED_TEXT_RE.search(line) and DYNAMIC_EXEC_RE.search(line)
    )
    if dynamic_exec_hits:
        findings.append(Finding("Encoded dynamic execution", dynamic_exec_hits[:10]))

    subprocess_hits = tuple(
        line
        for line in added_lines
        if SUBPROCESS_RE.search(line)
        and (ENCODED_TEXT_RE.search(line) or SUBPROCESS_PLAIN_BASE64_RE.search(line))
    )
    if subprocess_hits:
        findings.append(Finding("Encoded subprocess command", subprocess_hits[:10]))

    return tuple(findings)


def git_lines(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def is_python_startup_file(path: str) -> bool:
    name = Path(path).name
    return name.endswith(".pth") or name in PYTHON_STARTUP_FILES


def is_package_hook_file(path: str) -> bool:
    normalized = Path(path)
    return str(normalized) in INSTALL_HOOK_FILES


def extract_added_lines(diff_text: str) -> list[str]:
    added: list[str] = []
    current_path = ""
    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_path = raw_line.removeprefix("+++ b/")
            continue
        if current_path and is_lockfile_path(current_path):
            continue
        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            added.append(raw_line[1:])
    return added


def is_lockfile_path(path: str) -> bool:
    name = Path(path).name
    return name in LOCKFILE_NAMES or name.endswith(".lock")


if __name__ == "__main__":
    raise SystemExit(main())
