from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Final

COMMENT_MARKER: Final[str] = "<!-- pandas-booster:supply-chain-risk-guard -->"
BOT_LOGIN: Final[str] = "github-actions[bot]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize the supply-chain risk guard comment")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sync_parser = subparsers.add_parser("sync", help="Create or update the PR comment")
    sync_parser.add_argument("--pr-number", required=True, help="Pull request number")
    sync_parser.add_argument("--head-sha", required=True, help="Audited head commit")
    sync_parser.add_argument("--state", required=True, choices=("finding", "clear"))
    sync_parser.add_argument("--findings", type=Path, help="Markdown findings file")
    sync_parser.set_defaults(func=sync_command)
    return parser


def sync_command(args: argparse.Namespace) -> int:
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        print("::warning::GITHUB_REPOSITORY is not set; skipping risk guard comment")
        return 0

    existing_id = find_existing_comment(repository=repository, pr_number=args.pr_number)
    if existing_id is None and args.state == "clear":
        print("no existing supply-chain risk guard comment")
        return 0

    body = build_body(state=args.state, head_sha=args.head_sha, findings_path=args.findings)
    if existing_id is None:
        return create_comment(pr_number=args.pr_number, body=body)
    return update_comment(repository=repository, comment_id=existing_id, body=body)


def find_existing_comment(repository: str, pr_number: str) -> str | None:
    query = (
        f'.[] | select(.user.login == "{BOT_LOGIN}" and '
        f'(.body // "" | contains("{COMMENT_MARKER}"))) | .id'
    )
    result = run_gh(
        "api",
        "--paginate",
        f"repos/{repository}/issues/{pr_number}/comments",
        "--jq",
        query,
    )
    if result.returncode != 0:
        warn("Could not read existing supply-chain risk guard comments")
        return None

    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return ids[-1] if ids else None


def build_body(state: str, head_sha: str, findings_path: Path | None) -> str:
    if state == "finding":
        if findings_path is None:
            raise SystemExit("--findings is required when --state=finding")
        findings = findings_path.read_text(encoding="utf-8").strip()
        return "\n".join(
            [
                COMMENT_MARKER,
                "## Active Supply-chain Risk Findings",
                "",
                f"Head: `{head_sha}`",
                "",
                (
                    "This PR changed code paths that can run before ordinary imports "
                    "or execute encoded commands."
                ),
                "",
                findings,
                "",
                "This comment is managed by the pandas-booster supply-chain risk guard.",
                "",
            ]
        )

    return "\n".join(
        [
            COMMENT_MARKER,
            "## No Current Supply-chain Risk Findings",
            "",
            f"Head: `{head_sha}`",
            "",
            "Previous findings were resolved for this head.",
            "",
            "This comment is managed by the pandas-booster supply-chain risk guard.",
            "",
        ]
    )


def create_comment(pr_number: str, body: str) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        body_path = Path(tmp_dir) / "comment.md"
        body_path.write_text(body, encoding="utf-8")
        result = run_gh("pr", "comment", pr_number, "--body-file", str(body_path))
    if result.returncode != 0:
        warn("Could not create supply-chain risk guard comment")
    return 0


def update_comment(repository: str, comment_id: str, body: str) -> int:
    with tempfile.TemporaryDirectory() as tmp_dir:
        payload_path = Path(tmp_dir) / "comment.json"
        payload_path.write_text(json.dumps({"body": body}), encoding="utf-8")
        result = run_gh(
            "api",
            "--method",
            "PATCH",
            f"repos/{repository}/issues/comments/{comment_id}",
            "--input",
            str(payload_path),
        )
    if result.returncode != 0:
        warn("Could not update supply-chain risk guard comment")
    return 0


def run_gh(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["gh", *args],
        check=False,
        text=True,
        capture_output=True,
    )


def warn(message: str) -> None:
    print(f"::warning::{message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
