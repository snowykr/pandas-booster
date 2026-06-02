from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_native_audit_script_flags_high_signal_findings(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "sitecustomize.py").write_text("# startup hook\n", encoding="utf-8")
    (repo / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    encoded_command = "base" + "64 ZWNobyBoaQ== | sh"
    subprocess_call = "sub" + f"process.run('{encoded_command}', shell=True)"
    (repo / "payload.py").write_text(
        f"import subprocess\n{subprocess_call}\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add risky files")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "found=true" in result.stdout
    assert "Python startup hook" in findings
    assert "Package install hook" in findings
    assert "Encoded subprocess command" in findings


def test_native_audit_script_ignores_lockfile_only_matches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    lockfile_payload = "exec(base" + "64.b64decode('cHJpbnQoImhpIik='))\n"
    (repo / "uv.lock").write_text(lockfile_payload, encoding="utf-8")
    head = _commit_all(repo, "add lockfile")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    assert result.returncode == 0
    assert "found=false" in result.stdout
    assert not findings_path.exists()


def test_native_comment_script_updates_finding_to_clear(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    comment_path = repo_root / "scripts" / "supply_chain_comment.py"
    fake_bin = tmp_path / "bin"
    state_dir = tmp_path / "state"
    fake_bin.mkdir()
    state_dir.mkdir()
    gh_path = fake_bin / "gh"
    gh_path.write_text("\n".join(_fake_gh_script_lines()) + "\n", encoding="utf-8")
    gh_path.chmod(0o755)
    findings_path = tmp_path / "findings.md"
    findings_path.write_text("### Encoded subprocess command\npayload.py:1\n", encoding="utf-8")
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "STATE_DIR": str(state_dir),
        "GITHUB_REPOSITORY": "snowykr/pandas-booster",
    }

    finding = subprocess.run(
        [
            "python",
            str(comment_path),
            "sync",
            "--pr-number",
            "12",
            "--head-sha",
            "redsha",
            "--state",
            "finding",
            "--findings",
            str(findings_path),
        ],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    clear = subprocess.run(
        [
            "python",
            str(comment_path),
            "sync",
            "--pr-number",
            "12",
            "--head-sha",
            "greensha",
            "--state",
            "clear",
        ],
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    body = (state_dir / "comment.md").read_text(encoding="utf-8")
    assert finding.returncode == 0
    assert clear.returncode == 0
    assert "No Current Supply-chain Risk Findings" in body
    assert "Previous findings were resolved for this head." in body
    assert (state_dir / "action").read_text(encoding="utf-8").strip() == "patch"


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    return repo, _commit_all(repo, "baseline")


def _commit_all(repo: Path, message: str) -> str:
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", message)
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True)


def _run_audit(
    scanner_path: Path,
    repo: Path,
    base: str,
    head: str,
    findings_path: Path,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "python",
            str(scanner_path),
            "scan",
            "--base",
            base,
            "--head",
            head,
            "--findings",
            str(findings_path),
        ],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )


def _fake_gh_script_lines() -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'if [ "$1" = "api" ] && [ "${2:-}" = "--paginate" ]; then',
        '  if [ -f "$STATE_DIR/comment.md" ] && grep -q '
        "'pandas-booster:supply-chain-risk-guard' "
        '"$STATE_DIR/comment.md"; then echo 321; fi',
        "  exit 0",
        "fi",
        'if [ "$1" = "pr" ] && [ "$2" = "comment" ]; then',
        '  cp "$5" "$STATE_DIR/comment.md"; echo create > "$STATE_DIR/action"; exit 0',
        "fi",
        'if [ "$1" = "api" ] && [ "$2" = "--method" ] && [ "$3" = "PATCH" ]; then',
        "  python -c 'import json, pathlib, sys; body = "
        "json.loads(pathlib.Path(sys.argv[1]).read_text())[\"body\"]; "
        "pathlib.Path(sys.argv[2]).write_text(body, encoding=\"utf-8\")' "
        '"$6" "$STATE_DIR/comment.md"',
        '  echo patch > "$STATE_DIR/action"; exit 0',
        "fi",
        "exit 9",
    ]
