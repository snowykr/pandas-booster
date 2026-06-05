from __future__ import annotations

import os
import subprocess
import sys
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
    _write_fake_gh(fake_bin)
    findings_path = tmp_path / "findings.md"
    findings_path.write_text("### Encoded subprocess command\npayload.py:1\n", encoding="utf-8")
    env = _comment_env(
        fake_bin,
        STATE_DIR=str(state_dir),
        GITHUB_REPOSITORY="snowykr/pandas-booster",
    )

    finding = subprocess.run(
        [
            sys.executable,
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
            sys.executable,
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
            sys.executable,
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


def _comment_env(fake_bin: Path, **overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(fake_bin), env.get("PATH", "")])
    env["PANDAS_BOOSTER_GH"] = str(fake_bin / "gh.py")
    env.update(overrides)
    return env


def _write_fake_gh(fake_bin: Path) -> None:
    script_path = fake_bin / "gh.py"
    script_path.write_text(_fake_gh_python_script(), encoding="utf-8")

    posix_wrapper = fake_bin / "gh"
    posix_wrapper.write_text(
        f'#!/usr/bin/env sh\nexec "{sys.executable}" "{script_path}" "$@"\n',
        encoding="utf-8",
    )
    posix_wrapper.chmod(0o755)

    windows_wrapper = fake_bin / "gh.cmd"
    windows_wrapper.write_text(
        f'@echo off\r\n"{sys.executable}" "{script_path}" %*\r\n',
        encoding="utf-8",
    )


def _fake_gh_python_script() -> str:
    lines = [
        "from __future__ import annotations",
        "",
        "import json",
        "import os",
        "import shutil",
        "import sys",
        "from pathlib import Path",
        "",
        'MARKER = "pandas-booster:supply-chain-risk-guard"',
        "",
        "",
        "def main() -> int:",
        "    args = sys.argv[1:]",
        '    state_dir = Path(os.environ["STATE_DIR"])',
        '    if args[:2] == ["api", "--paginate"]:',
        '        comment_path = state_dir / "comment.md"',
        '        if comment_path.exists() and MARKER in comment_path.read_text(encoding="utf-8"):',
        '            print("321")',
        "        return 0",
        '    if len(args) >= 5 and args[:2] == ["pr", "comment"]:',
        '        shutil.copyfile(args[4], state_dir / "comment.md")',
        '        (state_dir / "action").write_text("create\\n", encoding="utf-8")',
        "        return 0",
        '    if len(args) >= 6 and args[:3] == ["api", "--method", "PATCH"]:',
        '        body = json.loads(Path(args[5]).read_text(encoding="utf-8"))["body"]',
        '        (state_dir / "comment.md").write_text(body, encoding="utf-8")',
        '        (state_dir / "action").write_text("patch\\n", encoding="utf-8")',
        "        return 0",
        "    return 9",
        "",
        "",
        'if __name__ == "__main__":',
        "    raise SystemExit(main())",
    ]
    return "\n".join(lines) + "\n"
