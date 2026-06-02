from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_supply_chain_workflow_uses_native_scripts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / "supply-chain-audit.yml"

    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "name: Supply Chain Risk Guard" in workflow_text
    assert "pull_request:" in workflow_text
    assert "pull-requests: write" in workflow_text
    assert "contents: read" in workflow_text
    assert "Setup Python" in workflow_text
    assert "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5" in workflow_text
    assert "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065" in workflow_text
    assert "actions/checkout@v4" not in workflow_text
    assert "actions/setup-python@v5" not in workflow_text
    assert "python \"$AUDIT_SCRIPT\" scan" in workflow_text
    assert "python \"$COMMENT_SCRIPT\" sync" in workflow_text
    assert "AUDIT_SCRIPT=" in workflow_text
    assert "COMMENT_SCRIPT=" in workflow_text
    assert "scripts/supply_chain_audit.py" in workflow_text
    assert "scripts/supply_chain_comment.py" in workflow_text
    assert "pandas-booster:supply-chain-risk-guard" in workflow_text
    assert "git show \"$BASE_SHA:scripts/supply_chain_audit.py\"" in workflow_text
    assert "git show \"$BASE_SHA:scripts/supply_chain_comment.py\"" in workflow_text
    assert "Base branch does not contain supply-chain risk guard scripts." in workflow_text
    assert "cp scripts/supply_chain_audit.py" not in workflow_text
    assert "cp scripts/supply_chain_comment.py" not in workflow_text
    legacy_script = "supply_chain" + "_audit.sh"
    assert f'cat > "$RUNNER_TEMP/{legacy_script}"' not in workflow_text
    assert f"$RUNNER_TEMP/{legacy_script}" not in workflow_text
    assert f"bash scripts/{legacy_script}" not in workflow_text
    assert "credential-" + "stealing" not in workflow_text
    assert "lite" + "llm" not in workflow_text.lower()
    assert "gh pr comment" not in workflow_text
    assert "Fail on risk findings" in workflow_text
    assert "BASE_SHA:" in workflow_text
    assert "HEAD_SHA:" in workflow_text
    assert "PR_NUMBER:" in workflow_text
    assert "$RUNNER_TEMP/findings.md" in workflow_text
    assert "paths:" not in workflow_text
    assert "pull_request_target" not in workflow_text


def test_supply_chain_workflow_cancels_stale_pr_runs() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / "supply-chain-audit.yml"

    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "concurrency:" in workflow_text
    assert "group: supply-chain-risk-guard-${{ github.event.pull_request.number }}" in workflow_text
    assert "cancel-in-progress: true" in workflow_text


def test_supply_chain_workflow_never_reads_github_event_inside_run_blocks() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / "supply-chain-audit.yml"

    workflow_text = workflow_path.read_text(encoding="utf-8")

    for block in _workflow_run_blocks(workflow_text):
        assert "github.event" not in block


def test_native_supply_chain_scanner_flags_python_execution_risks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "sitecustomize.py").write_text("# startup hook\n", encoding="utf-8")
    (repo / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    encoded_call = _encoded_exec_call()
    (repo / "payload.py").write_text(
        f"import base64\n{encoded_call}\n",
        encoding="utf-8",
    )
    (repo / "danger.pth").write_text("import payload\n", encoding="utf-8")
    head = _commit_all(repo, "add execution risks")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "found=true" in result.stdout
    assert "Python startup hook" in findings
    assert "Encoded dynamic execution" in findings
    assert "Package install hook" in findings
    assert "danger.pth" in findings
    assert "sitecustomize.py" in findings
    assert "credential-" + "stealing" not in findings
    assert "lite" + "llm" not in findings.lower()


def test_native_supply_chain_scanner_ignores_lockfile_only_matches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "uv.lock").write_text(
        _encoded_exec_call() + "\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add lockfile text")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    assert result.returncode == 0
    assert "found=false" in result.stdout
    assert not findings_path.exists()


def test_native_supply_chain_scanner_rejects_invalid_revisions(tmp_path: Path) -> None:
    repo, _base = _init_audit_repo(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    findings_path = tmp_path / "findings.md"

    result = _run_audit(scanner_path, repo, "invalid-base", "invalid-head", findings_path)

    assert result.returncode == 2
    assert "could not diff revisions" in result.stderr
    assert not findings_path.exists()


def test_native_supply_chain_scanner_flags_encoded_subprocess_calls(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "runner.py").write_text(
        "\n".join(
            [
                "import subprocess",
                _encoded_subprocess_call("run"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add encoded subprocess")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "found=true" in result.stdout
    assert "Encoded subprocess command" in findings
    assert "subprocess.run" in findings


def test_native_supply_chain_scanner_ignores_deleted_startup_hooks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, _initial = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "sitecustomize.py").write_text("# existing hook\n", encoding="utf-8")
    base = _commit_all(repo, "add existing hook")
    (repo / "sitecustomize.py").unlink()
    head = _commit_all(repo, "delete existing hook")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    assert result.returncode == 0
    assert "found=false" in result.stdout
    assert not findings_path.exists()


def test_native_supply_chain_scanner_flags_renamed_startup_hooks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, _initial = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "ordinary.txt").write_text("import payload\n", encoding="utf-8")
    base = _commit_all(repo, "add ordinary file")
    _run_git(repo, "mv", "ordinary.txt", "danger.pth")
    head = _commit_all(repo, "rename to startup hook")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Python startup hook" in findings
    assert "danger.pth" in findings


def test_native_supply_chain_scanner_flags_renamed_root_package_hooks(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, _initial = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "ordinary.txt").write_text("from setuptools import setup\n", encoding="utf-8")
    base = _commit_all(repo, "add ordinary file")
    _run_git(repo, "mv", "ordinary.txt", "setup.py")
    head = _commit_all(repo, "rename to package hook")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Package install hook" in findings
    assert "setup.py" in findings


def test_native_supply_chain_scanner_flags_root_package_hooks(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "setup.py").write_text("from setuptools import setup\n", encoding="utf-8")
    (repo / "setup.cfg").write_text("[metadata]\nname = sample\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "setup.py").write_text("ignored nested setup\n", encoding="utf-8")
    head = _commit_all(repo, "add package hooks")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Package install hook" in findings
    assert "setup.py" in findings
    assert "setup.cfg" in findings
    assert "pkg/setup.py" not in findings


def test_native_supply_chain_scanner_flags_usercustomize_and_init_pth(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "usercustomize.py").write_text("# user startup hook\n", encoding="utf-8")
    (repo / "__init__.pth").write_text("import payload\n", encoding="utf-8")
    head = _commit_all(repo, "add startup hooks")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Python startup hook" in findings
    assert "usercustomize.py" in findings
    assert "__init__.pth" in findings


def test_native_supply_chain_scanner_flags_encoded_eval_and_character_indicators(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "payload.py").write_text(
        "\n".join(
            [
                _encoded_eval_call(),
                _hex_exec_call(),
                _chr_exec_call(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add encoded eval variants")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Encoded dynamic execution" in findings
    assert "eval(base" + "64.b64decode" in findings
    assert "\\x70\\x72\\x69\\x6e\\x74" in findings
    assert "exec(" + "ch" + "r(112)" in findings


def test_native_supply_chain_scanner_flags_encoded_popen_call_and_plain_base64(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "runner.py").write_text(
        "\n".join(
            [
                "import subprocess",
                _encoded_subprocess_call("Popen"),
                _plain_base64_subprocess_call(),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add encoded subprocess variants")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "Encoded subprocess command" in findings
    assert "subprocess.Popen" in findings
    assert "subprocess.call" in findings
    assert "ZWNobyBoaQ==" in findings


def test_native_supply_chain_scanner_allows_ordinary_non_lock_changes(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    scanner_path = repo_root / "scripts" / "supply_chain_audit.py"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "src").mkdir()
    (repo / "src" / "ordinary.py").write_text(
        "def add(left: int, right: int) -> int:\n    return left + right\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add ordinary source")

    result = _run_audit(scanner_path, repo, base, head, findings_path)

    assert result.returncode == 0
    assert "found=false" in result.stdout
    assert not findings_path.exists()


def test_supply_chain_comment_warns_when_api_and_create_fail(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    comment_path = repo_root / "scripts" / "supply_chain_comment.py"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh_path = fake_bin / "gh"
    gh_path.write_text("\n".join(_failing_gh_script_lines()) + "\n", encoding="utf-8")
    gh_path.chmod(0o755)
    findings_path = tmp_path / "findings.md"
    findings_path.write_text("### Encoded dynamic execution\npayload.py:2\n", encoding="utf-8")
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "snowykr/pandas-booster",
    }

    result = subprocess.run(
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

    assert result.returncode == 0
    assert "::warning::Could not read existing supply-chain risk guard comments" in result.stderr
    assert "::warning::Could not create supply-chain risk guard comment" in result.stderr


def test_supply_chain_comment_warns_when_patch_fails(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    comment_path = repo_root / "scripts" / "supply_chain_comment.py"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    gh_path = fake_bin / "gh"
    gh_path.write_text("\n".join(_patch_failing_gh_script_lines()) + "\n", encoding="utf-8")
    gh_path.chmod(0o755)
    findings_path = tmp_path / "findings.md"
    findings_path.write_text("### Encoded dynamic execution\npayload.py:2\n", encoding="utf-8")
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": "snowykr/pandas-booster",
    }

    result = subprocess.run(
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

    assert result.returncode == 0
    assert "::warning::Could not update supply-chain risk guard comment" in result.stderr


def test_supply_chain_comment_lifecycle_updates_existing_alert_to_resolved(
    tmp_path: Path,
) -> None:
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
    findings_path.write_text("### Encoded dynamic execution\npayload.py:2\n", encoding="utf-8")
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
    resolved = subprocess.run(
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
    assert resolved.returncode == 0
    assert "No Current Supply-chain Risk Findings" in body
    assert "Previous findings were resolved for this head." in body
    assert "Active Supply-chain Risk Findings" not in body
    assert (state_dir / "action").read_text(encoding="utf-8").strip() == "patch"


def test_supply_chain_comment_lifecycle_skips_clear_without_existing_alert(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    comment_path = repo_root / "scripts" / "supply_chain_comment.py"
    fake_bin = tmp_path / "bin"
    state_dir = tmp_path / "state"
    fake_bin.mkdir()
    state_dir.mkdir()
    gh_path = fake_bin / "gh"
    gh_path.write_text("\n".join(_fake_gh_script_lines()) + "\n", encoding="utf-8")
    gh_path.chmod(0o755)
    env = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "STATE_DIR": str(state_dir),
        "GITHUB_REPOSITORY": "snowykr/pandas-booster",
    }

    result = subprocess.run(
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

    assert result.returncode == 0
    assert "no existing supply-chain risk guard comment" in result.stdout
    assert not (state_dir / "action").exists()


def _run_git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, text=True, capture_output=True)


def _commit_all(repo: Path, message: str) -> str:
    _run_git(repo, "add", ".")
    _run_git(repo, "commit", "-m", message)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()


def _init_audit_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "audit-repo"
    repo.mkdir()
    _run_git(repo, "init")
    _run_git(repo, "config", "user.email", "test@example.com")
    _run_git(repo, "config", "user.name", "Test User")
    (repo / "README.md").write_text("baseline\n", encoding="utf-8")
    return repo, _commit_all(repo, "baseline")


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


def _encoded_exec_call() -> str:
    decoder = "base" + "64.b64decode"
    return f"exec({decoder}('cHJpbnQoImhpIik='))"


def _encoded_eval_call() -> str:
    decoder = "base" + "64.b64decode"
    return f"eval({decoder}('cHJpbnQoImhpIik='))"


def _hex_exec_call() -> str:
    encoded_print = "\\x70\\x72\\x69\\x6e\\x74"
    return f"exec('{encoded_print}(1)')"


def _chr_exec_call() -> str:
    chr_call = "chr"
    return (
        f"exec({chr_call}(112) + {chr_call}(114) + {chr_call}(105) + "
        f"{chr_call}(110) + {chr_call}(116))"
    )


def _encoded_subprocess_call(method: str) -> str:
    decoder = "base" + "64.b64decode"
    return f"subprocess.{method}({decoder}('ZWNobyBoaQ=='), shell=True)"


def _plain_base64_subprocess_call() -> str:
    encoded_command = "base" + "64 ZWNobyBoaQ== | sh"
    return f"subprocess.call('{encoded_command}', shell=True)"


def _workflow_run_blocks(workflow_text: str) -> list[str]:
    blocks: list[str] = []
    lines = workflow_text.splitlines()
    for index, line in enumerate(lines):
        if line != "        run: |":
            continue

        block_lines: list[str] = []
        for block_line in lines[index + 1 :]:
            if not block_line.startswith("          "):
                break
            block_lines.append(block_line)
        blocks.append("\n".join(block_lines))
    return blocks


def _fake_gh_script_lines() -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'echo "gh $*" >> "$STATE_DIR/gh.log"',
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


def _failing_gh_script_lines() -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "exit 7",
    ]


def _patch_failing_gh_script_lines() -> list[str]:
    return [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'if [ "$1" = "api" ] && [ "${2:-}" = "--paginate" ]; then echo 321; exit 0; fi',
        "exit 8",
    ]
