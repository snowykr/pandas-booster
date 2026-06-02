from __future__ import annotations

import subprocess
from pathlib import Path


def test_supply_chain_audit_workflow_runs_high_signal_pr_scan() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow_path = repo_root / ".github" / "workflows" / "supply-chain-audit.yml"

    workflow_text = workflow_path.read_text(encoding="utf-8")

    assert "name: Supply Chain Audit" in workflow_text
    assert "pull_request:" in workflow_text
    assert "pull-requests: write" in workflow_text
    assert "contents: read" in workflow_text
    assert "$RUNNER_TEMP/supply_chain_audit.sh" in workflow_text
    assert "bash scripts/supply_chain_audit.sh" not in workflow_text
    assert "gh pr comment" in workflow_text
    assert "--body-file" in workflow_text
    assert "Fail on critical findings" in workflow_text
    assert "BASE_SHA:" in workflow_text
    assert "HEAD_SHA:" in workflow_text
    assert "PR_NUMBER:" in workflow_text
    assert "$RUNNER_TEMP/findings.md" in workflow_text
    assert "paths:" not in workflow_text
    assert "pull_request_target" not in workflow_text
    assert 'gh pr comment "${{' not in workflow_text


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


def test_supply_chain_audit_script_flags_critical_python_payloads(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "supply_chain_audit.sh"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "sitecustomize.py").write_text("# import-time hook\n", encoding="utf-8")
    encoded_call = (
        "exec("
        "base64.b64decode('cHJpbnQoImhpIik=')"
        ")"
    )
    (repo / "payload.py").write_text(
        f"import base64\n{encoded_call}\n",
        encoding="utf-8",
    )
    (repo / "danger.pth").write_text("import payload\n", encoding="utf-8")
    head = _commit_all(repo, "add critical payload")

    result = subprocess.run(
        [str(script_path), base, head, str(findings_path)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    findings = findings_path.read_text(encoding="utf-8")
    assert result.returncode == 1
    assert "CRITICAL: .pth file added or modified" in findings
    assert "CRITICAL: base64 decode + exec/eval combo" in findings
    assert "CRITICAL: Install-hook file added or modified" in findings
    assert "danger.pth" in findings
    assert "sitecustomize.py" in findings


def test_supply_chain_audit_script_rejects_invalid_revisions(tmp_path: Path) -> None:
    repo, _base = _init_audit_repo(tmp_path)
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "supply_chain_audit.sh"
    findings_path = tmp_path / "findings.md"

    result = subprocess.run(
        [str(script_path), "invalid-base", "invalid-head", str(findings_path)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert "could not diff revisions" in result.stderr
    assert not findings_path.exists()


def test_supply_chain_audit_script_ignores_lockfile_only_matches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script_path = repo_root / "scripts" / "supply_chain_audit.sh"
    repo, base = _init_audit_repo(tmp_path)
    findings_path = tmp_path / "findings.md"

    (repo / "uv.lock").write_text(
        "exec("
        "base64.b64decode('cHJpbnQoImhpIik=')"
        ")\n",
        encoding="utf-8",
    )
    head = _commit_all(repo, "add lockfile text")

    result = subprocess.run(
        [str(script_path), base, head, str(findings_path)],
        cwd=repo,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert not findings_path.exists()
