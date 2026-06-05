"""Shared helpers for release contract tests."""

from __future__ import annotations

from pathlib import Path


def _write_package_init(root: Path, version: str) -> None:
    package_dir = root / "python" / "pandas_booster"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")


def _job_block(workflow_text: str, job_name: str) -> str:
    marker = f"  {job_name}:"
    lines = workflow_text.splitlines()

    try:
        start = lines.index(marker)
    except ValueError as exc:
        jobs_start = next((idx for idx, line in enumerate(lines) if line == "jobs:"), None)
        available_jobs = [
            line.strip().removesuffix(":")
            for line in (lines[jobs_start + 1 :] if jobs_start is not None else [])
            if line.startswith("  ") and not line.startswith("    ") and line.endswith(":")
        ]
        context = f" Available jobs: {', '.join(available_jobs)}." if available_jobs else ""
        raise AssertionError(f"Workflow is missing expected job {job_name!r}.{context}") from exc

    block = [lines[start]]

    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("    "):
            break
        block.append(line)

    return "\n".join(block)


def _job_if_expression(workflow_text: str, job_name: str) -> str:
    block_lines = _job_block(workflow_text, job_name).splitlines()

    for idx, line in enumerate(block_lines):
        if line == "    if: >":
            expr_lines: list[str] = []
            for cont in block_lines[idx + 1 :]:
                if not cont.startswith("      "):
                    break
                expr_lines.append(cont.strip())
            return " ".join(expr_lines)

        if line.startswith("    if: "):
            return line.removeprefix("    if: ").strip()

    raise AssertionError(f"Job {job_name!r} is missing an if expression")


def _current_supply_chain_workflow_text() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root / ".github" / "workflows" / "supply-chain-audit.yml").read_text(
        encoding="utf-8"
    )
