from __future__ import annotations

import sys
from pathlib import Path

from ._report_output_helpers import (
    _BENCHMARK_PATH,
    _expected_benchmark_report_aggs,
    _loaded_generate_benchmark_docs_module,
)


def test_generate_benchmark_docs_builds_all_agg_command(tmp_path):
    with _loaded_generate_benchmark_docs_module() as module:
        args = module.parse_args(
            [
                "--samples",
                "1",
                "--cardinality",
                "standard",
                "--sort-mode",
                "sorted",
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        command = module.build_command(args)

    assert command[:2] == [sys.executable, str(_BENCHMARK_PATH)]
    assert command[command.index("--samples") + 1] == "1"
    assert command[command.index("--output") + 1] == str(tmp_path / "reports")
    assert command.count("--agg") == 9
    assert command[-18:] == [
        "--agg",
        "sum",
        "--agg",
        "mean",
        "--agg",
        "median",
        "--agg",
        "prod",
        "--agg",
        "std",
        "--agg",
        "var",
        "--agg",
        "min",
        "--agg",
        "max",
        "--agg",
        "count",
    ]


def test_generate_benchmark_docs_default_command_is_publication_quality_full_generation():
    with _loaded_generate_benchmark_docs_module() as module:
        args = module.parse_args([])
        command = module.build_command(args)

    assert args.samples == 20
    assert args.cardinality == "all"
    assert args.diagnostic == "none"
    assert args.sort_mode == "all"
    assert args.output == module.DEFAULT_OUTPUT_DIR
    assert command[:2] == [sys.executable, str(_BENCHMARK_PATH)]
    assert command[command.index("--cardinality") + 1] == "all"
    assert command[command.index("--diagnostic") + 1] == "none"
    assert command[command.index("--sort-mode") + 1] == "all"
    assert command[command.index("--samples") + 1] == "20"
    assert command[command.index("--output") + 1] == str(module.DEFAULT_OUTPUT_DIR)
    assert [command[index + 1] for index, token in enumerate(command) if token == "--agg"] == list(
        _expected_benchmark_report_aggs()
    )


def test_generate_benchmark_docs_main_returns_subprocess_code(monkeypatch, tmp_path):
    with _loaded_generate_benchmark_docs_module() as module:
        captured_command: list[str] = []
        captured_cwd: Path | None = None
        captured_check: bool | None = None
        expected_repo_root = module.REPO_ROOT

        def fake_run(command, *, cwd, check):
            nonlocal captured_command, captured_cwd, captured_check
            captured_command = list(command)
            captured_cwd = cwd
            captured_check = check

            class Completed:
                returncode = 7

            return Completed()

        monkeypatch.setattr(module.subprocess, "run", fake_run)

        exit_code = module.main(
            [
                "--samples",
                "1",
                "--cardinality",
                "standard",
                "--sort-mode",
                "sorted",
                "--output",
                str(tmp_path / "reports"),
            ]
        )
        assert exit_code == 7
        assert captured_check is False
        assert captured_cwd == expected_repo_root
        assert captured_command[:2] == [sys.executable, str(_BENCHMARK_PATH)]


def test_readme_documents_explicit_smoke_reports_separately_from_default_full_generation():
    readme = (_BENCHMARK_PATH.parent.parent / "README.md").read_text(encoding="utf-8")

    expected_snippets = (
        "# Run the checked-in publication-quality reports for all supported aggregations",
        "python benchmarks/generate_docs.py --samples 20 --cardinality all --sort-mode all",
        "# Run lightweight smoke reports when iterating locally",
        "python benchmarks/generate_docs.py --samples 1 --cardinality standard --sort-mode sorted",
        "# Run default sum benchmark only (standard + high)",
        "python benchmarks/benchmark.py --samples 20 --output benchmarks/reports",
    )

    for expected_snippet in expected_snippets:
        assert expected_snippet in readme
