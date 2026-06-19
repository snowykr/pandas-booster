from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from reporting import (  # noqa: E402
    BENCHMARK_INDEX_FILENAME,
    SUPPORTED_AGGS,
    collect_benchmark_environment,
    format_benchmark_index,
    is_generated_benchmark_report,
    render_generated_markdown,
    write_generated_report,
)

BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks" / "benchmark.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "reports"


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--cardinality",
        args.cardinality,
        "--diagnostic",
        args.diagnostic,
        "--sort-mode",
        args.sort_mode,
        "--samples",
        str(args.samples),
        "--output",
        str(args.output),
    ]
    for agg in SUPPORTED_AGGS:
        command.extend(("--agg", agg))
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-aggregation benchmark Markdown reports for all operations."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of cold/warm samples per benchmark case (default: 20).",
    )
    parser.add_argument(
        "--cardinality",
        choices=["all", "standard", "high"],
        default="all",
        help="Workload cardinality suite to run (default: all).",
    )
    parser.add_argument(
        "--diagnostic",
        choices=["none", "threshold"],
        default="none",
        help="Internal diagnostic suite to add (default: none).",
    )
    parser.add_argument(
        "--sort-mode",
        choices=["all", "sorted", "unsorted"],
        default="all",
        help="Which sort mode to run (default: all).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated benchmark Markdown reports (default: benchmarks/reports/).",
    )
    return parser.parse_args(argv)


def refresh_index_environment(output_dir: Path, elapsed_seconds: float) -> None:
    report_dir = output_dir if output_dir.is_absolute() else REPO_ROOT / output_dir
    index_path = report_dir / BENCHMARK_INDEX_FILENAME
    if not is_generated_benchmark_report(index_path):
        raise ValueError(
            "Benchmark index must already be a generated pandas-booster report before "
            "refreshing run environment metadata."
        )
    environment_lines = collect_benchmark_environment(elapsed_seconds)
    write_generated_report(
        index_path,
        render_generated_markdown(
            format_benchmark_index(list(SUPPORTED_AGGS), environment_lines=environment_lines)
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    elapsed_seconds = time.perf_counter() - start
    print(f"Benchmark elapsed seconds: {int(round(elapsed_seconds))}")
    if completed.returncode == 0:
        refresh_index_environment(args.output, elapsed_seconds)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
