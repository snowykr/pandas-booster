from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_DIR = Path(__file__).resolve().parent
if str(BENCHMARKS_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS_DIR))

from reporting import SUPPORTED_AGGS  # noqa: E402

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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_command(args)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
