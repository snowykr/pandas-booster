"""Benchmark command-line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Final

from profile_json import collect_stats_evidence
from profile_json_payload import save_profile_json
from reporting import SUPPORTED_AGGS, save_results_md
from runner import benchmark_worker, run_benchmarks

JsonScalar = None | bool | int | float | str
JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
_NOFOLLOW_FLAG: Final[int] = getattr(os, "O_NOFOLLOW", 0)


def _parse_worker_args(payload: str) -> dict[str, JsonValue]:
    parsed_payload = json.loads(payload)
    if not isinstance(parsed_payload, dict):
        raise ValueError("worker payload must be a JSON object")
    return parsed_payload


def _validate_worker_output_file(output_file: JsonValue) -> Path:
    if not isinstance(output_file, str):
        raise ValueError("output_file must be a string path")

    output_path = Path(output_file)
    if output_path.is_symlink():
        raise ValueError("output_file must not be a symlink")

    temp_root = Path(tempfile.gettempdir()).resolve(strict=True)
    resolved_parent = output_path.parent.resolve(strict=True)
    if resolved_parent != temp_root:
        raise ValueError("output_file must be directly inside the system temp directory")

    return output_path


def _write_worker_output(output_path: Path, result: Any) -> None:
    open_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | _NOFOLLOW_FLAG
    file_descriptor = os.open(output_path, open_flags, 0o600)
    with os.fdopen(file_descriptor, "w", encoding="utf-8") as file_handle:
        json.dump(result, file_handle)


def main(
    *,
    run_benchmarks_func=run_benchmarks,
    benchmark_worker_func=benchmark_worker,
    collect_stats_evidence_func=collect_stats_evidence,
    save_results_md_func=save_results_md,
    save_profile_json_func=save_profile_json,
):
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Official pandas-booster benchmarks with Cold/Warm measurement",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmarks/benchmark.py
      # Run default benchmarks (cardinality=all, diagnostic=none)
  python benchmarks/benchmark.py --cardinality all
      # Run core benchmarks only (standard + high)
  python benchmarks/benchmark.py --cardinality standard             # Standard only
  python benchmarks/benchmark.py --cardinality high                 # High only
  python benchmarks/benchmark.py --agg std --agg var                # Run only std/var benchmarks
  python benchmarks/benchmark.py --agg median                       # Run only median benchmarks
  python benchmarks/benchmark.py --agg prod                         # Run only product benchmarks
  python benchmarks/benchmark.py --agg min --agg max               # Run only min/max benchmarks
  python benchmarks/benchmark.py --diagnostic threshold --sort-mode unsorted
      # Add threshold diagnostics
  python benchmarks/benchmark.py --cardinality all --diagnostic threshold --sort-mode unsorted
      # Core + diagnostics
  python benchmarks/benchmark.py --sort-mode sorted                 # Sorted only
  python benchmarks/benchmark.py --cardinality high --sort-mode unsorted  # Combine
  python benchmarks/benchmark.py --output benchmarks/reports        # Save per-aggregation reports
  python benchmarks/benchmark.py --samples 10                       # Adjust sample count

Environment:
  PANDAS_BOOSTER_FORCE_PANDAS_SORT=1  # force Python sort_index() (panic button)
  PANDAS_BOOSTER_FORCE_PANDAS_SORT=0  # use Rust-side sorting (default)
        """,
    )
    parser.add_argument(
        "--cardinality",
        choices=["all", "standard", "high"],
        default="all",
        help="Workload cardinality suite to run (default: all = standard + high)",
    )
    parser.add_argument(
        "--diagnostic",
        choices=["none", "threshold"],
        default="none",
        help="Internal diagnostic suite to add (default: none)",
    )
    parser.add_argument(
        "--sort-mode",
        choices=["all", "sorted", "unsorted"],
        default="all",
        help="Which sort mode to run (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per benchmark (applies to both cold and warm, default: 5)",
    )
    parser.add_argument(
        "--agg",
        dest="aggs",
        action="append",
        choices=SUPPORTED_AGGS,
        help=(
            "Aggregation function to benchmark. Repeatable. "
            "Defaults to core=sum for benchmark reports; profile JSON defaults to std/var evidence."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help=(
            "Output directory for per-aggregation Markdown reports (e.g., benchmarks/reports). "
            "Existing README.md/<agg>.md files must already be benchmark-generated or the command "
            "will fail to avoid overwriting user files."
        ),
    )
    parser.add_argument(
        "--profile-json",
        type=str,
        help="Internal-use profile JSON output path for single-key std/var evidence",
    )
    parser.add_argument(
        "--worker",
        type=str,
        help="Worker mode (internal use): JSON args for single benchmark run",
    )

    args = parser.parse_args()

    if args.diagnostic == "threshold" and args.sort_mode != "unsorted":
        parser.error(
            "--diagnostic threshold requires --sort-mode unsorted "
            "(threshold diagnostics are sort=False boundary checks)"
        )

    # Default to Rust-side sorting for sort=True benchmarks unless explicitly forced off.
    if not args.worker:
        os.environ.setdefault("PANDAS_BOOSTER_FORCE_PANDAS_SORT", "0")
        print(
            "PANDAS_BOOSTER_FORCE_PANDAS_SORT="
            f"{os.environ.get('PANDAS_BOOSTER_FORCE_PANDAS_SORT')}",
            file=sys.stderr,
        )
        print(
            "Note: benchmarks default Rust-side sort=True unless you set "
            "PANDAS_BOOSTER_FORCE_PANDAS_SORT=1",
            file=sys.stderr,
        )

    if args.worker:
        worker_args = _parse_worker_args(args.worker)
        output_file = worker_args.pop("output_file", None)
        output_path = (
            _validate_worker_output_file(output_file)
            if output_file is not None
            else None
        )

        try:
            result = benchmark_worker_func(**worker_args)
        except Exception as e:
            raise e

        if output_path is not None:
            _write_worker_output(output_path, result)
        else:
            print(json.dumps(result))
        return

    all_results = run_benchmarks_func(
        cardinality=args.cardinality,
        diagnostic=args.diagnostic,
        sort_mode=args.sort_mode,
        n_samples=args.samples,
        aggs=args.aggs,
    )

    profile_stats_evidence: list[dict[str, Any]] = []
    if args.profile_json:
        profile_stats_evidence = collect_stats_evidence_func(
            args.samples,
            args.cardinality,
            args.sort_mode,
            args.aggs,
        )

    if args.output:
        result_aggs = {result["agg"] for result in all_results}
        output_stats_evidence = [
            item for item in profile_stats_evidence if item["agg"] in result_aggs
        ]
        if not args.profile_json and args.aggs is not None:
            output_stats_evidence = collect_stats_evidence_func(
                args.samples,
                args.cardinality,
                args.sort_mode,
                args.aggs,
            )

        save_results_md_func(
            all_results,
            output_stats_evidence,
            args.output,
            args.sort_mode,
            args.cardinality,
            args.diagnostic,
        )

    if args.profile_json:
        save_profile_json_func(
            profile_stats_evidence,
            args.profile_json,
            cardinality=args.cardinality,
            sort_mode=args.sort_mode,
            n_samples=args.samples,
            selected_aggs=args.aggs,
        )

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)
    print(f"Total benchmarks run: {len(all_results)}")

    return all_results
