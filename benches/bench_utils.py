"""Common utilities for pandas-booster benchmarks."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import tempfile
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BenchmarkStats:
    """Statistical summary of benchmark samples."""

    mean: float
    std: float
    min: float
    max: float
    samples: list[float]

    def format_ms(self, precision: int = 2) -> str:
        """Format mean ± std (min..max) in milliseconds."""
        ms_mean = self.mean * 1000
        ms_std = self.std * 1000
        ms_min = self.min * 1000
        ms_max = self.max * 1000
        return (
            f"{ms_mean:.{precision}f} ± {ms_std:.{precision}f} "
            f"({ms_min:.{precision}f}..{ms_max:.{precision}f})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.min,
            "max": self.max,
            "samples": self.samples,
        }


def compute_stats(samples: list[float]) -> BenchmarkStats:
    """Compute statistics from a list of samples (in seconds)."""
    if not samples:
        return BenchmarkStats(0.0, 0.0, 0.0, 0.0, [])

    if len(samples) == 1:
        mean = samples[0]
        std = 0.0
        min_val = samples[0]
        max_val = samples[0]
    else:
        mean = statistics.mean(samples)
        std = statistics.stdev(samples)
        min_val = min(samples)
        max_val = max(samples)

    return BenchmarkStats(mean, std, min_val, max_val, samples)


def run_worker_process(
    script_path: Path,
    worker_args: dict[str, Any],
    timeout: int = 300,
) -> dict[str, Any]:
    """Run a worker process and return its JSON output."""

    with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".json") as tmp:
        output_file = tmp.name

    # Add output_file to args so worker knows where to write
    worker_args = worker_args.copy()
    worker_args["output_file"] = output_file

    cmd = [
        sys.executable,
        str(script_path),
        "--worker",
        json.dumps(worker_args),
    ]

    result = None
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True,
            cwd=str(script_path.parent.parent),  # Ensure running from repo root
        )

        try:
            with open(output_file, "r") as f:
                content = f.read()
                if not content.strip():
                    raise json.JSONDecodeError("Empty file", content, 0)
                return json.loads(content)
        except (json.JSONDecodeError, FileNotFoundError):
            lines = result.stdout.strip().splitlines()
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
            raise

    except subprocess.CalledProcessError as e:
        print(f"Worker failed with return code {e.returncode}", file=sys.stderr)
        print(f"Stdout: {e.stdout}", file=sys.stderr)
        print(f"Stderr: {e.stderr}", file=sys.stderr)
        raise
    except json.JSONDecodeError as e:
        print(f"Failed to decode worker output: {e}", file=sys.stderr)
        if result:
            print(f"Stdout: {result.stdout}", file=sys.stderr)
        raise
    finally:
        if os.path.exists(output_file):
            try:
                os.unlink(output_file)
            except OSError:
                pass


def run_cold_warm_benchmark(
    script_path: Path,
    worker_args: dict[str, Any],
    n_samples: int = 5,
    timeout_per_worker: int = 300,
) -> dict[str, Any]:
    """Run benchmark with separate cold and warm measurements.

    Both Cold and Warm stats are calculated from n_samples fresh processes.

    - Cold: Fresh process -> 1st run
    - Warm: Fresh process -> Cold run (discard) -> Warmup run (discard) -> 1st Warm run
    """

    # 1. Collect Cold Samples (Fresh Process x N)
    cold_samples = []
    print(f"  Collecting {n_samples} cold samples...", end="", flush=True)
    for _ in range(n_samples):
        args = worker_args.copy()
        args["mode"] = "cold"
        res = run_worker_process(script_path, args, timeout=timeout_per_worker)
        cold_samples.append(res["cold_time_s"])
        print(".", end="", flush=True)
    print(" Done.")

    # 2. Collect Warm Samples (Fresh Process x N)
    warm_samples = []
    print(f"  Collecting {n_samples} warm samples...", end="", flush=True)
    for _ in range(n_samples):
        args = worker_args.copy()
        args["mode"] = "warm"
        res = run_worker_process(script_path, args, timeout=timeout_per_worker)
        warm_samples.append(res["warm_time_s"])
        print(".", end="", flush=True)
    print(" Done.")

    return {
        "cold_stats": compute_stats(cold_samples),
        "warm_stats": compute_stats(warm_samples),
    }
