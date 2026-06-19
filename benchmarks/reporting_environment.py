from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


def collect_benchmark_environment(elapsed_seconds: float) -> tuple[str, ...]:
    return (
        "## Environment & Configuration",
        "",
        "The following environment was used to generate the checked-in benchmark reports.",
        "",
        "- **Build Mode**: Release (`maturin develop --release`)",
        f"- **Machine**: {_format_machine()}",
        f"- **Threading**: {_format_threading()}",
        f"- **OS**: {_format_os()}",
        f"- **Python**: {platform.python_version()}",
        f"- **Pandas**: {_package_version('pandas')}",
        f"- **Polars**: {_package_version('polars')}",
        f"- **Benchmark Duration**: {format_benchmark_duration(elapsed_seconds)}",
    )


def format_benchmark_duration(elapsed_seconds: float) -> str:
    total_seconds = max(0, int(round(elapsed_seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        compact = f"{hours}h {minutes}m {seconds}s"
    elif minutes:
        compact = f"{minutes}m {seconds}s"
    else:
        compact = f"{seconds}s"
    return (
        f"{compact} wall-clock ({total_seconds:,} seconds), "
        "measured around generate_docs.py only"
    )


def _format_machine() -> str:
    fields = _lscpu_fields()
    model = fields.get("Model name") or platform.processor() or "unknown CPU"
    threads = _parse_int(fields.get("CPU(s)"))
    cores = _physical_core_count(fields)
    ram = _format_ram()
    cpu_detail = _format_cpu_detail(cores, threads)
    details = [model]
    if cpu_detail:
        details.append(cpu_detail)
    if ram:
        details.append(f"{ram} RAM")
    return ", ".join(details)


def _format_threading() -> str:
    rayon_threads = os.environ.get("RAYON_NUM_THREADS")
    if rayon_threads:
        return f"RAYON_NUM_THREADS={rayon_threads}"
    return "Default Rayon behavior (RAYON_NUM_THREADS unset; uses available logical cores)"


def _format_os() -> str:
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


def _package_version(package_name: str) -> str:
    try:
        package = __import__(package_name)
    except Exception as exc:  # pragma: no cover - depends on optional local env
        return f"unavailable ({exc!r})"
    return getattr(package, "__version__", "unknown")


def _lscpu_fields() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["lscpu"],
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, ValueError):
        return {}
    if completed.returncode != 0:
        return {}
    fields: dict[str, str] = {}
    for line in completed.stdout.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()
    return fields


def _physical_core_count(fields: dict[str, str]) -> int | None:
    cores_per_socket = _parse_int(fields.get("Core(s) per socket"))
    sockets = _parse_int(fields.get("Socket(s)"))
    if cores_per_socket and sockets:
        return cores_per_socket * sockets
    return None


def _format_cpu_detail(cores: int | None, threads: int | None) -> str | None:
    if cores and threads:
        if cores == threads:
            return f"{cores} CPU cores"
        return f"{cores} CPU cores / {threads} threads"
    if threads:
        return f"{threads} logical CPUs"
    if cores:
        return f"{cores} CPU cores"
    return None


def _format_ram() -> str | None:
    meminfo = Path("/proc/meminfo")
    try:
        text = meminfo.read_text(encoding="utf-8")
    except OSError:
        return None
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB$", text, flags=re.MULTILINE)
    if not match:
        return None
    gib = int(match.group(1)) / (1024 * 1024)
    if gib >= 10:
        return f"{gib:.0f} GiB"
    return f"{gib:.1f} GiB"


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None
