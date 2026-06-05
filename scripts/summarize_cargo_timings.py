from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Optional


class SummaryError(Exception):
    pass


@dataclass(frozen=True)
class Sample:
    label: str
    command: str
    raw_time: Path
    timing_html: Optional[Path]  # noqa: UP045 - Python 3.9 syntax compatibility.
    target_dir: str
    cache_state: str
    sample_index: int
    elapsed_seconds: float


@dataclass(frozen=True)
class UnitTiming:
    name: str
    duration_seconds: float
    kind: str


_TIME_RE = re.compile(
    r"(?m)^real\s+(?P<real>\d+(?:\.\d+)?)$\n^user\s+\d+(?:\.\d+)?$\n^sys\s+\d+(?:\.\d+)?$"
)
_UNIT_DATA_MARKER = "const UNIT_DATA = "


def _mapping(value: object, name: str) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    raise SummaryError(f"{name} must be an object")


def _sequence(value: object, name: str) -> list[object]:
    if isinstance(value, list):
        return value
    raise SummaryError(f"{name} must be an array")


def _string(value: object, name: str) -> str:
    if isinstance(value, str) and value:
        return value
    raise SummaryError(f"{name} must be a non-empty string")


def _integer(value: object, name: str) -> int:
    if isinstance(value, int):
        return value
    raise SummaryError(f"{name} must be an integer")


def _number(value: object, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise SummaryError(f"{name} must be numeric")


def _raw_seconds(path: Path) -> float:
    if not path.is_file():
        raise SummaryError(f"raw_time file is missing: {path}")
    match = _TIME_RE.search(path.read_text(encoding="utf-8"))
    if match is None:
        raise SummaryError(f"raw_time file must contain real/user/sys lines: {path}")
    return float(match.group("real"))


def _sample(entry: object) -> Sample:
    data = _mapping(entry, "sample")
    html_value = data.get("timing_html")
    timing_html = None
    if html_value is not None:
        timing_html = Path(_string(html_value, "sample.timing_html"))
        if not timing_html.is_file():
            raise SummaryError(f"timing_html file is missing: {timing_html}")
    raw_time = Path(_string(data.get("raw_time"), "sample.raw_time"))
    return Sample(
        label=_string(data.get("label"), "sample.label"),
        command=_string(data.get("command"), "sample.command"),
        raw_time=raw_time,
        timing_html=timing_html,
        target_dir=_string(data.get("target_dir"), "sample.target_dir"),
        cache_state=_string(data.get("cache_state"), "sample.cache_state"),
        sample_index=_integer(data.get("sample_index"), "sample.sample_index"),
        elapsed_seconds=_raw_seconds(raw_time),
    )


def _load_manifest(path: Path) -> tuple[list[Sample], dict[str, object]]:
    try:
        data = _mapping(json.loads(path.read_text(encoding="utf-8")), "manifest")
    except json.JSONDecodeError as exc:
        raise SummaryError(f"manifest is not valid JSON: {path}") from exc
    stats = _mapping(data.get("stats"), "stats")
    for key in ("median_seconds", "min_seconds", "max_seconds"):
        _number(stats.get(key), f"stats.{key}")
    samples = [_sample(entry) for entry in _sequence(data.get("samples"), "samples")]
    if not samples:
        raise SummaryError("samples must not be empty")
    return samples, stats


def _unit_data_json(source: str) -> str:
    start = source.find(_UNIT_DATA_MARKER)
    if start < 0:
        raise SummaryError("timing_html does not contain UNIT_DATA")
    idx = source.find("[", start)
    if idx < 0:
        raise SummaryError("UNIT_DATA array is missing")
    depth = 0
    for pos in range(idx, len(source)):
        char = source[pos]
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth == 0:
                return source[idx : pos + 1]
    raise SummaryError("UNIT_DATA array is unterminated")


def _unit_name(entry: dict[str, object]) -> str:
    name = _string(entry.get("name"), "unit.name")
    target = entry.get("target")
    target_suffix = target if isinstance(target, str) else ""
    return f"{name}{target_suffix}".strip()


def _units_from_html(path: Path) -> list[UnitTiming]:
    try:
        unit_entries = json.loads(_unit_data_json(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as exc:
        raise SummaryError(f"UNIT_DATA is not valid JSON in {path}") from exc
    units: list[UnitTiming] = []
    for raw_entry in _sequence(unit_entries, "UNIT_DATA"):
        entry = _mapping(raw_entry, "unit")
        base_name = _string(entry.get("name"), "unit.name")
        units.append(
            UnitTiming(
                name=_unit_name(entry),
                duration_seconds=_number(entry.get("duration"), "unit.duration"),
                kind="local-crate" if "pandas_booster" in base_name else "dependency",
            )
        )
    return units


def _aggregate_units(samples: Sequence[Sample]) -> list[UnitTiming]:
    totals: dict[str, UnitTiming] = {}
    for sample in samples:
        if sample.timing_html is None:
            continue
        for unit in _units_from_html(sample.timing_html):
            previous = totals.get(unit.name)
            duration = unit.duration_seconds
            if previous is not None:
                duration += previous.duration_seconds
            totals[unit.name] = UnitTiming(unit.name, duration, unit.kind)
    return sorted(totals.values(), key=lambda unit: unit.duration_seconds, reverse=True)


def _local_crate_dominates(units: Sequence[UnitTiming]) -> bool:
    local_duration = sum(unit.duration_seconds for unit in units if unit.kind == "local-crate")
    total_duration = sum(unit.duration_seconds for unit in units)
    return bool(units) and local_duration >= total_duration * 0.5 and units[0].kind == "local-crate"


def _summary(samples: Sequence[Sample], stats: dict[str, object]) -> dict[str, object]:
    elapsed = [sample.elapsed_seconds for sample in samples]
    units = _aggregate_units(samples)
    return {
        "samples": [
            {
                "label": sample.label,
                "command": sample.command,
                "raw_time": str(sample.raw_time),
                "timing_html": None if sample.timing_html is None else str(sample.timing_html),
                "target_dir": sample.target_dir,
                "cache_state": sample.cache_state,
                "sample_index": sample.sample_index,
                "elapsed_seconds": sample.elapsed_seconds,
            }
            for sample in samples
        ],
        "stats": {
            "median_seconds": _number(stats.get("median_seconds", median(elapsed)), "median"),
            "min_seconds": _number(stats.get("min_seconds", min(elapsed)), "min"),
            "max_seconds": _number(stats.get("max_seconds", max(elapsed)), "max"),
        },
        "local_crate_dominates": _local_crate_dominates(units),
        "units": [
            {"name": unit.name, "duration_seconds": unit.duration_seconds, "kind": unit.kind}
            for unit in units
        ],
    }


def _markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Rust Build Timing Baseline",
        "",
        "## Samples",
        "",
        "| Label | Command | Cache | Target dir | Seconds |",
        "| --- | --- | --- | --- | ---: |",
    ]
    for entry in _sequence(summary.get("samples"), "summary.samples"):
        sample = _mapping(entry, "summary.sample")
        lines.append(
            (
                "| {label} | `{command}` | {cache_state} | `{target_dir}` | "
                "{elapsed_seconds:.2f} |"
            ).format(
                label=_string(sample.get("label"), "summary.sample.label"),
                command=_string(sample.get("command"), "summary.sample.command"),
                cache_state=_string(sample.get("cache_state"), "summary.sample.cache_state"),
                target_dir=_string(sample.get("target_dir"), "summary.sample.target_dir"),
                elapsed_seconds=_number(
                    sample.get("elapsed_seconds"), "summary.sample.elapsed_seconds"
                ),
            )
        )
    stats = _mapping(summary.get("stats"), "summary.stats")
    lines.extend(
        [
            "",
            "## Stats",
            "",
            f"- median_seconds: {_number(stats.get('median_seconds'), 'median'):.2f}",
            f"- min_seconds: {_number(stats.get('min_seconds'), 'min'):.2f}",
            f"- max_seconds: {_number(stats.get('max_seconds'), 'max'):.2f}",
            f"- local_crate_dominates: {summary.get('local_crate_dominates')}",
            "",
            "## Top Units",
            "",
            "| Name | Kind | Duration seconds |",
            "| --- | --- | ---: |",
        ]
    )
    for entry in _sequence(summary.get("units"), "summary.units"):
        unit = _mapping(entry, "summary.unit")
        lines.append(
            "| {name} | {kind} | {duration_seconds:.2f} |".format(
                name=_string(unit.get("name"), "summary.unit.name"),
                kind=_string(unit.get("kind"), "summary.unit.kind"),
                duration_seconds=_number(unit.get("duration_seconds"), "summary.unit.duration"),
            )
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:  # noqa: UP045 - Python 3.9 syntax compatibility.
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-md", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    args = parser.parse_args(argv)
    samples, stats = _load_manifest(args.manifest)
    summary = _summary(samples, stats)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_markdown(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
