from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Optional

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "summarize_cargo_timings.py"


def _load_summary_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("summarize_cargo_timings", _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load timing helper from {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_raw_time(path: Path, seconds: float) -> None:
    path.write_text(
        "\n".join(
            [
                "Finished release [optimized] target(s) in 0.00s",
                f"real {seconds:.2f}",
                "user 0.10",
                "sys 0.02",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_timing_html(path: Path, *, pandas_seconds: float) -> None:
    path.write_text(
        "\n".join(
            [
                "<html>",
                "<body>",
                "<script>",
                "const UNIT_DATA = [",
                f'  {{"name": "pandas_booster", "target": "", "duration": {pandas_seconds:.1f}}},',
                '  {"name": "syn", "target": "", "duration": 2.5},',
                '  {"name": "quote", "target": "", "duration": 1.2},',
                '  {"name": "proc-macro2", "target": "", "duration": 1.1},',
                '  {"name": "pyo3-build-config", "target": " build script", "duration": 0.9},',
                '  {"name": "libc", "target": " build script", "duration": 0.6}',
                "];",
                "</script>",
                "</body>",
                "</html>",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _sample_dict(
    *,
    label: str,
    command: str,
    raw_time: Path,
    timing_html: Optional[Path],  # noqa: UP045 - Python 3.9 syntax compatibility.
    target_dir: str,
    cache_state: str,
    sample_index: int,
) -> dict[str, object]:
    return {
        "label": label,
        "command": command,
        "raw_time": str(raw_time),
        "timing_html": None if timing_html is None else str(timing_html),
        "target_dir": target_dir,
        "cache_state": cache_state,
        "sample_index": sample_index,
    }


def _write_manifest(
    path: Path, *, samples: list[dict[str, object]], stats: dict[str, float]
) -> None:
    path.write_text(
        json.dumps({"samples": samples, "stats": stats}, indent=2) + "\n",
        encoding="utf-8",
    )


def test_summary_rejects_malformed_manifest(tmp_path: Path) -> None:
    module = _load_summary_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"samples": []}) + "\n", encoding="utf-8")

    with pytest.raises(module.SummaryError, match="stats"):
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--out-md",
                str(tmp_path / "summary.md"),
                "--out-json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_summary_rejects_missing_timing_html(tmp_path: Path) -> None:
    module = _load_summary_module()
    raw_time_path = tmp_path / "clean.raw.txt"
    _write_raw_time(raw_time_path, 12.5)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        samples=[
            _sample_dict(
                label="Clean release sample 1",
                command="cargo build --release --features extension-module --timings",
                raw_time=raw_time_path,
                timing_html=tmp_path / "missing.html",
                target_dir=".omo/evidence/rust-build-speed/target-baseline-clean-1",
                cache_state="clean",
                sample_index=1,
            )
        ],
        stats={"median_seconds": 12.5, "min_seconds": 12.5, "max_seconds": 12.5},
    )

    with pytest.raises(module.SummaryError, match="timing_html"):
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--out-md",
                str(tmp_path / "summary.md"),
                "--out-json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_summary_rejects_malformed_time_log(tmp_path: Path) -> None:
    module = _load_summary_module()
    raw_time_path = tmp_path / "clean.raw.txt"
    raw_time_path.write_text("real 12.50\nuser 0.10\n", encoding="utf-8")
    timing_html_path = tmp_path / "clean.html"
    _write_timing_html(timing_html_path, pandas_seconds=8.0)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        samples=[
            _sample_dict(
                label="Clean release sample 1",
                command="cargo build --release --features extension-module --timings",
                raw_time=raw_time_path,
                timing_html=timing_html_path,
                target_dir=".omo/evidence/rust-build-speed/target-baseline-clean-1",
                cache_state="clean",
                sample_index=1,
            )
        ],
        stats={"median_seconds": 12.5, "min_seconds": 12.5, "max_seconds": 12.5},
    )

    with pytest.raises(module.SummaryError, match="raw_time"):
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--out-md",
                str(tmp_path / "summary.md"),
                "--out-json",
                str(tmp_path / "summary.json"),
            ]
        )


def test_summary_writes_expected_json_schema_and_markdown(tmp_path: Path) -> None:
    module = _load_summary_module()
    clean_one_raw = tmp_path / "clean-1.raw.txt"
    clean_two_raw = tmp_path / "clean-2.raw.txt"
    warm_raw = tmp_path / "warm.raw.txt"
    clean_one_html = tmp_path / "clean-1.html"
    clean_two_html = tmp_path / "clean-2.html"
    _write_raw_time(clean_one_raw, 12.5)
    _write_raw_time(clean_two_raw, 13.5)
    _write_raw_time(warm_raw, 4.5)
    _write_timing_html(clean_one_html, pandas_seconds=8.0)
    _write_timing_html(clean_two_html, pandas_seconds=9.0)

    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        samples=[
            _sample_dict(
                label="Clean release sample 1",
                command="cargo build --release --features extension-module --timings",
                raw_time=clean_one_raw,
                timing_html=clean_one_html,
                target_dir=".omo/evidence/rust-build-speed/target-baseline-clean-1",
                cache_state="clean",
                sample_index=1,
            ),
            _sample_dict(
                label="Clean release sample 2",
                command="cargo build --release --features extension-module --timings",
                raw_time=clean_two_raw,
                timing_html=clean_two_html,
                target_dir=".omo/evidence/rust-build-speed/target-baseline-clean-2",
                cache_state="clean",
                sample_index=2,
            ),
            _sample_dict(
                label="Warm release",
                command="cargo build --release --features extension-module",
                raw_time=warm_raw,
                timing_html=None,
                target_dir=".omo/evidence/rust-build-speed/target-baseline-warm",
                cache_state="warm",
                sample_index=1,
            ),
        ],
        stats={"median_seconds": 13.0, "min_seconds": 12.5, "max_seconds": 13.5},
    )

    out_md = tmp_path / "summary.md"
    out_json = tmp_path / "summary.json"
    assert (
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--out-md",
                str(out_md),
                "--out-json",
                str(out_json),
            ]
        )
        == 0
    )

    summary = json.loads(out_json.read_text(encoding="utf-8"))
    assert sorted(summary) == ["local_crate_dominates", "samples", "stats", "units"]
    assert summary["local_crate_dominates"] is True
    assert summary["stats"] == {
        "median_seconds": 13.0,
        "min_seconds": 12.5,
        "max_seconds": 13.5,
    }
    assert len(summary["samples"]) == 3
    assert summary["samples"][0]["elapsed_seconds"] == 12.5
    assert summary["units"][0] == {
        "name": "pandas_booster",
        "duration_seconds": 17.0,
        "kind": "local-crate",
    }
    assert {unit["name"] for unit in summary["units"][1:]} >= {
        "syn",
        "quote",
        "proc-macro2",
        "pyo3-build-config build script",
        "libc build script",
    }

    markdown = out_md.read_text(encoding="utf-8")
    assert "Clean release sample 1" in markdown
    assert "Warm release" in markdown
    assert "cargo build --release --features extension-module --timings" in markdown
    assert "local_crate_dominates: True" in markdown
