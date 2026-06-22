from __future__ import annotations

import json

from ._report_output_helpers import _make_breakdown, _make_result


def test_save_profile_json_adds_json_suffix_and_creates_parent_dir(
    benchmark_module,
    monkeypatch,
    tmp_path,
):
    payload = {"ok": True}
    captured_args: list[tuple[list[dict[str, object]], dict[str, object]]] = []

    def fake_build_profile_json_payload(
        evidence,
        *,
        cardinality,
        sort_mode,
        n_samples,
        selected_aggs,
    ):
        captured_args.append(
            (
                evidence,
                {
                    "cardinality": cardinality,
                    "sort_mode": sort_mode,
                    "n_samples": n_samples,
                    "selected_aggs": selected_aggs,
                },
            )
        )
        return payload

    monkeypatch.setattr(
        benchmark_module, "build_profile_json_payload", fake_build_profile_json_payload
    )

    output_path = tmp_path / "profiles" / "profile_output"
    benchmark_module.save_profile_json(
        [{"preset": "1key"}],
        str(output_path),
        cardinality="standard",
        sort_mode="sorted",
        n_samples=3,
        selected_aggs=["std"],
    )

    written_path = output_path.with_suffix(".json")
    assert written_path.exists()
    assert written_path.parent.is_dir()
    assert json.loads(written_path.read_text()) == payload
    assert captured_args == [
        (
            [{"preset": "1key"}],
            {
                "cardinality": "standard",
                "sort_mode": "sorted",
                "n_samples": 3,
                "selected_aggs": ["std"],
            },
        )
    ]


def test_main_profile_json_wires_evidence_collection_and_file_write(
    benchmark_module, monkeypatch, tmp_path
):
    results = [_make_result(benchmark_module, preset="1key", agg="std", sort=True)]
    evidence = [
        {
            "preset": "1key",
            "workload": "standard",
            "agg": "std",
            "sort": True,
            "execution": {
                "pandas": "pandas.groupby.std",
                "booster": "booster->rust.groupby_std_f64_sorted",
            },
            "result": results[0],
            "breakdown": _make_breakdown(benchmark_module),
        }
    ]
    captured_run_args: list[dict[str, object]] = []
    captured_evidence_args: list[tuple[int, str, str, list[str] | None]] = []

    def fake_run_benchmarks(*, cardinality, diagnostic, sort_mode, n_samples, aggs):
        captured_run_args.append(
            {
                "cardinality": cardinality,
                "diagnostic": diagnostic,
                "sort_mode": sort_mode,
                "n_samples": n_samples,
                "aggs": aggs,
            }
        )
        return results

    def fake_collect_stats_evidence(n_samples, cardinality, sort_mode, selected_aggs=None):
        captured_evidence_args.append((n_samples, cardinality, sort_mode, selected_aggs))
        return evidence

    profile_path = tmp_path / "profiles" / "std_profile"
    monkeypatch.setattr(benchmark_module, "run_benchmarks", fake_run_benchmarks)
    monkeypatch.setattr(benchmark_module, "collect_stats_evidence", fake_collect_stats_evidence)
    monkeypatch.setattr(
        benchmark_module.sys,
        "argv",
        [
            "benchmark.py",
            "--cardinality",
            "standard",
            "--sort-mode",
            "sorted",
            "--samples",
            "3",
            "--agg",
            "std",
            "--profile-json",
            str(profile_path),
        ],
    )

    assert benchmark_module.main() == results

    written_path = profile_path.with_suffix(".json")
    payload = json.loads(written_path.read_text())
    assert captured_run_args == [
        {
            "cardinality": "standard",
            "diagnostic": "none",
            "sort_mode": "sorted",
            "n_samples": 3,
            "aggs": ["std"],
        }
    ]
    assert captured_evidence_args == [(3, "standard", "sorted", ["std"])]
    assert payload["metadata"] == {
        "cardinality": "standard",
        "sort_mode": "sorted",
        "samples": 3,
        "selected_aggs": ["std"],
    }
    assert payload["cases"][0]["breakdown"]["execution"] == (
        "booster->rust.groupby_std_f64_sorted"
    )
