"""Benchmark profile JSON payload serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench_utils import BenchmarkStats
from runner import resolve_stats_evidence_aggs


def serialize_stats(stats: BenchmarkStats) -> dict[str, Any]:
    return stats.to_dict()


def serialize_phase_stats(phases: dict[str, BenchmarkStats]) -> dict[str, dict[str, Any]]:
    return {name: serialize_stats(stats) for name, stats in phases.items()}


def stats_mean_map(phases: dict[str, BenchmarkStats]) -> dict[str, float]:
    return {name: stats.mean for name, stats in phases.items()}


def summarize_profile_cases(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    profiled_cases = [case for case in cases if case["breakdown"] is not None]
    if not profiled_cases:
        return None

    phase_names = list(profiled_cases[0]["breakdown"]["phases"].keys())
    phase_means = {
        phase_name: sum(case["breakdown"]["phases"][phase_name].mean for case in profiled_cases)
        / len(profiled_cases)
        for phase_name in phase_names
    }
    first_breakdown = profiled_cases[0]["breakdown"]

    return {
        "preset": profiled_cases[0]["preset"],
        "workload": profiled_cases[0]["workload"],
        "sort": profiled_cases[0]["sort"],
        "aggs": [case["agg"] for case in profiled_cases],
        "phases": phase_means,
        "rust_total_s": sum(case["breakdown"]["rust_total_s"] for case in profiled_cases)
        / len(profiled_cases),
        "python_total_s": sum(case["breakdown"]["python_total_s"] for case in profiled_cases)
        / len(profiled_cases),
        "total_pipeline_s": sum(case["breakdown"]["total_pipeline_s"] for case in profiled_cases)
        / len(profiled_cases),
        "partial_group_total": first_breakdown["partial_group_total"],
        "final_group_count": first_breakdown["final_group_count"],
        "partial_to_final_ratio": first_breakdown["partial_to_final_ratio"],
        "per_agg": {
            case["agg"]: {
                "execution": case["breakdown"]["execution"],
                "phases": stats_mean_map(case["breakdown"]["phases"]),
                "rust_total_s": case["breakdown"]["rust_total_s"],
                "python_total_s": case["breakdown"]["python_total_s"],
                "total_pipeline_s": case["breakdown"]["total_pipeline_s"],
                "partial_group_total": case["breakdown"]["partial_group_total"],
                "final_group_count": case["breakdown"]["final_group_count"],
                "partial_to_final_ratio": case["breakdown"]["partial_to_final_ratio"],
            }
            for case in profiled_cases
        },
    }


def build_profile_json_payload(
    evidence: list[dict[str, Any]],
    *,
    cardinality: str,
    sort_mode: str,
    n_samples: int,
    selected_aggs: list[str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "metadata": {
            "cardinality": cardinality,
            "sort_mode": sort_mode,
            "samples": n_samples,
            "selected_aggs": resolve_stats_evidence_aggs(selected_aggs),
        },
        "cases": [],
    }

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    for item in evidence:
        grouped.setdefault((item["workload"], item["sort"]), []).append(item)
        breakdown = item["breakdown"]
        payload["cases"].append(
            {
                "preset": item["preset"],
                "workload": item["workload"],
                "agg": item["agg"],
                "sort": item["sort"],
                "execution": item["execution"],
                "result": {
                    "preset": item["result"]["preset"],
                    "n_rows": item["result"]["n_rows"],
                    "n_keys": item["result"]["n_keys"],
                    "key_cols": item["result"]["key_cols"],
                    "combo_cardinality": item["result"]["combo_cardinality"],
                    "group_ratio": item["result"]["group_ratio"],
                    "agg": item["result"]["agg"],
                    "sort": item["result"]["sort"],
                    "backends": {
                        backend_name: {
                            "cold_stats": serialize_stats(backend_data["cold_stats"]),
                            "warm_stats": serialize_stats(backend_data["warm_stats"]),
                            "cold_correctness": backend_data["cold_correctness"],
                            "warm_correctness": backend_data["warm_correctness"],
                        }
                        for backend_name, backend_data in item["result"]["backends"].items()
                    },
                },
                "breakdown": None
                if breakdown is None
                else {
                    "execution": breakdown["execution"],
                    "phases": serialize_phase_stats(breakdown["phases"]),
                    "phase_means": stats_mean_map(breakdown["phases"]),
                    "rust_total_s": breakdown["rust_total_s"],
                    "python_total_s": breakdown["python_total_s"],
                    "total_pipeline_s": breakdown["total_pipeline_s"],
                    "partial_group_total": breakdown["partial_group_total"],
                    "final_group_count": breakdown["final_group_count"],
                    "partial_to_final_ratio": breakdown["partial_to_final_ratio"],
                },
            }
        )

    for workload in ("standard", "high"):
        for sort in (False, True):
            cases = grouped.get((workload, sort), [])
            if not cases:
                continue
            suffix = "unsorted" if not sort else "sorted"
            summary = summarize_profile_cases(cases)
            if summary is not None:
                payload[f"single_key_{suffix}_{workload}"] = summary

    if "single_key_unsorted_high" in payload:
        payload["single_key_unsorted"] = payload["single_key_unsorted_high"]
    elif "single_key_unsorted_standard" in payload:
        payload["single_key_unsorted"] = payload["single_key_unsorted_standard"]

    if "single_key_sorted_high" in payload:
        payload["single_key_sorted"] = payload["single_key_sorted_high"]
    elif "single_key_sorted_standard" in payload:
        payload["single_key_sorted"] = payload["single_key_sorted_standard"]

    return payload


def save_profile_json(
    evidence: list[dict[str, Any]],
    output_path: str,
    *,
    cardinality: str,
    sort_mode: str,
    n_samples: int,
    selected_aggs: list[str] | None,
    build_profile_json_payload_func=build_profile_json_payload,
) -> None:
    path = Path(output_path)
    if not path.suffix:
        path = path.with_suffix(".json")
    elif path.suffix != ".json":
        print(f"Warning: Output path should have .json extension, got {path.suffix}")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_profile_json_payload_func(
        evidence,
        cardinality=cardinality,
        sort_mode=sort_mode,
        n_samples=n_samples,
        selected_aggs=selected_aggs,
    )
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Profile JSON saved to: {path}")
