"""Benchmark profile JSON payload serialization."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from bench_utils import BenchmarkStats
from profile_json_multi_key import MULTI_KEY_ROUTE_FIELDS, REQUIRED_MULTI_KEY_SORTED_PHASES
from runner import resolve_stats_evidence_aggs

PHASE_NAMES: tuple[str, ...] = (
    "prepare_inputs_s",
    "unique_build_s",
    "key_sort_s",
    "count_s",
    "buffer_setup_s",
    "scatter_s",
    "median_select_s",
    "local_build_s",
    "merge_s",
    "reorder_s",
    "materialize_s",
    "python_normalize_s",
    "python_series_build_s",
    "rust_total_s",
    "python_total_s",
    "total_pipeline_s",
)


def zero_phase_stats() -> BenchmarkStats:
    return BenchmarkStats(0.0, 0.0, 0.0, 0.0, [])


def normalize_phase_stats(
    phases: dict[str, BenchmarkStats], profile_kind: str = "single_key"
) -> dict[str, BenchmarkStats]:
    phase_names = (
        REQUIRED_MULTI_KEY_SORTED_PHASES if profile_kind == "multi_key_sorted" else PHASE_NAMES
    )
    zero = zero_phase_stats()
    return {name: phases.get(name, zero) for name in phase_names}


def serialize_stats(stats: BenchmarkStats) -> dict[str, Any]:
    return stats.to_dict()


def serialize_phase_stats(
    phases: dict[str, BenchmarkStats], profile_kind: str = "single_key"
) -> dict[str, dict[str, Any]]:
    return {
        name: serialize_stats(stats)
        for name, stats in normalize_phase_stats(phases, profile_kind).items()
    }


def stats_mean_map(
    phases: dict[str, BenchmarkStats], profile_kind: str = "single_key"
) -> dict[str, float]:
    return {name: stats.mean for name, stats in normalize_phase_stats(phases, profile_kind).items()}


def _collapse_route_metadata(profiled_cases: list[dict[str, Any]], field: str) -> str:
    values = {
        str(case["breakdown"].get(field, ""))
        for case in profiled_cases
        if case["breakdown"].get(field, "")
    }
    if len(values) == 1:
        return next(iter(values))
    if values:
        return "mixed"
    return ""


def summarize_profile_cases(cases: list[dict[str, Any]]) -> dict[str, Any] | None:
    profiled_cases = [case for case in cases if case["breakdown"] is not None]
    if not profiled_cases:
        return None

    normalized_phase_maps = [
        normalize_phase_stats(case["breakdown"]["phases"], _profile_kind(case))
        for case in profiled_cases
    ]
    phase_means = {
        phase_name: sum(phases[phase_name].mean for phases in normalized_phase_maps)
        / len(normalized_phase_maps)
        for phase_name in normalized_phase_maps[0]
    }
    first_breakdown = profiled_cases[0]["breakdown"]
    routes = {str(case["breakdown"]["route"]) for case in profiled_cases}
    route = routes.pop() if len(routes) == 1 else "mixed"

    return {
        "preset": profiled_cases[0]["preset"],
        "workload": profiled_cases[0]["workload"],
        "sort": profiled_cases[0]["sort"],
        "aggs": [case["agg"] for case in profiled_cases],
        "route": route,
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
        "sort_first_segment_scan_count": first_breakdown["sort_first_segment_scan_count"],
        "selected_sort_strategy": first_breakdown["selected_sort_strategy"],
        "sort_key_bit_widths": first_breakdown["sort_key_bit_widths"],
        "route_kind": _collapse_route_metadata(profiled_cases, "route_kind"),
        "route_reason": _collapse_route_metadata(profiled_cases, "route_reason"),
        **{
            field: first_breakdown[field]
            for field in MULTI_KEY_ROUTE_FIELDS
            if field in first_breakdown and field != "route_reason"
        },
        "per_agg": {
            case["agg"]: {
                "execution": case["breakdown"]["execution"],
                "route": case["breakdown"]["route"],
                "phases": stats_mean_map(case["breakdown"]["phases"], _profile_kind(case)),
                "rust_total_s": case["breakdown"]["rust_total_s"],
                "python_total_s": case["breakdown"]["python_total_s"],
                "total_pipeline_s": case["breakdown"]["total_pipeline_s"],
                "partial_group_total": case["breakdown"]["partial_group_total"],
                "final_group_count": case["breakdown"]["final_group_count"],
                "partial_to_final_ratio": case["breakdown"]["partial_to_final_ratio"],
                "sort_first_segment_scan_count": case["breakdown"]["sort_first_segment_scan_count"],
                "selected_sort_strategy": case["breakdown"]["selected_sort_strategy"],
                "sort_key_bit_widths": case["breakdown"]["sort_key_bit_widths"],
                "route_kind": case["breakdown"].get("route_kind", ""),
                "route_reason": case["breakdown"].get("route_reason", ""),
                **{
                    field: case["breakdown"][field]
                    for field in MULTI_KEY_ROUTE_FIELDS
                    if field in case["breakdown"] and field != "route_reason"
                },
            }
            for case in profiled_cases
        },
    }


def _profile_kind(case: dict[str, Any]) -> str:
    breakdown = case["breakdown"]
    if breakdown is None:
        return "unprofiled"
    return str(breakdown.get("profile_kind", "single_key"))


def _metadata_selected_aggs(
    evidence: list[dict[str, Any]],
    selected_aggs: list[str] | None,
) -> list[str]:
    profiled_aggs = {item["agg"] for item in evidence if item["breakdown"] is not None}
    stats_aggs = set(resolve_stats_evidence_aggs(selected_aggs))
    eligible_aggs = stats_aggs | profiled_aggs

    if selected_aggs is None:
        ordered_aggs = list(resolve_stats_evidence_aggs(None))
        for item in evidence:
            agg = item["agg"]
            if item["breakdown"] is not None and agg not in ordered_aggs:
                ordered_aggs.append(agg)
        return ordered_aggs

    ordered_selected_aggs: list[str] = []
    for agg in selected_aggs:
        if agg in eligible_aggs and agg not in ordered_selected_aggs:
            ordered_selected_aggs.append(agg)
    return ordered_selected_aggs


def _requires_selected_median_sorted_breakdown(
    item: dict[str, Any],
    *,
    selected_aggs: list[str] | None,
) -> bool:
    booster_execution = item["execution"]["booster"]
    return (
        selected_aggs is not None
        and "median" in resolve_stats_evidence_aggs(selected_aggs)
        and item["agg"] == "median"
        and item["sort"] is True
        and booster_execution == "booster->rust.groupby_median_f64_sorted"
        and (
            item["breakdown"] is None
            or not item["breakdown"].get("route_kind")
            or not item["breakdown"].get("route_reason")
        )
    )


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
            "selected_aggs": _metadata_selected_aggs(evidence, selected_aggs),
        },
        "cases": [],
    }

    grouped: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    multi_key_sorted: dict[str, list[dict[str, Any]]] = {}
    for item in evidence:
        if _requires_selected_median_sorted_breakdown(item, selected_aggs=selected_aggs):
            raise ValueError(
                "selected median sorted profile breakdown is unavailable; "
                "profile JSON would omit required median evidence"
            )

        if _profile_kind(item) == "multi_key_sorted":
            multi_key_sorted.setdefault(item["workload"], []).append(item)
        else:
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
                    "profile_kind": breakdown.get("profile_kind", "single_key"),
                    "route": breakdown["route"],
                    "execution": breakdown["execution"],
                    "route_kind": breakdown.get("route_kind", ""),
                    "route_reason": breakdown.get("route_reason", ""),
                    **{
                        field: breakdown[field]
                        for field in MULTI_KEY_ROUTE_FIELDS
                        if field in breakdown and field != "route_reason"
                    },
                    "phases": serialize_phase_stats(breakdown["phases"], _profile_kind(item)),
                    "phase_means": stats_mean_map(breakdown["phases"], _profile_kind(item)),
                    "rust_total_s": breakdown["rust_total_s"],
                    "python_total_s": breakdown["python_total_s"],
                    "total_pipeline_s": breakdown["total_pipeline_s"],
                    "partial_group_total": breakdown["partial_group_total"],
                    "final_group_count": breakdown["final_group_count"],
                    "partial_to_final_ratio": breakdown["partial_to_final_ratio"],
                    "sort_first_segment_scan_count": breakdown["sort_first_segment_scan_count"],
                    "selected_sort_strategy": breakdown["selected_sort_strategy"],
                    "sort_key_bit_widths": breakdown["sort_key_bit_widths"],
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

    for workload, cases in multi_key_sorted.items():
        summary = summarize_profile_cases(cases)
        if summary is not None:
            payload[f"multi_key_sorted_{workload}"] = summary

    for (workload, sort), cases in grouped.items():
        if workload in {"standard", "high"}:
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
