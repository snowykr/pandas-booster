from __future__ import annotations

from bench_utils import BenchmarkStats


def render_standard_table(results: list[dict]) -> str:
    preset_labels = {
        "1key": "Single-key",
        "2key": "2-key",
        "3key": "3-key",
        "4key": "4-key",
        "5key": "5-key",
    }
    return _render_result_table(results, list(preset_labels), preset_labels)


def render_high_table(results: list[dict]) -> str:
    preset_labels = {
        "high_cardinality_1key": "Single-key",
        "high_cardinality_2key": "2-key",
        "high_cardinality_3key": "3-key",
    }
    return _render_result_table(results, list(preset_labels), preset_labels)


def render_threshold_table(results: list[dict]) -> str:
    preset_labels = {
        "threshold_180k": "2-key (~180k elems)",
        "threshold_200k": "2-key (~200k elems)",
        "threshold_220k": "2-key (~220k elems)",
    }
    return _render_result_table(results, list(preset_labels), preset_labels)


def _format_backend_phase_cell(
    backends: dict,
    name: str,
    phase: str,
    pandas_mean: float,
) -> str:
    if name not in backends:
        return "-"
    stats: BenchmarkStats = backends[name][f"{phase}_stats"]
    mean_ms = stats.mean * 1000
    std_ms = stats.std * 1000
    speedup = pandas_mean / stats.mean if stats.mean > 0 else 0
    value = f"{mean_ms:.1f}±{std_ms:.1f}ms"
    if name == "pandas":
        return f"{value} (1.0x)"
    speedup_text = f"**{speedup:.1f}x**" if speedup >= 1.1 else f"{speedup:.1f}x"
    return f"{value} ({speedup_text})"


def _render_result_table(
    results: list[dict],
    preset_order: list[str],
    preset_labels: dict[str, str],
) -> str:
    if not results:
        return ""

    results_by_preset_sort: dict[tuple[str, bool], dict] = {}
    for r in results:
        results_by_preset_sort[(r["preset"], r["sort"])] = r

    lines = [
        "| Workload | Groups | Sort | Type | Pandas | Polars | Booster |",
        "|----------|--------|------|------|--------|--------|---------|",
    ]
    prev_label = None
    prev_groups = None
    prev_sort_str = None

    for preset in preset_order:
        for sort_val in [True, False]:
            key = (preset, sort_val)
            if key not in results_by_preset_sort:
                continue

            r = results_by_preset_sort[key]
            label = preset_labels[preset]
            groups = f"{r['combo_cardinality']:,}"
            sort_str = "True" if sort_val else "False"
            backends = r["backends"]
            if "pandas" not in backends:
                continue

            pandas_cold: BenchmarkStats = backends["pandas"]["cold_stats"]
            pandas_warm: BenchmarkStats = backends["pandas"]["warm_stats"]
            cold_cells = [
                _format_backend_phase_cell(backends, name, "cold", pandas_cold.mean)
                for name in ("pandas", "polars", "booster")
            ]
            warm_cells = [
                _format_backend_phase_cell(backends, name, "warm", pandas_warm.mean)
                for name in ("pandas", "polars", "booster")
            ]
            display_label = label if label != prev_label else ""
            display_groups = groups if groups != prev_groups else ""
            display_sort = sort_str if sort_str != prev_sort_str else ""
            lines.append(
                f"| {display_label} | {display_groups} | {display_sort} | Cold | "
                f"{' | '.join(cold_cells)} |"
            )
            lines.append(f"|  |  |  | Warm | {' | '.join(warm_cells)} |")
            prev_label = label
            prev_groups = groups
            prev_sort_str = sort_str

    return "\n".join(lines)
