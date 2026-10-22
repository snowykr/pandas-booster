"""Shared positive and negative checks for archived primary phase observations."""

from __future__ import annotations

import math


def primary_max_evidence(report, pins):
    rows = [
        row for row in report["raw"] if row["kind"] == "evidence" and row["case"]["agg"] == "max"
    ]
    assert len(rows) == 2
    assert {row["role"] for row in rows} == {"baseline", "candidate"}
    for row in rows:
        assert row["threads"] == 4
        assert row["case"] == {
            "name": "primary",
            "n": 5000000,
            "widths": [500, 500, 500],
            "dtype": "float64",
            "sort": True,
            "agg": "max",
            "seed": 42,
        }
        result = row["result"]
        expected_sha = pins[
            "baseline_sha" if row["role"] == "baseline" else "measured_candidate_sha"
        ]
        assert result["identity"]["sha"] == expected_sha
        assert result["status"] == "passed" and result["parity"] is True
        assert result["groups"] == 4901309
        for field in (
            "prepare_inputs_s",
            "python_normalize_s",
            "pandas_index_construction_s",
            "profile_pipeline_s",
            "public_pipeline_s",
        ):
            assert math.isfinite(result[field]) and result[field] > 0
        profile = result["profile"]
        for field in (
            "hash_build_s",
            "partition_scatter_s",
            "partition_aggregation_s",
            "flatten_s",
            "sort_key_construction_s",
            "radix_sort_s",
            "sorted_materialization_s",
            "conversion_s",
        ):
            assert math.isfinite(profile[field]) and profile[field] >= 0
        if row["role"] == "candidate":
            assert profile["route"] == "sort_first"
            assert profile["hash_first_aggregation_count"] == 0
            assert profile["post_aggregation_sort_count"] == 0
            # Inclusive permutation timing is intentionally excluded from this sum.
            leaves = sum(
                profile[field]
                for field in (
                    "routing_s",
                    "sort_key_construction_s",
                    "radix_sort_s",
                    "sort_first_boundary_s",
                    "sort_first_setup_s",
                    "sort_first_segment_scan_s",
                )
            )
            assert 0 <= profile["rust_elapsed_s"] - leaves < 0.01
        else:
            assert profile["route"] == "hash_first"
