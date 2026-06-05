from ._helpers import _REPO_ROOT


def test_pr12_regression_anchor_targets_still_exist():
    anchor_targets = {
        "discussion_r3196780027": (
            _REPO_ROOT / "tests" / "rust_stub_exports" / "test_dispatch_mapping.py",
            "test_prod_exact_dispatch_mapping_selects_expected_symbols",
        ),
        "discussion_r3196782200": (
            _REPO_ROOT / "tests" / "rust_stub_exports" / "test_export_surface.py",
            "test_rust_pyi_exports_match_expected_pyo3_surface",
        ),
        "discussion_r3197179732": (
            _REPO_ROOT / "src" / "groupby" / "api_float_tests.rs",
            "test_groupby_prod_f64_preserves_row_order_ieee_semantics",
        ),
        "discussion_r3197709485": (
            _REPO_ROOT / "tests" / "edge_cases" / "test_median_extremes.py",
            "test_single_key_float_median_large_even_middle_values_match_pandas",
        ),
    }

    for anchor, (path, needle) in anchor_targets.items():
        assert needle in path.read_text(), (
            f"{anchor} lost its characterization target {needle!r} in {path}"
        )
