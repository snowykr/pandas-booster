use super::convert::build_single_profile_dict;
use crate::groupby::SingleKeyPhaseProfile;
use pyo3::prelude::*;

#[test]
fn single_profile_dict_counts_direct_phases_and_conversion_once() -> PyResult<()> {
    pyo3::Python::initialize();

    Python::attach(|py| {
        let profile = SingleKeyPhaseProfile {
            local_build_s: 0.0,
            merge_s: 0.0,
            reorder_s: 0.0,
            materialize_s: 0.0,
            unique_build_s: 1.0,
            key_sort_s: 2.0,
            count_s: 0.0,
            buffer_setup_s: 2.0,
            scatter_s: 3.0,
            median_select_s: 4.0,
            route_kind: Some("direct_sparse"),
            route_reason: Some("sparse_exact_low_cardinality_memory_safe"),
            partial_group_total: 2,
            final_group_count: 2,
        };

        let dict = build_single_profile_dict(py, &profile, 0.5)?;
        let rust_total_s = dict
            .get_item("rust_total_s")?
            .expect("rust_total_s exists")
            .extract::<f64>()?;
        let materialize_s = dict
            .get_item("materialize_s")?
            .expect("materialize_s exists")
            .extract::<f64>()?;
        let buffer_setup_s = dict
            .get_item("buffer_setup_s")?
            .expect("buffer_setup_s exists")
            .extract::<f64>()?;

        let route_kind = dict
            .get_item("route_kind")?
            .expect("route_kind exists")
            .extract::<String>()?;
        let route_reason = dict
            .get_item("route_reason")?
            .expect("route_reason exists")
            .extract::<String>()?;

        assert_eq!(rust_total_s, 12.5);
        assert_eq!(materialize_s, 0.5);
        assert_eq!(buffer_setup_s, 2.0);
        assert_eq!(route_kind, "direct_sparse");
        assert_eq!(route_reason, "sparse_exact_low_cardinality_memory_safe");
        Ok(())
    })
}

#[test]
fn single_profile_dict_preserves_legacy_total_with_zero_direct_fields() -> PyResult<()> {
    pyo3::Python::initialize();

    Python::attach(|py| {
        let profile = SingleKeyPhaseProfile::legacy(1.0, 2.0, 3.0, 4.0, 5, 5);
        let dict = build_single_profile_dict(py, &profile, 0.25)?;
        let rust_total_s = dict
            .get_item("rust_total_s")?
            .expect("rust_total_s exists")
            .extract::<f64>()?;
        let unique_build_s = dict
            .get_item("unique_build_s")?
            .expect("unique_build_s exists")
            .extract::<f64>()?;
        let buffer_setup_s = dict
            .get_item("buffer_setup_s")?
            .expect("buffer_setup_s exists")
            .extract::<f64>()?;

        assert_eq!(rust_total_s, 10.25);
        assert_eq!(unique_build_s, 0.0);
        assert_eq!(buffer_setup_s, 0.0);
        Ok(())
    })
}
