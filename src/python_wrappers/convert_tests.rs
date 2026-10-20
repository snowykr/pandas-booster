use super::convert::{
    build_single_profile_dict, convert_multi_result_f64, convert_multi_result_i64,
};
use crate::groupby::SingleKeyPhaseProfile;
use crate::groupby_multi::{GroupByMultiResultF64, GroupByMultiResultI64};
use crate::radix_groupby::SMALL_DIRECT_THRESHOLD_ELEMS;
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;

fn multi_conversion_shapes() -> [(usize, usize); 6] {
    [
        (0, 2),
        (8, 2),
        (SMALL_DIRECT_THRESHOLD_ELEMS / 2, 2),
        (SMALL_DIRECT_THRESHOLD_ELEMS / 5 + 1, 5),
        (SMALL_DIRECT_THRESHOLD_ELEMS / 2 + 1, 2),
        (SMALL_DIRECT_THRESHOLD_ELEMS + 1, 1),
    ]
}

fn conversion_keys(n_groups: usize, n_keys: usize) -> Vec<i64> {
    (0..n_groups * n_keys).map(|i| i as i64 - 17).collect()
}

fn assert_conversion_keys(
    py: Python<'_>,
    columns: &[Py<PyArray1<i64>>],
    n_groups: usize,
    n_keys: usize,
    permuted: bool,
) {
    assert_eq!(columns.len(), n_keys);
    for (col, column) in columns.iter().enumerate() {
        let readonly = column.bind(py).readonly();
        let keys = readonly.as_slice().expect("contiguous key column");
        assert_eq!(keys.len(), n_groups);
        for (out_g, &key) in keys.iter().enumerate() {
            let src_g = if permuted {
                n_groups - 1 - out_g
            } else {
                out_g
            };
            assert_eq!(key, (src_g * n_keys + col) as i64 - 17);
        }
    }
}

#[test]
fn multi_f64_conversion_preserves_ownership_bits_and_lifetime() -> PyResult<()> {
    Python::initialize();
    let bits = [
        0_u64,
        (-0.0_f64).to_bits(),
        0x7ff8_0000_0000_0042,
        0xfff8_0000_0000_0017,
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        1_u64,
        (-3.5_f64).to_bits(),
    ];

    for (n_groups, n_keys) in multi_conversion_shapes() {
        for permuted in [false, true] {
            let mut values = Vec::with_capacity(n_groups + 19);
            values.extend((0..n_groups).map(|i| f64::from_bits(bits[i % bits.len()])));
            let original_ptr = values.as_ptr() as usize;
            let result = GroupByMultiResultF64 {
                keys_flat: conversion_keys(n_groups, n_keys),
                n_keys,
                values,
                perm: permuted.then(|| (0..n_groups).rev().collect()),
            };
            let (columns, output) = Python::attach(|py| {
                let (columns, output) = convert_multi_result_f64(py, result)?;
                if !permuted {
                    assert_eq!(output.data() as usize, original_ptr);
                } else if n_groups != 0 {
                    assert_ne!(output.data() as usize, original_ptr);
                }
                Ok::<_, PyErr>((
                    columns
                        .into_iter()
                        .map(|col| col.unbind())
                        .collect::<Vec<_>>(),
                    output.unbind(),
                ))
            })?;

            // The converter, its owned result, and its Python attachment have ended.
            Python::attach(|py| {
                assert_conversion_keys(py, &columns, n_groups, n_keys, permuted);
                let readonly = output.bind(py).readonly();
                let values = readonly.as_slice().expect("contiguous values");
                assert_eq!(values.len(), n_groups);
                for (out_g, &value) in values.iter().enumerate() {
                    let src_g = if permuted {
                        n_groups - 1 - out_g
                    } else {
                        out_g
                    };
                    assert_eq!(value.to_bits(), bits[src_g % bits.len()]);
                }
            });
        }
    }
    Ok(())
}

#[test]
fn multi_i64_conversion_preserves_ownership_values_and_lifetime() -> PyResult<()> {
    Python::initialize();
    let integers = [
        i64::MIN,
        i64::MAX,
        -1,
        0,
        1,
        -(1_i64 << 54),
        1_i64 << 54,
        37,
    ];

    for (n_groups, n_keys) in multi_conversion_shapes() {
        for permuted in [false, true] {
            let mut values = Vec::with_capacity(n_groups + 19);
            values.extend((0..n_groups).map(|i| integers[i % integers.len()]));
            let original_ptr = values.as_ptr() as usize;
            let result = GroupByMultiResultI64 {
                keys_flat: conversion_keys(n_groups, n_keys),
                n_keys,
                values,
                perm: permuted.then(|| (0..n_groups).rev().collect()),
            };
            let (columns, output) = Python::attach(|py| {
                let (columns, output) = convert_multi_result_i64(py, result)?;
                if !permuted {
                    assert_eq!(output.data() as usize, original_ptr);
                } else if n_groups != 0 {
                    assert_ne!(output.data() as usize, original_ptr);
                }
                Ok::<_, PyErr>((
                    columns
                        .into_iter()
                        .map(|col| col.unbind())
                        .collect::<Vec<_>>(),
                    output.unbind(),
                ))
            })?;

            Python::attach(|py| {
                assert_conversion_keys(py, &columns, n_groups, n_keys, permuted);
                let readonly = output.bind(py).readonly();
                let values = readonly.as_slice().expect("contiguous values");
                assert_eq!(values.len(), n_groups);
                for (out_g, &value) in values.iter().enumerate() {
                    let src_g = if permuted {
                        n_groups - 1 - out_g
                    } else {
                        out_g
                    };
                    assert_eq!(value, integers[src_g % integers.len()]);
                }
            });
        }
    }
    Ok(())
}

#[test]
fn single_profile_dict_counts_routing_direct_phases_and_conversion_once() -> PyResult<()> {
    pyo3::Python::initialize();

    Python::attach(|py| {
        let profile = SingleKeyPhaseProfile {
            local_build_s: 0.75,
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
        let local_build_s = dict
            .get_item("local_build_s")?
            .expect("local_build_s exists")
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

        assert_eq!(rust_total_s, 13.25);
        assert_eq!(local_build_s, 0.75);
        assert_eq!(materialize_s, 0.5);
        assert_eq!(buffer_setup_s, 2.0);
        assert_eq!(route_kind, "direct_sparse");
        assert_eq!(route_reason, "sparse_exact_low_cardinality_memory_safe");
        Ok(())
    })
}

#[test]
fn single_profile_dict_counts_fallback_routing_with_zero_direct_fields() -> PyResult<()> {
    pyo3::Python::initialize();

    Python::attach(|py| {
        let mut profile = SingleKeyPhaseProfile::legacy(1.0, 2.0, 3.0, 4.0, 5, 5);
        profile.local_build_s += 0.75;
        profile.route_kind = Some("existing_sorted_fallback");
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

        assert_eq!(rust_total_s, 11.0);
        assert_eq!(unique_build_s, 0.0);
        assert_eq!(buffer_setup_s, 0.0);
        Ok(())
    })
}
