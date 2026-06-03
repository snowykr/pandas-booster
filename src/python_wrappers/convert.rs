use crate::groupby::{GroupByResultF64, GroupByResultI64};
use crate::python_wrappers::shared::{
    MultiGroupByReturn, MultiGroupByReturnF64, MultiGroupByReturnI64, SingleGroupByReturn,
    SingleGroupByReturnF64, SingleGroupByReturnI64,
};
use crate::radix_groupby::SMALL_DIRECT_THRESHOLD_ELEMS;
use crate::{groupby, groupby_multi};
use numpy::{IntoPyArray, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::prelude::*;
use std::ptr::NonNull;

#[derive(Clone, Copy)]
struct WritePtr<T>(NonNull<T>);

// SAFETY: `WritePtr<T>` only wraps a writable pointer to a NumPy allocation
// owned by the current result conversion. The wrapper is only used for
// disjoint indexed writes, so sharing it across threads does not create
// concurrent aliasing of the same location.
unsafe impl<T: Send> Send for WritePtr<T> {}
// SAFETY: shared references to `WritePtr<T>` do not grant mutation beyond the
// indexed `write` method. The conversion code partitions indices so each
// thread writes to a unique slot, which makes shared access race-free.
unsafe impl<T: Send + Sync> Sync for WritePtr<T> {}

impl<T> WritePtr<T> {
    #[inline]
    unsafe fn write(self, pos: usize, value: T) {
        // SAFETY: `self` points to a NumPy allocation created for this result
        // and `pos` is always within bounds of that allocation. Each `pos` is
        // assigned exactly once by the outer scatter loop, so no aliasing write
        // races occur.
        unsafe { *self.0.as_ptr().add(pos) = value };
    }
}

pub(crate) fn convert_single_result_f64<'py>(
    py: Python<'py>,
    result: GroupByResultF64,
) -> PyResult<SingleGroupByReturnF64<'py>> {
    let GroupByResultF64 { keys, values } = result;
    let keys_1d = keys.into_pyarray(py);
    let values_1d = values.into_pyarray(py);
    Ok((keys_1d, values_1d))
}

pub(crate) fn build_single_profile_dict<'py>(
    py: Python<'py>,
    profile: &groupby::SingleKeyPhaseProfile,
    materialize_conversion_s: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let profile_dict = PyDict::new(py);
    let materialize_s = profile.materialize_s + materialize_conversion_s;
    let final_group_count = profile.final_group_count;
    let partial_group_total = profile.partial_group_total;
    let partial_to_final_ratio = if final_group_count == 0 {
        0.0
    } else {
        partial_group_total as f64 / final_group_count as f64
    };
    let rust_total_s = profile.local_build_s + profile.merge_s + profile.reorder_s + materialize_s;

    profile_dict.set_item("local_build_s", profile.local_build_s)?;
    profile_dict.set_item("merge_s", profile.merge_s)?;
    profile_dict.set_item("reorder_s", profile.reorder_s)?;
    profile_dict.set_item("materialize_s", materialize_s)?;
    profile_dict.set_item("rust_total_s", rust_total_s)?;
    profile_dict.set_item("partial_group_total", partial_group_total)?;
    profile_dict.set_item("final_group_count", final_group_count)?;
    profile_dict.set_item("partial_to_final_ratio", partial_to_final_ratio)?;
    Ok(profile_dict)
}

pub(crate) fn convert_single_result<'py>(
    py: Python<'py>,
    result: GroupByResultF64,
) -> PyResult<SingleGroupByReturn<'py>> {
    convert_single_result_f64(py, result)
}

pub(crate) fn convert_single_result_i64<'py>(
    py: Python<'py>,
    result: GroupByResultI64,
) -> PyResult<SingleGroupByReturnI64<'py>> {
    let GroupByResultI64 { keys, values } = result;
    let keys_1d = keys.into_pyarray(py);
    let values_1d = values.into_pyarray(py);
    Ok((keys_1d, values_1d))
}

pub(crate) fn convert_multi_result_f64<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResultF64,
) -> PyResult<MultiGroupByReturnF64<'py>> {
    let n_groups = result.values.len();
    let n_keys = result.n_keys;
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = result.perm.as_deref();
    if let Some(p) = perm {
        debug_assert_eq!(p.len(), n_groups);
    }

    // Fast path: for small outputs, build Rust Vec columns and transfer ownership into NumPy.
    // This avoids per-element pointer writes and Rayon overhead.
    if perm.is_none() && n_groups.saturating_mul(n_keys) <= SMALL_DIRECT_THRESHOLD_ELEMS {
        let mut key_cols: Vec<Vec<i64>> =
            (0..n_keys).map(|_| Vec::with_capacity(n_groups)).collect();
        for g in 0..n_groups {
            let base = g * n_keys;
            for (col, key_col) in key_cols.iter_mut().enumerate() {
                key_col.push(result.keys_flat[base + col]);
            }
        }

        let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = key_cols
            .into_iter()
            .map(|col| col.into_pyarray(py))
            .collect();
        let values_1d = result.values.into_pyarray(py);
        return Ok((key_arrays, values_1d));
    }

    // Allocate output arrays under the GIL.
    // SAFETY: we will initialize all elements before returning to Python.
    let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = (0..n_keys)
        .map(|_| unsafe { PyArray1::<i64>::new(py, n_groups, false) })
        .collect();
    let values_out = unsafe { PyArray1::<f64>::new(py, n_groups, false) };

    // Extract typed writable pointers under GIL; fill outside GIL.
    let key_ptrs: Vec<WritePtr<i64>> = key_arrays
        .iter()
        .map(|a| WritePtr(NonNull::new(a.data()).expect("numpy array data pointer is non-null")))
        .collect();
    let values_ptr =
        WritePtr(NonNull::new(values_out.data()).expect("numpy array data pointer is non-null"));

    let keys_flat = result.keys_flat;
    let values = result.values;

    py.detach(|| {
        // Rayon overhead can dominate for small outputs.
        if n_groups < 50_000 {
            for out_g in 0..n_groups {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe { values_ptr.write(out_g, values[src_g]) };
                for col in 0..n_keys {
                    unsafe { key_ptrs[col].write(out_g, keys_flat[base + col]) };
                }
            }
            return;
        }

        let chunk = 1usize.max(n_groups / (rayon::current_num_threads() * 8));
        (0..n_groups)
            .into_par_iter()
            .with_min_len(chunk)
            .for_each(|out_g| {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe { values_ptr.write(out_g, values[src_g]) };
                for col in 0..n_keys {
                    unsafe { key_ptrs[col].write(out_g, keys_flat[base + col]) };
                }
            });
    });

    Ok((key_arrays, values_out))
}

pub(crate) fn convert_multi_result_i64<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResultI64,
) -> PyResult<MultiGroupByReturnI64<'py>> {
    let n_groups = result.values.len();
    let n_keys = result.n_keys;
    debug_assert_eq!(result.keys_flat.len(), n_groups * n_keys);

    let perm = result.perm.as_deref();
    if let Some(p) = perm {
        debug_assert_eq!(p.len(), n_groups);
    }

    if perm.is_none() && n_groups.saturating_mul(n_keys) <= SMALL_DIRECT_THRESHOLD_ELEMS {
        let mut key_cols: Vec<Vec<i64>> =
            (0..n_keys).map(|_| Vec::with_capacity(n_groups)).collect();
        for g in 0..n_groups {
            let base = g * n_keys;
            for (col, key_col) in key_cols.iter_mut().enumerate() {
                key_col.push(result.keys_flat[base + col]);
            }
        }

        let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = key_cols
            .into_iter()
            .map(|col| col.into_pyarray(py))
            .collect();
        let values_1d = result.values.into_pyarray(py);
        return Ok((key_arrays, values_1d));
    }

    let key_arrays: Vec<Bound<'py, PyArray1<i64>>> = (0..n_keys)
        .map(|_| unsafe { PyArray1::<i64>::new(py, n_groups, false) })
        .collect();
    let values_out = unsafe { PyArray1::<i64>::new(py, n_groups, false) };

    let key_ptrs: Vec<WritePtr<i64>> = key_arrays
        .iter()
        .map(|a| WritePtr(NonNull::new(a.data()).expect("numpy array data pointer is non-null")))
        .collect();
    let values_ptr =
        WritePtr(NonNull::new(values_out.data()).expect("numpy array data pointer is non-null"));

    let keys_flat = result.keys_flat;
    let values = result.values;

    py.detach(|| {
        if n_groups < 50_000 {
            for out_g in 0..n_groups {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe { values_ptr.write(out_g, values[src_g]) };
                for col in 0..n_keys {
                    unsafe { key_ptrs[col].write(out_g, keys_flat[base + col]) };
                }
            }
            return;
        }

        let chunk = 1usize.max(n_groups / (rayon::current_num_threads() * 8));
        (0..n_groups)
            .into_par_iter()
            .with_min_len(chunk)
            .for_each(|out_g| {
                let src_g = perm.map_or(out_g, |p| p[out_g]);
                let base = src_g * n_keys;
                unsafe { values_ptr.write(out_g, values[src_g]) };
                for col in 0..n_keys {
                    unsafe { key_ptrs[col].write(out_g, keys_flat[base + col]) };
                }
            });
    });

    Ok((key_arrays, values_out))
}

pub(crate) fn convert_multi_result<'py>(
    py: Python<'py>,
    result: groupby_multi::GroupByMultiResultF64,
) -> PyResult<MultiGroupByReturn<'py>> {
    convert_multi_result_f64(py, result)
}

#[cfg(test)]
mod tests {
    use super::WritePtr;
    use std::ptr::NonNull;

    #[test]
    fn write_ptr_writes_unique_slots() {
        let mut values = vec![0_i64; 4];
        let ptr = WritePtr(NonNull::new(values.as_mut_ptr()).expect("vec pointer is non-null"));

        for (pos, value) in [7, 11, 13, 17].into_iter().enumerate() {
            // SAFETY: `pos` is within `values`, and each slot is written once.
            unsafe { ptr.write(pos, value) };
        }

        assert_eq!(values, [7, 11, 13, 17]);
    }
}
