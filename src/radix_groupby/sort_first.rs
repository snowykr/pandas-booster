use std::cmp::Ordering;

use crate::aggregation::{Aggregator, CountAggI64, MaxAggF64, MinAggF64, SumAggF64};

use super::result::GroupByMultiResult;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SortFirstDiagnostics {
    pub lexicographic_permutation_built: bool,
    pub segment_scan_count: usize,
    pub post_aggregation_sort_count: usize,
}

pub(super) fn sort_first_groupby_sum_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<(GroupByMultiResult<f64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<f64, SumAggF64, f64>(key_slices, values)
}

pub(super) fn sort_first_groupby_max_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<(GroupByMultiResult<f64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<f64, MaxAggF64, f64>(key_slices, values)
}

pub(super) fn sort_first_groupby_min_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<(GroupByMultiResult<f64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<f64, MinAggF64, f64>(key_slices, values)
}

pub(super) fn sort_first_groupby_count_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<(GroupByMultiResult<i64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<i64, CountAggI64, i64>(key_slices, values)
}

fn sort_first_groupby<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<(GroupByMultiResult<O>, SortFirstDiagnostics), String>
where
    T: Copy,
    O: Copy,
    A: Aggregator<T, O>,
{
    validate_inputs(key_slices, values.len())?;

    let permutation = build_lexicographic_permutation(key_slices, values.len());
    let result = aggregate_segments::<T, A, O>(key_slices, values, &permutation);
    let diagnostics = SortFirstDiagnostics {
        lexicographic_permutation_built: true,
        segment_scan_count: 1,
        post_aggregation_sort_count: 0,
    };

    Ok((result, diagnostics))
}

fn validate_inputs(key_slices: &[&[i64]], n_rows: usize) -> Result<(), String> {
    if key_slices.is_empty() {
        return Err("sort-first groupby requires at least one key column".to_owned());
    }

    for (idx, col) in key_slices.iter().enumerate() {
        if col.len() != n_rows {
            return Err(format!(
                "Key column {idx} has length {}, expected {n_rows}",
                col.len()
            ));
        }
    }

    Ok(())
}

fn build_lexicographic_permutation(key_slices: &[&[i64]], n_rows: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..n_rows).collect();
    permutation.sort_by(|left, right| compare_rows(key_slices, *left, *right));
    permutation
}

fn compare_rows(key_slices: &[&[i64]], left: usize, right: usize) -> Ordering {
    for col in key_slices {
        match col[left].cmp(&col[right]) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }

    left.cmp(&right)
}

fn aggregate_segments<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
    permutation: &[usize],
) -> GroupByMultiResult<O>
where
    T: Copy,
    O: Copy,
    A: Aggregator<T, O>,
{
    let n_keys = key_slices.len();
    let mut keys_flat = Vec::with_capacity(permutation.len().saturating_mul(n_keys));
    let mut out_values = Vec::with_capacity(permutation.len());
    let mut cursor = 0usize;

    while cursor < permutation.len() {
        let segment_first_row = permutation[cursor];
        for col in key_slices {
            keys_flat.push(col[segment_first_row]);
        }

        let mut accumulator = A::init();
        while cursor < permutation.len()
            && rows_have_same_key(key_slices, segment_first_row, permutation[cursor])
        {
            accumulator.update(values[permutation[cursor]]);
            cursor += 1;
        }
        out_values.push(accumulator.finalize_owned());
    }

    GroupByMultiResult {
        keys_flat,
        n_keys,
        values: out_values,
        perm: None,
    }
}

fn rows_have_same_key(key_slices: &[&[i64]], left: usize, right: usize) -> bool {
    key_slices.iter().all(|col| col[left] == col[right])
}
