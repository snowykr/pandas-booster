#[cfg(test)]
use std::cmp::Ordering;
#[cfg(test)]
use std::time::Instant;

#[cfg(test)]
use crate::aggregation::{Aggregator, CountAggI64, MaxAggF64, MinAggF64, SumAggF64};

#[cfg(test)]
use super::result::GroupByMultiResult;
#[cfg(test)]
use super::sort_first_routing::MAX_KEY_COLUMNS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SortFirstDiagnostics {
    pub lexicographic_permutation_built: bool,
    pub segment_scan_count: usize,
    pub post_aggregation_sort_count: usize,
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct SortFirstPhaseTimings {
    pub lexicographic_permutation_s: f64,
    pub segment_scan_s: f64,
}

#[cfg(test)]
pub(super) fn sort_first_groupby_sum_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<(GroupByMultiResult<f64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<f64, SumAggF64, f64>(key_slices, values)
}

#[cfg(test)]
pub(super) fn sort_first_groupby_min_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<(GroupByMultiResult<f64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<f64, MinAggF64, f64>(key_slices, values)
}

#[cfg(test)]
pub(super) fn sort_first_groupby_max_f64(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<(GroupByMultiResult<f64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<f64, MaxAggF64, f64>(key_slices, values)
}

#[cfg(test)]
pub(super) fn sort_first_groupby_count_i64(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<(GroupByMultiResult<i64>, SortFirstDiagnostics), String> {
    sort_first_groupby::<i64, CountAggI64, i64>(key_slices, values)
}

#[cfg(test)]
fn sort_first_groupby<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<(GroupByMultiResult<O>, SortFirstDiagnostics), String>
where
    T: Copy,
    O: Copy,
    A: Aggregator<T, O>,
{
    let (result, diagnostics, _timings) =
        sort_first_groupby_profiled::<T, A, O>(key_slices, values)?;
    Ok((result, diagnostics))
}

#[cfg(test)]
fn sort_first_groupby_profiled<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<
    (
        GroupByMultiResult<O>,
        SortFirstDiagnostics,
        SortFirstPhaseTimings,
    ),
    String,
>
where
    T: Copy,
    O: Copy,
    A: Aggregator<T, O>,
{
    validate_inputs(key_slices, values.len())?;

    let permutation_start = Instant::now();
    let permutation = build_lexicographic_permutation(key_slices, values.len());
    let lexicographic_permutation_s = permutation_start.elapsed().as_secs_f64();

    let segment_start = Instant::now();
    let result = aggregate_segments::<T, A, O>(key_slices, values, &permutation);
    let segment_scan_s = segment_start.elapsed().as_secs_f64();
    let diagnostics = SortFirstDiagnostics {
        lexicographic_permutation_built: true,
        segment_scan_count: 1,
        post_aggregation_sort_count: 0,
    };
    let timings = SortFirstPhaseTimings {
        lexicographic_permutation_s,
        segment_scan_s,
    };

    Ok((result, diagnostics, timings))
}

#[cfg(test)]
fn validate_inputs(key_slices: &[&[i64]], n_rows: usize) -> Result<(), String> {
    if key_slices.is_empty() {
        return Err("sort-first groupby requires at least one key column".to_owned());
    }

    if key_slices.len() > MAX_KEY_COLUMNS {
        return Err(format!(
            "sort-first groupby supports at most {MAX_KEY_COLUMNS} key columns"
        ));
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

#[cfg(test)]
fn build_lexicographic_permutation(key_slices: &[&[i64]], n_rows: usize) -> Vec<usize> {
    let mut permutation: Vec<usize> = (0..n_rows).collect();
    permutation.sort_by(|left, right| compare_rows(key_slices, *left, *right));
    permutation
}

#[cfg(test)]
fn compare_rows(key_slices: &[&[i64]], left: usize, right: usize) -> Ordering {
    for col in key_slices {
        match col[left].cmp(&col[right]) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }

    left.cmp(&right)
}

#[cfg(test)]
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

#[cfg(test)]
fn rows_have_same_key(key_slices: &[&[i64]], left: usize, right: usize) -> bool {
    key_slices.iter().all(|col| col[left] == col[right])
}
