#[cfg(test)]
use std::cmp::Ordering;
use std::time::Instant;

use super::result::GroupByMultiResult;
use super::sort_first_routing::{checked_route_vec, memory_budget};
use crate::aggregation::Aggregator;
use crate::radix_sort_multi_key::{sort_first_columnar_permutation, MultiKeySortProof};
use rayon::prelude::*;

#[cfg(test)]
use crate::aggregation::{CountAggI64, MaxAggF64, MinAggF64, SumAggF64};

#[cfg(test)]
use super::sort_first_routing::MAX_KEY_COLUMNS;

#[derive(Debug, Clone, Default)]
pub(super) struct ProductionSortFirstTimings {
    pub layout_s: f64,
    pub radix_s: f64,
    pub boundary_discovery_s: f64,
    pub starts_setup_s: f64,
    pub reduction_emission_s: f64,
}

// This is independent of the comparison-sort test prototype below. The radix
// permutation is stable, and each accumulator owns an entire group: no merge
// can change NaN or first-equal-maximum (signed zero) semantics.
pub(super) fn production_sort_first_groupby<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<
    (
        GroupByMultiResult<O>,
        SortFirstDiagnostics,
        ProductionSortFirstTimings,
        MultiKeySortProof,
    ),
    String,
>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O>,
{
    let budget = memory_budget(values.len(), key_slices.len())
        .map_err(|reason| format!("sort-first resource refusal: {reason:?}"))?;
    let (permutation, proof, layout_s, radix_s) = sort_first_columnar_permutation(key_slices)?;
    let n = permutation.len();
    #[cfg(test)]
    super::sort_first_routing::observe_scratch(
        "radix_returned",
        permutation.capacity() * std::mem::size_of::<usize>(),
    );
    let mut timings = ProductionSortFirstTimings {
        layout_s,
        radix_s,
        ..Default::default()
    };
    let clock = Instant::now();
    let mut flags = checked_route_vec::<u8>(n)?;
    flags.resize(n, 0);
    #[cfg(test)]
    super::sort_first_routing::observe_scratch(
        "boundary_live",
        permutation.capacity() * std::mem::size_of::<usize>() + flags.capacity(),
    );
    flags.par_iter_mut().enumerate().for_each(|(idx, flag)| {
        *flag = u8::from(
            idx == 0
                || !key_slices
                    .iter()
                    .all(|col| col[permutation[idx - 1]] == col[permutation[idx]]),
        );
    });
    timings.boundary_discovery_s = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let mut counts = checked_route_vec::<usize>(budget.chunks)?;
    counts.resize(budget.chunks, 0);
    counts
        .par_iter_mut()
        .zip(flags.par_chunks(budget.chunk_size))
        .for_each(|(count, block)| *count = block.iter().map(|flag| *flag as usize).sum());
    let groups = counts
        .iter()
        .try_fold(0usize, |sum, count| sum.checked_add(*count))
        .ok_or("sort-first group count overflow")?;
    let mut starts =
        checked_route_vec::<usize>(groups.checked_add(1).ok_or("sort-first starts overflow")?)?;
    starts.resize(groups + 1, n);
    #[cfg(test)]
    super::sort_first_routing::observe_scratch(
        "starts_live",
        (permutation.capacity() + counts.capacity() + starts.capacity())
            * std::mem::size_of::<usize>()
            + flags.capacity(),
    );
    rayon::scope(|scope| {
        let mut remaining = &mut starts[..groups];
        for (block_idx, (block, count)) in flags.chunks(budget.chunk_size).zip(&counts).enumerate()
        {
            let (destination, rest) = remaining.split_at_mut(*count);
            remaining = rest;
            scope.spawn(move |_| {
                for (slot, (offset, _)) in destination
                    .iter_mut()
                    .zip(block.iter().enumerate().filter(|(_, flag)| **flag != 0))
                {
                    *slot = block_idx * budget.chunk_size + offset;
                }
            });
        }
    });
    drop(counts);
    drop(flags);
    #[cfg(test)]
    super::sort_first_routing::observe_scratch(
        "boundary_released",
        (permutation.capacity() + starts.capacity()) * std::mem::size_of::<usize>(),
    );
    timings.starts_setup_s = clock.elapsed().as_secs_f64();
    let clock = Instant::now();
    let k = key_slices.len();
    let key_len = groups
        .checked_mul(k)
        .ok_or("sort-first output size overflow")?;
    let mut keys_flat = checked_route_vec::<i64>(key_len)?;
    keys_flat.resize(key_len, 0);
    let mut out_values = checked_route_vec::<O>(groups)?;
    if let Some(&value) = values.first() {
        let mut initial = A::init();
        initial.update(value);
        out_values.resize(groups, initial.finalize_owned());
    }
    keys_flat
        .par_chunks_mut(k)
        .zip(out_values.par_iter_mut())
        .enumerate()
        .for_each(|(group, (keys, output))| {
            let first = permutation[starts[group]];
            for (slot, col) in keys.iter_mut().zip(key_slices) {
                *slot = col[first];
            }
            let mut accumulator = A::init();
            for row in &permutation[starts[group]..starts[group + 1]] {
                accumulator.update(values[*row]);
            }
            *output = accumulator.finalize_owned();
        });
    drop(starts);
    drop(permutation);
    #[cfg(test)]
    super::sort_first_routing::observe_scratch("conversion_ready", 0);
    timings.reduction_emission_s = clock.elapsed().as_secs_f64();
    Ok((
        GroupByMultiResult {
            keys_flat,
            n_keys: k,
            values: out_values,
            perm: None,
        },
        SortFirstDiagnostics {
            lexicographic_permutation_built: true,
            segment_scan_count: 1,
            post_aggregation_sort_count: 0,
        },
        timings,
        proof,
    ))
}

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
