use crate::aggregation::{
    CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MedianAggF64,
    MedianAggI64, MinAggF64, MinAggI64, ProdAggF64, ProdAggI64, StdAggF64, StdAggI64, SumAggF64,
    SumAggI64, VarAggF64, VarAggI64,
};

use super::dispatch::{radix_groupby_sorted, radix_groupby_sorted_with_diagnostics};
use super::profile::profile_radix_groupby_sorted;
use super::result::{GroupByMultiResult, ProfiledGroupByMultiResult};
use super::sort_first_routing::SortFirstReducer;

pub fn radix_groupby_sum_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, SumAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_prod_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, ProdAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_mean_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, MeanAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_median_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, MedianAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_var_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, VarAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_std_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, StdAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_min_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<f64, MinAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_max_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<f64>, String> {
    let (result, _diagnostics) = radix_groupby_max_f64_sorted_with_diagnostics(key_slices, values)?;
    Ok(result)
}

pub(super) fn radix_groupby_max_f64_sorted_with_diagnostics(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<
    (
        GroupByMultiResult<f64>,
        super::dispatch::SortedDispatchDiagnostics,
    ),
    String,
> {
    radix_groupby_sorted_with_diagnostics::<f64, MaxAggF64, f64>(
        key_slices,
        values,
        SortFirstReducer::MaxF64,
    )
}

pub fn profile_radix_groupby_max_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<ProfiledGroupByMultiResult<f64>, String> {
    profile_radix_groupby_sorted::<f64, MaxAggF64, f64>(key_slices, values)
}

pub fn radix_groupby_sum_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, SumAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_prod_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, ProdAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_mean_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<i64, MeanAggI64, f64>(key_slices, values)
}

pub fn radix_groupby_median_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<i64, MedianAggI64, f64>(key_slices, values)
}

pub fn radix_groupby_var_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<i64, VarAggI64, f64>(key_slices, values)
}

pub fn radix_groupby_std_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<f64>, String> {
    radix_groupby_sorted::<i64, StdAggI64, f64>(key_slices, values)
}

pub fn radix_groupby_min_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    radix_groupby_sorted::<i64, MinAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_max_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    let (result, _diagnostics) = radix_groupby_max_i64_sorted_with_diagnostics(key_slices, values)?;
    Ok(result)
}

pub(super) fn radix_groupby_max_i64_sorted_with_diagnostics(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<
    (
        GroupByMultiResult<i64>,
        super::dispatch::SortedDispatchDiagnostics,
    ),
    String,
> {
    radix_groupby_sorted_with_diagnostics::<i64, MaxAggI64, i64>(
        key_slices,
        values,
        SortFirstReducer::MaxI64,
    )
}

pub fn profile_radix_groupby_max_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<ProfiledGroupByMultiResult<i64>, String> {
    profile_radix_groupby_sorted::<i64, MaxAggI64, i64>(key_slices, values)
}

pub fn radix_groupby_count_f64_sorted(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<GroupByMultiResult<i64>, String> {
    let (result, _diagnostics) =
        radix_groupby_count_f64_sorted_with_diagnostics(key_slices, values)?;
    Ok(result)
}

pub(super) fn radix_groupby_count_f64_sorted_with_diagnostics(
    key_slices: &[&[i64]],
    values: &[f64],
) -> Result<
    (
        GroupByMultiResult<i64>,
        super::dispatch::SortedDispatchDiagnostics,
    ),
    String,
> {
    radix_groupby_sorted_with_diagnostics::<f64, CountAggF64, i64>(
        key_slices,
        values,
        SortFirstReducer::CountF64,
    )
}

pub fn radix_groupby_count_i64_sorted(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<GroupByMultiResult<i64>, String> {
    let (result, _diagnostics) =
        radix_groupby_count_i64_sorted_with_diagnostics(key_slices, values)?;
    Ok(result)
}

pub(super) fn radix_groupby_count_i64_sorted_with_diagnostics(
    key_slices: &[&[i64]],
    values: &[i64],
) -> Result<
    (
        GroupByMultiResult<i64>,
        super::dispatch::SortedDispatchDiagnostics,
    ),
    String,
> {
    radix_groupby_sorted_with_diagnostics::<i64, CountAggI64, i64>(
        key_slices,
        values,
        SortFirstReducer::CountI64,
    )
}
