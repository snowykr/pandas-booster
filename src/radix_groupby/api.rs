use crate::aggregation::{
    CountAggF64, CountAggI64, MaxAggF64, MaxAggI64, MeanAggF64, MeanAggI64, MedianAggF64,
    MedianAggI64, MinAggF64, MinAggI64, ProdAggF64, ProdAggI64, StdAggF64, StdAggI64, SumAggF64,
    SumAggI64, VarAggF64, VarAggI64,
};

use super::dispatch::{
    radix_groupby_dispatch, radix_groupby_firstseen_u32, radix_groupby_firstseen_u64,
};
use super::result::GroupByMultiResult;

macro_rules! impl_radix_dispatch {
    ($name:ident, $val_type:ty, $agg:ty, $out_type:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult<$out_type>, String> {
            radix_groupby_dispatch::<$val_type, $agg, $out_type>(key_slices, values)
        }
    };
}

impl_radix_dispatch!(radix_groupby_sum_f64, f64, SumAggF64, f64);
impl_radix_dispatch!(radix_groupby_prod_f64, f64, ProdAggF64, f64);
impl_radix_dispatch!(radix_groupby_mean_f64, f64, MeanAggF64, f64);
impl_radix_dispatch!(radix_groupby_median_f64, f64, MedianAggF64, f64);
impl_radix_dispatch!(radix_groupby_var_f64, f64, VarAggF64, f64);
impl_radix_dispatch!(radix_groupby_std_f64, f64, StdAggF64, f64);
impl_radix_dispatch!(radix_groupby_min_f64, f64, MinAggF64, f64);
impl_radix_dispatch!(radix_groupby_max_f64, f64, MaxAggF64, f64);

impl_radix_dispatch!(radix_groupby_sum_i64, i64, SumAggI64, i64);
impl_radix_dispatch!(radix_groupby_prod_i64, i64, ProdAggI64, i64);
impl_radix_dispatch!(radix_groupby_mean_i64, i64, MeanAggI64, f64);
impl_radix_dispatch!(radix_groupby_median_i64, i64, MedianAggI64, f64);
impl_radix_dispatch!(radix_groupby_var_i64, i64, VarAggI64, f64);
impl_radix_dispatch!(radix_groupby_std_i64, i64, StdAggI64, f64);
impl_radix_dispatch!(radix_groupby_min_i64, i64, MinAggI64, i64);
impl_radix_dispatch!(radix_groupby_max_i64, i64, MaxAggI64, i64);

impl_radix_dispatch!(radix_groupby_count_f64, f64, CountAggF64, i64);
impl_radix_dispatch!(radix_groupby_count_i64, i64, CountAggI64, i64);

// Public API - first-seen ordered (for sort=False semantics)

macro_rules! impl_radix_firstseen_u32 {
    ($name:ident, $val_type:ty, $agg:ty, $out_type:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult<$out_type>, String> {
            radix_groupby_firstseen_u32::<$val_type, $agg, $out_type>(key_slices, values)
        }
    };
}

macro_rules! impl_radix_firstseen_u64 {
    ($name:ident, $val_type:ty, $agg:ty, $out_type:ty) => {
        pub fn $name(
            key_slices: &[&[i64]],
            values: &[$val_type],
        ) -> Result<GroupByMultiResult<$out_type>, String> {
            radix_groupby_firstseen_u64::<$val_type, $agg, $out_type>(key_slices, values)
        }
    };
}

impl_radix_firstseen_u32!(radix_groupby_sum_f64_firstseen_u32, f64, SumAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_prod_f64_firstseen_u32, f64, ProdAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_mean_f64_firstseen_u32, f64, MeanAggF64, f64);
impl_radix_firstseen_u32!(
    radix_groupby_median_f64_firstseen_u32,
    f64,
    MedianAggF64,
    f64
);
impl_radix_firstseen_u32!(radix_groupby_var_f64_firstseen_u32, f64, VarAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_std_f64_firstseen_u32, f64, StdAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_min_f64_firstseen_u32, f64, MinAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_max_f64_firstseen_u32, f64, MaxAggF64, f64);
impl_radix_firstseen_u32!(radix_groupby_count_f64_firstseen_u32, f64, CountAggF64, i64);

impl_radix_firstseen_u32!(radix_groupby_sum_i64_firstseen_u32, i64, SumAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_prod_i64_firstseen_u32, i64, ProdAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_mean_i64_firstseen_u32, i64, MeanAggI64, f64);
impl_radix_firstseen_u32!(
    radix_groupby_median_i64_firstseen_u32,
    i64,
    MedianAggI64,
    f64
);
impl_radix_firstseen_u32!(radix_groupby_var_i64_firstseen_u32, i64, VarAggI64, f64);
impl_radix_firstseen_u32!(radix_groupby_std_i64_firstseen_u32, i64, StdAggI64, f64);
impl_radix_firstseen_u32!(radix_groupby_min_i64_firstseen_u32, i64, MinAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_max_i64_firstseen_u32, i64, MaxAggI64, i64);
impl_radix_firstseen_u32!(radix_groupby_count_i64_firstseen_u32, i64, CountAggI64, i64);

impl_radix_firstseen_u64!(radix_groupby_sum_f64_firstseen_u64, f64, SumAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_prod_f64_firstseen_u64, f64, ProdAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_mean_f64_firstseen_u64, f64, MeanAggF64, f64);
impl_radix_firstseen_u64!(
    radix_groupby_median_f64_firstseen_u64,
    f64,
    MedianAggF64,
    f64
);
impl_radix_firstseen_u64!(radix_groupby_var_f64_firstseen_u64, f64, VarAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_std_f64_firstseen_u64, f64, StdAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_min_f64_firstseen_u64, f64, MinAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_max_f64_firstseen_u64, f64, MaxAggF64, f64);
impl_radix_firstseen_u64!(radix_groupby_count_f64_firstseen_u64, f64, CountAggF64, i64);

impl_radix_firstseen_u64!(radix_groupby_sum_i64_firstseen_u64, i64, SumAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_prod_i64_firstseen_u64, i64, ProdAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_mean_i64_firstseen_u64, i64, MeanAggI64, f64);
impl_radix_firstseen_u64!(
    radix_groupby_median_i64_firstseen_u64,
    i64,
    MedianAggI64,
    f64
);
impl_radix_firstseen_u64!(radix_groupby_var_i64_firstseen_u64, i64, VarAggI64, f64);
impl_radix_firstseen_u64!(radix_groupby_std_i64_firstseen_u64, i64, StdAggI64, f64);
impl_radix_firstseen_u64!(radix_groupby_min_i64_firstseen_u64, i64, MinAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_max_i64_firstseen_u64, i64, MaxAggI64, i64);
impl_radix_firstseen_u64!(radix_groupby_count_i64_firstseen_u64, i64, CountAggI64, i64);
