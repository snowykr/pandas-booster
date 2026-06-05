use crate::aggregation::Aggregator;

use super::engine::radix_groupby_engine;
use super::firstseen_u32::radix_groupby_engine_firstseen_u32;
use super::firstseen_u64::radix_groupby_engine_firstseen_u64;
use super::keys::{CompositeKeyOps, FixedKeyOps, RadixKeyOps};
use super::order::sort_groupby_result;
use super::result::GroupByMultiResult;

pub(super) fn radix_groupby_fixed<const N: usize, T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_engine::<FixedKeyOps<N>, T, A, O>(FixedKeyOps::<N>::new(N), key_slices, values)
}

pub(super) fn radix_groupby<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_engine::<CompositeKeyOps, T, A, O>(
        CompositeKeyOps::new(key_slices.len()),
        key_slices,
        values,
    )
}

pub(super) fn radix_groupby_dispatch<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => radix_groupby_fixed::<1, T, A, O>(key_slices, values),
        2 => radix_groupby_fixed::<2, T, A, O>(key_slices, values),
        3 => radix_groupby_fixed::<3, T, A, O>(key_slices, values),
        4 => radix_groupby_fixed::<4, T, A, O>(key_slices, values),
        5 => radix_groupby_fixed::<5, T, A, O>(key_slices, values),
        6 => radix_groupby_fixed::<6, T, A, O>(key_slices, values),
        7 => radix_groupby_fixed::<7, T, A, O>(key_slices, values),
        8 => radix_groupby_fixed::<8, T, A, O>(key_slices, values),
        9 => radix_groupby_fixed::<9, T, A, O>(key_slices, values),
        10 => radix_groupby_fixed::<10, T, A, O>(key_slices, values),
        _ => radix_groupby::<T, A, O>(key_slices, values),
    }
}

pub(super) fn radix_groupby_dispatch_firstseen_u32<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<1>, T, A, O>(
            FixedKeyOps::<1>::new(1),
            key_slices,
            values,
        ),
        2 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<2>, T, A, O>(
            FixedKeyOps::<2>::new(2),
            key_slices,
            values,
        ),
        3 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<3>, T, A, O>(
            FixedKeyOps::<3>::new(3),
            key_slices,
            values,
        ),
        4 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<4>, T, A, O>(
            FixedKeyOps::<4>::new(4),
            key_slices,
            values,
        ),
        5 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<5>, T, A, O>(
            FixedKeyOps::<5>::new(5),
            key_slices,
            values,
        ),
        6 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<6>, T, A, O>(
            FixedKeyOps::<6>::new(6),
            key_slices,
            values,
        ),
        7 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<7>, T, A, O>(
            FixedKeyOps::<7>::new(7),
            key_slices,
            values,
        ),
        8 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<8>, T, A, O>(
            FixedKeyOps::<8>::new(8),
            key_slices,
            values,
        ),
        9 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<9>, T, A, O>(
            FixedKeyOps::<9>::new(9),
            key_slices,
            values,
        ),
        10 => radix_groupby_engine_firstseen_u32::<FixedKeyOps<10>, T, A, O>(
            FixedKeyOps::<10>::new(10),
            key_slices,
            values,
        ),
        _ => radix_groupby_engine_firstseen_u32::<CompositeKeyOps, T, A, O>(
            CompositeKeyOps::new(key_slices.len()),
            key_slices,
            values,
        ),
    }
}

pub(super) fn radix_groupby_dispatch_firstseen_u64<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    match key_slices.len() {
        1 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<1>, T, A, O>(
            FixedKeyOps::<1>::new(1),
            key_slices,
            values,
        ),
        2 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<2>, T, A, O>(
            FixedKeyOps::<2>::new(2),
            key_slices,
            values,
        ),
        3 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<3>, T, A, O>(
            FixedKeyOps::<3>::new(3),
            key_slices,
            values,
        ),
        4 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<4>, T, A, O>(
            FixedKeyOps::<4>::new(4),
            key_slices,
            values,
        ),
        5 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<5>, T, A, O>(
            FixedKeyOps::<5>::new(5),
            key_slices,
            values,
        ),
        6 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<6>, T, A, O>(
            FixedKeyOps::<6>::new(6),
            key_slices,
            values,
        ),
        7 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<7>, T, A, O>(
            FixedKeyOps::<7>::new(7),
            key_slices,
            values,
        ),
        8 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<8>, T, A, O>(
            FixedKeyOps::<8>::new(8),
            key_slices,
            values,
        ),
        9 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<9>, T, A, O>(
            FixedKeyOps::<9>::new(9),
            key_slices,
            values,
        ),
        10 => radix_groupby_engine_firstseen_u64::<FixedKeyOps<10>, T, A, O>(
            FixedKeyOps::<10>::new(10),
            key_slices,
            values,
        ),
        _ => radix_groupby_engine_firstseen_u64::<CompositeKeyOps, T, A, O>(
            CompositeKeyOps::new(key_slices.len()),
            key_slices,
            values,
        ),
    }
}

pub(super) fn radix_groupby_firstseen_u32<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_dispatch_firstseen_u32::<T, A, O>(key_slices, values)
}

pub(super) fn radix_groupby_firstseen_u64<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    radix_groupby_dispatch_firstseen_u64::<T, A, O>(key_slices, values)
}

pub(super) fn radix_groupby_sorted<T, A, O>(
    key_slices: &[&[i64]],
    values: &[T],
) -> Result<GroupByMultiResult<O>, String>
where
    T: Copy + Send + Sync,
    O: Copy + Send + Sync,
    A: Aggregator<T, O> + Clone + Default + Send,
{
    let mut result = radix_groupby_dispatch::<T, A, O>(key_slices, values)?;
    sort_groupby_result(&mut result);
    Ok(result)
}

// Public API - unsorted (fastest)
