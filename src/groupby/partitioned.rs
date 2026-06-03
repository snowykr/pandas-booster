use ahash::AHashMap;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::hash::{Hash, Hasher};

use crate::aggregation::Aggregator;
use crate::radix_groupby::stable_scatter_by_partition;

use super::chunks::validate_firstseen_deterministic_inputs;
use super::order::{FirstSeenMaterializedResult, FirstSeenRowIndex};
use super::result::GroupByResult;

pub(super) struct SingleKeyPartitionState<A, I> {
    gid_map: AHashMap<i64, usize>,
    keys_by_gid: Vec<i64>,
    aggs: Vec<A>,
    first_seen: Vec<I>,
}

impl<A, I> SingleKeyPartitionState<A, I> {
    #[inline]
    pub(super) fn len(&self) -> usize {
        self.aggs.len()
    }
}

#[inline]
fn hash_single_key(key: i64) -> u64 {
    use ahash::AHasher;

    let mut hasher = AHasher::default();
    key.hash(&mut hasher);
    hasher.finish()
}

fn compute_single_key_hashes(keys: &[i64]) -> Vec<u64> {
    let mut hashes = vec![0u64; keys.len()];
    hashes
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, hash_out)| {
            *hash_out = hash_single_key(keys[row]);
        });
    hashes
}

fn build_partitioned_firstseen_state<T, O, A, I>(
    perm_rows: &[usize],
    keys: &[i64],
    values: &[T],
) -> SingleKeyPartitionState<A, I>
where
    T: Copy,
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    let mut gid_map: AHashMap<i64, usize> = AHashMap::default();
    let mut keys_by_gid: Vec<i64> = Vec::new();
    let mut aggs: Vec<A> = Vec::new();
    let mut first_seen: Vec<I> = Vec::new();

    for &row in perm_rows {
        let key = keys[row];
        let value = values[row];
        let row_index = I::from_row_index(row);

        if let Some(&gid) = gid_map.get(&key) {
            if row_index < first_seen[gid] {
                first_seen[gid] = row_index;
            }
            aggs[gid].update(value);
        } else {
            let gid = aggs.len();
            gid_map.insert(key, gid);
            keys_by_gid.push(key);

            let mut agg = A::init();
            agg.update(value);
            aggs.push(agg);
            first_seen.push(row_index);
        }
    }

    SingleKeyPartitionState {
        gid_map,
        keys_by_gid,
        aggs,
        first_seen,
    }
}

pub(super) fn build_partitioned_deterministic_firstseen_states<T, O, A, I>(
    keys: &[i64],
    values: &[T],
) -> PyResult<Vec<SingleKeyPartitionState<A, I>>>
where
    T: Copy + Send + Sync,
    A: Aggregator<T, O> + Send,
    I: FirstSeenRowIndex,
{
    let n_rows = validate_firstseen_deterministic_inputs::<T, I>(keys, values)?;
    if n_rows == 0 {
        return Ok(Vec::new());
    }

    let hashes = compute_single_key_hashes(keys);
    let (perm, offsets) = stable_scatter_by_partition(&hashes);

    Ok((0..offsets.len() - 1)
        .into_par_iter()
        .map(|partition| {
            let start = offsets[partition];
            let end = offsets[partition + 1];
            build_partitioned_firstseen_state::<T, O, A, I>(&perm[start..end], keys, values)
        })
        .collect())
}

pub(super) fn materialize_partitioned_deterministic_firstseen_states<T, O, A, I>(
    states: Vec<SingleKeyPartitionState<A, I>>,
) -> FirstSeenMaterializedResult<I, O>
where
    O: Copy,
    A: Aggregator<T, O>,
    I: FirstSeenRowIndex,
{
    let total_groups: usize = states.iter().map(SingleKeyPartitionState::len).sum();
    let mut result_keys = Vec::with_capacity(total_groups);
    let mut result_values = Vec::with_capacity(total_groups);
    let mut first_seen = Vec::with_capacity(total_groups);

    for state in states {
        let SingleKeyPartitionState {
            gid_map,
            keys_by_gid,
            aggs,
            first_seen: state_first_seen,
        } = state;

        debug_assert_eq!(gid_map.len(), keys_by_gid.len());
        debug_assert_eq!(keys_by_gid.len(), aggs.len());
        debug_assert_eq!(keys_by_gid.len(), state_first_seen.len());

        for ((key, agg), first) in keys_by_gid
            .into_iter()
            .zip(aggs.into_iter())
            .zip(state_first_seen.into_iter())
        {
            result_keys.push(key);
            result_values.push(agg.finalize_owned());
            first_seen.push(first);
        }
    }

    FirstSeenMaterializedResult {
        result: GroupByResult {
            keys: result_keys,
            values: result_values,
        },
        first_seen,
    }
}
