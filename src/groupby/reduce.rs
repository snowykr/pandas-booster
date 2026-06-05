use ahash::AHashMap;
use rayon::prelude::*;
use std::collections::hash_map::Entry;

use crate::aggregation::Aggregator;

pub(super) trait PairwiseReduceValue<T, O, A>: Send
where
    A: Aggregator<T, O>,
{
    fn merge_value(left: &mut Self, right: Self);
}

impl<T, O, A> PairwiseReduceValue<T, O, A> for A
where
    A: Aggregator<T, O> + Send,
{
    #[inline]
    fn merge_value(left: &mut Self, right: Self) {
        left.merge(right);
    }
}

impl<T, O, A> PairwiseReduceValue<T, O, A> for (A, u32)
where
    A: Aggregator<T, O> + Send,
{
    #[inline]
    fn merge_value(left: &mut Self, right: Self) {
        left.1 = left.1.min(right.1);
        left.0.merge(right.0);
    }
}

impl<T, O, A> PairwiseReduceValue<T, O, A> for (A, u64)
where
    A: Aggregator<T, O> + Send,
{
    #[inline]
    fn merge_value(left: &mut Self, right: Self) {
        left.1 = left.1.min(right.1);
        left.0.merge(right.0);
    }
}

pub(super) fn reduce_partial_maps_pairwise<T, O, A, V>(
    mut partials: Vec<AHashMap<i64, V>>,
) -> AHashMap<i64, V>
where
    T: Copy + Send + Sync,
    O: Copy,
    A: Aggregator<T, O> + Send,
    V: PairwiseReduceValue<T, O, A>,
{
    if partials.is_empty() {
        return AHashMap::default();
    }

    while partials.len() > 1 {
        let carry = if partials.len() % 2 == 1 {
            partials.pop()
        } else {
            None
        };

        let mut iter = partials.into_iter();
        let mut pairs = Vec::with_capacity(iter.len() / 2);
        while let Some(left) = iter.next() {
            let right = iter.next().expect("pairwise merge requires right map");
            pairs.push((left, right));
        }

        let mut next: Vec<AHashMap<i64, V>> = pairs
            .into_par_iter()
            .map(|(mut left, right)| {
                for (k, v) in right {
                    match left.entry(k) {
                        Entry::Occupied(mut e) => {
                            V::merge_value(e.get_mut(), v);
                        }
                        Entry::Vacant(e) => {
                            e.insert(v);
                        }
                    }
                }
                left
            })
            .collect();

        if let Some(map) = carry {
            next.push(map);
        }
        partials = next;
    }

    partials
        .pop()
        .expect("non-empty partial maps must produce one map")
}
