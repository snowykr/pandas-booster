use super::legacy::{
    parallel_groupby, parallel_groupby_firstseen_u32, parallel_groupby_firstseen_u64,
};
use super::partitioned::{
    build_partitioned_deterministic_firstseen_states,
    materialize_partitioned_deterministic_firstseen_states,
};
use super::test_support::{NonCloneVecAgg, OwnedOnlyAgg};
use ahash::AHashMap;
use rayon::ThreadPoolBuilder;

#[test]
fn test_parallel_groupby_reduce_merges_owned_aggregators_without_clone() {
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| {
        let n = 80_000usize;
        let keys = vec![7_i64; n];
        let values: Vec<i64> = (0..n as i64).collect();

        let result = parallel_groupby::<i64, NonCloneVecAgg, usize>(&keys, &values).unwrap();

        assert_eq!(result.keys, vec![7]);
        assert_eq!(result.values, vec![n]);
    });
}

#[test]
fn test_firstseen_materialization_uses_owned_finalize() {
    let keys = vec![2_i64, 1, 2, 1, 3];
    let values = vec![10_i64, 20, 30, 40, 50];

    let result =
        parallel_groupby_firstseen_u32::<i64, OwnedOnlyAgg, usize>(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![2, 1, 3]);
    assert_eq!(result.values, vec![2, 2, 1]);

    let result =
        parallel_groupby_firstseen_u64::<i64, OwnedOnlyAgg, usize>(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![2, 1, 3]);
    assert_eq!(result.values, vec![2, 2, 1]);
}

#[test]
fn test_partitioned_firstseen_materialization_uses_owned_finalize() {
    let keys = vec![5_i64, 4, 5, 3, 4, 2];
    let values = vec![10_i64, 20, 30, 40, 50, 60];
    let states = build_partitioned_deterministic_firstseen_states::<i64, usize, OwnedOnlyAgg, u32>(
        &keys, &values,
    )
    .unwrap();

    let materialized =
        materialize_partitioned_deterministic_firstseen_states::<i64, usize, OwnedOnlyAgg, u32>(
            states,
        );

    let mut counts = AHashMap::new();
    for (key, value) in materialized
        .result
        .keys
        .iter()
        .copied()
        .zip(materialized.result.values.iter().copied())
    {
        counts.insert(key, value);
    }

    assert_eq!(counts[&5], 2);
    assert_eq!(counts[&4], 2);
    assert_eq!(counts[&3], 1);
    assert_eq!(counts[&2], 1);
}
