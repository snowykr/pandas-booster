use super::legacy::{
    parallel_groupby, parallel_groupby_firstseen_u32, parallel_groupby_firstseen_u64,
};
use super::partitioned::{
    build_partitioned_deterministic_firstseen_states,
    materialize_partitioned_deterministic_firstseen_states,
};
use super::scalar_firstseen::{
    clear_scalar_firstseen_route_for_test, parallel_groupby_firstseen_deterministic_low_u32,
    parallel_groupby_firstseen_deterministic_low_u64, parallel_groupby_firstseen_legacy_low_u32,
    parallel_groupby_firstseen_legacy_low_u64, take_scalar_firstseen_route_for_test,
    ScalarFirstseenRoute,
};
use super::test_support::{NonCloneVecAgg, OwnedOnlyAgg};
use super::{
    parallel_groupby_count_f64_firstseen_u32, parallel_groupby_count_f64_firstseen_u64,
    parallel_groupby_count_i64_firstseen_u32, parallel_groupby_count_i64_firstseen_u64,
    parallel_groupby_max_f64_firstseen_u32, parallel_groupby_max_f64_firstseen_u64,
    parallel_groupby_max_i64_firstseen_u32, parallel_groupby_max_i64_firstseen_u64,
    parallel_groupby_mean_f64_firstseen_u32, parallel_groupby_mean_f64_firstseen_u64,
    parallel_groupby_mean_i64_firstseen_u32, parallel_groupby_mean_i64_firstseen_u64,
    parallel_groupby_min_f64_firstseen_u32, parallel_groupby_min_f64_firstseen_u64,
    parallel_groupby_min_i64_firstseen_u32, parallel_groupby_min_i64_firstseen_u64,
    parallel_groupby_sum_f64_firstseen_u32, parallel_groupby_sum_f64_firstseen_u64,
    parallel_groupby_sum_i64_firstseen_u32, parallel_groupby_sum_i64_firstseen_u64,
};
use crate::aggregation::{MinAggF64, SumAggF64};
use ahash::AHashMap;
use rayon::ThreadPoolBuilder;

fn observe_scalar_firstseen_route<R>(
    label: &str,
    run: impl FnOnce() -> pyo3::PyResult<R>,
    failures: &mut Vec<String>,
) {
    clear_scalar_firstseen_route_for_test();

    if let Err(err) = run() {
        panic!("{label} should execute before route assertion, got {err}");
    }

    let observed = take_scalar_firstseen_route_for_test();
    if observed != Some(ScalarFirstseenRoute::Partitioned) {
        failures.push(format!("{label} => {observed:?}"));
    }
}

macro_rules! observe_partitioned_routes {
    ($keys:expr, $values:expr, $failures:expr; $($label:literal => $route:path),+ $(,)?) => {
        $(
            observe_scalar_firstseen_route($label, || $route($keys, $values), $failures);
        )+
    };
}

fn assert_last_scalar_firstseen_route(label: &str, expected: ScalarFirstseenRoute) {
    let observed = take_scalar_firstseen_route_for_test();
    assert_eq!(observed, Some(expected), "{label}");
}

#[test]
fn deterministic_low_wrapper_selects_low_route_for_low_cardinality() {
    let keys = vec![2_i64, 1, 2, 3];
    let values = vec![10.0_f64, 20.0, 30.0, 40.0];

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_deterministic_low_u32::<f64, SumAggF64, f64>(&keys, &values)
            .unwrap();

    assert_eq!(result.keys, vec![2, 1, 3]);
    assert_eq!(result.values, vec![40.0, 20.0, 40.0]);
    assert_last_scalar_firstseen_route(
        "u32 deterministic low",
        ScalarFirstseenRoute::DeterministicLow,
    );

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_deterministic_low_u64::<f64, SumAggF64, f64>(&keys, &values)
            .unwrap();

    assert_eq!(result.keys, vec![2, 1, 3]);
    assert_eq!(result.values, vec![40.0, 20.0, 40.0]);
    assert_last_scalar_firstseen_route(
        "u64 deterministic low",
        ScalarFirstseenRoute::DeterministicLow,
    );
}

#[test]
fn deterministic_low_wrapper_selects_partitioned_for_high_uniqueness() {
    let n = 5_000usize;
    let keys: Vec<i64> = (0..n).map(|row| row as i64).collect();
    let values: Vec<f64> = (0..n).map(|row| row as f64).collect();

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_deterministic_low_u32::<f64, SumAggF64, f64>(&keys, &values)
            .unwrap();

    assert_eq!(result.keys.len(), n);
    assert_last_scalar_firstseen_route("u32 deterministic high", ScalarFirstseenRoute::Partitioned);

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_deterministic_low_u64::<f64, SumAggF64, f64>(&keys, &values)
            .unwrap();

    assert_eq!(result.keys.len(), n);
    assert_last_scalar_firstseen_route("u64 deterministic high", ScalarFirstseenRoute::Partitioned);
}

#[test]
fn legacy_low_wrapper_selects_low_route_for_low_cardinality() {
    let keys = vec![2_i64, 1, 2, 3];
    let values = vec![10.0_f64, 20.0, 30.0, 40.0];

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_legacy_low_u32::<f64, MinAggF64, f64>(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![2, 1, 3]);
    assert_eq!(result.values, vec![10.0, 20.0, 40.0]);
    assert_last_scalar_firstseen_route("u32 legacy low", ScalarFirstseenRoute::LegacyLow);

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_legacy_low_u64::<f64, MinAggF64, f64>(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![2, 1, 3]);
    assert_eq!(result.values, vec![10.0, 20.0, 40.0]);
    assert_last_scalar_firstseen_route("u64 legacy low", ScalarFirstseenRoute::LegacyLow);
}

#[test]
fn legacy_low_wrapper_selects_partitioned_for_high_uniqueness() {
    let n = 5_000usize;
    let keys: Vec<i64> = (0..n).map(|row| row as i64).collect();
    let values: Vec<f64> = (0..n).map(|row| row as f64).collect();

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_legacy_low_u32::<f64, MinAggF64, f64>(&keys, &values).unwrap();

    assert_eq!(result.keys.len(), n);
    assert_last_scalar_firstseen_route("u32 legacy high", ScalarFirstseenRoute::Partitioned);

    clear_scalar_firstseen_route_for_test();
    let result =
        parallel_groupby_firstseen_legacy_low_u64::<f64, MinAggF64, f64>(&keys, &values).unwrap();

    assert_eq!(result.keys.len(), n);
    assert_last_scalar_firstseen_route("u64 legacy high", ScalarFirstseenRoute::Partitioned);
}

#[test]
fn scalar_firstseen_high_uniqueness_routes_target_aggs_to_partitioned() {
    let n = 20_000usize;
    let keys: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let values_i64: Vec<i64> = (0..n).map(|i| i as i64).collect();
    let mut failures = Vec::new();

    observe_partitioned_routes!(&keys, &values, &mut failures;
        "sum_f64_firstseen_u32" => parallel_groupby_sum_f64_firstseen_u32,
        "sum_f64_firstseen_u64" => parallel_groupby_sum_f64_firstseen_u64,
        "mean_f64_firstseen_u32" => parallel_groupby_mean_f64_firstseen_u32,
        "mean_f64_firstseen_u64" => parallel_groupby_mean_f64_firstseen_u64,
        "min_f64_firstseen_u32" => parallel_groupby_min_f64_firstseen_u32,
        "min_f64_firstseen_u64" => parallel_groupby_min_f64_firstseen_u64,
        "max_f64_firstseen_u32" => parallel_groupby_max_f64_firstseen_u32,
        "max_f64_firstseen_u64" => parallel_groupby_max_f64_firstseen_u64,
        "count_f64_firstseen_u32" => parallel_groupby_count_f64_firstseen_u32,
        "count_f64_firstseen_u64" => parallel_groupby_count_f64_firstseen_u64,
    );
    observe_partitioned_routes!(&keys, &values_i64, &mut failures;
        "sum_i64_firstseen_u32" => parallel_groupby_sum_i64_firstseen_u32,
        "sum_i64_firstseen_u64" => parallel_groupby_sum_i64_firstseen_u64,
        "mean_i64_firstseen_u32" => parallel_groupby_mean_i64_firstseen_u32,
        "mean_i64_firstseen_u64" => parallel_groupby_mean_i64_firstseen_u64,
        "min_i64_firstseen_u32" => parallel_groupby_min_i64_firstseen_u32,
        "min_i64_firstseen_u64" => parallel_groupby_min_i64_firstseen_u64,
        "max_i64_firstseen_u32" => parallel_groupby_max_i64_firstseen_u32,
        "max_i64_firstseen_u64" => parallel_groupby_max_i64_firstseen_u64,
        "count_i64_firstseen_u32" => parallel_groupby_count_i64_firstseen_u32,
        "count_i64_firstseen_u64" => parallel_groupby_count_i64_firstseen_u64,
    );

    assert!(
        failures.is_empty(),
        "structural route assertion failed: expected Partitioned for every \
         high-uniqueness scalar first-seen target agg; observed {}",
        failures.join(", ")
    );
}

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
