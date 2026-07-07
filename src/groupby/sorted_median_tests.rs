use super::sorted_median::{
    groupby_median_f64_sorted_direct, groupby_median_f64_sorted_direct_with_stats,
    groupby_median_i64_sorted_direct, reset_sorted_median_parallel_slice_median_count,
    sorted_median_parallel_slice_median_count,
};
use rayon::ThreadPoolBuilder;

fn assert_f64_bits(actual: f64, expected: f64) {
    assert_eq!(actual.to_bits(), expected.to_bits());
}

fn make_issue_shape_parallel_median_data() -> (Vec<i64>, Vec<f64>) {
    let group_count = 4_096usize;
    let rows_per_group = 9usize;
    let mut keys = Vec::with_capacity(group_count * rows_per_group);
    let mut values = Vec::with_capacity(group_count * rows_per_group);

    for row in 0..rows_per_group {
        for group in 0..group_count {
            let shuffled = (group * 8_191 + row * 131) % group_count;
            keys.push(shuffled as i64 - 2_048);
            values.push(if row == 0 && group % 17 == 0 {
                f64::NAN
            } else {
                (row as f64 * 0.5) + shuffled as f64
            });
        }
    }

    (keys, values)
}

#[test]
fn sorted_median_direct_f64_preserves_sorted_keys_and_pandas_edges() {
    // Given: unsorted rows with NaNs, an all-NaN group, infinities, signed
    // zero, and an even group that intentionally overflows during averaging.
    let keys = vec![3, 1, 2, 3, 1, 2, 4, 1, 5, 5, 6, 6, 7, 7];
    let values = vec![
        f64::NAN,
        f64::INFINITY,
        -0.0,
        f64::NAN,
        2.0,
        0.0,
        f64::NAN,
        -f64::INFINITY,
        f64::MAX,
        f64::MAX,
        -f64::MAX,
        -f64::MAX,
        8.0,
        f64::NAN,
    ];

    // When: the direct sorted median engine materializes medians.
    let result = groupby_median_f64_sorted_direct(&keys, &values).unwrap();

    // Then: keys are sorted and median semantics match the existing helper.
    assert_eq!(result.keys, vec![1, 2, 3, 4, 5, 6, 7]);
    assert_eq!(result.values[0], 2.0);
    assert_f64_bits(result.values[1], -0.0);
    assert!(result.values[2].is_nan());
    assert!(result.values[3].is_nan());
    assert_eq!(result.values[4], f64::INFINITY);
    assert_eq!(result.values[5], f64::NEG_INFINITY);
    assert_eq!(result.values[6], 8.0);
}

#[test]
fn sorted_median_direct_f64_preserves_pandas_signed_zero_bits() {
    let keys = vec![1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    let values = vec![-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -1.0, 0.0, -0.0, 1.0];

    let result = groupby_median_f64_sorted_direct(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![1, 2, 3]);
    assert_f64_bits(result.values[0], -0.0);
    assert_f64_bits(result.values[1], -0.0);
    assert_f64_bits(result.values[2], 0.0);
}

#[test]
fn sorted_median_direct_i64_preserves_sorted_keys_and_extremes() {
    // Given: unsorted rows with repeated keys and i64 middle-value extremes.
    let keys = vec![10, -1, 10, 2, -1, 2, 3, 3, 4];
    let values = vec![
        i64::MAX,
        -9,
        i64::MAX - 2,
        i64::MIN,
        4,
        i64::MAX,
        (1_i64 << 53) + 1,
        (1_i64 << 53) + 3,
        42,
    ];

    // When: the direct sorted median engine materializes medians.
    let result = groupby_median_i64_sorted_direct(&keys, &values).unwrap();

    // Then: sorted output and f64-returning i64 median semantics are preserved.
    assert_eq!(result.keys, vec![-1, 2, 3, 4, 10]);
    assert_eq!(result.values[0], -2.5);
    assert_eq!(result.values[1], 0.0);
    assert_eq!(
        result.values[2],
        ((1_i64 << 53) + 1) as f64 / 2.0 + ((1_i64 << 53) + 3) as f64 / 2.0
    );
    assert_eq!(result.values[3], 42.0);
    assert_eq!(result.values[4], (i64::MAX - 1) as f64);
}

#[test]
fn sorted_median_direct_empty_and_mismatched_inputs() {
    // Given: empty inputs and mismatched input lengths.
    pyo3::Python::initialize();

    let empty_keys: Vec<i64> = Vec::new();
    let empty_values: Vec<f64> = Vec::new();
    let mismatched_keys = vec![1, 2, 1];
    let mismatched_values = vec![1.0, 2.0];

    // When: the direct engine receives empty and invalid shapes.
    let empty = groupby_median_f64_sorted_direct(&empty_keys, &empty_values).unwrap();
    let error = groupby_median_f64_sorted_direct(&mismatched_keys, &mismatched_values).unwrap_err();

    // Then: empty inputs produce empty output and invalid lengths are rejected.
    assert!(empty.keys.is_empty());
    assert!(empty.values.is_empty());
    assert!(
        error
            .to_string()
            .contains("keys and values must have same length"),
        "unexpected error: {error}"
    );
}

#[test]
fn sorted_median_direct_is_thread_stable() {
    // Given: enough unsorted rows to expose accidental thread-local merge order.
    let mut keys = Vec::with_capacity(6_000);
    let mut values = Vec::with_capacity(6_000);
    for row in 0..6_000_i64 {
        let key = (row * 37).rem_euclid(251) - 125;
        keys.push(key);
        values.push(((row * 17).rem_euclid(1_009) - 500) as f64);
    }

    // When: the direct engine runs repeatedly in this process.
    let first = groupby_median_f64_sorted_direct(&keys, &values).unwrap();
    let second = groupby_median_f64_sorted_direct(&keys, &values).unwrap();

    // Then: output ordering and median bits are deterministic.
    assert_eq!(first.keys, second.keys);
    assert_eq!(
        first
            .values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        second
            .values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn sorted_median_direct_issue_shape_has_no_chunk_local_buffers() {
    // Given: an issue-shaped unsorted workload with NaNs skipped for f64.
    let keys = vec![8, 3, 8, 1, 3, 8, 5, 1, 5, 9, 9, 9];
    let values = vec![
        10.0,
        f64::NAN,
        2.0,
        7.0,
        11.0,
        4.0,
        3.0,
        1.0,
        5.0,
        f64::NAN,
        6.0,
        8.0,
    ];

    // When: the stats-bearing direct engine runs.
    let profiled = groupby_median_f64_sorted_direct_with_stats(&keys, &values).unwrap();

    // Then: stats prove one final contiguous value buffer, not chunk-local
    // median buffers later merged by group.
    assert_eq!(profiled.result.keys, vec![1, 3, 5, 8, 9]);
    assert_eq!(profiled.result.values, vec![4.0, 11.0, 4.0, 4.0, 7.0]);
    assert_eq!(
        profiled.stats.partial_group_total,
        profiled.stats.final_group_count
    );
    assert_eq!(
        profiled.stats.accepted_value_count,
        profiled.stats.value_buffer_len
    );
    assert_eq!(
        profiled.stats.scatter_write_count,
        profiled.stats.accepted_value_count
    );
}
#[test]
fn sorted_median_direct_remaps_first_seen_to_sorted_keys_and_retains_all_nan_group() {
    // Given: first-seen order differs from sorted-key order, and key 3 has no
    // accepted f64 values but must remain a visible output group.
    let keys = vec![5, 1, 3, 5, 1, 3, 5, 1];
    let values = vec![10.0, 8.0, f64::NAN, 2.0, 4.0, f64::NAN, 6.0, 12.0];

    // When: the stats-bearing direct engine runs.
    let profiled = groupby_median_f64_sorted_direct_with_stats(&keys, &values).unwrap();

    // Then: output follows sorted keys, not first-seen or hash-map iteration
    // order, and the all-NaN group survives as an empty median slice.
    assert_eq!(profiled.result.keys, vec![1, 3, 5]);
    assert_eq!(profiled.result.values[0], 8.0);
    assert!(profiled.result.values[1].is_nan());
    assert_eq!(profiled.result.values[2], 6.0);
    assert_eq!(profiled.stats.final_group_count, 3);
    assert_eq!(profiled.stats.accepted_value_count, 6);
    assert_eq!(
        profiled.stats.accepted_value_count,
        profiled.stats.value_buffer_len
    );
    assert_eq!(
        profiled.stats.scatter_write_count,
        profiled.stats.accepted_value_count
    );
    assert!(profiled.stats.unique_build_s >= 0.0);
    assert!(profiled.stats.key_sort_s >= 0.0);
    assert_eq!(profiled.stats.count_s, 0.0);
    assert!(profiled.stats.buffer_setup_s >= 0.0);
    assert!(profiled.stats.scatter_s >= 0.0);
    assert!(profiled.stats.median_select_s >= 0.0);
}

#[test]
fn sorted_median_direct_uses_parallel_slice_medians_for_issue_shape() {
    // Given: an issue-shaped workload with many independent sorted-key groups.
    let (keys, values) = make_issue_shape_parallel_median_data();
    reset_sorted_median_parallel_slice_median_count();

    // When: the stats-bearing direct engine runs.
    let profiled = groupby_median_f64_sorted_direct_with_stats(&keys, &values).unwrap();

    // Then: every final group is materialized from the Rayon slice-median path.
    assert_eq!(profiled.stats.final_group_count, 4_096);
    assert!(
        sorted_median_parallel_slice_median_count() >= profiled.stats.final_group_count,
        "parallel slice median calls should cover every final group"
    );
}

#[test]
fn sorted_median_direct_issue_shape_matches_between_one_and_eight_threads() {
    // Given: the same issue-shaped data driven through explicit Rayon pools.
    let (keys, values) = make_issue_shape_parallel_median_data();
    let one_thread_pool = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let eight_thread_pool = ThreadPoolBuilder::new().num_threads(8).build().unwrap();

    // When: the direct engine runs with one and eight Rayon workers.
    let one_thread = one_thread_pool
        .install(|| groupby_median_f64_sorted_direct(&keys, &values))
        .unwrap();
    let eight_threads = eight_thread_pool
        .install(|| groupby_median_f64_sorted_direct(&keys, &values))
        .unwrap();

    // Then: sorted output and median bits are stable across thread counts.
    assert_eq!(one_thread.keys, eight_threads.keys);
    assert_eq!(
        one_thread
            .values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        eight_threads
            .values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn sorted_median_direct_empty_profile_ratio_is_zero() {
    // Given: empty sorted-median profile inputs.
    let keys: Vec<i64> = Vec::new();
    let values: Vec<f64> = Vec::new();

    // When: the stats-bearing direct engine profiles the empty input.
    let profiled = groupby_median_f64_sorted_direct_with_stats(&keys, &values).unwrap();
    let ratio = if profiled.stats.final_group_count == 0 {
        0.0
    } else {
        profiled.stats.partial_group_total as f64 / profiled.stats.final_group_count as f64
    };

    // Then: empty structural counts do not produce NaN or infinity.
    assert!(profiled.result.keys.is_empty());
    assert!(profiled.result.values.is_empty());
    assert_eq!(profiled.stats.partial_group_total, 0);
    assert_eq!(profiled.stats.final_group_count, 0);
    assert_eq!(ratio, 0.0);
    assert_eq!(profiled.stats.unique_build_s, 0.0);
    assert_eq!(profiled.stats.key_sort_s, 0.0);
    assert_eq!(profiled.stats.count_s, 0.0);
    assert_eq!(profiled.stats.buffer_setup_s, 0.0);
    assert_eq!(profiled.stats.scatter_s, 0.0);
    assert_eq!(profiled.stats.median_select_s, 0.0);
}
