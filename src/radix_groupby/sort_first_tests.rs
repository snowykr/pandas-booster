use super::result::GroupByMultiResult;
use super::test_support::{key_at_out, value_at_out};

fn assert_sort_first_shape(diagnostics: &super::sort_first::SortFirstDiagnostics) {
    assert!(diagnostics.lexicographic_permutation_built);
    assert_eq!(diagnostics.segment_scan_count, 1);
    assert_eq!(diagnostics.post_aggregation_sort_count, 0);
}

fn rows_f64(result: &GroupByMultiResult<f64>) -> Vec<((i64, i64), f64)> {
    (0..result.values.len())
        .map(|out_g| {
            (
                (key_at_out(result, out_g, 0), key_at_out(result, out_g, 1)),
                value_at_out(result, out_g),
            )
        })
        .collect()
}

fn rows_i64(result: &GroupByMultiResult<i64>) -> Vec<((i64, i64), i64)> {
    (0..result.values.len())
        .map(|out_g| {
            (
                (key_at_out(result, out_g, 0), key_at_out(result, out_g, 1)),
                value_at_out(result, out_g),
            )
        })
        .collect()
}

#[test]
fn sort_first_sum_f64_deduplicates_keys_without_first_seen_order_leakage() {
    // Given duplicated keys whose first-seen order differs from lexicographic order.
    let k1 = vec![2i64, 1, 2, 1, 1, 0, 0, 0];
    let k2 = vec![0i64, 5, 0, 5, 4, 9, 9, 9];
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0, 1e16, 1.0, -1e16];
    let key_slices: Vec<&[i64]> = vec![&k1, &k2];

    // When planned sort-first sum groups by lexicographic row permutation.
    let (result, diagnostics) = super::sort_first::sort_first_groupby_sum_f64(&key_slices, &values)
        .expect("planned sort-first sum should accept two i64 key columns");

    // Then output order is lexicographic, not first-seen, and equal-key rows stay stable.
    assert_sort_first_shape(&diagnostics);
    assert_eq!(
        rows_f64(&result),
        vec![
            ((0, 9), 0.0),
            ((1, 4), 50.0),
            ((1, 5), 60.0),
            ((2, 0), 40.0),
        ]
    );
}

#[test]
fn sort_first_max_f64_orders_negative_keys_with_signed_lexicographic_sort() {
    // Given negative and positive multi-key tuples.
    let k1 = vec![-1i64, -2, 0, -1, 1];
    let k2 = vec![5i64, 3, -1, -5, 2];
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let key_slices: Vec<&[i64]> = vec![&k1, &k2];

    // When planned sort-first max scans sorted contiguous segments.
    let (result, diagnostics) = super::sort_first::sort_first_groupby_max_f64(&key_slices, &values)
        .expect("planned sort-first max should accept signed i64 keys");

    // Then signed lexicographic key order is preserved without a post aggregation sort.
    assert_sort_first_shape(&diagnostics);
    assert_eq!(
        rows_f64(&result),
        vec![
            ((-2, 3), 20.0),
            ((-1, -5), 40.0),
            ((-1, 5), 10.0),
            ((0, -1), 30.0),
            ((1, 2), 50.0),
        ]
    );
}

#[test]
fn sort_first_min_f64_scans_already_sorted_segments_once() {
    // Given rows already in lexicographic order with duplicate contiguous groups.
    let k1 = vec![-2i64, -2, -1, 0, 0, 1];
    let k2 = vec![0i64, 0, 5, 2, 2, 9];
    let values = vec![7.0, 3.0, 4.0, 10.0, -2.0, 8.0];
    let key_slices: Vec<&[i64]> = vec![&k1, &k2];

    // When planned sort-first min receives pre-sorted rows.
    let (result, diagnostics) = super::sort_first::sort_first_groupby_min_f64(&key_slices, &values)
        .expect("planned sort-first min should scan already sorted rows");

    // Then it still reports one segment scan and emits the existing lexicographic order.
    assert_sort_first_shape(&diagnostics);
    assert_eq!(
        rows_f64(&result),
        vec![
            ((-2, 0), 3.0),
            ((-1, 5), 4.0),
            ((0, 2), -2.0),
            ((1, 9), 8.0)
        ]
    );
}

#[test]
fn sort_first_count_i64_reorders_reverse_sorted_input_without_first_seen_leakage() {
    // Given reverse-sorted rows where first-seen order is the exact wrong output order.
    let k1 = vec![3i64, 2, 2, 1, 1, 0];
    let k2 = vec![0i64, 9, 9, 5, 4, 3];
    let values = vec![100i64, 200, 300, 400, 500, 600];
    let key_slices: Vec<&[i64]> = vec![&k1, &k2];

    // When planned sort-first count scans the lexicographic row permutation.
    let (result, diagnostics) =
        super::sort_first::sort_first_groupby_count_i64(&key_slices, &values)
            .expect("planned sort-first count should accept i64 values");

    // Then count output is lexicographic and cannot leak first-seen group order.
    assert_sort_first_shape(&diagnostics);
    assert_eq!(
        rows_i64(&result),
        vec![
            ((0, 3), 1),
            ((1, 4), 1),
            ((1, 5), 1),
            ((2, 9), 2),
            ((3, 0), 1)
        ]
    );
}

#[test]
fn sort_first_sum_f64_keeps_high_uniqueness_like_keys_in_lexicographic_order() {
    // Given mostly unique keys, approximating the high-cardinality sorted workload shape.
    let k1 = vec![9i64, 8, 7, 6, 5, 4, 3, 2, 1, 0, 5, 8];
    let k2 = vec![0i64, 1, 2, 3, 4, 5, 6, 7, 8, 9, 4, 1];
    let values = vec![
        90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0, 0.0, 5.0, 8.0,
    ];
    let key_slices: Vec<&[i64]> = vec![&k1, &k2];

    // When planned sort-first sum aggregates mostly one-row segments.
    let (result, diagnostics) = super::sort_first::sort_first_groupby_sum_f64(&key_slices, &values)
        .expect("planned sort-first sum should support high-uniqueness-like fixtures");

    // Then every emitted group is already in lexicographic order after one segment scan.
    assert_sort_first_shape(&diagnostics);
    assert_eq!(
        rows_f64(&result),
        vec![
            ((0, 9), 0.0),
            ((1, 8), 10.0),
            ((2, 7), 20.0),
            ((3, 6), 30.0),
            ((4, 5), 40.0),
            ((5, 4), 55.0),
            ((6, 3), 60.0),
            ((7, 2), 70.0),
            ((8, 1), 88.0),
            ((9, 0), 90.0),
        ]
    );
}
