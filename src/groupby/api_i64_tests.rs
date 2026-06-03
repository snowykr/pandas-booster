use super::*;
use ahash::AHashMap;
use rayon::ThreadPoolBuilder;

#[test]
fn test_groupby_sum_i64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values: Vec<i64> = vec![1, 2, 3, 4, 5];
    let result = parallel_groupby_sum_i64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, i64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert_eq!(map[&1], 9);
    assert_eq!(map[&2], 6);
}

#[test]
fn test_groupby_prod_i64_wraps_and_firstseen_u64() {
    let keys = vec![9, 1, 9, 1, 2];
    let values = vec![i64::MAX, 3, 2, 4, 5];
    let result = parallel_groupby_prod_i64_firstseen_u64(&keys, &values).unwrap();

    assert_eq!(result.keys, vec![9, 1, 2]);
    assert_eq!(result.values[0], i64::MAX.wrapping_mul(2));
    assert_eq!(result.values[1], 12);
    assert_eq!(result.values[2], 5);

    let sorted = parallel_groupby_prod_i64_sorted(&keys, &values).unwrap();
    assert_eq!(sorted.keys, vec![1, 2, 9]);
    assert_eq!(sorted.values, vec![12, 5, i64::MAX.wrapping_mul(2)]);
}

#[test]
fn test_groupby_min_i64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values: Vec<i64> = vec![5, 2, 3, 4, 1];
    let result = parallel_groupby_min_i64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, i64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert_eq!(map[&1], 1);
    assert_eq!(map[&2], 2);
}

#[test]
fn test_groupby_max_i64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values: Vec<i64> = vec![5, 2, 3, 4, 1];
    let result = parallel_groupby_max_i64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, i64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert_eq!(map[&1], 5);
    assert_eq!(map[&2], 4);
}

#[test]
fn test_groupby_count_f64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
    let result = parallel_groupby_count_f64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, i64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert_eq!(map[&1], 2);
    assert_eq!(map[&2], 2);
}

#[test]
fn test_groupby_count_i64() {
    let keys = vec![1, 2, 1, 2, 1];
    let values: Vec<i64> = vec![1, 2, 3, 4, 5];
    let result = parallel_groupby_count_i64(&keys, &values).unwrap();

    let mut map: AHashMap<i64, i64> = AHashMap::new();
    for (k, v) in result.keys.iter().zip(result.values.iter()) {
        map.insert(*k, *v);
    }

    assert_eq!(map[&1], 3);
    assert_eq!(map[&2], 2);
}

#[test]
fn test_groupby_firstseen_order_u32_across_chunks() {
    // Force multiple Rayon threads so chunking happens.
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| {
        let n = 120_000;
        let mut keys = vec![0i64; n];
        // Ensure first-seen occurrences are far apart.
        keys[0] = 10;
        keys[50_000] = 20;
        keys[100_000] = 30;

        // Fill the rest with repetitions (no new groups).
        for (i, k) in keys.iter_mut().enumerate().skip(1) {
            if i == 50_000 || i == 100_000 {
                continue;
            }
            *k = match i % 3 {
                0 => 10,
                1 => 20,
                _ => 30,
            };
        }
        let values = vec![1.0f64; n];

        let result = parallel_groupby_sum_f64_firstseen_u32(&keys, &values).unwrap();
        assert_eq!(result.keys, vec![10, 20, 30]);
    });
}

#[test]
fn test_groupby_firstseen_order_u64_across_chunks() {
    // Same as the u32 test, but exercising the u64 first-seen path.
    let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    pool.install(|| {
        let n = 120_000;
        let mut keys = vec![0i64; n];
        keys[0] = 10;
        keys[50_000] = 20;
        keys[100_000] = 30;

        for (i, k) in keys.iter_mut().enumerate().skip(1) {
            if i == 50_000 || i == 100_000 {
                continue;
            }
            *k = match i % 3 {
                0 => 10,
                1 => 20,
                _ => 30,
            };
        }
        let values = vec![1.0f64; n];

        let result = parallel_groupby_sum_f64_firstseen_u64(&keys, &values).unwrap();
        assert_eq!(result.keys, vec![10, 20, 30]);
    });
}
