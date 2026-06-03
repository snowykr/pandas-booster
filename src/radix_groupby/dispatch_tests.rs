use super::dispatch::{radix_groupby, radix_groupby_fixed};
use super::*;
use crate::aggregation::SumAggF64;

#[test]
fn test_equivalence_fixed_vs_generic() {
    // Test N=2..=10
    let n_rows = 1000;

    for n_keys in 2..=10 {
        let keys: Vec<Vec<i64>> = (0..n_keys)
            .map(|k| {
                (0..n_rows)
                    .map(|i| ((i as i64) * (k as i64 + 1)) % 20)
                    .collect()
            })
            .collect();

        let values: Vec<f64> = (0..n_rows).map(|i| i as f64).collect();
        let key_slices: Vec<&[i64]> = keys.iter().map(|v| v.as_slice()).collect();

        // Generic result
        let res_generic = radix_groupby::<f64, SumAggF64, f64>(&key_slices, &values).unwrap();

        // Fixed result
        let res_fixed = match n_keys {
            2 => radix_groupby_fixed::<2, f64, SumAggF64, f64>(&key_slices, &values),
            3 => radix_groupby_fixed::<3, f64, SumAggF64, f64>(&key_slices, &values),
            4 => radix_groupby_fixed::<4, f64, SumAggF64, f64>(&key_slices, &values),
            5 => radix_groupby_fixed::<5, f64, SumAggF64, f64>(&key_slices, &values),
            6 => radix_groupby_fixed::<6, f64, SumAggF64, f64>(&key_slices, &values),
            7 => radix_groupby_fixed::<7, f64, SumAggF64, f64>(&key_slices, &values),
            8 => radix_groupby_fixed::<8, f64, SumAggF64, f64>(&key_slices, &values),
            9 => radix_groupby_fixed::<9, f64, SumAggF64, f64>(&key_slices, &values),
            10 => radix_groupby_fixed::<10, f64, SumAggF64, f64>(&key_slices, &values),
            _ => panic!("Unsupported N"),
        }
        .unwrap();

        assert_eq!(
            res_generic.values.len(),
            res_fixed.values.len(),
            "N={}: Group counts differ",
            n_keys
        );

        // Convert to sorted vectors for comparison
        let sort_result = |res: &GroupByMultiResult<f64>| {
            let n_groups = res.values.len();
            let mut pairs = Vec::with_capacity(n_groups);
            for i in 0..n_groups {
                let mut k = Vec::new();
                for j in 0..n_keys {
                    k.push(res.keys_flat[i * n_keys + j]);
                }
                pairs.push((k, res.values[i]));
            }
            pairs.sort_by(|a, b| a.0.cmp(&b.0));
            pairs
        };

        let sorted_generic = sort_result(&res_generic);
        let sorted_fixed = sort_result(&res_fixed);

        for i in 0..sorted_generic.len() {
            assert_eq!(
                sorted_generic[i].0, sorted_fixed[i].0,
                "N={}: Key mismatch at index {}",
                n_keys, i
            );
            assert!(
                (sorted_generic[i].1 - sorted_fixed[i].1).abs() < 1e-10,
                "N={}: Value mismatch at index {}",
                n_keys,
                i
            );
        }
    }
}
