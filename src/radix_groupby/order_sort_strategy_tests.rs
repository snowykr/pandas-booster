fn tuple_sorted_perm(keys_flat: &[i64], n_keys: usize) -> Vec<usize> {
    let n_groups = keys_flat.len() / n_keys;
    let mut perm: Vec<usize> = (0..n_groups).collect();
    perm.sort_unstable_by(|&left, &right| {
        let left_keys = &keys_flat[left * n_keys..(left + 1) * n_keys];
        let right_keys = &keys_flat[right * n_keys..(right + 1) * n_keys];
        left_keys.cmp(right_keys).then(left.cmp(&right))
    });
    perm
}

#[test]
fn selected_sort_strategy_uses_packed_for_bounded_three_key_domain() {
    let keys_flat = vec![499, 10, 0, 0, 499, 10, 10, 0, 499, 250, 250, 250];

    let (perm, proof) = crate::radix_sort::multi_key_sort_perm_with_proof(&keys_flat, 3);

    assert_eq!(
        proof.strategy,
        crate::radix_sort::MultiKeySortStrategy::PackedU64
    );
    assert_eq!(proof.bit_widths, vec![9, 9, 9]);
    assert_eq!(proof.total_bit_width, 27);
    assert_eq!(perm, tuple_sorted_perm(&keys_flat, 3));
}

#[test]
fn packed_bit_width_proof_handles_constants_and_negative_domains() {
    let constant_keys = vec![7, 7, 7, 7, 7, 7];
    let (_perm, constant_proof) =
        crate::radix_sort::multi_key_sort_perm_with_proof(&constant_keys, 2);
    assert_eq!(
        constant_proof.strategy,
        crate::radix_sort::MultiKeySortStrategy::PackedU64
    );
    assert_eq!(constant_proof.bit_widths, vec![0, 0]);
    assert_eq!(constant_proof.total_bit_width, 0);

    let negative_keys = vec![-250, -1, 249, -250, 0, 249, -1, 0];
    let (perm, negative_proof) =
        crate::radix_sort::multi_key_sort_perm_with_proof(&negative_keys, 2);
    assert_eq!(
        negative_proof.strategy,
        crate::radix_sort::MultiKeySortStrategy::PackedU64
    );
    assert_eq!(negative_proof.bit_widths, vec![9, 9]);
    assert_eq!(negative_proof.total_bit_width, 18);
    assert_eq!(perm, tuple_sorted_perm(&negative_keys, 2));
}

#[test]
fn selected_sort_strategy_falls_back_for_non_packable_extreme_ranges() {
    let keys_flat = vec![i64::MIN, i64::MAX, i64::MAX, i64::MIN, 0, 0, -1, 1];

    let (perm, proof) = crate::radix_sort::multi_key_sort_perm_with_proof(&keys_flat, 2);

    assert_eq!(
        proof.strategy,
        crate::radix_sort::MultiKeySortStrategy::FusedMultiKeyRadix
    );
    assert_eq!(proof.bit_widths, vec![64, 64]);
    assert_eq!(perm, tuple_sorted_perm(&keys_flat, 2));
}

#[test]
fn fused_and_packed_permutations_match_tuple_oracle_for_varied_key_counts() {
    let cases: Vec<(usize, Vec<i64>)> = vec![
        (1, vec![0, i64::MIN, i64::MAX, -7, 7]),
        (
            2,
            vec![i64::MIN, i64::MIN, i64::MAX, i64::MAX, -1, 2, -1, 1, 0, 0],
        ),
        (
            3,
            vec![3, 2, 1, -1, 0, 1, -1, 0, 0, i64::MAX, i64::MIN, 0, 3, 2, 1],
        ),
        (
            5,
            vec![
                1,
                2,
                3,
                4,
                5,
                1,
                2,
                3,
                4,
                4,
                -5,
                -4,
                -3,
                -2,
                -1,
                i64::MAX,
                0,
                0,
                0,
                i64::MIN,
            ],
        ),
    ];

    for (n_keys, keys_flat) in cases {
        let (perm, _proof) = crate::radix_sort::multi_key_sort_perm_with_proof(&keys_flat, n_keys);
        assert_eq!(perm, tuple_sorted_perm(&keys_flat, n_keys));
    }
}

#[test]
#[cfg(not(miri))]
fn production_rayon_multi_chunk_paths_match_tuple_oracle() {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("test thread pool builds");

    let packed_keys: Vec<i64> = (0..4096)
        .flat_map(|idx| {
            let key = i64::from((idx * 37 + 11) & 0x1FF);
            [key, 511 - key, i64::from(idx & 0x1FF)]
        })
        .collect();
    let fused_keys: Vec<i64> = (0..4096)
        .flat_map(|idx| {
            [
                if idx % 2 == 0 { i64::MIN } else { i64::MAX },
                i64::from((idx * 97 + 13) & 0xFFFF),
            ]
        })
        .collect();

    pool.install(|| {
        let (packed_perm, packed_proof) =
            crate::radix_sort::multi_key_sort_perm_with_proof(&packed_keys, 3);
        assert_eq!(
            packed_proof.strategy,
            crate::radix_sort::MultiKeySortStrategy::PackedU64
        );
        assert_eq!(packed_perm, tuple_sorted_perm(&packed_keys, 3));

        let (fused_perm, fused_proof) =
            crate::radix_sort::multi_key_sort_perm_with_proof(&fused_keys, 2);
        assert_eq!(
            fused_proof.strategy,
            crate::radix_sort::MultiKeySortStrategy::FusedMultiKeyRadix
        );
        assert_eq!(fused_perm, tuple_sorted_perm(&fused_keys, 2));
    });
}
