use crate::radix_sort::i64_to_sortable_u64;
use crate::radix_sort_digit::radix_sort_perm_by_digit_par;

fn checked_columnar_base(n: usize, threads: usize, k: usize) -> Result<usize, String> {
    use crate::radix_sort_digit::checked_route_digit_workspace_bytes;
    let threads = threads.max(1);
    let chunk_size = (n / threads).max(1024);
    let chunks = n / chunk_size + usize::from(!n.is_multiple_of(chunk_size));
    let workspace = checked_route_digit_workspace_bytes(n, threads)?;
    let extra = threads
        .checked_add(chunks)
        .and_then(|sum| sum.checked_add(k))
        .and_then(|sum| sum.checked_mul(4096))
        .and_then(|sum| sum.checked_add(65536))
        .ok_or_else(|| "columnar route workspace overflow".to_string())?;
    let permutations = n
        .checked_mul(2 * std::mem::size_of::<usize>())
        .ok_or_else(|| "columnar route workspace overflow".to_string())?;
    check_columnar_base(permutations, workspace, extra)
}

fn check_columnar_base(permutations: usize, digits: usize, extra: usize) -> Result<usize, String> {
    crate::radix_sort_digit::check_route_workspace(&[permutations, digits, extra])
}

fn check_columnar_packed(base: usize, packed: usize) -> Result<usize, String> {
    crate::radix_sort_digit::check_route_workspace(&[base, packed])
}

/// Sort input rows directly from columns, without an n-by-k key copy.
/// Layout/packing and radix timings are disjoint; ties retain input row order.
pub(crate) fn sort_first_columnar_permutation(
    key_slices: &[&[i64]],
) -> Result<(Vec<usize>, MultiKeySortProof, f64, f64), String> {
    use crate::radix_sort_digit::{radix_sort_perm_by_digit_checked, scratch_vec as route_vec};
    use rayon::prelude::*;
    use std::time::Instant;

    let layout_start = Instant::now();
    let k = key_slices.len();
    if !(2..=10).contains(&k) {
        return Err("columnar route requires 2..=10 key columns".to_string());
    }
    let n = key_slices[0].len();
    if key_slices.iter().any(|column| column.len() != n) {
        return Err("columnar key lengths differ".to_string());
    }
    let threads = rayon::current_num_threads().max(1);
    let base = checked_columnar_base(n, threads, k)?;

    let mut ranges = route_vec(k)?;
    ranges.resize(k, (0i64, 0i64));
    ranges
        .par_iter_mut()
        .zip(key_slices.par_iter())
        .for_each(|(range, column)| {
            if !column.is_empty() {
                *range = column.par_iter().copied().map(|key| (key, key)).reduce(
                    || (i64::MAX, i64::MIN),
                    |left, right| (left.0.min(right.0), left.1.max(right.1)),
                );
            }
        });
    let mut bit_widths = route_vec(k)?;
    let mut total_bit_width = 0u32;
    for &(min, max) in ranges.iter() {
        let width = bit_width_for_inclusive_range(i128::from(max) - i128::from(min))
            .ok_or_else(|| "columnar key range overflow".to_string())?;
        total_bit_width = total_bit_width
            .checked_add(width)
            .ok_or_else(|| "columnar key width overflow".to_string())?;
        bit_widths.push(width);
    }
    let strategy = if total_bit_width <= 64 {
        MultiKeySortStrategy::PackedU64
    } else {
        MultiKeySortStrategy::FusedMultiKeyRadix
    };
    let mut packed = route_vec::<u64>(0)?;
    if strategy == MultiKeySortStrategy::PackedU64 && total_bit_width != 0 {
        let packed_bytes = n
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| "columnar packing workspace overflow".to_string())?;
        check_columnar_packed(base, packed_bytes)?;
        packed = route_vec(n)?;
        packed.resize(n, 0);
        packed.par_iter_mut().enumerate().for_each(|(row, slot)| {
            let mut remaining = total_bit_width;
            let mut key = 0u64;
            for col in 0..k {
                let width = bit_widths[col];
                if width != 0 {
                    remaining -= width;
                    let offset =
                        (i128::from(key_slices[col][row]) - i128::from(ranges[col].0)) as u64;
                    key |= offset << remaining;
                }
            }
            *slot = key;
        });
    }
    let layout_s = layout_start.elapsed().as_secs_f64();
    let radix_start = Instant::now();
    let permutation = match strategy {
        MultiKeySortStrategy::PackedU64 => radix_sort_perm_by_digit_checked(
            n,
            total_bit_width.div_ceil(8) as usize,
            |row, pass| ((packed[row] >> (pass * 8)) & 255) as usize,
        )?,
        MultiKeySortStrategy::FusedMultiKeyRadix => {
            radix_sort_perm_by_digit_checked(n, k * 8, |row, pass| {
                let col = k - 1 - pass / 8;
                ((i64_to_sortable_u64(key_slices[col][row]) >> ((pass % 8) * 8)) & 255) as usize
            })?
        }
    };
    let radix_s = radix_start.elapsed().as_secs_f64();
    #[cfg(test)]
    let bit_widths = bit_widths.into_vec();
    Ok((
        permutation,
        MultiKeySortProof {
            strategy,
            bit_widths,
            total_bit_width,
        },
        layout_s,
        radix_s,
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct MultiKeySortProof {
    pub(crate) strategy: MultiKeySortStrategy,
    pub(crate) bit_widths: Vec<u32>,
    pub(crate) total_bit_width: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MultiKeySortStrategy {
    PackedU64,
    FusedMultiKeyRadix,
}

impl MultiKeySortStrategy {
    pub(crate) fn label(self) -> &'static str {
        match self {
            MultiKeySortStrategy::PackedU64 => "packed_u64",
            MultiKeySortStrategy::FusedMultiKeyRadix => "fused_multi_key_radix",
        }
    }
}

pub(crate) fn radix_sort_perm_by_multi_i64_keys_flat_par(
    keys_flat: &[i64],
    n_keys: usize,
) -> Vec<usize> {
    debug_assert!(n_keys > 0);
    debug_assert_eq!(keys_flat.len() % n_keys, 0);
    let n_groups = keys_flat.len() / n_keys;
    radix_sort_perm_by_digit_par(n_groups, n_keys * 8, |idx, pass| {
        let col = n_keys - 1 - (pass / 8);
        let byte_pass = pass % 8;
        let key = keys_flat[idx * n_keys + col];
        ((i64_to_sortable_u64(key) >> (byte_pass * 8)) & 0xFF) as usize
    })
}

pub(crate) fn multi_key_sort_perm_with_proof(
    keys_flat: &[i64],
    n_keys: usize,
) -> (Vec<usize>, MultiKeySortProof) {
    debug_assert!(n_keys > 0);
    debug_assert_eq!(keys_flat.len() % n_keys, 0);
    let n_groups = keys_flat.len() / n_keys;

    if n_groups <= 1 {
        return (
            (0..n_groups).collect(),
            MultiKeySortProof {
                strategy: MultiKeySortStrategy::PackedU64,
                bit_widths: vec![0; n_keys],
                total_bit_width: 0,
            },
        );
    }

    if let Some((packed_keys, bit_widths, total_bit_width)) =
        try_pack_multi_key_lex_u64(keys_flat, n_keys)
    {
        let perm = if total_bit_width == 0 {
            (0..n_groups).collect()
        } else {
            radix_sort_perm_by_u64_significant_bits_par(&packed_keys, total_bit_width)
        };
        return (
            perm,
            MultiKeySortProof {
                strategy: MultiKeySortStrategy::PackedU64,
                bit_widths,
                total_bit_width,
            },
        );
    }

    (
        radix_sort_perm_by_multi_i64_keys_flat_par(keys_flat, n_keys),
        MultiKeySortProof {
            strategy: MultiKeySortStrategy::FusedMultiKeyRadix,
            bit_widths: multi_key_bit_widths(keys_flat, n_keys).unwrap_or_else(|| vec![64; n_keys]),
            total_bit_width: 64u32.saturating_mul(n_keys as u32),
        },
    )
}

pub(crate) fn multi_key_sort_perm_with_profile(
    keys_flat: &[i64],
    n_keys: usize,
) -> (Vec<usize>, MultiKeySortProof, f64, f64) {
    use std::time::Instant;

    debug_assert!(n_keys > 0);
    debug_assert_eq!(keys_flat.len() % n_keys, 0);
    let n_groups = keys_flat.len() / n_keys;

    if n_groups <= 1 {
        return (
            (0..n_groups).collect(),
            MultiKeySortProof {
                strategy: MultiKeySortStrategy::PackedU64,
                bit_widths: vec![0; n_keys],
                total_bit_width: 0,
            },
            0.0,
            0.0,
        );
    }

    let construction_start = Instant::now();
    let packed_attempt = try_pack_multi_key_lex_u64(keys_flat, n_keys);
    let mut key_construction_s = construction_start.elapsed().as_secs_f64();

    if let Some((packed_keys, bit_widths, total_bit_width)) = packed_attempt {
        let radix_start = Instant::now();
        let perm = if total_bit_width == 0 {
            (0..n_groups).collect()
        } else {
            radix_sort_perm_by_u64_significant_bits_par(&packed_keys, total_bit_width)
        };
        return (
            perm,
            MultiKeySortProof {
                strategy: MultiKeySortStrategy::PackedU64,
                bit_widths,
                total_bit_width,
            },
            key_construction_s,
            radix_start.elapsed().as_secs_f64(),
        );
    }

    let widths_start = Instant::now();
    let bit_widths = multi_key_bit_widths(keys_flat, n_keys).unwrap_or_else(|| vec![64; n_keys]);
    key_construction_s += widths_start.elapsed().as_secs_f64();

    let radix_start = Instant::now();
    let perm = radix_sort_perm_by_multi_i64_keys_flat_par(keys_flat, n_keys);
    (
        perm,
        MultiKeySortProof {
            strategy: MultiKeySortStrategy::FusedMultiKeyRadix,
            bit_widths,
            total_bit_width: 64u32.saturating_mul(n_keys as u32),
        },
        key_construction_s,
        radix_start.elapsed().as_secs_f64(),
    )
}

fn try_pack_multi_key_lex_u64(
    keys_flat: &[i64],
    n_keys: usize,
) -> Option<(Vec<u64>, Vec<u32>, u32)> {
    let n_groups = keys_flat.len() / n_keys;
    let (mins, bit_widths, total_bit_width) = multi_key_pack_layout_u64(keys_flat, n_keys)?;

    let mut packed = Vec::with_capacity(n_groups);
    for group in 0..n_groups {
        let mut key = 0u64;
        let mut remaining = total_bit_width;
        for col in 0..n_keys {
            let width = bit_widths[col];
            if width == 0 {
                continue;
            }
            remaining = remaining.checked_sub(width)?;
            let offset = (keys_flat[group * n_keys + col] as i128).checked_sub(mins[col])?;
            let offset = u64::try_from(offset).ok()?;
            if width < 64 && offset >= (1u64 << width) {
                return None;
            }
            key |= offset << remaining;
        }
        packed.push(key);
    }

    Some((packed, bit_widths, total_bit_width))
}

fn multi_key_pack_layout_u64(
    keys_flat: &[i64],
    n_keys: usize,
) -> Option<(Vec<i128>, Vec<u32>, u32)> {
    if n_keys == 0 || !keys_flat.len().is_multiple_of(n_keys) {
        return None;
    }

    let n_groups = keys_flat.len() / n_keys;
    let mut mins = vec![i128::MAX; n_keys];
    let mut maxs = vec![i128::MIN; n_keys];

    for group in 0..n_groups {
        for col in 0..n_keys {
            let key = keys_flat[group * n_keys + col] as i128;
            mins[col] = mins[col].min(key);
            maxs[col] = maxs[col].max(key);
        }
    }

    let mut bit_widths = Vec::with_capacity(n_keys);
    let mut total = 0u32;
    for col in 0..n_keys {
        let range = maxs[col].checked_sub(mins[col])?;
        let width = bit_width_for_inclusive_range(range)?;
        total = total.checked_add(width)?;
        if total > 64 {
            return None;
        }
        bit_widths.push(width);
    }

    Some((mins, bit_widths, total))
}

fn multi_key_bit_widths(keys_flat: &[i64], n_keys: usize) -> Option<Vec<u32>> {
    multi_key_pack_layout_any_width(keys_flat, n_keys).map(|(_, widths)| widths)
}

fn multi_key_pack_layout_any_width(
    keys_flat: &[i64],
    n_keys: usize,
) -> Option<(Vec<i128>, Vec<u32>)> {
    if n_keys == 0 || !keys_flat.len().is_multiple_of(n_keys) {
        return None;
    }

    let n_groups = keys_flat.len() / n_keys;
    let mut mins = vec![i128::MAX; n_keys];
    let mut maxs = vec![i128::MIN; n_keys];

    for group in 0..n_groups {
        for col in 0..n_keys {
            let key = keys_flat[group * n_keys + col] as i128;
            mins[col] = mins[col].min(key);
            maxs[col] = maxs[col].max(key);
        }
    }

    let mut bit_widths = Vec::with_capacity(n_keys);
    for col in 0..n_keys {
        let range = maxs[col].checked_sub(mins[col])?;
        bit_widths.push(bit_width_for_inclusive_range(range)?);
    }

    Some((mins, bit_widths))
}

fn bit_width_for_inclusive_range(range: i128) -> Option<u32> {
    let range = u128::try_from(range).ok()?;
    if range == 0 {
        return Some(0);
    }
    let values = range.checked_add(1)?;
    Some(u128::BITS - (values - 1).leading_zeros())
}

fn radix_sort_perm_by_u64_significant_bits_par(keys: &[u64], significant_bits: u32) -> Vec<usize> {
    let passes = usize::try_from(significant_bits.div_ceil(8)).unwrap_or(8);
    radix_sort_perm_by_digit_par(keys.len(), passes, |idx, pass| {
        ((keys[idx] >> (pass * 8)) & 0xFF) as usize
    })
}

#[cfg(test)]
mod columnar_tests {
    use super::{sort_first_columnar_permutation, MultiKeySortStrategy};

    #[test]
    fn columnar_base_and_packing_guards_include_exact_cap_and_reject_overflow() {
        const K: usize = 512 * 1024 * 1024;
        for live in [K - 1, K, K + 1] {
            assert_eq!(
                super::check_columnar_base(live - 30, 10, 20).is_ok(),
                live <= K
            );
            assert_eq!(super::check_columnar_packed(live - 8, 8).is_ok(), live <= K);
        }
        assert!(super::check_columnar_base(usize::MAX, 1, 0).is_err());
        assert!(super::check_columnar_packed(usize::MAX, 1).is_err());
        use crate::radix_sort_digit::route_allocation_test::{with_failure, Failure};
        let (result, snapshot) = with_failure(1, Failure::Reserve, || {
            super::checked_columnar_base(usize::MAX, 4, 2)
        });
        assert!(result.is_err());
        assert_eq!(snapshot.attempts, 0);
    }

    #[test]
    fn columnar_allocation_failures_release_layout_packing_and_radix_scratch() {
        use crate::radix_sort_digit::route_allocation_test::{with_failure, Failure};
        let first: Vec<i64> = (0..4097).map(|row| row % 31).collect();
        let second: Vec<i64> = (0..4097).map(|row| row % 7).collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        pool.install(|| {
            let (success, snapshot) = with_failure(usize::MAX, Failure::Reserve, || {
                sort_first_columnar_permutation(&[&first, &second])
            });
            assert!(success.is_ok());
            assert_eq!(snapshot.live, 0);
            for failure in [Failure::Reserve, Failure::Capacity] {
                for at in 1..=snapshot.attempts {
                    let (result, failed) = with_failure(at, failure, || {
                        sort_first_columnar_permutation(&[&first, &second])
                    });
                    assert!(result.is_err(), "missing failure at allocation {at}");
                    let error = result.unwrap_err();
                    match failure {
                        Failure::Reserve => assert!(error.contains("injected route reserve")),
                        Failure::Capacity => assert!(error.contains("exceeded modeled capacity")),
                    }
                    assert_eq!(failed.attempts, at, "allocation retried after error");
                    assert_eq!(failed.live, 0, "scratch retained after allocation {at}");
                    if at > 1 {
                        assert!(failed.peak > 0);
                    }
                }
            }
        });
    }

    fn check(columns: &[Vec<i64>], strategy: MultiKeySortStrategy) {
        let slices: Vec<&[i64]> = columns.iter().map(Vec::as_slice).collect();
        let mut expected: Vec<usize> = (0..columns[0].len()).collect();
        expected.sort_by(|&left, &right| {
            columns
                .iter()
                .map(|column| column[left])
                .cmp(columns.iter().map(|column| column[right]))
        });
        for threads in [1, 2, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let (actual, proof, layout, radix) =
                pool.install(|| sort_first_columnar_permutation(&slices).unwrap());
            assert_eq!(actual, expected);
            assert_eq!(proof.strategy, strategy);
            assert!(layout >= 0.0 && radix >= 0.0);
        }
    }

    #[test]
    fn packed_signed_columns_keep_equal_row_order_across_uneven_chunks() {
        check(
            &[
                (0..10_003).map(|row| (row * 37 % 127) - 63).collect(),
                (0..10_003).map(|row| (row * 11 % 31) - 15).collect(),
            ],
            MultiKeySortStrategy::PackedU64,
        );
    }

    #[test]
    fn fused_columns_cover_full_signed_range_and_duplicate_tuples() {
        let keys = [i64::MIN, i64::MAX, -1, 0, 1];
        check(
            &[
                (0..10_003).map(|row| keys[row % keys.len()]).collect(),
                (0..10_003)
                    .map(|row| keys[(row / 3) % keys.len()])
                    .collect(),
            ],
            MultiKeySortStrategy::FusedMultiKeyRadix,
        );
    }

    #[test]
    fn one_full_width_column_can_pack_without_shift_by_64() {
        check(
            &[vec![i64::MAX, i64::MIN, 0, i64::MIN], vec![7; 4]],
            MultiKeySortStrategy::PackedU64,
        );
    }

    #[test]
    fn empty_singleton_and_constant_columns_are_identity() {
        for n in [0, 1, 10_003] {
            let column = vec![i64::MIN; n];
            let (perm, proof, _, _) = sort_first_columnar_permutation(&[&column, &column]).unwrap();
            assert_eq!(perm, (0..n).collect::<Vec<_>>());
            assert_eq!(proof.total_bit_width, 0);
        }
    }

    #[test]
    fn malformed_column_layout_is_rejected() {
        assert!(sort_first_columnar_permutation(&[]).is_err());
        assert!(sort_first_columnar_permutation(&[&[1], &[]]).is_err());
        assert!(sort_first_columnar_permutation(&[&[] as &[i64]; 11]).is_err());
    }
}
