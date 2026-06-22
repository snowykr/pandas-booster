use crate::radix_sort::i64_to_sortable_u64;
use crate::radix_sort_digit::radix_sort_perm_by_digit_par;

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
