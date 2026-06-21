use ahash::AHashSet;

const PARTITIONED_ENGINE_SAMPLE_SIZE: usize = 16_384;
const PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES: usize = 4_096;
const DIRECT_SORTED_MEDIAN_MAX_ROWS: usize = 300_000;
pub(super) const DIRECT_SORTED_MEDIAN_MAX_DENSE_KEY_RANGE: usize = 65_536;

fn estimate_sample_unique_keys(keys: &[i64]) -> usize {
    let sample_size = keys.len().min(PARTITIONED_ENGINE_SAMPLE_SIZE);
    if sample_size == 0 {
        return 0;
    }

    let stride = keys.len().div_ceil(sample_size);
    let mut seen = AHashSet::with_capacity(sample_size);
    let mut row = 0usize;
    let mut sampled = 0usize;

    while row < keys.len() && sampled < sample_size {
        seen.insert(keys[row]);
        row += stride;
        sampled += 1;
    }

    seen.len()
}

#[inline]
pub(super) fn should_use_partitioned_firstseen_engine(keys: &[i64]) -> bool {
    let sample_size = keys.len().min(PARTITIONED_ENGINE_SAMPLE_SIZE);
    sample_size > PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES
        && estimate_sample_unique_keys(keys) >= PARTITIONED_ENGINE_MIN_SAMPLE_UNIQUES
}

#[inline]
pub(super) fn should_use_partitioned_std_var_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}

#[inline]
pub(super) fn should_use_partitioned_median_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}

#[inline]
pub(super) fn should_use_direct_sorted_median_engine(keys: &[i64]) -> bool {
    if should_use_partitioned_median_engine(keys) {
        return false;
    }

    keys.len() <= DIRECT_SORTED_MEDIAN_MAX_ROWS || dense_sorted_median_key_range_len(keys).is_some()
}

pub(super) fn dense_sorted_median_key_range_len(keys: &[i64]) -> Option<usize> {
    let (&first, rest) = keys.split_first()?;
    let mut min_key = first;
    let mut max_key = first;

    for key in rest.iter().copied() {
        min_key = min_key.min(key);
        max_key = max_key.max(key);
    }

    let span = i128::from(max_key) - i128::from(min_key) + 1;
    if span <= DIRECT_SORTED_MEDIAN_MAX_DENSE_KEY_RANGE as i128 {
        usize::try_from(span).ok()
    } else {
        None
    }
}

#[inline]
pub(super) fn should_use_partitioned_prod_engine(keys: &[i64]) -> bool {
    should_use_partitioned_firstseen_engine(keys)
}
