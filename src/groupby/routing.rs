use ahash::AHashSet;

const STD_VAR_ENGINE_SAMPLE_SIZE: usize = 16_384;
const STD_VAR_ENGINE_MIN_SAMPLE_UNIQUES: usize = 4_096;

fn estimate_sample_unique_keys(keys: &[i64]) -> usize {
    let sample_size = keys.len().min(STD_VAR_ENGINE_SAMPLE_SIZE);
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
pub(super) fn should_use_partitioned_std_var_engine(keys: &[i64]) -> bool {
    let sample_size = keys.len().min(STD_VAR_ENGINE_SAMPLE_SIZE);
    sample_size > STD_VAR_ENGINE_MIN_SAMPLE_UNIQUES
        && estimate_sample_unique_keys(keys) >= STD_VAR_ENGINE_MIN_SAMPLE_UNIQUES
}

#[inline]
pub(super) fn should_use_partitioned_median_engine(keys: &[i64]) -> bool {
    should_use_partitioned_std_var_engine(keys)
}
