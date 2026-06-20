#[derive(Debug)]
pub struct GroupByMultiResult<V> {
    /// Row-major flat keys: keys for group g occupy `keys_flat[g*n_keys..(g+1)*n_keys]`.
    pub keys_flat: Vec<i64>,
    pub n_keys: usize,
    pub values: Vec<V>,
    /// Output ordering permutation.
    ///
    /// If present, output group at position `out_g` is sourced from group `perm[out_g]`.
    /// If None, `keys_flat`/`values` are already materialized in output order,
    /// so identity mapping is implied and consumers must not apply any additional permutation.
    pub perm: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
pub struct MultiKeySortedPhaseProfile {
    pub hash_build_s: f64,
    pub partition_scatter_s: f64,
    pub partition_aggregation_s: f64,
    pub flatten_s: f64,
    pub sort_key_construction_s: f64,
    pub radix_sort_s: f64,
    pub sorted_materialization_s: f64,
    pub partial_group_total: usize,
    pub final_group_count: usize,
}

#[derive(Debug)]
pub struct ProfiledGroupByMultiResult<V> {
    pub result: GroupByMultiResult<V>,
    pub profile: MultiKeySortedPhaseProfile,
}
