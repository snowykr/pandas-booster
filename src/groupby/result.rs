/// Result container for groupby operations, holding key-value pairs.
#[derive(Debug)]
pub struct GroupByResult<V> {
    pub keys: Vec<i64>,
    pub values: Vec<V>,
}

#[derive(Debug, Clone)]
pub struct SingleKeyPhaseProfile {
    pub local_build_s: f64,
    pub merge_s: f64,
    pub reorder_s: f64,
    pub materialize_s: f64,
    pub unique_build_s: f64,
    pub key_sort_s: f64,
    pub count_s: f64,
    pub buffer_setup_s: f64,
    pub scatter_s: f64,
    pub median_select_s: f64,
    pub partial_group_total: usize,
    pub final_group_count: usize,
}

impl SingleKeyPhaseProfile {
    pub(crate) fn legacy(
        local_build_s: f64,
        merge_s: f64,
        reorder_s: f64,
        materialize_s: f64,
        partial_group_total: usize,
        final_group_count: usize,
    ) -> Self {
        Self {
            local_build_s,
            merge_s,
            reorder_s,
            materialize_s,
            unique_build_s: 0.0,
            key_sort_s: 0.0,
            count_s: 0.0,
            buffer_setup_s: 0.0,
            scatter_s: 0.0,
            median_select_s: 0.0,
            partial_group_total,
            final_group_count,
        }
    }
}

#[derive(Debug)]
pub struct ProfiledGroupByResult<V> {
    pub result: GroupByResult<V>,
    pub profile: SingleKeyPhaseProfile,
}

pub type GroupByResultF64 = GroupByResult<f64>;
pub type GroupByResultI64 = GroupByResult<i64>;
