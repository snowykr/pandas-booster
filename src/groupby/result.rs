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
    pub partial_group_total: usize,
    pub final_group_count: usize,
}

#[derive(Debug)]
pub struct ProfiledGroupByResult<V> {
    pub result: GroupByResult<V>,
    pub profile: SingleKeyPhaseProfile,
}

pub type GroupByResultF64 = GroupByResult<f64>;
pub type GroupByResultI64 = GroupByResult<i64>;
