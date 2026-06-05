//! Parallel groupby implementation using Rayon's map-reduce pattern.
//!
//! Public names are re-exported here so Python wrappers keep the same call
//! surface while single-key groupby internals live in focused submodules.

mod api;
mod api_count;
mod api_f64;
mod api_f64_stats;
mod api_i64;
mod chunks;
mod deterministic;
mod engine;
mod legacy;
mod order;
mod partitioned;
mod profile;
mod reduce;
mod result;
mod routing;

#[cfg(test)]
mod api_float_tests;
#[cfg(test)]
mod api_i64_tests;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod engine_tests;
#[cfg(test)]
mod routing_tests;
#[cfg(test)]
mod test_support;

pub use api::*;
pub use result::{
    GroupByResult, GroupByResultF64, GroupByResultI64, ProfiledGroupByResult, SingleKeyPhaseProfile,
};
