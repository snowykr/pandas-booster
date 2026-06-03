//! Radix Partitioning Engine for high-performance multi-key groupby.
//!
//! Public names are re-exported here while radix internals live in focused
//! submodules for result shape, key operations, partitioning, ordering,
//! engines, dispatch, and API entrypoints.

mod api;
mod api_sorted;
mod dispatch;
mod engine;
mod firstseen_u32;
mod firstseen_u64;
mod keys;
mod order;
mod partition;
mod result;

#[cfg(test)]
mod api_basic_tests;
#[cfg(test)]
mod api_stats_tests;
#[cfg(test)]
mod dispatch_tests;
#[cfg(test)]
mod order_tests;
#[cfg(test)]
mod partition_tests;
#[cfg(test)]
mod test_support;

pub use api::*;
pub use api_sorted::*;
pub(crate) use partition::{stable_scatter_by_partition, SMALL_DIRECT_THRESHOLD_ELEMS};
pub use result::GroupByMultiResult;
