//! Aggregation primitives for parallel groupby operations.
//!
//! This module defines the [`Aggregator`] trait and implementations for common
//! groupby aggregations including sum, product, mean, median, variance, standard
//! deviation, min, max, and count on primitive numeric types.

#[path = "median.rs"]
pub(crate) mod median;

mod float;
mod integer;
mod variance;

pub use float::{
    CountAggF64, MaxAggF64, MeanAggF64, MedianAggF64, MinAggF64, ProdAggF64, SumAggF64,
};
pub use integer::{
    CountAggI64, MaxAggI64, MeanAggI64, MedianAggI64, MinAggI64, ProdAggI64, SumAggI64,
};
pub use variance::{StdAggF64, StdAggI64, VarAggF64, VarAggI64};

#[cfg(test)]
mod float_tests;
#[cfg(test)]
mod integer_tests;
#[cfg(test)]
mod variance_tests;

/// Core trait for streaming aggregation with support for parallel merge.
///
/// Implementors must be thread-safe (`Send + Sync`) to support Rayon's parallel fold/reduce.
pub trait Aggregator<T, O>: Send + Sync {
    /// Creates a new aggregator with identity state.
    fn init() -> Self;
    /// Incorporates a single value into the aggregation.
    fn update(&mut self, value: T);
    /// Merges another aggregator's state into this one (for parallel reduction).
    fn merge(&mut self, other: Self);
    /// Computes the final aggregated result.
    fn finalize(&self) -> O;
    /// Computes the final result when the accumulator is no longer needed.
    ///
    /// The default implementation preserves the immutable finalization contract,
    /// while Vec-backed aggregators can override this to consume their buffers
    /// and avoid a materialization-time clone. Implementations must return the
    /// same logical result as [`Aggregator::finalize`] for the same state.
    fn finalize_owned(self) -> O
    where
        Self: Sized,
    {
        self.finalize()
    }
}
