use crate::python_wrappers::convert::{
    build_single_profile_dict, convert_single_result, convert_single_result_f64,
    convert_single_result_i64,
};
use crate::python_wrappers::shared::{
    validate_inputs, validate_inputs_len, SingleGroupByProfileReturnF64, SingleGroupByReturn,
    SingleGroupByReturnF64, SingleGroupByReturnI64, FALLBACK_THRESHOLD,
};
use crate::{groupby, zero_copy};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::time::Instant;

mod abi;
mod firstseen_f64_u32;
mod firstseen_f64_u64;
mod firstseen_i64;
mod generated;
mod sorted_f64;
mod sorted_i64;
mod unsorted;

pub(crate) use abi::*;
pub(crate) use firstseen_f64_u32::*;
pub(crate) use firstseen_f64_u64::*;
pub(crate) use firstseen_i64::*;
pub(crate) use generated::*;
pub(crate) use sorted_f64::*;
pub(crate) use sorted_i64::*;
pub(crate) use unsorted::*;
