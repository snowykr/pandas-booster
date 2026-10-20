use crate::python_wrappers::convert::{
    convert_multi_result, convert_multi_result_f64, convert_multi_result_i64,
};
use crate::python_wrappers::profile::build_multi_sorted_profile_dict;
use crate::python_wrappers::shared::{
    validate_multi_inputs, validate_multi_inputs_len, MultiGroupByProfileReturnF64,
    MultiGroupByProfileReturnI64, MultiGroupByReturn, MultiGroupByReturnF64, MultiGroupByReturnI64,
};
use crate::{groupby_multi, zero_copy};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use std::time::Instant;

mod firstseen_f64;
mod firstseen_i64;
mod generated;
mod sorted_f64;
mod sorted_i64;
mod unsorted_f64;
mod unsorted_i64;

pub(crate) use firstseen_f64::*;
pub(crate) use firstseen_i64::*;
pub(crate) use generated::*;
pub(crate) use sorted_f64::*;
pub(crate) use sorted_i64::*;
pub(crate) use unsorted_f64::*;
pub(crate) use unsorted_i64::*;
