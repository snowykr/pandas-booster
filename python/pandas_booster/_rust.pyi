from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

SingleReturnF64 = tuple[NDArray[np.int64], NDArray[np.float64]]
SingleReturnI64 = tuple[NDArray[np.int64], NDArray[np.int64]]
SingleProfile = dict[str, float | int]
SingleProfileReturnF64 = tuple[NDArray[np.int64], NDArray[np.float64], SingleProfile]

MultiKeys = list[NDArray[np.int64]]
MultiReturnF64 = tuple[MultiKeys, NDArray[np.float64]]
MultiReturnI64 = tuple[MultiKeys, NDArray[np.int64]]

def get_fallback_threshold() -> int: ...
def get_thread_count() -> int: ...
def groupby_sum_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_mean_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_var_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_std_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_min_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_max_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_count_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnI64: ...
def groupby_sum_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_mean_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_var_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_std_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_min_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_max_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_count_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnI64: ...
def profile_groupby_var_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleProfileReturnF64: ...
def profile_groupby_std_f64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleProfileReturnF64: ...
def groupby_sum_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_sum_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_mean_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_mean_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_var_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_var_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_std_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_std_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_min_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_min_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_max_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_max_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnF64: ...
def groupby_count_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnI64: ...
def groupby_count_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleReturnI64: ...
def profile_groupby_var_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleProfileReturnF64: ...
def profile_groupby_var_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleProfileReturnF64: ...
def profile_groupby_std_f64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleProfileReturnF64: ...
def profile_groupby_std_f64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> SingleProfileReturnF64: ...
def groupby_sum_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_mean_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_var_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_std_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_min_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_max_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_count_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_sum_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_mean_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_var_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_std_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_min_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_max_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_count_i64_sorted(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_sum_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_sum_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_mean_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_mean_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_var_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_var_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_std_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_std_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnF64: ...
def groupby_min_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_min_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_max_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_max_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_count_i64_firstseen_u32(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_count_i64_firstseen_u64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> SingleReturnI64: ...
def groupby_multi_sum_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_mean_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_var_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_std_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_min_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_max_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_count_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnI64: ...
def groupby_multi_sum_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_mean_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_var_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_std_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_min_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_max_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_count_f64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnI64: ...
def groupby_multi_sum_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_sum_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_mean_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_var_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_std_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_mean_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_var_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_std_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_min_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_min_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_max_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_max_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnF64: ...
def groupby_multi_count_f64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnI64: ...
def groupby_multi_count_f64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> MultiReturnI64: ...
def groupby_multi_sum_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_mean_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_var_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_std_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_min_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_max_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_count_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_sum_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_mean_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_var_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_std_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_min_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_max_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_count_i64_sorted(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_sum_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_sum_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_mean_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_var_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_std_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_mean_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_var_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_std_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnF64: ...
def groupby_multi_min_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_min_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_max_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_max_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_count_i64_firstseen_u32(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
def groupby_multi_count_i64_firstseen_u64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> MultiReturnI64: ...
