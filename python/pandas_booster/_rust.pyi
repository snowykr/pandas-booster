from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


def get_fallback_threshold() -> int: ...
def get_thread_count() -> int: ...


def groupby_sum_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_mean_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_min_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_max_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_count_f64(
    keys: NDArray[np.int64],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...


def groupby_sum_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_mean_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_min_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_max_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
def groupby_count_i64(
    keys: NDArray[np.int64],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...


def groupby_multi_sum_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_mean_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_min_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_max_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_count_f64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.float64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...


def groupby_multi_sum_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_mean_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_min_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_max_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...

def groupby_multi_count_i64(
    key_arrays: Sequence[NDArray[np.int64]],
    values: NDArray[np.int64],
) -> tuple[NDArray[np.int64], NDArray[np.float64]]: ...
