from __future__ import annotations

import logging
import sys
import threading
import warnings
from collections.abc import Sequence
from typing import Any, NoReturn

import numpy as np

from ._config import abi_skew_notice_enabled, strict_abi_enabled

logger = logging.getLogger(__name__)


ABI_SKEW_PREFIX = "pandas-booster ABI skew"


class PandasBoosterAbiSkewWarning(RuntimeWarning):
    """Emitted when pandas-booster detects a Python/Rust wheel ABI skew."""


class PandasBoosterKeyShapeSkewError(ValueError):
    """Raised for recognized Rust/Python ABI skew.

    This error is intentionally narrow: it is only raised for ABI skew detected
    by Python-side validation of the Rust return payload (e.g. unexpected tuple
    shape, mismatched container types, unexpected ndarray dimensions/shapes).

    Callers may catch this to fall back to pandas in non-strict mode without
    masking unrelated Rust execution bugs.
    """


_WARNED_ABI_SKEW = False
_WARNED_ABI_SKEW_LOCK = threading.Lock()


def _best_effort_warning_stacklevel() -> int:
    """Return a stacklevel that best attributes warnings to user code.

    The stacklevel for `warnings.warn` is counted from the call site of `warnings.warn`.
    We walk Python frames and skip internal modules (`pandas_booster.*` and `pandas.*`)
    to point the warning at the first external frame (typically user/test code).

    This function must never raise.
    """
    try:
        # Frame of the `warnings.warn` call site (`emit_abi_skew_notice_once`).
        frame = sys._getframe(1)
    except Exception:
        return 4

    stacklevel = 1
    max_depth = 64
    while frame is not None and stacklevel < max_depth:
        module_name = frame.f_globals.get("__name__", "")
        if module_name == "pandas" or module_name.startswith("pandas."):
            frame = frame.f_back
            stacklevel += 1
            continue
        if module_name == "pandas_booster" or module_name.startswith("pandas_booster."):
            frame = frame.f_back
            stacklevel += 1
            continue
        break

    return stacklevel


def _abi_skew_message(*, context: str, detail: str) -> str:
    return (
        f"{ABI_SKEW_PREFIX} ({context}): {detail} Reinstall matching pandas-booster wheel versions."
    )


def emit_abi_skew_notice_once(message: str) -> None:
    """Best-effort warn-once; must never raise.

    Some environments configure warnings as errors (e.g. PYTHONWARNINGS=error).
    ABI skew must not crash execution in non-strict mode, so we swallow any
    Warning raised by warnings.warn.
    """
    if not abi_skew_notice_enabled():
        return

    global _WARNED_ABI_SKEW
    with _WARNED_ABI_SKEW_LOCK:
        if _WARNED_ABI_SKEW:
            return
        # Flip the gate before emitting so warning-as-error can't cause repeated raises.
        _WARNED_ABI_SKEW = True
    try:
        # Best-effort attribution to user code; only executed once per process.
        stacklevel = _best_effort_warning_stacklevel()
        warnings.warn(message, category=PandasBoosterAbiSkewWarning, stacklevel=stacklevel)
    except Warning:
        # Fall back to logging for observability.
        logger.warning(message)


def raise_abi_skew(*, context: str, detail: str) -> NoReturn:
    msg = _abi_skew_message(context=context, detail=detail)
    emit_abi_skew_notice_once(msg)
    raise PandasBoosterKeyShapeSkewError(msg)


def normalize_multi_keys_cols(
    keys_cols: Any,
    *,
    n_groups: int,
    n_keys: int,
    context: str,
    strict: bool | None = None,
) -> list[np.ndarray]:
    """Normalize multi-key groupby keys to `list[np.ndarray]` (one 1D array per key).

    Supported inputs:
    - Current ABI: sequence of length `n_keys`, each element 1D length `n_groups`.
    - Legacy ABI: 2D ndarray with exact shape `(n_groups, n_keys)`.

    Any other shape/container is treated as recognized ABI skew and raises
    `PandasBoosterKeyShapeSkewError`.

    Strict mode:
    - If `PANDAS_BOOSTER_STRICT_ABI=1`, *any* ABI skew (including legacy 2D keys)
      raises and must not be normalized.
    """
    if strict is None:
        strict = strict_abi_enabled()

    # Legacy ABI: single 2D ndarray shaped (n_groups, n_keys)
    if isinstance(keys_cols, np.ndarray):
        if strict:
            raise_abi_skew(
                context=context,
                detail=(
                    f"detected legacy 2D keys ndarray shape={keys_cols.shape}; "
                    "strict ABI mode enabled."
                ),
            )

        if keys_cols.ndim != 2:
            raise_abi_skew(
                context=context,
                detail=(
                    f"expected multi-key keys as a sequence of {n_keys} 1D arrays or a 2D ndarray "
                    f"shaped ({n_groups}, {n_keys}); got ndarray ndim={keys_cols.ndim} "
                    f"shape={keys_cols.shape}."
                ),
            )

        if keys_cols.shape != (n_groups, n_keys):
            raise_abi_skew(
                context=context,
                detail=(
                    f"detected 2D keys ndarray ndim=2 shape={keys_cols.shape}, but expected legacy "
                    f"shape ({n_groups}, {n_keys})."
                ),
            )

        emit_abi_skew_notice_once(
            _abi_skew_message(
                context=context,
                detail=(
                    f"detected legacy 2D keys ndarray shape=({n_groups}, {n_keys}); normalizing to "
                    "per-column 1D keys."
                ),
            )
        )

        # Copy only on the legacy path for stable downstream expectations.
        return [np.ascontiguousarray(keys_cols[:, i]) for i in range(n_keys)]

    # Current ABI: sequence of per-column 1D arrays
    if not isinstance(keys_cols, Sequence):
        raise_abi_skew(
            context=context,
            detail=(
                "expected multi-key keys as a sequence of "
                f"{n_keys} 1D arrays (or legacy 2D ndarray). "
                f"Got type={type(keys_cols)!r}."
            ),
        )

    if len(keys_cols) != n_keys:
        raise_abi_skew(
            context=context,
            detail=(f"expected {n_keys} key columns, got {len(keys_cols)}."),
        )

    out: list[np.ndarray] = []
    for i in range(n_keys):
        arr = np.asarray(keys_cols[i])
        if arr.ndim != 1 or arr.shape[0] != n_groups:
            raise_abi_skew(
                context=context,
                detail=(
                    f"keys_cols[{i}] expected 1D length {n_groups}, got ndim={arr.ndim} "
                    f"shape={arr.shape}."
                ),
            )
        out.append(arr)
    return out
