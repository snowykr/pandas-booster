from __future__ import annotations

import inspect
from functools import wraps
from numbers import Integral
from typing import Any

import pandas as pd

from .accessor import BoosterAccessor
from .proxy import BoosterDataFrameGroupBy

__all__ = ["BoosterAccessor", "activate", "deactivate", "is_active"]
__version__ = "0.1.2"

_original_groupby = None
_is_active = False


def _make_proxy_groupby(original_fn):
    sig = inspect.signature(original_fn)

    try:
        from pandas._libs import lib as _pd_lib  # type: ignore
    except Exception:  # pragma: no cover
        _pd_lib = None

    def _is_no_default(value: object) -> bool:
        return _pd_lib is not None and value is _pd_lib.no_default

    def _normalize_axis(axis: object) -> int | None:
        if _is_no_default(axis):
            return 0
        try:
            if isinstance(axis, str):
                axis_lower = axis.lower()
                if axis_lower in ("index", "rows"):
                    return 0
                if axis_lower in ("columns", "cols"):
                    return 1
                return None

            if isinstance(axis, bool):
                return None

            if not isinstance(axis, Integral):
                return None

            axis_num = int(axis)
        except Exception:
            return None
        return axis_num if axis_num in (0, 1) else None

    @wraps(original_fn)
    def proxy_groupby(self, by=None, *args: Any, **kwargs: Any) -> Any:
        # Always call pandas first to preserve native semantics/errors.
        gb_obj = original_fn(self, by, *args, **kwargs)

        try:
            bound = sig.bind(self, by, *args, **kwargs)
            provided = set(bound.arguments.keys())
            bound.apply_defaults()
            params = bound.arguments
        except Exception:
            return gb_obj

        axis = params.get("axis", 0)
        axis_num = _normalize_axis(axis)
        if axis_num is None:
            return gb_obj
        if axis_num != 0:
            return gb_obj

        try:
            level = params.get("level", None)
            as_index = bool(params.get("as_index", True))
            sort = bool(params.get("sort", True))
            dropna = bool(params.get("dropna", True))
        except Exception:
            return gb_obj

        if level is not None or not as_index or not dropna:
            return gb_obj

        # Conservative safety: if the user opts into behaviors we don't
        # explicitly support, do not proxy.
        if "squeeze" in provided:
            return gb_obj
        if "observed" in provided:
            observed = params.get("observed", False)
            try:
                observed_bool = bool(observed)
            except Exception:
                return gb_obj
            if not _is_no_default(observed) and observed_bool:
                return gb_obj
        if "group_keys" in provided:
            group_keys = params.get("group_keys", True)
            if _is_no_default(group_keys) or group_keys is not True:
                return gb_obj

        if by is None:
            return gb_obj

        by_cols = (
            [by] if isinstance(by, str) else list(by) if isinstance(by, (list, tuple)) else None
        )

        if by_cols is None:
            return gb_obj

        for col in by_cols:
            if not isinstance(col, str) or col not in self.columns:
                return gb_obj

        return BoosterDataFrameGroupBy(
            original_groupby=gb_obj,
            df=self,
            by_cols=by_cols,
            sort=sort,
        )

    # Preserve pandas signature for introspection tools.
    proxy_groupby.__signature__ = sig  # type: ignore[attr-defined]
    return proxy_groupby


def activate() -> None:
    """Enable global DataFrame.groupby monkey-patching."""
    global _original_groupby, _is_active

    if _is_active:
        return

    _original_groupby = pd.DataFrame.groupby
    pd.DataFrame.groupby = _make_proxy_groupby(_original_groupby)
    _is_active = True


def deactivate() -> None:
    """Disable global DataFrame.groupby monkey-patching."""
    global _original_groupby, _is_active

    if not _is_active or _original_groupby is None:
        return

    pd.DataFrame.groupby = _original_groupby
    _original_groupby = None
    _is_active = False


def is_active() -> bool:
    """Return True if groupby monkey-patching is active."""
    return _is_active
