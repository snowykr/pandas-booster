from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

from .accessor import BoosterAccessor
from .proxy import BoosterDataFrameGroupBy

__all__ = ["BoosterAccessor", "activate", "deactivate", "is_active"]
__version__ = "0.1.0"

_original_groupby = None
_is_active = False


def _make_proxy_groupby(original_fn):
    @wraps(original_fn)
    def proxy_groupby(self, by=None, **kwargs) -> Any:
        gb_obj = original_fn(self, by=by, **kwargs)

        axis = kwargs.get("axis", 0)
        level = kwargs.get("level", None)
        as_index = kwargs.get("as_index", True)
        sort = kwargs.get("sort", True)
        dropna = kwargs.get("dropna", True)

        if axis != 0 or level is not None or not as_index or not dropna:
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

    return proxy_groupby


def activate() -> None:
    global _original_groupby, _is_active

    if _is_active:
        return

    _original_groupby = pd.DataFrame.groupby
    setattr(pd.DataFrame, "groupby", _make_proxy_groupby(_original_groupby))
    _is_active = True


def deactivate() -> None:
    global _original_groupby, _is_active

    if not _is_active or _original_groupby is None:
        return

    setattr(pd.DataFrame, "groupby", _original_groupby)
    _original_groupby = None
    _is_active = False


def is_active() -> bool:
    return _is_active
