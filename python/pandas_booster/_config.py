from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def force_pandas_sort_enabled() -> bool:
    """Return True if Python-side sort_index() is forced for sort=True.

    Emergency toggle:
    - unset / 0: OFF (default)
    - 1: ON
    - also accepts: true/yes/on (case-insensitive)
    """
    value = os.getenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY
