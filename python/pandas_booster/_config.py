from __future__ import annotations

import os

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def rust_sort_enabled() -> bool:
    """Return True if Rust-side sort=True is enabled.

    Emergency toggle:
    - unset / 0: OFF (default)
    - 1: ON
    - also accepts: true/yes/on (case-insensitive)
    """
    value = os.getenv("PANDAS_BOOSTER_RUST_SORT")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY
