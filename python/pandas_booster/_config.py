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


def strict_abi_enabled() -> bool:
    """Return True if ABI skew should raise instead of falling back.

    - unset / 0: OFF (default)
    - 1: ON
    - also accepts: true/yes/on (case-insensitive)
    """
    value = os.getenv("PANDAS_BOOSTER_STRICT_ABI")
    if value is None:
        return False
    return value.strip().lower() in _TRUTHY


def abi_skew_notice_enabled() -> bool:
    """Return True if ABI-skew notices should be emitted.

    - unset: ON (default)
    - truthy (1/true/yes/on): ON
    - anything else (including 0/false/no/off): OFF
    """
    value = os.getenv("PANDAS_BOOSTER_ABI_SKEW_NOTICE")
    if value is None:
        return True
    return value.strip().lower() in _TRUTHY
