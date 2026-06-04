from __future__ import annotations

import conftest


def test_wheel_smoke_mode_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("PANDAS_BOOSTER_WHEEL_SMOKE", raising=False)
    assert not conftest.wheel_smoke_mode()


def test_wheel_smoke_mode_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("PANDAS_BOOSTER_WHEEL_SMOKE", "1")
    assert conftest.wheel_smoke_mode()
