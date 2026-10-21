from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def procfs(driver, monkeypatch):
    source = SimpleNamespace(text="VmRSS:\t100 kB\nVmHWM:\t200 kB\n", fail_on=set(), reads=0)
    read_text = Path.read_text

    def read_status(path, *args, **kwargs):
        if path.as_posix() == "/proc/self/status":
            source.reads += 1
            if source.reads in source.fail_on:
                raise FileNotFoundError("simulated missing Linux procfs")
            return source.text
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(driver.sys, "platform", "linux")
    monkeypatch.setattr(Path, "read_text", read_status)
    return source
