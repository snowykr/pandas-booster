from __future__ import annotations

import importlib.util
import sys
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

import pytest

_BENCHMARK_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "benchmark.py"


@contextmanager
def _loaded_benchmark_module() -> Generator[ModuleType, None, None]:
    sys_path_snapshot = list(sys.path)
    try:
        spec = importlib.util.spec_from_file_location("benchmark_under_test", _BENCHMARK_PATH)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.path[:] = sys_path_snapshot


@pytest.fixture(scope="module")
def benchmark_module():
    with _loaded_benchmark_module() as module:
        yield module
