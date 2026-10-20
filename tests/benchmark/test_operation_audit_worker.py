from __future__ import annotations

import sys
import types

import pytest

from . import test_operation_audit

audit = test_operation_audit.audit
runtime = test_operation_audit.runtime


@pytest.mark.parametrize("native,sort", [(False, True), (False, False), (True, False)])
@pytest.mark.parametrize("include_nans", [False, True])
def test_worker_checks_real_nan_count_parity(
    audit, tmp_path, monkeypatch, native, sort, include_nans
):
    import numpy as np
    import pandas as pd

    package = tmp_path / "runtime/lib/pandas_booster"
    package.mkdir(parents=True)
    extension = package / "_rust.so"
    rust = types.ModuleType("pandas_booster._rust")
    rust.__file__ = str(extension)
    rust.get_thread_count = lambda: 1
    booster = types.ModuleType("pandas_booster")
    booster.__file__ = str(package / "__init__.py")
    booster._rust = rust
    monkeypatch.setitem(sys.modules, "pandas_booster", booster)
    monkeypatch.setitem(sys.modules, "pandas_booster._rust", rust)
    calls = []

    def count(frame, keys, value, sort):
        assert frame[value].dtype == np.dtype("float64")
        assert frame[value].isna().any()
        grouped = frame.groupby(keys, sort=sort)[value]
        correct, erroneous = grouped.count(), grouped.size()
        assert (correct != erroneous).any()
        calls.append("native" if native else "public")
        return erroneous if include_nans else correct

    class PublicAccessor:
        def __init__(self, frame):
            self.frame = frame

        def groupby(self, keys, value, agg, sort):
            assert agg == "count"
            return count(self.frame, keys, value, sort)

    def native_count(columns, values):
        keys = [f"k{i}" for i in range(len(columns))]
        frame = pd.DataFrame(dict(zip(keys, columns)), copy=False)
        frame["value"] = values
        result = count(frame, keys, "value", False)
        return [
            result.index.get_level_values(i).to_numpy() for i in range(len(keys))
        ], result.to_numpy()

    rust.groupby_multi_count_f64_firstseen_u64 = native_count
    monkeypatch.setattr(pd.DataFrame, "booster", property(PublicAccessor), raising=False)
    runtime = {
        "python": str(tmp_path / "runtime/bin/python"),
        "python_identity": "controlled",
        "package": str(package),
        "extension": str(extension),
        "dependencies": [],
        "manifest": {"source_root": str(tmp_path / "source")},
    }
    monkeypatch.setattr(audit, "validate_provenance", lambda runtime: None)
    monkeypatch.setattr(
        audit,
        "sys",
        types.SimpleNamespace(
            executable=runtime["python"], flags=types.SimpleNamespace(isolated=True)
        ),
    )
    monkeypatch.setattr(audit.route, "clean_env", lambda threads: dict(audit.os.environ))
    monkeypatch.setattr(audit.route, "python_identity", lambda: "controlled")
    monkeypatch.setattr(audit.route.importlib.metadata, "distributions", lambda: [])
    request = {
        "purpose": audit.PURPOSE,
        "kind": "operations",
        "runtime": runtime,
        "threads": 1,
        "rows": 4096,
        "agg": "count",
        "sort": sort,
        "native_u64": native,
        "dtype": "float64",
    }
    if include_nans:
        with pytest.raises(AssertionError, match="Series are different"):
            audit.audit_worker(request)
    else:
        result = audit.audit_worker(request)
        assert result["parity"] == "exact"
        assert result["purpose"] == audit.PURPOSE
        assert result["dtype"] == "float64"
        assert result["appearance_order_checked"] is (not sort)
    assert calls == ["native" if native else "public"]
