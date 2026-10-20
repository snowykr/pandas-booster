from __future__ import annotations

import base64
import csv
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def driver():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "route_evidence.py"
    spec = importlib.util.spec_from_file_location("route_evidence_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_does_not_inherit_overrides(driver, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/unexpected/source")
    monkeypatch.setenv("PANDAS_BOOSTER_FORCE_PANDAS_SORT", "1")
    monkeypatch.setenv("RUSTFLAGS", "-C opt-level=0")
    environment = driver.clean_env(4)
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert "RUSTFLAGS" not in environment
    assert "PANDAS_BOOSTER_FORCE_PANDAS_SORT" not in environment
    assert environment["RAYON_NUM_THREADS"] == "4"


def test_python_identity_records_actual_executable_and_full_build(driver):
    import platform

    identity = driver.python_identity()
    assert identity["version"] == sys.version
    assert identity["implementation"] == platform.python_implementation()
    assert identity["build"] == list(platform.python_build())
    assert identity["executable"] == str(Path(sys.executable).resolve())
    assert identity["executable_sha256"] == driver.digest(Path(sys.executable).resolve())


def test_stamp_wheel_embeds_provenance_and_rehashes_record(driver, tmp_path):
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pandas_booster/__init__.py", "value = 1\n")
        archive.writestr("package.dist-info/RECORD", "obsolete")
    manifest = {"sha": driver.BASELINE, "release": True, "sources": {"src/lib.rs": "abc"}}
    driver.stamp_wheel(wheel, manifest)
    with zipfile.ZipFile(wheel) as archive:
        assert json.loads(archive.read(driver.MANIFEST)) == manifest
        rows = list(csv.reader(io.StringIO(archive.read("package.dist-info/RECORD").decode())))
        assert rows[-1] == ["package.dist-info/RECORD", "", ""]
        for name, hashed, size in rows[:-1]:
            content = archive.read(name)
            expected = base64.urlsafe_b64encode(driver.hashlib.sha256(content).digest()).rstrip(
                b"="
            )
            assert hashed == "sha256=" + expected.decode()
            assert size == str(len(content))


@pytest.fixture
def runtime(driver, tmp_path, monkeypatch):
    root = tmp_path / "artifact"
    source = root / "source"
    source.mkdir(parents=True)
    (source / "lib.rs").write_text("fn main() {}\n")
    git_sources = driver.files_manifest(source)
    (source / "Cargo.lock").write_text("# explicit external lock\nversion = 4\n")
    cargo_lock_sha256 = driver.digest(source / "Cargo.lock")
    (root / "build.log").write_text("release build\n")
    package = root / "runtime/lib/site-packages/pandas_booster"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    extension = package / "_rust.so"
    extension.write_bytes(b"test extension")
    python = root / "runtime/bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test interpreter")
    lock = root / "requirements.lock"
    lock.write_text("locked dependencies\n")
    wheel = root / "package.whl"
    wheel.write_bytes(b"wheel bytes")
    build_environment = dict(
        driver.clean_env(), CARGO_TARGET_DIR=str(root / "cargo-target"), RUSTFLAGS=""
    )
    build_environment["PATH"] = (
        str(Path.home() / ".cargo/bin") + driver.os.pathsep + driver.os.defpath
    )
    manifest = {
        "sha": driver.BASELINE,
        "sources": driver.files_manifest(source),
        "source_root": str(source),
        "release": True,
        "git_sources": git_sources,
        "cargo_lock_sha256": cargo_lock_sha256,
        "snapshot_overlays": {
            "Cargo.lock": {
                "origin": "explicit --cargo-lock input",
                "input_path": "/external/Cargo.lock",
                "sha256": cargo_lock_sha256,
                "git_sha256": None,
            }
        },
        "build_command": ["maturin", "build", "--release", "--locked"],
        "build_environment": build_environment,
        "lock_sha256": driver.digest(lock),
        "build_log_sha256": driver.digest(root / "build.log"),
        "rustc": "rustc test verbose identity",
        "cargo": "cargo test identity",
    }
    driver.save(package / "_route_evidence_manifest.json", manifest)
    descriptor = {
        "python": str(python),
        "package": str(package),
        "extension": str(extension),
        "manifest": manifest,
        "wheel": str(wheel),
        "wheel_sha256": driver.digest(wheel),
        "lock": str(lock),
        "installed_files": driver.files_manifest(package),
        "dependencies": [["numpy", "1.26.4"]],
    }
    fake_package = ModuleType("pandas_booster")
    fake_package.__path__ = [str(package)]
    fake_package.__file__ = str(package / "__init__.py")
    fake_rust = ModuleType("pandas_booster._rust")
    fake_rust.__file__ = str(extension)
    fake_rust.get_thread_count = lambda: 4
    fake_package._rust = fake_rust
    monkeypatch.setitem(sys.modules, "pandas_booster", fake_package)
    monkeypatch.setitem(sys.modules, "pandas_booster._rust", fake_rust)
    monkeypatch.setattr(driver.sys, "executable", str(python))
    descriptor["python_identity"] = driver.python_identity()
    monkeypatch.setattr(driver.sys, "flags", SimpleNamespace(isolated=1))
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    monkeypatch.setattr(
        driver.importlib.metadata,
        "distributions",
        lambda: [SimpleNamespace(metadata={"Name": "numpy"}, version="1.26.4")],
    )
    return descriptor


def test_runtime_identity_validates_artifacts(driver, runtime, monkeypatch):
    # Pytest adds PYTEST_CURRENT_TEST after fixture setup; establish the exact
    # worker environment in the call phase, as the subprocess launcher does.
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    identity = driver.validate_runtime(runtime, 4)
    assert identity["sha"] == driver.BASELINE
    assert identity["wheel_sha256"] == runtime["wheel_sha256"]
    assert identity["python_identity"] == runtime["python_identity"]


@pytest.mark.parametrize("missing", [False, True])
def test_runtime_rejects_interpreter_identity_skew(driver, runtime, monkeypatch, missing):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    if missing:
        del runtime["python_identity"]
    else:
        Path(runtime["python"]).write_bytes(b"different interpreter")
    with pytest.raises(ValueError, match="Python identity"):
        driver.validate_runtime(runtime, 4)


@pytest.mark.parametrize(
    "target,match",
    [
        ("wheel", "Wheel SHA256"),
        ("lock", "Dependency lock"),
        ("extension", "Installed package/extension hash"),
    ],
)
def test_runtime_rejects_mutated_artifacts(driver, runtime, target, match, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    Path(runtime[target]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match=match):
        driver.validate_runtime(runtime, 4)


def test_runtime_rejects_source_tampering(driver, runtime, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    (Path(runtime["manifest"]["source_root"]) / "lib.rs").write_text("changed\n")
    with pytest.raises(ValueError, match="source snapshot mismatch"):
        driver.validate_runtime(runtime, 4)


def test_runtime_rejects_cargo_lock_tampering(driver, runtime, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    (Path(runtime["manifest"]["source_root"]) / "Cargo.lock").write_text("changed\n")
    with pytest.raises(ValueError, match="Cargo lock mismatch"):
        driver.validate_runtime(runtime, 4)


def test_runtime_rejects_cargo_lock_attributed_to_commit(driver, runtime, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    runtime["manifest"]["snapshot_overlays"]["Cargo.lock"]["origin"] = "git archive"
    driver.save(Path(runtime["package"]) / "_route_evidence_manifest.json", runtime["manifest"])
    with pytest.raises(ValueError, match="overlay provenance mismatch"):
        driver.validate_runtime(runtime, 4)


def test_runtime_rejects_override_and_dependency_skew(driver, runtime, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    monkeypatch.setitem(driver.os.environ, "PYTHONPATH", "/source")
    with pytest.raises(ValueError, match="environment differs"):
        driver.validate_runtime(runtime, 4)
    monkeypatch.delitem(driver.os.environ, "PYTHONPATH")
    runtime["dependencies"] = [["numpy", "wrong"]]
    with pytest.raises(ValueError, match="dependency inventory mismatch"):
        driver.validate_runtime(runtime, 4)


def test_runtime_rejects_source_imports(driver, runtime, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    runtime["package"] = runtime["manifest"]["source_root"]
    with pytest.raises(ValueError, match="import path"):
        driver.validate_runtime(runtime, 4)


def test_runtime_rejects_nonrelease_flags(driver, runtime, monkeypatch):
    monkeypatch.setattr(driver.os, "environ", driver.clean_env(4))
    runtime["manifest"]["build_command"].remove("--release")
    # Keep embedded receipt consistent to test flags, not the earlier provenance gate.
    package = Path(runtime["package"])
    driver.save(package / "_route_evidence_manifest.json", runtime["manifest"])
    with pytest.raises(ValueError, match="release build evidence"):
        driver.validate_runtime(runtime, 4)


def test_exact_parity_rejects_zero_sign_and_metadata(driver):
    import pandas as pd

    expected = pd.Series([0.0], name="value", index=pd.Index([1], name="key"))
    with pytest.raises(AssertionError):
        driver.exact_equal(pd.Series([-0.0], name="value", index=expected.index), expected)
    with pytest.raises(AssertionError):
        driver.exact_equal(expected.rename("different"), expected)
    with pytest.raises(AssertionError):
        driver.exact_equal(expected.astype("float32"), expected)
