from __future__ import annotations

import importlib.util
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def audit():
    path = Path(__file__).resolve().parents[2] / "benchmarks" / "operation_audit.py"
    spec = importlib.util.spec_from_file_location("operation_audit_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seal(audit, descriptor):
    """Keep secondary receipts consistent so negative tests reach the intended gate."""
    package = Path(descriptor["package"])
    audit.route.save(package / "_route_evidence_manifest.json", descriptor["manifest"])
    with zipfile.ZipFile(descriptor["wheel"], "w") as wheel:
        wheel.writestr(audit.route.MANIFEST, json.dumps(descriptor["manifest"]))
        wheel.writestr(
            "pandas_booster/" + Path(descriptor["extension"]).name,
            Path(descriptor["extension"]).read_bytes(),
        )
    descriptor["wheel_sha256"] = audit.route.digest(Path(descriptor["wheel"]))
    descriptor["installed_files"] = audit.route.files_manifest(package)


@pytest.fixture
def runtime(audit, tmp_path, monkeypatch):
    root = tmp_path / "diagnostic"
    source, package = root / "source", root / "runtime/lib/pandas_booster"
    source.mkdir(parents=True)
    package.mkdir(parents=True)
    revision = "a" * 40
    original = {}
    for name, anchor, count, _ in audit.EDITS + [audit.CANDIDATE_EDIT]:
        original[name] = original.get(name, b"") + ((anchor + "\n") * count).encode()
    original["Cargo.toml"] = b"[package]\nname='diagnostic-fixture'\n"
    stream = io.BytesIO()
    with tarfile.open(
        fileobj=stream, mode="w", format=tarfile.PAX_FORMAT, pax_headers={"comment": revision}
    ) as archive:
        for name, content in original.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    data = stream.getvalue()
    (root / "source.tar").write_bytes(data)
    monkeypatch.setattr(audit, "git_archive", lambda repo, rev: data)
    resulting, patch = audit.overlay(original, revision)
    patch_path = root / "operation-observers.json"
    patch_path.write_bytes(audit.patch_bytes(patch))
    receipt = {
        "revision": revision,
        "path": str(patch_path),
        "sha256": audit.route.digest(patch_path),
        "changed_files": {
            name: {"before": audit.sha(original[name]), "after": audit.sha(resulting[name])}
            for name in original
            if resulting[name] != original[name]
        },
    }
    cargo = root / "external.Cargo.lock"
    cargo.write_bytes(b"version = 4\n")
    resulting["Cargo.lock"] = cargo.read_bytes()
    for name, content in resulting.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (root / "build.log").write_text("release build\n")
    lock = root / "dependencies.lock"
    lock.write_text("hash-locked dependencies\n")
    extension = package / "_rust.so"
    extension.write_bytes(b"diagnostic native extension")
    environment = dict(
        audit.route.clean_env(), CARGO_TARGET_DIR=str(root / "cargo-target"), RUSTFLAGS=""
    )
    environment["PATH"] = str(Path.home() / ".cargo/bin") + audit.os.pathsep + audit.os.defpath
    manifest = {
        "purpose": audit.PURPOSE,
        "sha": revision,
        "repo": str(root / "repo"),
        "source_root": str(source),
        "archive_sha256": audit.sha(data),
        "git_sources": audit.hashes(original),
        "sources": audit.hashes(resulting),
        "observer_overlay": receipt,
        "cargo_lock_sha256": audit.route.digest(cargo),
        "snapshot_overlays": {
            "operation-observers": receipt,
            "Cargo.lock": {
                "origin": "explicit --cargo-lock input",
                "input_path": str(cargo),
                "sha256": audit.route.digest(cargo),
                "git_sha256": None,
            },
        },
        "release": True,
        "build_command": ["maturin", "build", "--release", "--locked"],
        "build_environment": environment,
        "build_log_sha256": audit.route.digest(root / "build.log"),
        "lock_sha256": audit.route.digest(lock),
    }
    descriptor = {
        "purpose": audit.PURPOSE,
        "manifest": manifest,
        "package": str(package),
        "extension": str(extension),
        "extension_sha256": audit.route.digest(extension),
        "wheel": str(root / "audit.whl"),
        "lock": str(lock),
    }
    seal(audit, descriptor)
    return descriptor
