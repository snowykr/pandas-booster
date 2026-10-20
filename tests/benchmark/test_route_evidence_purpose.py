from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import test_route_evidence

driver = test_route_evidence.driver
runtime = test_route_evidence.runtime


@pytest.mark.parametrize("role", ["baseline", "candidate"])
@pytest.mark.parametrize("mode", ["timing", "rss", "scaling", "evidence"])
@pytest.mark.parametrize("mutation", ["descriptor", "manifest", "embedded", "unknown-overlay"])
def test_run_preflights_both_roles_before_launch(
    driver, runtime, tmp_path, monkeypatch, role, mode, mutation
):
    runtimes = {name: copy.deepcopy(runtime) for name in ("baseline", "candidate")}
    if mutation == "descriptor":
        runtimes[role]["purpose"] = "audit-only"
    elif mutation == "manifest":
        runtimes[role]["manifest"]["purpose"] = "audit-only"
    elif mutation == "embedded":
        driver.save(
            Path(runtime["package"]) / "_route_evidence_manifest.json",
            dict(runtime["manifest"], purpose="audit-only"),
        )
    else:
        # Matching receipt cannot authorize arbitrary source overlays.
        for descriptor in runtimes.values():
            descriptor["manifest"]["snapshot_overlays"]["src/observer.rs"] = {}
        driver.save(
            Path(runtime["package"]) / "_route_evidence_manifest.json",
            runtimes[role]["manifest"],
        )
    paths = {}
    for name, descriptor in runtimes.items():
        paths[name] = str(tmp_path / (name + ".json"))
        driver.save(Path(paths[name]), descriptor)
    launches = []
    monkeypatch.setattr(driver, "command", lambda *a, **k: launches.append(a))
    args = SimpleNamespace(
        **paths,
        output=str(tmp_path / "result.json"),
        host="n100",
        threads=[1],
        quick=True,
        cases=None,
        memory_only=mode == "rss",
        refusal_rows=5000000,
    )
    with pytest.raises(ValueError, match="purpose|non-Cargo"):
        driver.run(args)
    assert launches == []


@pytest.mark.parametrize("kind", ["cold", "warm", "rss", "scaling", "evidence"])
@pytest.mark.parametrize("mutation", ["audit", "request", "unknown-overlay", "embedded"])
def test_direct_worker_cannot_bypass_purpose(
    driver, runtime, tmp_path, kind, mutation, monkeypatch
):
    request = {"purpose": "acceptance", "runtime": runtime, "kind": kind}
    if mutation == "audit":
        runtime["purpose"] = "audit-only"
    elif mutation == "request":
        request["purpose"] = "audit-only"
    elif mutation == "unknown-overlay":
        runtime["manifest"]["snapshot_overlays"]["src/observer.rs"] = {}
        driver.save(Path(runtime["package"]) / "_route_evidence_manifest.json", runtime["manifest"])
    else:
        embedded = dict(runtime["manifest"], purpose="audit-only")
        driver.save(Path(runtime["package"]) / "_route_evidence_manifest.json", embedded)
    path = tmp_path / "request.json"
    driver.save(path, request)
    launches = []
    monkeypatch.setattr(driver, "validate_runtime", lambda *a: launches.append(a))
    with pytest.raises(ValueError, match="purpose|non-Cargo"):
        driver._worker(path, {})
    assert launches == []


def test_acceptance_preflight_valid_control(driver, runtime, tmp_path, monkeypatch):
    class PreflightCompleteError(Exception):
        pass

    paths = {}
    for role in ("baseline", "candidate"):
        descriptor = copy.deepcopy(runtime)
        package = tmp_path / role / "package"
        package.mkdir(parents=True)
        driver.save(package / "_route_evidence_manifest.json", descriptor["manifest"])
        descriptor["package"] = str(package)
        paths[role] = str(tmp_path / (role + ".json"))
        driver.save(Path(paths[role]), descriptor)
    checked = []
    validate = driver.validate_acceptance

    def check(descriptor):
        validate(descriptor)
        checked.append(descriptor["package"])

    def stop_after_preflight(runtimes):
        assert set(runtimes) == {"baseline", "candidate"}
        raise PreflightCompleteError

    monkeypatch.setattr(driver, "validate_acceptance", check)
    monkeypatch.setattr(driver, "validate_comparison", stop_after_preflight)
    with pytest.raises(PreflightCompleteError):
        driver.run(SimpleNamespace(**paths, output=str(tmp_path / "control.json")))
    assert len(set(checked)) == 2


def test_legacy_runtime_requires_rebuild(driver, runtime):
    del runtime["purpose"]
    with pytest.raises(ValueError, match="rebuild"):
        driver.validate_acceptance(runtime)
