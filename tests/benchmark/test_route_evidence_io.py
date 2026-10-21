from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from . import test_route_evidence

driver = test_route_evidence.driver


def test_save_ignores_preplanted_temporary_symlink(driver, tmp_path):
    destination = tmp_path / "report.json"
    victim = tmp_path / "victim.txt"
    victim.write_text("do not overwrite\n")
    planted = destination.with_suffix(destination.suffix + ".tmp")
    planted.symlink_to(victim)

    driver.save(destination, {"status": "incomplete"})

    assert victim.read_text() == "do not overwrite\n"
    assert planted.is_symlink()
    assert planted.resolve() == victim
    assert json.loads(destination.read_text()) == {"status": "incomplete"}
    assert set(tmp_path.iterdir()) == {destination, victim, planted}


def test_save_closes_same_directory_exclusive_file_before_atomic_replace(
    driver, tmp_path, monkeypatch
):
    destination = tmp_path / "nested" / "report.json"
    driver.save(destination, {"previous": True})
    previous = destination.read_bytes()
    replace = driver.os.replace
    publications = []

    def inspect_replace(source, target):
        temporary = Path(source)
        assert target == destination
        assert temporary.parent == destination.parent
        assert temporary != destination.with_suffix(".json.tmp")
        assert stat.S_IMODE(temporary.stat().st_mode) == 0o600
        assert destination.read_bytes() == previous
        assert temporary.read_text() == '{\n  "status": "adverse"\n}\n'
        replace(source, target)
        publications.append(temporary)

    monkeypatch.setattr(driver.os, "replace", inspect_replace)

    driver.save(destination, {"status": "adverse"})

    assert len(publications) == 1
    assert not publications[0].exists()
    assert destination.read_text() == '{\n  "status": "adverse"\n}\n'
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("stage", ["write", "replace"])
def test_save_io_failure_preserves_destination_and_cleans_temporary(
    driver, tmp_path, monkeypatch, existing, stage
):
    destination = tmp_path / "report.json"
    if existing:
        destination.write_bytes(b"previous evidence\n")
    error = OSError(f"injected {stage} failure")
    temporary_paths = []
    temporary_handles = []
    named_temporary_file = driver.tempfile.NamedTemporaryFile

    def temporary_file(*args, **kwargs):
        handle = named_temporary_file(*args, **kwargs)
        temporary_paths.append(Path(handle.name))
        temporary_handles.append(handle)
        if stage == "write":
            write = handle.write

            def fail_write(content):
                write(content[:5])
                raise error

            monkeypatch.setattr(handle, "write", fail_write)
        return handle

    def fail_replace(source, target):
        assert temporary_handles[0].closed
        assert Path(source).read_text() == '{\n  "new": true\n}\n'
        assert target == destination
        raise error

    monkeypatch.setattr(driver.tempfile, "NamedTemporaryFile", temporary_file)
    if stage == "replace":
        monkeypatch.setattr(driver.os, "replace", fail_replace)

    with pytest.raises(OSError) as caught:
        driver.save(destination, {"new": True})

    assert caught.value is error
    assert len(temporary_paths) == 1
    assert temporary_handles[0].closed
    assert not temporary_paths[0].exists()
    if existing:
        assert destination.read_bytes() == b"previous evidence\n"
        assert list(tmp_path.iterdir()) == [destination]
    else:
        assert not destination.exists()
        assert list(tmp_path.iterdir()) == []


def test_save_serialization_failure_preserves_destination_and_cleans_temporary(driver, tmp_path):
    destination = tmp_path / "report.json"
    destination.write_bytes(b"previous evidence\n")

    with pytest.raises(ValueError, match="Out of range float"):
        driver.save(destination, {"invalid": float("nan")})

    assert destination.read_bytes() == b"previous evidence\n"
    assert list(tmp_path.iterdir()) == [destination]
