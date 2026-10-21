"""Independent operation observer; NEVER acceptance timing/RSS evidence (Python 3.9+).

Fresh diagnostic snapshots only; never reuse or relabel acceptance runtimes::

    python benchmarks/operation_audit.py prepare --repo . --sha 291bc4a \
        --lock /absolute/requirements.lock --cargo-lock /absolute/Cargo.lock \
        --root /tmp/pr20-audit-base
    python benchmarks/operation_audit.py prepare --repo . --sha REV \
        --lock /absolute/requirements.lock --cargo-lock /absolute/Cargo.lock \
        --root /tmp/pr20-audit-new
    python benchmarks/operation_audit.py run --baseline /tmp/pr20-audit-base/runtime.json \
        --candidate /tmp/pr20-audit-new/runtime.json --output /tmp/pr20-audit.json

Observers emit stderr events immediately at physical operations, not route labels.
Zero counts are admitted only after immutable source/observer validation and a
successful exact-parity call; positive baseline and unsorted controls are required.
Native u64 uses the actual exported u64 kernel on bounded inputs, not >2**32 rows.
Unsorted allocator overhead/bound is unknown. No elapsed time or RSS is collected.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

# -I does not put the script's directory on sys.path.
_SPEC = importlib.util.spec_from_file_location(
    "operation_audit_route", Path(__file__).with_name("route_evidence.py")
)
assert _SPEC is not None and _SPEC.loader is not None
route = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(route)
PURPOSE = "audit-only"
PREFIX = "PB_OPERATION_AUDIT:"
# Each anchor's exact multiplicity is part of the diagnostic contract.
EDITS = [
    ("src/radix_groupby/engine.rs", "    let mut hashes = vec![0u64; n_rows];", 1, "hash_vectors"),
    (
        "src/radix_groupby/engine.rs",
        "            let mut local_map: AHashMap<Ops::Key, A> = AHashMap::new();",
        1,
        "hash_maps",
    ),
    (
        "src/radix_groupby/firstseen_u32.rs",
        "    let mut hashes = vec![0u64; n_rows];",
        1,
        "hash_vectors",
    ),
    (
        "src/radix_groupby/firstseen_u64.rs",
        "    let mut hashes = vec![0u64; n_rows];",
        1,
        "hash_vectors",
    ),
    (
        "src/radix_groupby/firstseen_u32.rs",
        "            let mut gid_map: AHashMap<Ops::Key, u32> = AHashMap::new();",
        1,
        "hash_maps",
    ),
    (
        "src/radix_groupby/firstseen_u64.rs",
        "            let mut gid_map: AHashMap<Ops::Key, u32> = AHashMap::new();",
        1,
        "hash_maps",
    ),
    (
        "src/radix_groupby/order.rs",
        "    let perm = if n_groups < RADIX_SORT_THRESHOLD {",
        2,
        "output_sort",
    ),
    (
        "src/radix_groupby/order.rs",
        "    let perm = radix_sort_perm_by_u32(first_seen);",
        1,
        "firstseen_u32",
    ),
    (
        "src/radix_groupby/order.rs",
        "    let perm = radix_sort_perm_by_u64(first_seen);",
        1,
        "firstseen_u64",
    ),
]
CANDIDATE_EDIT = (
    "src/radix_groupby/sort_first.rs",
    "    let n = permutation.len();",
    1,
    "sort_first",
)
EVENTS = sorted({edit[3] for edit in EDITS} | {"sort_first"})


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_sources(data: bytes, revision: str) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        if archive.pax_headers.get("comment") != revision:
            raise ValueError("Archive revision mismatch")
        sources = {}
        for member in archive.getmembers():
            path = Path(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe git archive member")
            if member.isfile():
                stream = archive.extractfile(member)
                assert stream is not None
                sources[member.name] = stream.read()
        return sources


def hashes(sources: dict[str, bytes]) -> dict[str, str]:
    return {name: sha(content) for name, content in sorted(sources.items())}


def overlay(
    original: dict[str, bytes], revision: str
) -> tuple[dict[str, bytes], list[dict[str, Any]]]:
    """Only the fixed observer edits are permitted; unmatched edits fail closed."""
    sources = dict(original)
    patch = []
    edits = EDITS if revision == route.BASELINE else EDITS + [CANDIDATE_EDIT]
    for name, anchor, count, event in edits:
        before = sources[name].decode()
        if before.count(anchor) != count:
            raise ValueError("Unmatched diagnostic operation: " + name + ": " + anchor)
        indentation = anchor[: len(anchor) - len(anchor.lstrip())]
        replacement = indentation + 'eprintln!("' + PREFIX + event + '");\n' + anchor
        sources[name] = before.replace(anchor, replacement).encode()
        patch.append(
            {"path": name, "before": anchor, "after": replacement, "count": count, "event": event}
        )
    return sources, patch


def patch_bytes(patch: list[dict[str, Any]]) -> bytes:
    return (json.dumps(patch, sort_keys=True, indent=2) + "\n").encode()


def git_archive(repo: Path, revision: str) -> bytes:
    return subprocess.run(
        ["git", "archive", "--format=tar", revision],
        cwd=str(repo),
        env=route.clean_env(),
        check=True,
        capture_output=True,
    ).stdout


def validate_provenance(descriptor: dict[str, Any]) -> None:
    """Reconstruct declared edits from original git bytes, never trust hash lists alone."""
    manifest = descriptor["manifest"]
    if descriptor.get("purpose") != PURPOSE or manifest.get("purpose") != PURPOSE:
        raise ValueError("Audit purpose must be audit-only")
    package = Path(descriptor["package"])
    if json.loads((package / "_route_evidence_manifest.json").read_text()) != manifest:
        raise ValueError("Embedded audit provenance mismatch")
    revision = manifest["sha"]
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Audit revision must be a full immutable git SHA")
    root = Path(manifest["source_root"]).parent
    original_data = (root / "source.tar").read_bytes()
    if sha(original_data) != manifest["archive_sha256"]:
        raise ValueError("Original source archive mismatch")
    # Independently bind original bytes to the immutable git object, not a mutable label.
    if git_archive(Path(manifest["repo"]), revision) != original_data:
        raise ValueError("Git revision/archive mismatch")
    original = archive_sources(original_data, revision)
    if hashes(original) != manifest["git_sources"]:
        raise ValueError("Original source hashes mismatch")
    resulting, patch = overlay(original, revision)
    receipt = manifest["observer_overlay"]
    expected_patch = patch_bytes(patch)
    if (
        receipt["revision"] != revision
        or receipt["sha256"] != sha(expected_patch)
        or Path(receipt["path"]).read_bytes() != expected_patch
    ):
        raise ValueError("Observer patch/revision mismatch")
    changed = {
        name: {"before": sha(original[name]), "after": sha(resulting[name])}
        for name in original
        if original[name] != resulting[name]
    }
    if receipt["changed_files"] != changed:
        raise ValueError("Observer changed-file manifest mismatch")
    if set(manifest["snapshot_overlays"]) != {"Cargo.lock", "operation-observers"}:
        raise ValueError("Unknown audit snapshot overlays")
    if manifest["snapshot_overlays"]["operation-observers"] != receipt:
        raise ValueError("Undeclared observer overlay")
    cargo = manifest["snapshot_overlays"]["Cargo.lock"]
    lock = Path(cargo["input_path"])
    if (
        cargo["origin"] != "explicit --cargo-lock input"
        or cargo["git_sha256"] != manifest["git_sources"].get("Cargo.lock")
        or route.digest(lock) != cargo["sha256"]
        or cargo["sha256"] != manifest["cargo_lock_sha256"]
    ):
        raise ValueError("Audit Cargo lock mismatch")
    resulting["Cargo.lock"] = lock.read_bytes()
    if hashes(resulting) != manifest["sources"]:
        raise ValueError("Undeclared audit source overlay")
    if route.files_manifest(Path(manifest["source_root"])) != manifest["sources"]:
        raise ValueError("Audit resulting sources mismatch")
    if route.digest(root / "build.log") != manifest["build_log_sha256"]:
        raise ValueError("Audit build log mismatch")
    if route.digest(Path(descriptor["lock"])) != manifest["lock_sha256"]:
        raise ValueError("Audit dependency lock mismatch")
    if route.digest(Path(descriptor["wheel"])) != descriptor["wheel_sha256"]:
        raise ValueError("Audit wheel hash mismatch")
    with zipfile.ZipFile(descriptor["wheel"]) as wheel:
        if json.loads(wheel.read(route.MANIFEST)) != manifest:
            raise ValueError("Audit wheel embedded provenance mismatch")
        extension_member = "pandas_booster/" + Path(descriptor["extension"]).name
        if sha(wheel.read(extension_member)) != descriptor["extension_sha256"]:
            raise ValueError("Audit wheel/extension binding mismatch")
    installed = {
        str(p.relative_to(package)): route.digest(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    if installed != descriptor["installed_files"]:
        raise ValueError("Audit installed extension/package mismatch")
    if route.digest(Path(descriptor["extension"])) != descriptor["extension_sha256"]:
        raise ValueError("Audit extension hash mismatch")
    environment = dict(route.clean_env(), CARGO_TARGET_DIR=str(root / "cargo-target"), RUSTFLAGS="")
    environment["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + os.defpath
    if (
        manifest["build_environment"] != environment
        or not manifest["release"]
        or "--release" not in manifest["build_command"]
        or "--locked" not in manifest["build_command"]
    ):
        raise ValueError("Audit build provenance mismatch")


def prepare(args: argparse.Namespace) -> None:
    root, repo = Path(args.root).resolve(), Path(args.repo).resolve()
    if root.exists():
        raise ValueError("Audit root must not exist; never overlay acceptance snapshots")
    lock, cargo = Path(args.lock).resolve(), Path(args.cargo_lock).resolve()
    if not lock.is_file() or not cargo.is_file():
        raise ValueError("Explicit dependency and Cargo locks required")
    revision = route.command(
        ["git", "rev-parse", args.sha + "^{commit}"], repo, route.clean_env()
    ).strip()
    data = git_archive(repo, revision)
    original = archive_sources(data, revision)
    sources, patch = overlay(original, revision)
    root.mkdir(parents=True)
    (root / "source.tar").write_bytes(data)
    patch_path = root / "operation-observers.json"
    patch_path.write_bytes(patch_bytes(patch))
    receipt = {
        "revision": revision,
        "path": str(patch_path),
        "sha256": route.digest(patch_path),
        "changed_files": {
            name: {"before": sha(original[name]), "after": sha(sources[name])}
            for name in original
            if sources[name] != original[name]
        },
    }
    source = root / "source"
    sources["Cargo.lock"] = cargo.read_bytes()
    for name, content in sources.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    shutil.copyfile(lock, root / "dependencies.lock")
    env = route.clean_env()
    for name in ("build", "runtime"):
        route.command([sys.executable, "-m", "venv", str(root / name)], root, env)
        route.command(
            [
                str(root / name / "bin/python"),
                "-I",
                "-m",
                "pip",
                "install",
                "--require-hashes",
                "-r",
                str(root / "dependencies.lock"),
            ],
            root,
            env,
        )
    build_python = str(root / "build/bin/python")
    build_env = dict(env, CARGO_TARGET_DIR=str(root / "cargo-target"), RUSTFLAGS="")
    build_env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + os.defpath
    build_command = [
        build_python,
        "-I",
        "-m",
        "maturin",
        "build",
        "--release",
        "--locked",
        "--out",
        str(root / "wheels"),
        "--interpreter",
        build_python,
    ]
    route.command(build_command, source, build_env, transcript=root / "build.log")
    if route.files_manifest(source) != hashes(sources):
        raise ValueError("Audit build changed source snapshot")
    wheels = list((root / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected one diagnostic wheel")
    manifest = {
        "purpose": PURPOSE,
        "sha": revision,
        "repo": str(repo),
        "source_root": str(source),
        "archive_sha256": sha(data),
        "git_sources": hashes(original),
        "sources": hashes(sources),
        "observer_overlay": receipt,
        "cargo_lock_sha256": route.digest(cargo),
        "snapshot_overlays": {
            "operation-observers": receipt,
            "Cargo.lock": {
                "origin": "explicit --cargo-lock input",
                "input_path": str(cargo),
                "sha256": route.digest(cargo),
                "git_sha256": hashes(original).get("Cargo.lock"),
            },
        },
        "build_command": build_command,
        "build_environment": build_env,
        "release": True,
        "build_log_sha256": route.digest(root / "build.log"),
        "lock_sha256": route.digest(root / "dependencies.lock"),
        "rustc": route.command(["rustc", "-Vv"], root, build_env),
        "cargo": route.command(["cargo", "-V"], root, build_env),
    }
    route.stamp_wheel(wheels[0], manifest)
    python = str(root / "runtime/bin/python")
    route.command([python, "-I", "-m", "pip", "install", "--no-deps", str(wheels[0])], root, env)
    inventory = json.loads(route.command([python, "-I", "-c", route.INVENTORY], root, env))
    descriptor = dict(
        inventory,
        purpose=PURPOSE,
        manifest=manifest,
        python=python,
        wheel=str(wheels[0]),
        wheel_sha256=route.digest(wheels[0]),
        extension_sha256=route.digest(Path(inventory["extension"])),
        lock=str(root / "dependencies.lock"),
    )
    validate_provenance(descriptor)
    route.save(root / "runtime.json", descriptor)
    for target in list(source.rglob("*")) + [source, patch_path, root / "source.tar"]:
        target.chmod(0o555 if target.is_dir() else 0o444)
    print(root / "runtime.json")


def audit_worker(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("purpose") != PURPOSE or request.get("kind") != "operations":
        raise ValueError("Audit worker only accepts audit-only operations requests")
    runtime = request["runtime"]
    validate_provenance(runtime)
    if (
        dict(os.environ) != route.clean_env(request["threads"])
        or not sys.flags.isolated
        or Path(sys.executable).absolute() != Path(runtime["python"]).absolute()
        or route.python_identity() != runtime["python_identity"]
    ):
        raise ValueError("Audit worker interpreter/environment mismatch")
    import numpy as np
    import pandas as pd
    import pandas_booster
    import pandas_booster._rust as rust

    package = Path(pandas_booster.__file__).resolve().parent
    if (
        str(package) != runtime["package"]
        or str(Path(rust.__file__).resolve()) != runtime["extension"]
    ):
        raise ValueError("Audit imported wrong package/extension")
    # Do not resolve the interpreter symlink when finding its virtualenv root.
    runtime_root = Path(runtime["python"]).absolute().parent.parent
    if (
        runtime_root not in package.parents
        or runtime_root not in Path(rust.__file__).resolve().parents
    ):
        raise ValueError("Audit imported outside installed runtime")
    source = Path(runtime["manifest"]["source_root"]).resolve()
    if Path.cwd() == source or source in Path.cwd().parents:
        raise ValueError("Audit worker must run outside source snapshots")
    if rust.get_thread_count() != request["threads"]:
        raise ValueError("Audit Rayon thread mismatch")
    dependencies = sorted(
        [d.metadata["Name"].lower(), d.version] for d in route.importlib.metadata.distributions()
    )
    if dependencies != runtime["dependencies"]:
        raise ValueError("Audit dependencies mismatch")
    n = request["rows"]
    # Match the binding primary key generator, while additionally auditing NaNs.
    rng = np.random.RandomState(42)
    keys = ["k0", "k1", "k2"]
    arrays = {key: rng.randint(0, 500, n, dtype=np.int64) for key in keys}
    if request["dtype"] == "float64":
        values = rng.random_sample(n) * 1000
        values[::97] = np.nan
    else:
        values = rng.randint(0, 10000, n, dtype=np.int64)
    arrays["value"] = values
    input_hashes = {key: sha(array.tobytes()) for key, array in arrays.items()}
    frame = pd.DataFrame(arrays, copy=False)
    agg, sort = request["agg"], request["sort"]
    expected = frame.groupby(keys, sort=sort)["value"].agg(agg)
    if request["native_u64"]:
        suffix = "f64" if request["dtype"] == "float64" else "i64"
        operation = getattr(rust, "groupby_multi_" + agg + "_" + suffix + "_firstseen_u64")
        columns, values = operation([frame[k].to_numpy() for k in keys], frame["value"].to_numpy())
        actual = pd.Series(
            values, index=pd.MultiIndex.from_arrays(columns, names=keys), name="value"
        )
    else:
        actual = frame.booster.groupby(keys, "value", agg, sort=sort)
    route.exact_equal(actual, expected)
    return {
        "purpose": PURPOSE,
        "kind": "operations",
        "parity": "exact",
        "groups": len(actual),
        "dtype": request["dtype"],
        "rows": n,
        "input_sha256": input_hashes,
        "key_generator": "RandomState(42); three int64 keys, cardinality500 each",
        "float_nan_stride": 97 if request["dtype"] == "float64" else None,
        "appearance_order_checked": not sort,
        "allocator_bound": "unknown",
        "native_u64_scope": "bounded actual u64 kernel; not >2**32 rows"
        if request["native_u64"]
        else None,
    }


def check_events(events: dict[str, int], baseline: bool, sort: bool, native_u64: bool) -> None:
    if sort:
        if baseline:
            if not all(events[name] > 0 for name in ("hash_vectors", "hash_maps", "output_sort")):
                raise ValueError("Baseline sorted-count positive physical control failed")
        elif events["sort_first"] == 0 or any(
            count for name, count in events.items() if name != "sort_first"
        ):
            raise ValueError("Candidate sorted count performed hash/output/first-seen operations")
    elif (
        events["hash_vectors"] == 0
        or events["hash_maps"] == 0
        or events["output_sort"] != 0
        or events["sort_first"] != 0
        or events["firstseen_u64" if native_u64 else "firstseen_u32"] == 0
    ):
        raise ValueError("Unsorted physical first-seen/hash control failed")


def run(args: argparse.Namespace) -> None:
    output = Path(args.output).resolve()
    artifact = output.with_suffix(".artifacts")
    if output.exists() or artifact.exists():
        raise ValueError("Audit evidence output must be fresh")
    runtimes = {
        role: json.loads(Path(getattr(args, role)).read_text())
        for role in ("baseline", "candidate")
    }
    for runtime in runtimes.values():
        validate_provenance(runtime)
        source = Path(runtime["manifest"]["source_root"]).resolve()
        if artifact == source or source in artifact.parents:
            raise ValueError("Audit output must be outside source snapshots")
    route.validate_comparison(runtimes)
    if runtimes["baseline"]["manifest"]["sha"] != route.BASELINE:
        raise ValueError("Audit baseline must be 291bc4a")
    for key in ("lock_sha256", "cargo_lock_sha256"):
        if runtimes["baseline"]["manifest"][key] != runtimes["candidate"]["manifest"][key]:
            raise ValueError("Audit paired locks differ")
    artifact.mkdir(parents=True)
    # Copy both tools so direct workers under -I have a stable, paired driver.
    for name in ("operation_audit.py", "route_evidence.py"):
        shutil.copyfile(Path(__file__).with_name(name), artifact / name)
    report = {
        "purpose": PURPOSE,
        "acceptance_usable": False,
        "allocator_bound": "unknown",
        "runtimes": runtimes,
        "observations": [],
        "status": "INCOMPLETE",
    }
    route.save(output, report)
    failures = []
    for role, runtime in runtimes.items():
        cases = [
            (agg, sort, native, dtype)
            for dtype in ("int64", "float64")
            for agg, sort, native in [
                ("count", True, False),
                ("count", False, False),
                ("max", False, False),
                ("count", False, True),
                ("max", False, True),
            ]
        ]
        for agg, sort, native, dtype in cases:
            name = (
                role
                + "-"
                + agg
                + ("-sorted" if sort else "-unsorted")
                + ("-u64" if native else "-public")
                + "-"
                + dtype
            )
            request = {
                "purpose": PURPOSE,
                "kind": "operations",
                "runtime": runtime,
                "threads": args.threads,
                "rows": args.rows,
                "agg": agg,
                "sort": sort,
                "native_u64": native,
                "dtype": dtype,
            }
            request_path, response = (
                artifact / (name + "-request.json"),
                artifact / (name + "-response.json"),
            )
            route.save(request_path, request)
            process = subprocess.run(
                [
                    runtime["python"],
                    "-I",
                    str(artifact / "operation_audit.py"),
                    "worker",
                    str(request_path),
                    str(response),
                ],
                cwd=str(artifact),
                env=route.clean_env(args.threads),
                capture_output=True,
                text=True,
            )
            (artifact / (name + "-stdout.log")).write_text(process.stdout)
            (artifact / (name + "-stderr.log")).write_text(process.stderr)
            counts = Counter(
                line[len(PREFIX) :]
                for line in process.stderr.splitlines()
                if line.startswith(PREFIX)
            )
            observation = {
                "name": name,
                "request": str(request_path),
                "response": str(response),
                "returncode": process.returncode,
                "events": {name: counts[name] for name in EVENTS},
            }
            try:
                if process.returncode:
                    raise ValueError("Audit worker failed; inspect retained stderr")
                observation["result"] = json.loads(response.read_text())
                if (
                    observation["result"].get("purpose") != PURPOSE
                    or observation["result"].get("parity") != "exact"
                ):
                    raise ValueError("Missing successful audit parity receipt")
                if set(counts) - set(EVENTS):
                    raise ValueError("Unknown observer event")
                check_events(observation["events"], role == "baseline", sort, native)
            except (ValueError, OSError) as error:
                observation["error"] = str(error)
                failures.append(name + ": " + str(error))
            report["observations"].append(observation)
            route.save(output, report)
    report.update(status="FAILED" if failures else "AUDIT_PASS", failures=failures)
    route.save(output, report)
    print(output)
    if failures:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser(
        "prepare", help="fresh git-archive diagnostic-only build with exact observers"
    )
    for name in ("repo", "sha", "lock", "cargo-lock", "root"):
        build.add_argument("--" + name, required=True)
    collect = commands.add_parser(
        "run", help="audit sorted count and unsorted count/max; NEVER acceptance evidence"
    )
    for name in ("baseline", "candidate", "output"):
        collect.add_argument("--" + name, required=True)
    collect.add_argument("--threads", type=int, default=1)
    collect.add_argument("--rows", type=int, default=200000)
    worker = commands.add_parser(
        "worker", help="internal audit-only operations worker; no timing/RSS modes"
    )
    worker.add_argument("request", type=Path)
    worker.add_argument("response", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        route.save(args.response, audit_worker(json.loads(args.request.read_text())))
    else:
        if args.rows < 4096 or args.threads < 1:
            parser.error("Require at least 4096 rows and a positive thread count")
        run(args)


if __name__ == "__main__":
    main()
