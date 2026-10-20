"""Isolated wheel evidence driver (Python 3.9+).

Prepare each immutable revision, then run from any directory::

    python benchmarks/route_evidence.py prepare --repo . --sha REV \
        --lock /absolute/hash-locked-requirements.txt \
        --cargo-lock /absolute/Cargo.lock --root /tmp/evidence/REV
    python benchmarks/route_evidence.py run --baseline /tmp/base/runtime.json \
        --candidate /tmp/new/runtime.json --output /tmp/evidence/result.json

The dependency lock must include maturin and identical runtime dependencies, with
hashes. Supply the same explicit Cargo.lock to both revisions; it is recorded as
an external snapshot input, not attributed to the git commit. Preparation never
uses editable installs. Existing runtime.json files can
be consumed instead; every installed artifact is revalidated in every worker.
A successful command is not certification: inspect status and unmet_gates. OCI
and baseline count audit evidence cannot be replaced by local timings.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import inspect
import io
import json
import math
import os
import random
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path
from typing import Any

BASELINE = "291bc4af284119fd84e5408fd411dd6b4e0171d4"
MIB = 1024 * 1024
MANIFEST = "pandas_booster/_route_evidence_manifest.json"
LEAVES = (
    "routing_s",
    "sort_key_construction_s",
    "radix_sort_s",
    "sort_first_boundary_s",
    "sort_first_setup_s",
    "sort_first_segment_scan_s",
)
FALLBACK_LEAVES = (
    "hash_build_s",
    "partition_scatter_s",
    "partition_aggregation_s",
    "flatten_s",
    "sorted_materialization_s",
)


def digest(path: Path) -> str:
    """Return the SHA256 of a file without loading it into memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def save(path: Path, value: Any) -> None:
    """Atomically persist evidence, including adverse and incomplete results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def clean_env(threads: int = 1) -> dict[str, str]:
    """Use an explicit environment rather than inherited routing overrides."""
    return {
        "PATH": os.defpath,
        "HOME": str(Path.home()),
        "LANG": "C.UTF-8",
        "RAYON_NUM_THREADS": str(threads),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PANDAS_BOOSTER_STRICT_ABI": "1",
        "PYTHONHASHSEED": "42",
    }


def command(args: list[str], cwd: Path, env: dict[str, str], transcript: Path | None = None) -> str:
    """Run a checked command, retaining stderr on failure."""
    result = subprocess.run(args, cwd=str(cwd), env=env, text=True, capture_output=True)
    if transcript is not None:
        transcript.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{args!r}: exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result.stdout


def files_manifest(root: Path) -> dict[str, str]:
    """Hash all immutable source files."""
    return {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def stamp_wheel(wheel: Path, manifest: dict[str, Any]) -> None:
    """Embed source/build provenance and rebuild the wheel RECORD hashes."""
    with zipfile.ZipFile(wheel) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    record = next(name for name in members if name.endswith(".dist-info/RECORD"))
    members[MANIFEST] = json.dumps(manifest, sort_keys=True).encode()
    rows = []
    for name, content in sorted(members.items()):
        if name != record:
            hashed = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=")
            rows.append([name, "sha256=" + hashed.decode(), str(len(content))])
    rows.append([record, "", ""])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    members[record] = output.getvalue().encode()
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def prepare(args: argparse.Namespace) -> None:
    """Build a release wheel from a git archive and install into a fresh venv."""
    root = Path(args.root).resolve()
    if root.exists():
        raise ValueError("Preparation root must not exist (immutable, non-reusable builds)")
    repo, lock = Path(args.repo).resolve(), Path(args.lock).resolve()
    cargo_lock = Path(args.cargo_lock).resolve()
    if not cargo_lock.is_file():
        raise ValueError("An existing explicit Cargo.lock is required; never resolve implicitly")
    root.mkdir(parents=True)
    source = root / "source"
    source.mkdir()
    env = clean_env()
    sha = command(["git", "rev-parse", args.sha + "^{commit}"], repo, env).strip()
    archive_path = root / "source.tar"
    command(["git", "archive", "--format=tar", "--output=" + str(archive_path), sha], repo, env)
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            destination = (source / member.name).resolve()
            if source not in destination.parents or member.issym() or member.islnk():
                raise ValueError("Unsafe source archive member: " + member.name)
        archive.extractall(source)  # Paths and links checked above, including on Python 3.9.
    git_sources = files_manifest(source)
    shutil.copyfile(cargo_lock, source / "Cargo.lock")
    cargo_lock_sha256 = digest(source / "Cargo.lock")
    sources = files_manifest(source)
    shutil.copyfile(lock, root / "dependencies.lock")
    for name in ("build", "runtime"):
        command([sys.executable, "-m", "venv", str(root / name)], root, env)
        command(
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
    # rustup/cargo need the explicit toolchain bin, not the caller's arbitrary PATH.
    build_env["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + os.defpath
    build_args = [
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
    command(build_args, source, build_env, transcript=root / "build.log")
    if files_manifest(source) != sources:
        raise ValueError("Build modified immutable source snapshot")
    wheels = list((root / "wheels").glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Expected exactly one release wheel")
    manifest = {
        "sha": sha,
        "sources": sources,
        "source_root": str(source),
        "git_sources": git_sources,
        "cargo_lock_sha256": cargo_lock_sha256,
        "snapshot_overlays": {
            "Cargo.lock": {
                "origin": "explicit --cargo-lock input",
                "input_path": str(cargo_lock),
                "sha256": cargo_lock_sha256,
                "git_sha256": git_sources.get("Cargo.lock"),
            }
        },
        "lock_sha256": digest(root / "dependencies.lock"),
        "build_command": build_args,
        "build_environment": build_env,
        "release": True,
        "rustc": command(["rustc", "-Vv"], root, build_env),
        "cargo": command(["cargo", "-V"], root, build_env),
        "build_log_sha256": digest(root / "build.log"),
    }
    stamp_wheel(wheels[0], manifest)
    python = str(root / "runtime/bin/python")
    command([python, "-I", "-m", "pip", "install", "--no-deps", str(wheels[0])], root, env)
    inventory = command([python, "-I", "-c", INVENTORY], root, env)
    descriptor = dict(
        json.loads(inventory),
        manifest=manifest,
        wheel=str(wheels[0]),
        wheel_sha256=digest(wheels[0]),
        python=python,
        lock=str(root / "dependencies.lock"),
    )
    save(root / "runtime.json", descriptor)
    for path in source.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    source.chmod(0o555)
    print(root / "runtime.json")


def python_identity() -> dict[str, object]:
    """Fingerprint the actual interpreter, not its virtualenv symlink."""
    import hashlib
    import platform
    import sys
    from pathlib import Path

    executable = Path(sys.executable).resolve(strict=True)
    return {
        "implementation": platform.python_implementation(),
        "version": sys.version,
        "build": list(platform.python_build()),
        "executable": str(executable),
        "executable_sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }


INVENTORY = (
    "from __future__ import annotations\n"
    + inspect.getsource(python_identity)
    + """
import hashlib, importlib.metadata, json, pathlib
import pandas_booster, pandas_booster._rust
p = pathlib.Path(pandas_booster.__file__).resolve().parent
print(json.dumps({
 'python_identity': python_identity(),
 'package': str(p), 'extension': str(pathlib.Path(pandas_booster._rust.__file__).resolve()),
 'installed_files': {str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
    for f in sorted(p.rglob('*')) if f.is_file() and '__pycache__' not in f.parts},
 'dependencies': sorted((d.metadata['Name'].lower(), d.version)
    for d in importlib.metadata.distributions())}))
"""
)


def validate_runtime(descriptor: dict[str, Any], threads: int) -> dict[str, Any]:
    """Fail closed on interpreter, wheel, source, installed files, or env skew."""
    import pandas_booster
    import pandas_booster._rust as rust

    if Path(sys.executable).absolute() != Path(descriptor["python"]).absolute():
        raise ValueError("Unexpected runtime interpreter")
    if descriptor.get("python_identity") != python_identity():
        raise ValueError("Runtime Python identity missing or different")
    expected_env = clean_env(threads)
    if dict(os.environ) != expected_env:
        raise ValueError("Worker environment differs from explicit environment")
    if not sys.flags.isolated:
        raise ValueError("Worker must run under -I")
    package = Path(pandas_booster.__file__).resolve().parent
    extension = Path(rust.__file__).resolve()
    if str(package) != descriptor["package"] or str(extension) != descriptor["extension"]:
        raise ValueError("Unexpected package/extension import path")
    runtime = Path(descriptor["python"]).parent.parent.resolve()
    if runtime not in package.parents or runtime not in extension.parents:
        raise ValueError("Package/extension imported outside installed runtime")
    manifest = descriptor["manifest"]
    source_root = Path(manifest["source_root"]).resolve()
    if Path.cwd() == source_root or source_root in Path.cwd().parents:
        raise ValueError("Worker must run outside source snapshots")
    if json.loads((package / "_route_evidence_manifest.json").read_text()) != manifest:
        raise ValueError("Embedded provenance mismatch")
    if (
        not manifest["release"]
        or "--release" not in manifest["build_command"]
        or "--locked" not in manifest["build_command"]
    ):
        raise ValueError("Missing release build evidence")
    build_environment = dict(
        clean_env(), CARGO_TARGET_DIR=str(source_root.parent / "cargo-target"), RUSTFLAGS=""
    )
    build_environment["PATH"] = str(Path.home() / ".cargo/bin") + os.pathsep + os.defpath
    if manifest["build_environment"] != build_environment:
        raise ValueError("Unexpected build flags or environment")
    if digest(Path(descriptor["wheel"])) != descriptor["wheel_sha256"]:
        raise ValueError("Wheel SHA256 mismatch")
    if digest(Path(descriptor["lock"])) != manifest["lock_sha256"]:
        raise ValueError("Dependency lock mismatch")
    if digest(source_root / "Cargo.lock") != manifest["cargo_lock_sha256"]:
        raise ValueError("Cargo lock mismatch")
    overlay = manifest["snapshot_overlays"]["Cargo.lock"]
    if (
        overlay["origin"] != "explicit --cargo-lock input"
        or overlay["sha256"] != manifest["cargo_lock_sha256"]
        or overlay["git_sha256"] != manifest["git_sources"].get("Cargo.lock")
    ):
        raise ValueError("Cargo lock overlay provenance mismatch")
    expected_sources = dict(
        manifest["git_sources"], **{"Cargo.lock": manifest["cargo_lock_sha256"]}
    )
    if expected_sources != manifest["sources"]:
        raise ValueError("Snapshot differs from git sources plus explicit Cargo lock")
    if digest(Path(manifest["source_root"]).parent / "build.log") != manifest["build_log_sha256"]:
        raise ValueError("Build log mismatch")
    if files_manifest(Path(manifest["source_root"])) != manifest["sources"]:
        raise ValueError("Immutable source snapshot mismatch")
    installed = {
        str(p.relative_to(package)): digest(p)
        for p in sorted(package.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }
    if installed != descriptor["installed_files"]:
        raise ValueError("Installed package/extension hash mismatch")
    dependencies = sorted(
        [d.metadata["Name"].lower(), d.version] for d in importlib.metadata.distributions()
    )
    if dependencies != descriptor["dependencies"]:
        raise ValueError("Installed dependency inventory mismatch")
    if rust.get_thread_count() != threads:
        raise ValueError("Rayon pool does not match requested threads")
    return {
        "sha": manifest["sha"],
        "wheel_sha256": descriptor["wheel_sha256"],
        "extension_sha256": digest(extension),
        "dependencies": dependencies,
        "environment": dict(os.environ),
        "python": sys.executable,
        "python_identity": descriptor["python_identity"],
        "rustc": manifest.get("rustc"),
        "cargo": manifest.get("cargo"),
    }


def validate_comparison(runtimes: dict[str, Any]) -> None:
    """Require actual matching interpreter and recorded build toolchain identities."""
    baseline, candidate = runtimes["baseline"], runtimes["candidate"]
    required = ("implementation", "version", "build", "executable", "executable_sha256")
    for runtime in runtimes.values():
        identity = runtime.get("python_identity", {})
        if not isinstance(identity, dict) or any(not identity.get(k) for k in required):
            raise ValueError("Missing Python identity")
    if baseline["python_identity"] != candidate["python_identity"]:
        raise ValueError("Baseline/candidate Python identities differ")
    for tool in ("rustc", "cargo"):
        identities = [runtime["manifest"].get(tool) for runtime in runtimes.values()]
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("Missing build toolchain identity: " + tool)
        if baseline["manifest"][tool] != candidate["manifest"][tool]:
            raise ValueError("Baseline/candidate build toolchain differs: " + tool)


def paired_summary(pairs: list[dict[str, float]], draws: int = 10000) -> dict[str, Any]:
    """Bootstrap matched pairs, using ratio of medians and seed 42."""
    if not pairs or draws < 1:
        raise ValueError("At least one pair and bootstrap draw required")
    for pair in pairs:
        if any(not math.isfinite(pair[k]) or pair[k] <= 0 for k in ("baseline", "candidate")):
            raise ValueError("Durations must be finite and positive")
    rng = random.Random(42)
    samples = []
    for _ in range(draws):
        selected = rng.choices(pairs, k=len(pairs))
        samples.append(
            statistics.median(p["candidate"] for p in selected)
            / statistics.median(p["baseline"] for p in selected)
        )
    samples.sort()
    # Nearest-rank empirical one-sided 95th percentile (definition persisted).
    return {
        "ratio": statistics.median(p["candidate"] for p in pairs)
        / statistics.median(p["baseline"] for p in pairs),
        "u95": samples[math.ceil(0.95 * draws) - 1],
        "pairs": len(pairs),
        "bootstrap_draws": draws,
        "bootstrap_seed": 42,
        "quantile": "nearest_rank",
    }


def case_specs(quick: bool = False) -> list[dict[str, Any]]:
    """Return the binding primary/control/domain matrix."""
    primary_n, domain_n = (100003, 100003) if quick else (5000000, 1000000)
    definitions = [
        ("primary", primary_n, [500] * 3, "float64", True),
        ("low", primary_n, [10] * 3, "float64", True),
        ("ordinary", primary_n, [50, 100, 5], "float64", True),
        ("unsorted", primary_n, [500] * 3, "float64", False),
        ("packed2", domain_n, [1000000, 2], "int64", True),
        ("packed10", domain_n, [1000000] + [2] * 9, "int64", True),
        ("fused3", domain_n, [0] * 3, "float64", True),
    ]
    return [
        {
            "name": name,
            "n": n,
            "widths": widths,
            "dtype": dtype,
            "sort": sort,
            "agg": agg,
            "seed": 42,
        }
        for name, n, widths, dtype, sort in definitions
        for agg in ("max", "count")
    ]


def create_dataset(case: dict[str, Any], directory: Path) -> dict[str, Any]:
    """Persist read-only input arrays and exact group count outside all workers."""
    import numpy as np
    import pandas as pd

    directory.mkdir(parents=True)
    rng = np.random.RandomState(case["seed"])
    n = case["n"]
    arrays = {}
    for i, width in enumerate(case["widths"]):
        if case["name"].startswith(("packed", "refused")):
            # Unique, bounded joint tuples: ten keys still fit in 29 packed bits.
            array = np.arange(n, dtype=np.int64) if i == 0 else rng.randint(0, 2, n)
        elif width == 0:
            array = rng.randint(
                np.iinfo(np.int64).min, np.iinfo(np.int64).max, size=n, dtype=np.int64
            )
        else:
            array = rng.randint(0, width, size=n, dtype=np.int64)
        arrays[f"k{i}"] = array
    arrays["value"] = (
        rng.random_sample(n) * 1000
        if case["dtype"] == "float64"
        else rng.randint(0, 10000, n, dtype=np.int64)
    )
    for name, array in arrays.items():
        path = directory / (name + ".npy")
        np.save(path, array, allow_pickle=False)
        path.chmod(0o444)
    groups = len(pd.DataFrame({k: v for k, v in arrays.items() if k != "value"}).drop_duplicates())
    if case["name"] in ("primary", "unsorted") and n == 5000000 and groups != 4901309:
        raise ValueError(f"Primary seed42 group count mismatch: {groups}")
    manifest = {
        "files": files_manifest(directory),
        "groups": groups,
        "input_bytes": sum(a.nbytes for a in arrays.values()),
    }
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    save(directory / "dataset.json", manifest)
    (directory / "dataset.json").chmod(0o444)
    return manifest


def rss() -> dict[str, int]:
    """Read Linux resident and high-water bytes (not ru_maxrss platform units)."""
    values = {}
    for line in Path("/proc/self/status").read_text().splitlines():
        key, _, value = line.partition(":")
        if key in ("VmRSS", "VmHWM"):
            values[key] = int(value.split()[0]) * 1024
    if set(values) != {"VmRSS", "VmHWM"}:
        raise ValueError("Linux RSS evidence unavailable")
    return values


def exact_equal(actual: Any, expected: Any) -> None:
    """Assert exact pandas metadata/value parity, including signed zeros."""
    import numpy as np
    import pandas as pd

    pd.testing.assert_series_equal(
        actual,
        expected,
        check_exact=True,
        check_dtype=True,
        check_names=True,
        check_index_type=True,
        check_categorical=True,
    )
    if actual.dtype.kind == "f":
        zero = expected.to_numpy() == 0
        np.testing.assert_array_equal(
            np.signbit(actual.to_numpy()[zero]), np.signbit(expected.to_numpy()[zero])
        )


def profile_gaps(profile: dict[str, Any], candidate: bool, admitted: bool) -> list[str]:
    """Validate route/leaf evidence without substituting success-shaped defaults."""
    gaps = []
    expected = "sort_first" if candidate and admitted else "hash_first"
    if profile.get("route") != expected:
        gaps.append("route mismatch: expected " + expected)
    if candidate and not admitted:
        for field in ("hash_first_aggregation_count", "post_aggregation_sort_count"):
            if field not in profile or profile[field] < 1:
                gaps.append("missing actual fallback operation evidence: " + field)
    if candidate and admitted:
        for field in LEAVES + (
            "estimated_peak_bytes",
            "hash_first_aggregation_count",
            "post_aggregation_sort_count",
            "distinct_lower_bound",
            "rayon_threads",
            "route_reason",
            "rust_elapsed_s",
        ):
            if field not in profile:
                gaps.append("missing candidate diagnostic: " + field)
        if profile.get("hash_first_aggregation_count") != 0:
            gaps.append("admitted route lacks zero hash-engine call evidence")
        if profile.get("post_aggregation_sort_count") != 0:
            gaps.append("admitted route lacks zero output-sort call evidence")
        if (
            "estimated_peak_bytes" in profile
            and not 0 <= profile["estimated_peak_bytes"] <= 512 * MIB
        ):
            gaps.append("modeled route peak exceeds 512 MiB")
    return gaps


def phase_gaps(profile: dict[str, Any], outer: float) -> list[str]:
    """Require disjoint leaf accounting, never inclusive permutation timing."""
    required = LEAVES + FALLBACK_LEAVES + ("conversion_s", "rust_total_s", "rust_elapsed_s")
    missing = ["missing timing: " + key for key in required if key not in profile]
    if missing:
        return missing
    if any(not math.isfinite(profile[k]) or profile[k] < 0 for k in required):
        return ["nonfinite or negative phase timing"]
    leaves = sum(profile[k] for k in LEAVES + FALLBACK_LEAVES)
    tolerance = max(1e-6, outer * 0.01)
    gaps = []
    if abs(leaves - profile["rust_total_s"]) > tolerance:
        gaps.append("Rust total is not the sum of disjoint leaves")
    if leaves > profile["rust_elapsed_s"] + tolerance:
        gaps.append("leaf sum exceeds independent Rust elapsed")
    if leaves + profile["conversion_s"] > outer + tolerance:
        gaps.append("leaf sum plus conversion exceeds independent profile outer")
    return gaps


def failure_strings(value: Any) -> list[str]:
    """Find explicit worker failure strings, including nested profile payloads."""
    if isinstance(value, str):
        return [value] if value.lower().startswith("fail:") else []
    if isinstance(value, dict):
        return [failure for item in value.values() for failure in failure_strings(item)]
    if isinstance(value, (list, tuple)):
        return [failure for item in value for failure in failure_strings(item)]
    return []


def worker(request_path: Path, response_path: Path) -> None:
    """Execute exactly one isolated timing, parity/profile, or memory observation."""
    result: dict[str, Any] = {"status": "failed", "stage": "startup"}
    try:
        _worker(request_path, result)
        result.update(status="passed", stage="complete")
    except Exception as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        save(response_path, result)


def _worker(request_path: Path, result: dict[str, Any]) -> None:
    import numpy as np
    import pandas as pd
    import pandas_booster  # noqa: F401 -- registers public accessor
    import pandas_booster._rust as rust

    request = json.loads(request_path.read_text())
    result["stage"] = "runtime_validation"
    identity = validate_runtime(request["runtime"], request["threads"])
    pre_input = rss()
    result["stage"] = "dataset_validation"
    directory = Path(request["dataset"])
    manifest = json.loads((directory / "dataset.json").read_text())
    if manifest != request["dataset_manifest"]:
        raise ValueError("Dataset manifest changed since collection started")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected or (directory / name).stat().st_mode & 0o222:
            raise ValueError("Dataset fingerprint mismatch")
    arrays = {
        Path(name).stem: np.load(directory / name, allow_pickle=False) for name in manifest["files"]
    }
    for array in arrays.values():
        array.setflags(write=False)
    frame = pd.DataFrame(arrays, copy=False)
    keys = [f"k{i}" for i in range(len(request["case"]["widths"]))]
    case = request["case"]
    result.update(
        identity=identity,
        fingerprint=manifest["fingerprint"],
        groups=manifest["groups"],
        pre_input=pre_input,
    )

    def public_call() -> Any:
        return frame.booster.groupby(keys, "value", case["agg"], sort=case["sort"])

    if request["kind"] in ("cold", "warm", "rss", "scaling"):
        result["stage"] = "ordinary_call"
        if request["kind"] in ("warm", "scaling"):
            for _ in range(2):
                warmup = public_call()
                del warmup
        cpu_start, start = time.process_time(), time.perf_counter()
        output = public_call()
        result.update(
            elapsed_s=time.perf_counter() - start,
            cpu_s=time.process_time() - cpu_start,
            rss=rss(),
            output_groups=len(output),
        )
        result["cpu_to_wall_ratio"] = result["cpu_s"] / result["elapsed_s"]
        result["thread_saturation"] = result["cpu_to_wall_ratio"] / request["threads"]
        result["stage"] = "ordinary_output_validation"
        if len(output) != manifest["groups"]:
            raise ValueError("Ordinary-call output group count mismatch")
        # No pandas reference exists in either ordinary timing or RSS workers.
    else:
        result["stage"] = "reference_validation"
        reference = getattr(frame.groupby(keys, sort=case["sort"])["value"], case["agg"])()
        if len(reference) != manifest["groups"]:
            raise ValueError("Exact group count differs from dataset manifest")
        public_start = time.perf_counter()
        public_cpu_start = time.process_time()
        result["stage"] = "public_call"
        public_output = public_call()
        result["public_pipeline_s"] = time.perf_counter() - public_start
        result.update(public_cpu_s=time.process_time() - public_cpu_start, rss=rss())
        result["stage"] = "public_parity"
        exact_equal(public_output, reference)
        result["parity"] = True
        suffix = "i64" if case["dtype"] == "int64" else "f64"
        export = f"profile_groupby_multi_{case['agg']}_{suffix}_sorted"
        gaps = []
        if not case["sort"]:
            gaps.append(
                "unsorted route requires independent operation audit; no sorted substitution"
            )
        elif not hasattr(rust, export):
            gaps.append("missing profile export: " + export + "; explicit source/audit gap")
        else:
            pipeline_start = time.perf_counter()
            preparation_start = time.perf_counter()
            profile_keys = [np.ascontiguousarray(arrays[k], dtype=np.int64) for k in keys]
            profile_values = np.ascontiguousarray(arrays["value"])
            preparation_s = time.perf_counter() - preparation_start
            start = time.perf_counter()
            cpu_start = time.process_time()
            result["stage"] = "profile_call"
            out_keys, out_values, profile = getattr(rust, export)(profile_keys, profile_values)
            outer = time.perf_counter() - start
            result.update(
                profile=profile,
                profile_outer_s=outer,
                profile_cpu_s=time.process_time() - cpu_start,
                rss=rss(),
                prepare_inputs_s=preparation_s,
            )
            result["stage"] = "profile_normalization"
            normalize_start = time.perf_counter()
            columns = list(out_keys)
            normalized_values = np.asarray(out_values)
            normalize_s = time.perf_counter() - normalize_start
            index_start = time.perf_counter()
            index = pd.MultiIndex.from_arrays(columns, names=keys)
            profiled_output = pd.Series(normalized_values, index=index, name="value")
            index_s = time.perf_counter() - index_start
            result.update(
                prepare_inputs_s=preparation_s,
                python_normalize_s=normalize_s,
                pandas_index_construction_s=index_s,
                profile_pipeline_s=time.perf_counter() - pipeline_start,
            )
            result["stage"] = "profile_parity"
            exact_equal(profiled_output, reference)
            result["stage"] = "profile_validation"
            admitted = case["name"] in ("primary", "packed2", "packed10", "fused3")
            gaps.extend(profile_gaps(profile, request["candidate"], admitted))
            if request["candidate"]:
                if profile.get("rayon_threads") != request["threads"]:
                    gaps.append("Profile thread count mismatch")
                if admitted:
                    lower = profile.get("distinct_lower_bound")
                    if lower is None or not 0.90 * case["n"] <= lower <= manifest["groups"]:
                        gaps.append(
                            "Admission lower bound missing, too low, or exceeds exact groups"
                        )
                    strategy = "fused_multi_key_radix" if case["name"] == "fused3" else "packed_u64"
                    if profile.get("selected_sort_strategy") != strategy:
                        gaps.append("Enabled-domain physical strategy mismatch: " + strategy)
                gaps.extend(phase_gaps(profile, outer))
                if all(k in profile for k in LEAVES + FALLBACK_LEAVES + ("conversion_s",)):
                    result["profile_residual_s"] = (
                        outer
                        - sum(float(profile[k]) for k in LEAVES + FALLBACK_LEAVES)
                        - float(profile["conversion_s"])
                    )
            gaps.extend(failure_strings(profile))
        result["unmet_gates"] = gaps


def memory_gaps(
    candidate: dict[str, Any],
    baseline_median: float,
    input_bytes: int,
    modeled_peak: int,
    threads: int,
    admitted: bool = True,
) -> list[str]:
    """Apply the per-worker binding analytical and measured memory envelopes."""
    if modeled_peak < 0 or (admitted and modeled_peak > 512 * MIB):
        return ["analytical route peak outside 512 MiB cap"]
    known = candidate["pre_input"]["VmRSS"] + input_bytes + modeled_peak
    envelope = min(known + 128 * MIB + 8 * MIB * threads, 1.10 * baseline_median + 32 * MIB)
    return ["RSS envelope exceeded"] if candidate["rss"]["VmHWM"] > envelope else []


def run(args: argparse.Namespace) -> None:
    """Collect paired processes and preserve all observations before gate evaluation."""
    output = Path(args.output).resolve()
    artifact = output.parent / (output.stem + "-artifacts")
    original_source = Path(__file__).resolve().parent.parent
    if artifact == original_source or original_source in artifact.parents:
        raise ValueError("Evidence output and workers must be outside the source checkout")
    if output.exists() or artifact.exists():
        raise ValueError("Evidence destinations must not exist; never overwrite prior runs")
    artifact.mkdir(parents=True)
    driver = artifact / "worker.py"
    shutil.copyfile(Path(__file__).resolve(), driver)
    runtimes = {
        role: json.loads(Path(getattr(args, role)).read_text())
        for role in ("baseline", "candidate")
    }
    validate_comparison(runtimes)
    for runtime in runtimes.values():
        if not Path(runtime["python"]).is_absolute():
            raise ValueError("Runtime interpreter must be an absolute path")
        source = Path(runtime["manifest"]["source_root"]).resolve()
        if artifact == source or source in artifact.parents:
            raise ValueError("Evidence workers must run outside both source snapshots")
    if runtimes["baseline"]["manifest"]["sha"] != BASELINE:
        raise ValueError("Baseline must be merged PR19 SHA " + BASELINE)
    if runtimes["baseline"]["dependencies"] != runtimes["candidate"]["dependencies"]:
        raise ValueError("Runtime dependency inventories differ")
    if (
        runtimes["baseline"]["manifest"]["lock_sha256"]
        != runtimes["candidate"]["manifest"]["lock_sha256"]
    ):
        raise ValueError("Runtime dependency locks differ")
    if (
        runtimes["baseline"]["manifest"]["cargo_lock_sha256"]
        != runtimes["candidate"]["manifest"]["cargo_lock_sha256"]
    ):
        raise ValueError("Baseline/candidate Cargo locks differ")
    if runtimes["baseline"]["python"] == runtimes["candidate"]["python"]:
        raise ValueError("Baseline and candidate must have separate runtime environments")
    if (
        runtimes["baseline"]["manifest"]["source_root"]
        == runtimes["candidate"]["manifest"]["source_root"]
    ):
        raise ValueError("Baseline and candidate must have separate immutable snapshots")
    if runtimes["baseline"]["wheel"] == runtimes["candidate"]["wheel"]:
        raise ValueError("Baseline and candidate must have separate wheel artifacts")
    report: dict[str, Any] = {
        "status": "DRAFT",
        "host": args.host,
        "runtimes": runtimes,
        "raw": [],
        "summary": [],
        "unmet_gates": [],
        "artifact_directory": str(artifact),
    }
    report["machine"] = {
        "uname": list(os.uname()),
        "cpuinfo": Path("/proc/cpuinfo").read_text(),
        "affinity": sorted(os.sched_getaffinity(0)),
    }
    gaps = report["unmet_gates"]
    selected_cases = args.cases
    memory_only = args.memory_only
    cases = [
        case
        for case in case_specs(args.quick)
        if selected_cases is None or case["name"] in selected_cases
    ]
    report["collection"] = {"cases": selected_cases, "memory_only": memory_only}
    if selected_cases is not None:
        gaps.append("Explicit case subset: incomplete certification matrix")
    if memory_only:
        gaps.append("Memory-only collection: timing/scaling absent; incomplete certification")
    if args.quick:
        gaps.append("Quick exploration is explicitly uncertified")
    if args.host != "oci":
        gaps.append("OCI primary/control/domain/scaling evidence absent")
    if args.host == "n100" and "N100" not in Path("/proc/cpuinfo").read_text():
        gaps.append("Host CPU is not verified as Intel N100")
    required_threads = {1, 2, 4, 8, 16, 32, 64} if args.host == "oci" else {1, 2, 4}
    if not required_threads.issubset(args.threads):
        gaps.append("Required host thread matrix incomplete")
    save(output, report)
    serial = 0

    def observe(
        case: dict[str, Any], directory: Path, threads: int, kind: str, role: str, pair: int
    ) -> dict[str, Any]:
        nonlocal serial
        serial += 1
        request_path = artifact / f"{serial:06d}-request.json"
        response_path = artifact / f"{serial:06d}-response.json"
        request = {
            "runtime": runtimes[role],
            "case": case,
            "dataset": str(directory),
            "dataset_manifest": json.loads((directory / "dataset.json").read_text()),
            "threads": threads,
            "kind": kind,
            "candidate": role == "candidate",
        }
        save(request_path, request)
        observation = {
            "case": case,
            "threads": threads,
            "kind": kind,
            "role": role,
            "pair": pair,
            "request": str(request_path),
            "response": str(response_path),
        }
        try:
            command(
                [
                    runtimes[role]["python"],
                    "-I",
                    str(driver),
                    "worker",
                    str(request_path),
                    str(response_path),
                ],
                artifact,
                clean_env(threads),
            )
            observation["result"] = json.loads(response_path.read_text())
            if observation["result"].get("status") != "passed":
                raise ValueError("Worker response did not pass")
            gaps.extend(failure_strings(observation["result"]))
            for gap in observation["result"].get("unmet_gates", []):
                gaps.append(f"{case['name']}/{case['agg']}/T{threads}/{role}: {gap}")
        except (RuntimeError, ValueError, OSError) as error:
            observation["error"] = str(error)
            if response_path.is_file():
                try:
                    observation["result"] = json.loads(response_path.read_text())
                except (ValueError, OSError) as partial_error:
                    observation["partial_response_error"] = str(partial_error)
            gaps.append("Worker failed: " + str(request_path))
        report["raw"].append(observation)
        save(output, report)
        return {} if "error" in observation else observation["result"]

    try:
        for case in cases:
            directory = artifact / (case["name"] + "-" + case["agg"])
            dataset = create_dataset(case, directory)
            for threads in args.threads:
                profiles = {
                    role: observe(case, directory, threads, "evidence", role, -1)
                    for role in runtimes
                }
                for mode in () if memory_only else ("cold", "warm"):
                    pairs = []
                    for pair in range(2 if args.quick else 40):
                        roles = (
                            ("baseline", "candidate")
                            if pair % 2 == 0
                            else ("candidate", "baseline")
                        )
                        values = {
                            role: observe(case, directory, threads, mode, role, pair)
                            for role in roles
                        }
                        if all("elapsed_s" in values[role] for role in roles):
                            if (
                                values["baseline"]["fingerprint"]
                                != values["candidate"]["fingerprint"]
                            ):
                                gaps.append("Paired dataset fingerprint mismatch")
                            pairs.append({role: values[role]["elapsed_s"] for role in roles})
                    if len(pairs) != (2 if args.quick else 40):
                        gaps.append("Incomplete paired samples")
                        continue
                    summary = paired_summary(pairs)
                    summary.update(case=case, threads=threads, mode=mode)
                    report["summary"].append(summary)
                    gate_threads = (32, 64) if args.host == "oci" else (4,)
                    domain = case["name"] in ("packed2", "packed10", "fused3")
                    if threads in gate_threads and not (domain and threads == 64):
                        primary_warm = case["name"] == "primary" and mode == "warm"
                        passed = (
                            summary["ratio"] <= 0.90 and summary["u95"] < 1.0
                            if primary_warm
                            else summary["u95"] <= 1.05
                        )
                        if not passed:
                            gaps.append(
                                f"Performance gate: {case['name']}/{case['agg']}/{mode}/T{threads}"
                            )
                if memory_only or threads in (1, max(args.threads)):
                    measurements = {
                        role: [
                            observe(case, directory, threads, "rss", role, i)
                            for i in range(1 if args.quick else 10)
                        ]
                        for role in runtimes
                    }
                    baseline_rss = [
                        r["rss"]["VmHWM"] for r in measurements["baseline"] if "rss" in r
                    ]
                    peak = profiles["candidate"].get("profile", {}).get("estimated_peak_bytes")
                    if peak is None or len(baseline_rss) != (1 if args.quick else 10):
                        gaps.append("Missing modeled route peak or baseline RSS evidence")
                    else:
                        for measured in measurements["candidate"]:
                            if "rss" in measured:
                                gaps.extend(
                                    memory_gaps(
                                        measured,
                                        statistics.median(baseline_rss),
                                        dataset["input_bytes"],
                                        peak,
                                        threads,
                                        case["name"]
                                        in ("primary", "packed2", "packed10", "fused3"),
                                    )
                                )
            save(output, report)
        # Larger resource refusal is diagnostic, not an admitted-domain speed claim.
        if not args.quick and selected_cases is None:
            for agg in ("max", "count"):
                refused = {
                    "name": "refused10",
                    "n": args.refusal_rows,
                    "widths": [args.refusal_rows] + [2] * 9,
                    "dtype": "int64",
                    "sort": True,
                    "agg": agg,
                    "seed": 42,
                }
                directory = artifact / ("refused10-" + agg)
                create_dataset(refused, directory)
                for threads in (1, max(args.threads)):
                    for role in runtimes:
                        observe(refused, directory, threads, "evidence", role, -1)
                        for i in range(10):
                            observe(refused, directory, threads, "rss", role, i)
        if args.host == "oci" and not memory_only:
            for case in [c for c in cases if c["name"] == "primary"]:
                directory = artifact / (case["name"] + "-" + case["agg"])
                pairs = []
                for pair in range(2 if args.quick else 40):
                    order = (1, 32) if pair % 2 == 0 else (32, 1)
                    measured = {
                        threads: observe(case, directory, threads, "scaling", "candidate", pair)
                        for threads in order
                    }
                    if all("elapsed_s" in measured[t] for t in order):
                        pairs.append(
                            {
                                "baseline": measured[1]["elapsed_s"],
                                "candidate": measured[32]["elapsed_s"],
                            }
                        )
                if len(pairs) != 40:
                    gaps.append("Missing 40 OCI scaling pairs: " + case["agg"])
                else:
                    scaling = paired_summary(pairs)
                    report.setdefault("scaling", {})[case["agg"]] = scaling
                    if scaling["u95"] >= 1:
                        gaps.append("OCI 32-vs-1 warm scaling failed: " + case["agg"])
    except (ValueError, OSError, RuntimeError) as error:
        gaps.append("Collection aborted: " + str(error))
    report["unmet_gates"] = sorted(set(gaps))
    if selected_cases is not None or memory_only:
        report["status"] = "DRAFT"
    else:
        report["status"] = "UNCERTIFIED" if args.quick else ("DRAFT" if gaps else "HOST_GATES_PASS")
    # Host evidence alone cannot certify the union of N100 and OCI gates.
    report["certified"] = False
    save(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "status": report["status"],
                "unmet_gates": report["unmet_gates"],
            },
            indent=2,
        )
    )


def main() -> None:
    """Dispatch preparation, standalone collection, or internal workers."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("prepare", help="build immutable snapshot wheel and runtime")
    for name in ("repo", "sha", "lock", "cargo-lock", "root"):
        build.add_argument("--" + name, required=True)
    collect = sub.add_parser("run", help="collect raw evidence; missing gates remain DRAFT")
    for name in ("baseline", "candidate", "output"):
        collect.add_argument("--" + name, required=True)
    collect.add_argument("--host", choices=("n100", "oci"), default="n100")
    collect.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    collect.add_argument("--quick", action="store_true", help="2 pairs; never certified")
    collect.add_argument(
        "--cases",
        nargs="+",
        choices=("primary", "low", "ordinary", "unsorted", "packed2", "packed10", "fused3"),
        help="collect only named cases; always an incomplete DRAFT matrix",
    )
    collect.add_argument(
        "--memory-only",
        action="store_true",
        help="profiles and RSS only, without cold/warm or scaling pairs; always DRAFT",
    )
    collect.add_argument(
        "--refusal-rows",
        type=int,
        default=5000000,
        help="larger 10-key resource-refusal diagnostic rows",
    )
    leaf = sub.add_parser("worker", help=argparse.SUPPRESS)
    leaf.add_argument("request", type=Path)
    leaf.add_argument("response", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "worker":
        worker(args.request, args.response)
    else:
        if any(t < 1 for t in args.threads) or len(args.threads) != len(set(args.threads)):
            parser.error("Thread counts must be unique positive integers")
        if args.refusal_rows < 5000000:
            parser.error("Refusal diagnostics must use at least 5m rows")
        run(args)


if __name__ == "__main__":
    main()
