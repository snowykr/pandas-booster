# PROJECT GUIDANCE

## Purpose

`pandas-booster` is a Rust-backed Python package that accelerates selected Pandas
groupby reductions through a PyO3 extension. The main rule is correctness first:
preserve Pandas semantics, and fall back conservatively when a case is outside the
certified Rust dispatch domain.

## Layout

```text
src/                    Rust kernels, PyO3 registration, NumPy conversion
python/pandas_booster/   public Python API, dispatch policy, fallbacks
tests/                  pytest behavior, ABI, release, and contract tests
benchmarks/             benchmark runner and generated report pipeline
scripts/                release and supply-chain contract tools
```

## Important Rules

- Development commands assume an activated virtual environment, preferably `.venv`,
  unless they are run through `uv run`.
- Python source lives under `python/`; pytest adds that directory through
  `pyproject.toml`.
- Keep Python syntax compatible with Python 3.9.
- Keep pandas compatibility rules centralized in
  `python/pandas_booster/_groupby_policy.py`; do not scatter eligibility checks.
- Keep environment-variable parsing in `python/pandas_booster/_config.py`.
- If Rust exports change, update all three surfaces together:
  `src/python_wrappers/register.rs`, `python/pandas_booster/_rust.pyi`, and
  `tests/rust_stub_exports/`.
- Do not weaken release or supply-chain contract tests to satisfy workflow edits.
- Do not overwrite generated benchmark reports unless they carry the repo
  provenance marker.

## Behavior Notes

- `activate()` monkey-patches `pd.DataFrame.groupby`; it must remain reversible via
  `deactivate()`.
- Recognized ABI skew falls back to pandas unless `PANDAS_BOOSTER_STRICT_ABI=1`.
- `PANDAS_BOOSTER_FORCE_PANDAS_FLOAT_GROUPBY=1` forces pandas for certified
  single-key float reductions when bit-for-bit pandas identity is required.
- Single-key float `prod` uses a no-merge Rust path to preserve per-key row-order
  product semantics.
- `sort=False` means Pandas appearance order, not hash-map iteration order.

## Common Commands

```bash
uv sync --extra bench --extra dev
uv run --with "maturin>=1.13,<2.0" maturin develop --release

source .venv/bin/activate
maturin develop --release

cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
basedpyright --project pyrightconfig.json
ruff check python tests scripts benchmarks
pytest tests/ -v --strict-markers -m "not stress"
python scripts/check_release_contract.py
```
