# Stage 9 Merge Report

## Important note on the two source ZIPs

The two uploaded ZIPs did not match their stated descriptions, so this merge
was done by inspecting actual file contents rather than trusting the labels:

- **The ZIP described as "complete Stage 8 project"** actually contained a
  small, self-contained sandbox: `main.py`, `src/config.py`,
  `src/data_loader.py`, `src/manifest.py`, `src/experiments/{storage,
  quality_stability, stability, runner, statistical_analysis}.py`, one test
  file, and three tiny CSV outputs. Its own `STAGE_9_COMPLETION_SUMMARY.md`
  explicitly lists `storage.py`, `quality_stability.py`, `stability.py`,
  `runner.py`, `config.py`, `data_loader.py`, and `manifest.py` as **"Created
  Supporting Modules"** — i.e. minimal mock stand-ins written so
  `statistical_analysis.py` could be developed and unit-tested in isolation.
  It contained no `data/`, `docs/`, `notebooks/`, `src/algorithms/`,
  `src/metrics/`, `config.yaml`, `README.md`, or most of the Stage 1–8 tests.
- **The ZIP described as "Stage 9 implementation"** actually contained,
  nested inside it, the real complete Stage 1–8 project (`data/` with all 70
  LFR instances and all 4 real datasets, `docs/`, `notebooks/`,
  `src/algorithms/`, `src/metrics/`, `config.yaml`, `README.md`, full test
  suite), plus an embedded stale copy of the mock sandbox above.

So the actual base and the actual Stage 9 payload were the reverse of how the
two files were labeled. This report describes what was merged, using
"base" = the real Stage 1–8 project and "sandbox" = the mock package,
regardless of which upload each came from.

I verified this conclusion (not just inferred it) before merging: the
sandbox's `main.py`, `experiments/__init__.py`, and `statistical_analysis.py`
all call `resolve(cfg, "results_dir")` with the real base's 2-argument
signature (`cfg["paths"][key]`), not the sandbox's own 3-argument
`resolve(cfg, key, default)` reading flat `cfg.get(key)`. The sandbox's
`quality_stability.py` also invents algorithms (`leiden`, `oslom`) that don't
exist anywhere in this project (`src/algorithms/` only has `lpa`,
`semi_sync`, `flpa`, `slpa`) — confirming those particular files are
fabricated placeholders, not real Stage 8 code, and must not overwrite the
real ones.

## What was preserved unchanged from the base (Stage 1–8)

Everything: `data/` (raw datasets + all 70 LFR instances), `docs/`,
`notebooks/`, `config.yaml`, `README.md`, `.gitignore`, `pytest.ini`,
`requirements*.txt`, `results/` (existing processed/raw outputs),
`src/algorithms/`, `src/metrics/`, `src/graph.py`, `src/lfr.py`,
`src/preprocessing.py`, `src/statistics.py`, `src/visualization.py`,
`src/experiments/{quality_stability, quality_stability_figures, runner,
stability, storage}.py`, `src/config.py`, `src/data_loader.py`,
`src/manifest.py`, and the full existing `tests/` suite (algorithms, data
loader, experiments, LFR, metrics, preprocessing, quality-stability,
stability).

## Stage 9 files added/modified on top of the base

- **`src/experiments/statistical_analysis.py`** — added (new file). Its
  imports (`GRAPH_META_COLUMNS`, `HARD_ALGORITHMS`, `OVERLAPPING_ALGORITHMS`,
  `load_runs` from `.quality_stability`; `_applicable_metrics`,
  `_summary_stats_from_pairwise`, `SUMMARY_METRIC_STATS` from `.stability`)
  all resolve correctly against the real base modules.
- **`tests/test_statistical_analysis.py`** — replaced. The base's own copy
  was a one-line placeholder (`# Test file for statistical analysis`); the
  sandbox's 302-line real test suite was used instead.
- **`main.py`** — updated. Diffed line-by-line against the base's `main.py`
  first: every existing command (`prepare`, `generate-lfr`, `run`,
  `stability`, `quality-stability`) is untouched. The only additions are the
  `atomic_write_text` import, the new `cmd_statistical_analysis()` function,
  the `statistical-analysis` argparse choice, and the `--type` flag.
- **`src/experiments/__init__.py`** — updated. Diffed the same way: the only
  change is exposing `statistical_analysis` and its three public functions
  alongside the existing exports.
- **`STAGE_9_SUMMARY.md`, `STAGE_9_COMPLETION_SUMMARY.md`** — added (new
  docs, carried over as-is from the sandbox).

## Conflict found and how it was resolved

**Conflict:** `tests/test_statistical_analysis.py` builds its own fake `cfg`
dict in 5 tests, shaped for the sandbox's flat `resolve()`
(`{"results_dir": ..., "real_datasets": ..., "lfr": ...}`). Against the
base's real `resolve(cfg, key)` (`cfg["paths"][key]`), this raised
`KeyError: 'paths'` in 5 tests.

**Resolution:** this is a mechanical test-fixture mismatch, not a design
decision — the test bodies, assertions, and the code under test are
untouched. I changed only the shape of the 5 fake `cfg` dicts from
`{"results_dir": str(results_dir), ...}` to
`{"paths": {"results_dir": str(results_dir)}, ...}` so they match the real
project's actual config schema. Nothing else in the test file was altered.

**Excluded (not merged in):** the sandbox's own `config.py`, `data_loader.py`,
`manifest.py`, `storage.py`, `stability.py`, `quality_stability.py`,
`runner.py` (mock stand-ins, replaced by the real base versions — see
above), `test_temp.py` / `tests/simple_test.py` (throwaway scratch files,
each just a one-line `print(...)`), the embedded stale zip/tar.gz archives,
and `results/processed/{statistical_tests,posthoc_tests,qs_summary}.csv`
from the sandbox — these three CSVs contain small, round-number synthetic
fixture values (e.g. `n_blocks=3`, `VI=0.8/0.2/0.3`) used only to unit-test
the statistics code, not real output from the actual dataset. Including them
would misrepresent fabricated numbers as real Stage 9 results, which the
task explicitly ruled out. Running `python main.py statistical-analysis`
against the real data (already present in this project) will produce the
genuine versions of these three files.

## Test results

- `pytest tests/test_statistical_analysis.py`: **10 passed** (after the cfg
  fixture fix above).
- Full suite `pytest`: **154 passed, 2 failed, 2 skipped**. The 2 failures
  (`test_identical_hard_partitions_give_perfect_agreement`,
  `test_identical_cover_partitions_give_perfect_agreement` in
  `tests/test_stability.py`) are **pre-existing and unrelated to this
  merge** — they fail identically when run against the untouched base
  project by itself, in this sandbox's library versions (numpy 2.4.4,
  pandas 3.0.2, pytest 9.1.1). The cause is a `pytest.approx` /
  `pandas.Series` comparison returning a scalar instead of elementwise in
  this newer pandas version. No Stage 1–8 file was modified to work around
  this, per the instruction not to redesign existing code.

## Confirmations

- No dataset was regenerated: all 70 LFR instances and all 4 real datasets
  are byte-for-byte copies from the base project.
- The 8,880-run experiment was **not** run.
- The 70 LFR graphs were **not** regenerated.
- No experimental results were fabricated; the 3 synthetic sandbox CSVs were
  deliberately excluded (see above) rather than passed off as real output.
- No duplicate/nested project directories remain in the final archive.
- All `__pycache__` and `.pytest_cache` directories were removed before
  zipping.
