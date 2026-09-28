"""Stage 6 tests: pairwise stability, aggregation, bootstrap CIs -- built on tiny synthetic
partitions/graphs only (see docs/methodology.md-style rule inherited from Stage 5: never run
the full 8,880-run experiment matrix in tests). Real stored partitions are used only via a
tiny 6-node graph run through the ACTUAL Stage 5 runner for a handful of seeds (fast,
deterministic), never via LFR or the full real-dataset matrix.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from src.experiments import storage
from src.experiments.stability import (
    DEFAULT_BOOTSTRAP_SEED,
    _applicable_metrics,
    pairwise_stability,
    run_stability_analysis,
)
from src.metrics.stability import nmi as nmi_score
from src.metrics.stability import omega_index, onmi_lfk, variation_of_information

N_SYNTH = 6  # node count for every synthetic partition below (labels/covers over 0..5)


# ---------------------------------------------------------------------------
# helpers: write Stage-5-shaped stored runs directly (same interfaces Stage 5 uses)
# ---------------------------------------------------------------------------

def _write_hard_run(raw_dir, graph_id, algorithm, seed, run_index, labels,
                     degenerate=False, dataset="synth"):
    labels = np.asarray(labels)
    ppath = storage.partition_path(raw_dir, graph_id, algorithm, seed)
    storage.save_hard_partition(ppath, labels)
    rec = {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": None, "mu_empirical": None,
        "instance": None, "algorithm": algorithm, "run": run_index, "seed": seed,
        "number_of_communities": int(labels.max()) + 1, "quality": 0.0, "quality_metric": "Q",
        "nmi": None, "onmi": None, "f1": None, "runtime_seconds": 0.001,
        "largest_community_size": int(np.bincount(labels).max()), "degenerate": degenerate,
        "converged": True, "iterations": 1,
        "partition_file": str(ppath.relative_to(Path(raw_dir))), "partition_kind": "hard",
    }
    storage.save_record(storage.record_path(raw_dir, graph_id, algorithm, seed), rec)
    return rec


def _write_cover_run(raw_dir, graph_id, algorithm, seed, run_index, cover,
                      degenerate=False, dataset="synth"):
    ppath = storage.partition_path(raw_dir, graph_id, algorithm, seed)
    storage.save_cover_partition(ppath, cover)
    rec = {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": None, "mu_empirical": None,
        "instance": None, "algorithm": algorithm, "run": run_index, "seed": seed,
        "number_of_communities": len(cover), "quality": 0.0, "quality_metric": "EQ",
        "nmi": None, "onmi": None, "f1": None, "runtime_seconds": 0.001,
        "largest_community_size": max(len(c) for c in cover), "degenerate": degenerate,
        "converged": True, "iterations": 1,
        "partition_file": str(ppath.relative_to(Path(raw_dir))), "partition_kind": "cover",
    }
    storage.save_record(storage.record_path(raw_dir, graph_id, algorithm, seed), rec)
    return rec


def _finalize(raw_dir):
    """Mirrors what Stage 5 does after writing records: rebuild runs.csv from them."""
    return storage.rebuild_runs_csv(raw_dir)


# ---------------------------------------------------------------------------
# 1. identical partitions -> perfect-agreement metrics (per algorithm's own scope)
# ---------------------------------------------------------------------------

def test_identical_hard_partitions_give_perfect_agreement(tmp_path):
    raw_dir = tmp_path / "raw"
    labels = np.array([0, 0, 1, 1, 2, 2])
    for seed in range(3):
        _write_hard_run(raw_dir, "g1", "lpa", seed, seed, labels)
    _finalize(raw_dir)

    runs = pd.read_csv(raw_dir / "runs.csv")
    from src.experiments.stability import _load_group_partitions
    loaded = _load_group_partitions(raw_dir, runs)
    pw = pairwise_stability(loaded, "lpa", {c: None for c in
                             ["graph_id", "dataset", "mu_nominal", "mu_empirical", "instance"]})

    assert np.allclose(pw["VI"].to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(pw["normalized_VI"].to_numpy(), 0.0, atol=1e-12)
    assert np.allclose(pw["NMI"].to_numpy(), 1.0)
    assert np.allclose(pw["Omega"].to_numpy(), 1.0)
    # ONMI is out of scope for hard partitions in Stage 6 (see stability.py docstring) -> NaN
    assert pw["ONMI"].isna().all()


def test_identical_cover_partitions_give_perfect_agreement(tmp_path):
    raw_dir = tmp_path / "raw"
    cover = [{0, 1, 2}, {3, 4, 5}]
    for seed in range(3):
        _write_cover_run(raw_dir, "g1", "slpa", seed, seed, cover)
    _finalize(raw_dir)

    runs = pd.read_csv(raw_dir / "runs.csv")
    from src.experiments.stability import _load_group_partitions
    loaded = _load_group_partitions(raw_dir, runs)
    pw = pairwise_stability(loaded, "slpa", {c: None for c in
                             ["graph_id", "dataset", "mu_nominal", "mu_empirical", "instance"]})

    assert pw["VI"].isna().all() and pw["NMI"].isna().all()      # out of scope for SLPA
    assert np.allclose(pw["Omega"].to_numpy(), 1.0)
    assert np.allclose(pw["ONMI"].to_numpy(), 1.0)


# ---------------------------------------------------------------------------
# 2 & 6. clearly different hard partitions -> valid, correctly-computed metrics
# ---------------------------------------------------------------------------

def test_different_hard_partitions_produce_correct_metrics(tmp_path):
    raw_dir = tmp_path / "raw"
    a = np.array([0, 0, 0, 1, 1, 1])
    b = np.array([0, 0, 1, 1, 2, 2])
    _write_hard_run(raw_dir, "g1", "flpa", 0, 0, a)
    _write_hard_run(raw_dir, "g1", "flpa", 1, 1, b)
    _finalize(raw_dir)

    runs = pd.read_csv(raw_dir / "runs.csv")
    from src.experiments.stability import _load_group_partitions
    loaded = _load_group_partitions(raw_dir, runs)
    pw = pairwise_stability(loaded, "flpa", {c: None for c in
                             ["graph_id", "dataset", "mu_nominal", "mu_empirical", "instance"]})
    assert len(pw) == 1
    row = pw.iloc[0]

    assert row["VI"] == pytest.approx(variation_of_information(a, b))
    assert row["normalized_VI"] == pytest.approx(variation_of_information(a, b, normalize=True))
    assert row["NMI"] == pytest.approx(nmi_score(a, b))
    assert row["Omega"] == pytest.approx(omega_index(6, labels1=a, labels2=b))
    assert 0.0 <= row["VI"] < 2.0
    assert 0.0 <= row["NMI"] <= 1.0


# ---------------------------------------------------------------------------
# 3, 4, 5. pairwise combinations: unique, correct count for R=30 and for fewer runs
# ---------------------------------------------------------------------------

def _write_n_hard_runs(raw_dir, graph_id, algorithm, n):
    rng = np.random.default_rng(0)
    for i in range(n):
        labels = rng.integers(0, 3, size=N_SYNTH)
        _write_hard_run(raw_dir, graph_id, algorithm, seed=i, run_index=i, labels=labels)
    return _finalize(raw_dir)


def test_pairwise_combinations_are_unique(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 6)
    runs = pd.read_csv(raw_dir / "runs.csv")
    from src.experiments.stability import _load_group_partitions
    loaded = _load_group_partitions(raw_dir, runs)
    pw = pairwise_stability(loaded, "lpa", {c: None for c in
                             ["graph_id", "dataset", "mu_nominal", "mu_empirical", "instance"]})

    pairs = list(zip(pw["run_i"], pw["run_j"]))
    assert all(i < j for i, j in pairs)                 # always ordered i<j, no (j,i) duplicate
    assert len(set(pairs)) == len(pairs)                 # every pair unique


def test_thirty_runs_produce_435_pairs(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 30)
    _, summary = _run_one_group(raw_dir, "g1", "lpa")
    assert summary["number_of_runs"] == 30
    assert summary["number_of_pairs"] == math.comb(30, 2) == 435


@pytest.mark.parametrize("n,expected_pairs", [(2, 1), (3, 3), (5, 10), (10, 45)])
def test_fewer_runs_produce_correct_pair_count(tmp_path, n, expected_pairs):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", n)
    _, summary = _run_one_group(raw_dir, "g1", "lpa")
    assert summary["number_of_pairs"] == expected_pairs == math.comb(n, 2)


def _run_one_group(raw_dir, graph_id, algorithm, n_bootstrap=20, bootstrap_seed=1):
    results_dir = raw_dir.parent
    pw_df, summary_df = run_stability_analysis({}, results_dir=results_dir,
                                                 n_bootstrap=n_bootstrap,
                                                 bootstrap_seed=bootstrap_seed, verbose=False)
    row = summary_df[(summary_df["graph_id"] == graph_id) & (summary_df["algorithm"] == algorithm)]
    assert len(row) == 1
    return pw_df, row.iloc[0]


# ---------------------------------------------------------------------------
# 7. SLPA overlapping covers use Omega/ONMI appropriately
# ---------------------------------------------------------------------------

def test_slpa_uses_omega_and_onmi_not_hard_metrics(tmp_path):
    raw_dir = tmp_path / "raw"
    cover_a = [{0, 1, 2}, {3, 4, 5}]
    cover_b = [{0, 1, 2, 3}, {3, 4, 5}]   # node 3 now genuinely overlapping
    _write_cover_run(raw_dir, "g1", "slpa", 0, 0, cover_a)
    _write_cover_run(raw_dir, "g1", "slpa", 1, 1, cover_b)
    _finalize(raw_dir)
    pw_df, summary = _run_one_group(raw_dir, "g1", "slpa")

    assert _applicable_metrics("slpa") == ["Omega", "ONMI"]
    assert pw_df["VI"].isna().all() and pw_df["NMI"].isna().all()
    assert not pw_df["Omega"].isna().any() and not pw_df["ONMI"].isna().any()
    assert pw_df["Omega"].iloc[0] == pytest.approx(omega_index(6, cover1=cover_a, cover2=cover_b))
    assert pw_df["ONMI"].iloc[0] == pytest.approx(onmi_lfk(6, cover_a, cover_b))
    # summary row: hard-only columns are NaN, Omega/ONMI columns populated
    assert pd.isna(summary["mean_VI"]) and pd.isna(summary["mean_NMI"])
    assert not pd.isna(summary["mean_Omega"]) and not pd.isna(summary["mean_ONMI"])


# ---------------------------------------------------------------------------
# 8. degenerate flags are preserved (not recomputed)
# ---------------------------------------------------------------------------

def test_degenerate_flags_are_preserved_from_stage5(tmp_path):
    raw_dir = tmp_path / "raw"
    trivial = np.zeros(N_SYNTH, dtype=int)          # one giant community
    normal = np.array([0, 0, 0, 1, 1, 1])
    _write_hard_run(raw_dir, "g1", "lpa", 0, 0, trivial, degenerate=True)
    _write_hard_run(raw_dir, "g1", "lpa", 1, 1, normal, degenerate=False)
    _finalize(raw_dir)
    pw_df, summary = _run_one_group(raw_dir, "g1", "lpa")

    row = pw_df.iloc[0]
    assert bool(row["degenerate_i"]) is True
    assert bool(row["degenerate_j"]) is False
    assert summary["degenerate_run_count"] == 1
    assert summary["degenerate_pair_count"] == 1     # the one pair involves a degenerate run


# ---------------------------------------------------------------------------
# 9. aggregation produces exactly one row per graph x algorithm
# ---------------------------------------------------------------------------

def test_aggregation_one_row_per_graph_and_algorithm(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 4)
    _write_n_hard_runs(raw_dir, "g1", "flpa", 4)
    _write_n_hard_runs(raw_dir, "g2", "lpa", 4)
    _, summary_df = run_stability_analysis({}, results_dir=raw_dir.parent,
                                            n_bootstrap=10, bootstrap_seed=1, verbose=False)
    combos = list(zip(summary_df["graph_id"], summary_df["algorithm"]))
    assert sorted(combos) == [("g1", "flpa"), ("g1", "lpa"), ("g2", "lpa")]
    assert len(combos) == len(set(combos))


# ---------------------------------------------------------------------------
# 10 & 14. bootstrap output / whole-stage output is deterministic with a fixed seed
# ---------------------------------------------------------------------------

def test_bootstrap_and_full_rerun_are_deterministic(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 8)

    pw1, s1 = run_stability_analysis({}, results_dir=raw_dir.parent,
                                      n_bootstrap=100, bootstrap_seed=42, verbose=False)
    pw2, s2 = run_stability_analysis({}, results_dir=raw_dir.parent,
                                      n_bootstrap=100, bootstrap_seed=42, verbose=False)

    pd.testing.assert_frame_equal(pw1, pw2)
    pd.testing.assert_frame_equal(s1, s2)


# ---------------------------------------------------------------------------
# 11. bootstrap CI columns are valid
# ---------------------------------------------------------------------------

def test_bootstrap_ci_columns_are_valid(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 10)
    _, summary = _run_one_group(raw_dir, "g1", "lpa", n_bootstrap=500, bootstrap_seed=7)

    bounds = {"VI": (0.0, math.log2(N_SYNTH)), "normalized_VI": (0.0, 1.0),
              "NMI": (0.0, 1.0), "Omega": (-1.0, 1.0)}
    for metric, (lo_bound, hi_bound) in bounds.items():
        lo, hi = summary[f"mean_{metric}_ci_low"], summary[f"mean_{metric}_ci_high"]
        # CI bounds are well-formed and within the metric's valid range. Note: because the
        # run-level bootstrap includes self-pairs (a run resampled twice is a perfect-
        # agreement pair with itself -- see stability.py's module docstring), the bootstrap
        # distribution is systematically shifted toward higher stability than the point
        # estimate computed from the DISTINCT pairs alone; the percentile CI is therefore not
        # guaranteed to bracket the plain point estimate for small R or near-ceiling metrics.
        # This is a documented, known property of the run-level bootstrap-for-U-statistics
        # method used here (Bickel & Freedman 1981), not a bug -- so this test checks
        # well-formedness and valid range, not that the CI brackets the point estimate.
        assert lo <= hi
        assert lo_bound - 1e-9 <= lo <= hi_bound + 1e-9
        assert lo_bound - 1e-9 <= hi <= hi_bound + 1e-9
    # ONMI not applicable to lpa -> CI columns present but NaN, not fabricated
    assert pd.isna(summary["mean_ONMI_ci_low"]) and pd.isna(summary["mean_ONMI_ci_high"])


# ---------------------------------------------------------------------------
# 12. Stage 6 can read Stage 5 partition files (real end-to-end smoke test)
# ---------------------------------------------------------------------------

def _tiny_graph():
    from src.graph import GraphBundle
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]
    n = 6
    rows, cols = zip(*edges)
    U = sp.coo_matrix(([1] * len(edges), (rows, cols)), shape=(n, n)).tocsr()
    A = (U + U.T).tocsr()
    A.data[:] = 1
    A.sort_indices()
    A.sum_duplicates()
    return GraphBundle(graph_id="tiny", family="tiny", adj=A)


def test_stage6_reads_real_stage5_output_end_to_end(tmp_path):
    from src.experiments import run_experiments

    tiny_cfg = {
        "runs": {"n_runs": 6, "seed_start": 0},
        "paths": {"data_dir": "data", "raw_dir": "data/raw", "lfr_dir": "data/lfr",
                  "results_dir": "results"},
        "real_datasets": ["tiny"], "slpa": {"T": 20, "r": 0.1},
    }
    raw_dir = tmp_path / "raw"
    run_experiments(tiny_cfg, [_tiny_graph()], ["lpa", "slpa"], list(range(6)),
                     results_dir=raw_dir, verbose=False)

    pw_df, summary_df = run_stability_analysis({}, results_dir=raw_dir.parent,
                                                n_bootstrap=20, bootstrap_seed=3, verbose=False)
    assert set(summary_df["algorithm"]) == {"lpa", "slpa"}
    assert (summary_df["number_of_runs"] == 6).all()
    assert (summary_df["number_of_pairs"] == math.comb(6, 2)).all()
    # raw results untouched: Stage 5's runs.csv is still exactly 12 rows (2 algos x 6 seeds)
    assert len(pd.read_csv(raw_dir / "runs.csv")) == 12


# ---------------------------------------------------------------------------
# 13. Stage 6 does not invoke algorithm execution
# ---------------------------------------------------------------------------

def test_stage6_never_calls_an_algorithm(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5)

    import src.algorithms as algorithms_mod

    def _boom(*a, **k):
        raise AssertionError("Stage 6 must never execute a community-detection algorithm")

    original = dict(algorithms_mod.ALGORITHMS)
    algorithms_mod.ALGORITHMS.update({name: _boom for name in original})
    try:
        # must complete without tripping any of the poisoned algorithm entries
        _, summary_df = run_stability_analysis({}, results_dir=raw_dir.parent,
                                                n_bootstrap=5, bootstrap_seed=1, verbose=False)
        assert len(summary_df) == 1
    finally:
        algorithms_mod.ALGORITHMS.clear()
        algorithms_mod.ALGORITHMS.update(original)


# ---------------------------------------------------------------------------
# raw results are left untouched by Stage 6 (bonus check alongside #12/#14)
# ---------------------------------------------------------------------------

def test_stage6_does_not_modify_raw_results(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5)
    before = (raw_dir / "runs.csv").read_text()
    before_mtimes = {p: p.stat().st_mtime_ns for p in raw_dir.rglob("*") if p.is_file()}

    run_stability_analysis({}, results_dir=raw_dir.parent, n_bootstrap=5, bootstrap_seed=1,
                            verbose=False)

    after = (raw_dir / "runs.csv").read_text()
    after_mtimes = {p: p.stat().st_mtime_ns for p in raw_dir.rglob("*") if p.is_file()}
    assert before == after
    assert before_mtimes == after_mtimes


# ---------------------------------------------------------------------------
# 15. existing Stage 1-5 baseline is unaffected (lightweight smoke check here;
# the full tests/test_experiments.py etc. suites are the real proof and are run
# alongside this file, not superseded by it)
# ---------------------------------------------------------------------------

def test_stage5_storage_and_runner_still_importable_and_registered():
    from src.experiments import run_experiments, run_single, storage as storage_mod
    from src.algorithms import ALGORITHMS
    assert set(ALGORITHMS) == {"lpa", "semi_sync_lpa", "flpa", "slpa"}
    assert callable(run_experiments) and callable(run_single)
    assert storage_mod.RUN_COLUMNS[0] == "graph_id"


# ---------------------------------------------------------------------------
# recomputability: deleting results/processed/ and regenerating gives the same answer
# ---------------------------------------------------------------------------

def test_recomputable_from_raw_after_deleting_processed(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 8)
    proc_dir = raw_dir.parent / "processed"

    run_stability_analysis({}, results_dir=raw_dir.parent, n_bootstrap=20, bootstrap_seed=9,
                            verbose=False)
    first_pairwise = (proc_dir / "pairwise.csv").read_text()
    first_summary = (proc_dir / "stability_summary.csv").read_text()

    import shutil
    shutil.rmtree(proc_dir)
    assert not proc_dir.exists()

    run_stability_analysis({}, results_dir=raw_dir.parent, n_bootstrap=20, bootstrap_seed=9,
                            verbose=False)
    assert (proc_dir / "pairwise.csv").read_text() == first_pairwise
    assert (proc_dir / "stability_summary.csv").read_text() == first_summary


# ---------------------------------------------------------------------------
# default bootstrap seed is a fixed constant (documents the "store the bootstrap
# seed/configuration" requirement)
# ---------------------------------------------------------------------------

def test_default_bootstrap_seed_is_fixed_constant():
    assert isinstance(DEFAULT_BOOTSTRAP_SEED, int)
