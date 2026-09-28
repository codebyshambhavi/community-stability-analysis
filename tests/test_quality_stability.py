"""Stage 7 tests: quality x stability join, aggregation, and correlation -- built on tiny
synthetic partitions only (never the full 8,880-run matrix; see docs/methodology.md-style
rule inherited from Stages 5-6). Stage 6's own summary is produced here via its real
`run_stability_analysis` entry point (not re-implemented), exactly as Stage 7 will see it in
production.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import storage
from src.experiments.quality_stability import (
    _safe_spearman,
    aggregate_quality,
    build_quality_stability_summary,
    compute_correlations,
    run_quality_stability_analysis,
)
from src.experiments.stability import run_stability_analysis

N_SYNTH = 6  # node count for every synthetic partition below (labels/covers over 0..5)


# ---------------------------------------------------------------------------
# helpers: write Stage-5-shaped stored runs directly (mirrors tests/test_stability.py)
# ---------------------------------------------------------------------------

def _write_hard_run(raw_dir, graph_id, algorithm, seed, run_index, labels, quality=0.3,
                     nmi=None, degenerate=False, dataset="synth", mu_nominal=None,
                     mu_empirical=None, instance=None):
    labels = np.asarray(labels)
    ppath = storage.partition_path(raw_dir, graph_id, algorithm, seed)
    storage.save_hard_partition(ppath, labels)
    rec = {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": mu_nominal,
        "mu_empirical": mu_empirical, "instance": instance,
        "algorithm": algorithm, "run": run_index, "seed": seed,
        "number_of_communities": int(labels.max()) + 1, "quality": quality, "quality_metric": "Q",
        "nmi": nmi, "onmi": None, "f1": None, "runtime_seconds": 0.001,
        "largest_community_size": int(np.bincount(labels).max()), "degenerate": degenerate,
        "converged": True, "iterations": 1,
        "partition_file": str(ppath.relative_to(Path(raw_dir))), "partition_kind": "hard",
    }
    storage.save_record(storage.record_path(raw_dir, graph_id, algorithm, seed), rec)
    return rec


def _write_cover_run(raw_dir, graph_id, algorithm, seed, run_index, cover, quality=0.3,
                      onmi=None, degenerate=False, dataset="synth"):
    ppath = storage.partition_path(raw_dir, graph_id, algorithm, seed)
    storage.save_cover_partition(ppath, cover)
    rec = {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": None, "mu_empirical": None,
        "instance": None, "algorithm": algorithm, "run": run_index, "seed": seed,
        "number_of_communities": len(cover), "quality": quality, "quality_metric": "EQ",
        "nmi": None, "onmi": onmi, "f1": None, "runtime_seconds": 0.001,
        "largest_community_size": max(len(c) for c in cover), "degenerate": degenerate,
        "converged": True, "iterations": 1,
        "partition_file": str(ppath.relative_to(Path(raw_dir))), "partition_kind": "cover",
    }
    storage.save_record(storage.record_path(raw_dir, graph_id, algorithm, seed), rec)
    return rec


def _finalize(raw_dir):
    return storage.rebuild_runs_csv(raw_dir)


def _write_n_hard_runs(raw_dir, graph_id, algorithm, n, quality_fn=None, nmi_fn=None,
                        dataset="synth", mu_nominal=None, mu_empirical=None, instance=None):
    rng = np.random.default_rng(0)
    for i in range(n):
        labels = rng.integers(0, 3, size=N_SYNTH)
        q = quality_fn(i) if quality_fn else 0.3
        nmi = nmi_fn(i) if nmi_fn else None
        _write_hard_run(raw_dir, graph_id, algorithm, seed=i, run_index=i, labels=labels,
                         quality=q, nmi=nmi, dataset=dataset, mu_nominal=mu_nominal,
                         mu_empirical=mu_empirical, instance=instance)
    return _finalize(raw_dir)


def _write_n_cover_runs(raw_dir, graph_id, algorithm, n, quality_fn=None, onmi_fn=None, dataset="synth"):
    rng = np.random.default_rng(1)
    for i in range(n):
        # two-community cover with a bit of overlap noise across runs
        c1 = set(rng.choice(6, size=3, replace=False).tolist())
        c2 = set(range(6)) - c1 | ({int(rng.integers(0, 6))} if rng.random() < 0.5 else set())
        q = quality_fn(i) if quality_fn else 0.3
        onmi = onmi_fn(i) if onmi_fn else None
        _write_cover_run(raw_dir, graph_id, algorithm, seed=i, run_index=i, cover=[c1, c2],
                          quality=q, onmi=onmi, dataset=dataset)
    return _finalize(raw_dir)


def _run_stage6(raw_dir, n_bootstrap=10, bootstrap_seed=1):
    return run_stability_analysis({}, results_dir=raw_dir.parent, n_bootstrap=n_bootstrap,
                                   bootstrap_seed=bootstrap_seed, verbose=False)


# ---------------------------------------------------------------------------
# 1 & 2. Stage 5 + Stage 6 join correctly, one row per graph x algorithm
# ---------------------------------------------------------------------------

def test_join_produces_one_row_per_graph_and_algorithm(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5)
    _write_n_hard_runs(raw_dir, "g1", "flpa", 5)
    _write_n_hard_runs(raw_dir, "g2", "lpa", 5)
    _run_stage6(raw_dir)

    summary = build_quality_stability_summary(raw_dir.parent)
    combos = list(zip(summary["graph_id"], summary["algorithm"]))
    assert sorted(combos) == [("g1", "flpa"), ("g1", "lpa"), ("g2", "lpa")]
    assert len(combos) == len(set(combos))


def test_quality_and_stability_values_match_their_sources(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 6, quality_fn=lambda i: 0.1 * i)
    pw_df, stab_summary = _run_stage6(raw_dir)

    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary[(summary["graph_id"] == "g1") & (summary["algorithm"] == "lpa")].iloc[0]
    stab_row = stab_summary[(stab_summary["graph_id"] == "g1") & (stab_summary["algorithm"] == "lpa")].iloc[0]

    runs = pd.read_csv(raw_dir / "runs.csv")
    expected_mean_q = runs.loc[runs["quality_metric"] == "Q", "quality"].mean()
    assert row["mean_Q"] == pytest.approx(expected_mean_q)
    assert row["mean_Omega"] == pytest.approx(stab_row["mean_Omega"])
    assert row["mean_stability_VI"] == pytest.approx(stab_row["mean_VI"])


# ---------------------------------------------------------------------------
# 3. missing quality metrics remain missing rather than becoming fabricated zeros
# ---------------------------------------------------------------------------

def test_missing_quality_metric_stays_nan_not_zero(tmp_path):
    raw_dir = tmp_path / "raw"
    # hard algorithm: only Q is ever produced -> mean_EQ must be NaN, never 0
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5)
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary[(summary["graph_id"] == "g1") & (summary["algorithm"] == "lpa")].iloc[0]
    assert pd.isna(row["mean_EQ"])
    assert not pd.isna(row["mean_Q"])


def test_empty_inputs_produce_empty_schema_not_errors(tmp_path):
    empty_summary = build_quality_stability_summary(tmp_path / "results")
    assert list(empty_summary.columns) or empty_summary.empty
    assert len(empty_summary) == 0
    empty_corr = compute_correlations(empty_summary)
    assert len(empty_corr) == 0


# ---------------------------------------------------------------------------
# 4. missing ground truth handled correctly (NaN, never invented)
# ---------------------------------------------------------------------------

def test_missing_ground_truth_stays_nan(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "dolphins_like", "lpa", 5, nmi_fn=None)  # nmi always None
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary[summary["graph_id"] == "dolphins_like"].iloc[0]
    assert pd.isna(row["mean_ground_truth_NMI"])
    assert pd.isna(row["std_ground_truth_NMI"])


def test_present_ground_truth_is_aggregated(tmp_path):
    raw_dir = tmp_path / "raw"
    vals = [0.5, 0.6, 0.7, 0.8]
    _write_n_hard_runs(raw_dir, "karate_like", "lpa", 4, nmi_fn=lambda i: vals[i])
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary[summary["graph_id"] == "karate_like"].iloc[0]
    assert row["mean_ground_truth_NMI"] == pytest.approx(np.mean(vals))


# ---------------------------------------------------------------------------
# 5. SLPA uses its overlapping quality/stability metrics, not hard-partition ones
# ---------------------------------------------------------------------------

def test_slpa_uses_eq_and_overlap_stability_metrics(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_cover_runs(raw_dir, "g1", "slpa", 5, onmi_fn=lambda i: 0.5)
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary[(summary["graph_id"] == "g1") & (summary["algorithm"] == "slpa")].iloc[0]

    assert pd.isna(row["mean_Q"])            # SLPA never contributes Q
    assert not pd.isna(row["mean_EQ"])
    assert pd.isna(row["mean_stability_VI"]) and pd.isna(row["mean_stability_NMI"])  # out of scope
    assert not pd.isna(row["mean_Omega"]) and not pd.isna(row["mean_ONMI"])
    assert row["mean_ground_truth_ONMI"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 6. degenerate counts preserved from Stage 6, not recomputed
# ---------------------------------------------------------------------------

def test_degenerate_counts_are_preserved(tmp_path):
    raw_dir = tmp_path / "raw"
    trivial = np.zeros(N_SYNTH, dtype=int)
    normal = np.array([0, 0, 0, 1, 1, 1])
    _write_hard_run(raw_dir, "g1", "lpa", 0, 0, trivial, degenerate=True)
    _write_hard_run(raw_dir, "g1", "lpa", 1, 1, normal, degenerate=False)
    _finalize(raw_dir)
    _run_stage6(raw_dir)

    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary.iloc[0]
    assert row["degenerate_run_count"] == 1
    assert row["degenerate_pair_count"] == 1


# ---------------------------------------------------------------------------
# 7. stability direction is carried through unmodified (never silently inverted)
# ---------------------------------------------------------------------------

def test_stability_metrics_are_not_inverted(tmp_path):
    raw_dir = tmp_path / "raw"
    identical = np.array([0, 0, 0, 1, 1, 1])
    for seed in range(3):
        _write_hard_run(raw_dir, "g1", "lpa", seed, seed, identical)
    _finalize(raw_dir)
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary.iloc[0]
    # identical partitions every run -> VI must be (near) 0 (lower = more stable) and
    # Omega/NMI must be (near) 1 (higher = more stable); if a silent inversion crept in
    # these values would be swapped.
    assert row["mean_stability_VI"] == pytest.approx(0.0, abs=1e-9)
    assert row["mean_Omega"] == pytest.approx(1.0)
    assert row["mean_stability_NMI"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 8. Spearman calculation works on a small known fixture
# ---------------------------------------------------------------------------

def test_spearman_on_known_monotonic_fixture():
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    y = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])  # perfectly increasing -> rho = 1
    rho, p, n = _safe_spearman(x, y)
    assert rho == pytest.approx(1.0)
    assert n == 5
    assert p < 0.05

    y_dec = pd.Series([50.0, 40.0, 30.0, 20.0, 10.0])  # perfectly decreasing -> rho = -1
    rho2, _, _ = _safe_spearman(x, y_dec)
    assert rho2 == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# 9. constant-variable correlation is handled safely (no crash, no fabricated rho)
# ---------------------------------------------------------------------------

def test_constant_variable_correlation_is_nan_not_error():
    x = pd.Series([0.5, 0.5, 0.5, 0.5])
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    rho, p, n = _safe_spearman(x, y)
    assert np.isnan(rho) and np.isnan(p)
    assert n == 4


def test_too_few_points_correlation_is_nan_not_error():
    x = pd.Series([1.0, 2.0])
    y = pd.Series([3.0, 4.0])
    rho, p, n = _safe_spearman(x, y)
    assert np.isnan(rho) and np.isnan(p)


def test_compute_correlations_runs_without_error_on_small_summary(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(4):
        _write_n_hard_runs(raw_dir, f"g{i}", "lpa", 3, quality_fn=lambda s, i=i: 0.1 * i)
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    corr = compute_correlations(summary)
    assert set(CORRELATION_COLUMNS_CHECK).issubset(corr.columns)
    # no exception raised is the primary assertion; also sanity-check row shape
    assert (corr["n_graphs"] >= 0).all()


CORRELATION_COLUMNS_CHECK = ["subset", "algorithm", "quality_metric", "stability_metric",
                              "n_graphs", "spearman_rho", "p_value"]


# ---------------------------------------------------------------------------
# 10. LFR mu / instance identifiers are preserved through the join
# ---------------------------------------------------------------------------

def test_lfr_mu_and_instance_preserved(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "lfr_mu30_i02", "flpa", 4, dataset="lfr",
                        mu_nominal=0.3, mu_empirical=0.28, instance=2)
    _run_stage6(raw_dir)
    summary = build_quality_stability_summary(raw_dir.parent)
    row = summary.iloc[0]
    assert row["dataset"] == "lfr"
    assert row["mu_nominal"] == pytest.approx(0.3)
    assert row["mu_empirical"] == pytest.approx(0.28)
    assert row["instance"] == 2


# ---------------------------------------------------------------------------
# 11. Stage 7 never executes a community-detection algorithm
# ---------------------------------------------------------------------------

def test_quality_stability_never_calls_an_algorithm(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5)
    _run_stage6(raw_dir)

    import src.algorithms as algorithms_mod

    def _boom(*a, **k):
        raise AssertionError("Stage 7 must never execute a community-detection algorithm")

    original = dict(algorithms_mod.ALGORITHMS)
    algorithms_mod.ALGORITHMS.update({name: _boom for name in original})
    try:
        summary, corr = run_quality_stability_analysis({}, results_dir=raw_dir.parent, verbose=False)
        assert len(summary) == 1
    finally:
        algorithms_mod.ALGORITHMS.clear()
        algorithms_mod.ALGORITHMS.update(original)


# ---------------------------------------------------------------------------
# 12. Stage 7 does not modify raw Stage 5/6 results
# ---------------------------------------------------------------------------

def test_quality_stability_does_not_modify_raw_or_stage6_results(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5)
    _run_stage6(raw_dir)

    results_dir = raw_dir.parent
    proc_dir = results_dir / "processed"
    before_raw = (raw_dir / "runs.csv").read_text()
    before_stab = (proc_dir / "stability_summary.csv").read_text()
    before_pairwise = (proc_dir / "pairwise.csv").read_text()
    before_mtimes = {p: p.stat().st_mtime_ns for p in raw_dir.rglob("*") if p.is_file()}

    run_quality_stability_analysis({}, results_dir=results_dir, verbose=False)

    assert (raw_dir / "runs.csv").read_text() == before_raw
    assert (proc_dir / "stability_summary.csv").read_text() == before_stab
    assert (proc_dir / "pairwise.csv").read_text() == before_pairwise
    after_mtimes = {p: p.stat().st_mtime_ns for p in raw_dir.rglob("*") if p.is_file()}
    assert before_mtimes == after_mtimes


# ---------------------------------------------------------------------------
# 13. Re-running Stage 7 is deterministic
# ---------------------------------------------------------------------------

def test_quality_stability_is_deterministic(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 6, quality_fn=lambda i: 0.05 * i, nmi_fn=lambda i: 0.1 * i)
    _write_n_cover_runs(raw_dir, "g2", "slpa", 6, quality_fn=lambda i: 0.05 * i)
    _run_stage6(raw_dir)

    results_dir = raw_dir.parent
    s1, c1 = run_quality_stability_analysis({}, results_dir=results_dir, verbose=False)
    s2, c2 = run_quality_stability_analysis({}, results_dir=results_dir, verbose=False)
    pd.testing.assert_frame_equal(s1, s2)
    pd.testing.assert_frame_equal(c1, c2)


def test_recomputable_after_deleting_stage7_outputs(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 6, quality_fn=lambda i: 0.05 * i)
    _run_stage6(raw_dir)
    results_dir = raw_dir.parent
    proc_dir = results_dir / "processed"

    run_quality_stability_analysis({}, results_dir=results_dir, verbose=False)
    first_summary = (proc_dir / "quality_stability_summary.csv").read_text()
    first_corr = (proc_dir / "quality_stability_correlations.csv").read_text()

    (proc_dir / "quality_stability_summary.csv").unlink()
    (proc_dir / "quality_stability_correlations.csv").unlink()

    run_quality_stability_analysis({}, results_dir=results_dir, verbose=False)
    assert (proc_dir / "quality_stability_summary.csv").read_text() == first_summary
    assert (proc_dir / "quality_stability_correlations.csv").read_text() == first_corr


# ---------------------------------------------------------------------------
# writes both output files with the expected columns present
# ---------------------------------------------------------------------------

def test_output_files_written_with_expected_columns(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 5, quality_fn=lambda i: 0.1 * i)
    _run_stage6(raw_dir)
    results_dir = raw_dir.parent
    proc_dir = results_dir / "processed"

    run_quality_stability_analysis({}, results_dir=results_dir, verbose=False)
    assert (proc_dir / "quality_stability_summary.csv").exists()
    assert (proc_dir / "quality_stability_correlations.csv").exists()

    summary_written = pd.read_csv(proc_dir / "quality_stability_summary.csv")
    for col in ["graph_id", "dataset", "algorithm", "mean_Q", "mean_Omega",
                "degenerate_run_count", "degenerate_pair_count"]:
        assert col in summary_written.columns


# ---------------------------------------------------------------------------
# --graphs / --algorithms filtering (CLI-level support)
# ---------------------------------------------------------------------------

def test_graph_and_algorithm_filters_restrict_output(tmp_path):
    raw_dir = tmp_path / "raw"
    _write_n_hard_runs(raw_dir, "g1", "lpa", 4)
    _write_n_hard_runs(raw_dir, "g2", "flpa", 4)
    _run_stage6(raw_dir)
    results_dir = raw_dir.parent

    summary, _ = run_quality_stability_analysis({}, graph_ids=["g1"], results_dir=results_dir, verbose=False)
    assert set(summary["graph_id"]) == {"g1"}

    summary2, _ = run_quality_stability_analysis({}, algorithms=["flpa"], results_dir=results_dir, verbose=False)
    assert set(summary2["algorithm"]) == {"flpa"}
