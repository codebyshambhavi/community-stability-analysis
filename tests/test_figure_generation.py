"""Stage 10 tests: final figure/table generation.

Built on small, deterministic fixtures written directly in the on-disk shape Stages
1/5/6/7/9 leave behind (mirrors tests/test_quality_stability.py and
tests/test_statistical_analysis.py) -- never the full 8,880-run matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments import figure_generation as fg
from src.experiments.storage import atomic_write_text

TINY_CFG = {
    "runs": {"n_runs": 5, "seed_start": 0},
    "paths": {"data_dir": "data", "raw_dir": "data/raw", "lfr_dir": "data/lfr", "results_dir": "results"},
    "real_datasets": ["karate", "dolphins"],
    "lfr": {"n_instances": 2, "mu_levels": [0.1, 0.2]},
    "slpa": {"T": 20, "r": 0.1},
}


# ---------------------------------------------------------------------------
# Fixture builders (write processed CSVs directly -- no algorithm ever runs)
# ---------------------------------------------------------------------------

def _write_stability_summary(results_dir: Path, rows: list[dict]) -> None:
    proc = results_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    atomic_write_text(proc / "stability_summary.csv", df.to_csv(index=False))


def _write_qs_summary(results_dir: Path, rows: list[dict]) -> None:
    proc = results_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    atomic_write_text(proc / "quality_stability_summary.csv", df.to_csv(index=False))


def _write_runs(results_dir: Path, rows: list[dict]) -> None:
    raw = results_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=[
        "graph_id", "dataset", "mu_nominal", "mu_empirical", "instance", "algorithm", "run", "seed",
        "number_of_communities", "quality", "quality_metric", "nmi", "onmi", "f1",
        "runtime_seconds", "largest_community_size", "degenerate", "converged", "iterations",
        "partition_file", "partition_kind",
    ])
    atomic_write_text(raw / "runs.csv", df.to_csv(index=False))


def _write_preprocessing_log(results_dir: Path, log: dict) -> None:
    proc = results_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    (proc / "preprocessing_log.json").write_text(json.dumps(log), encoding="utf-8")


def _write_lfr_instances(results_dir: Path, rows: list[dict]) -> None:
    proc = results_dir / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    atomic_write_text(proc / "lfr_instances.csv", df.to_csv(index=False))


def _hard_stability_row(graph_id, algorithm, dataset="karate", mu_nominal=np.nan, mu_empirical=np.nan,
                         instance=np.nan, mean_omega=0.8, mean_vi=0.2, mean_nmi=0.85,
                         mean_ncomm=3.0, degenerate=0):
    return {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": mu_nominal, "mu_empirical": mu_empirical,
        "instance": instance, "algorithm": algorithm, "number_of_runs": 5, "number_of_pairs": 10,
        "mean_VI": mean_vi, "median_VI": mean_vi, "std_VI": 0.01, "mean_normalized_VI": mean_vi / 2,
        "mean_NMI": mean_nmi, "std_NMI": 0.01, "mean_Omega": mean_omega, "std_Omega": 0.01,
        "mean_ONMI": np.nan, "std_ONMI": np.nan,
        "degenerate_run_count": degenerate, "degenerate_pair_count": 0,
        "mean_number_of_communities": mean_ncomm, "std_number_of_communities": 0.5,
        "n_bootstrap": 100,
    }


def _cover_stability_row(graph_id, algorithm="slpa", dataset="karate", mean_omega=0.7, mean_onmi=0.6,
                          mean_ncomm=4.0):
    return {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": np.nan, "mu_empirical": np.nan,
        "instance": np.nan, "algorithm": algorithm, "number_of_runs": 5, "number_of_pairs": 10,
        "mean_VI": np.nan, "median_VI": np.nan, "std_VI": np.nan, "mean_normalized_VI": np.nan,
        "mean_NMI": np.nan, "std_NMI": np.nan, "mean_Omega": mean_omega, "std_Omega": 0.01,
        "mean_ONMI": mean_onmi, "std_ONMI": 0.01,
        "degenerate_run_count": 0, "degenerate_pair_count": 0,
        "mean_number_of_communities": mean_ncomm, "std_number_of_communities": 0.4,
        "n_bootstrap": 100,
    }


def _qs_row(graph_id, algorithm, dataset="karate", mu_nominal=np.nan, mu_empirical=np.nan,
            mean_q=0.4, mean_eq=np.nan, mean_omega=0.8, mean_community_count=3.0,
            ground_truth_nmi=np.nan):
    return {
        "graph_id": graph_id, "dataset": dataset, "mu_nominal": mu_nominal, "mu_empirical": mu_empirical,
        "instance": np.nan, "algorithm": algorithm,
        "mean_Q": mean_q, "std_Q": 0.02, "mean_EQ": mean_eq, "std_EQ": np.nan,
        "mean_ground_truth_NMI": ground_truth_nmi, "std_ground_truth_NMI": np.nan,
        "mean_ground_truth_ONMI": np.nan, "std_ground_truth_ONMI": np.nan,
        "n_quality_runs": 5,
        "mean_stability_VI": 0.2, "mean_stability_normalized_VI": 0.1, "mean_stability_NMI": 0.85,
        "mean_Omega": mean_omega, "mean_ONMI": np.nan,
        "stability_VI_ci_low": 0.1, "stability_VI_ci_high": 0.3,
        "stability_normalized_VI_ci_low": np.nan, "stability_normalized_VI_ci_high": np.nan,
        "stability_NMI_ci_low": np.nan, "stability_NMI_ci_high": np.nan,
        "stability_Omega_ci_low": mean_omega - 0.05, "stability_Omega_ci_high": mean_omega + 0.05,
        "stability_ONMI_ci_low": np.nan, "stability_ONMI_ci_high": np.nan,
        "mean_community_count": mean_community_count, "std_community_count": 0.5,
        "number_of_stability_runs": 5, "number_of_stability_pairs": 10,
        "degenerate_run_count": 0, "degenerate_pair_count": 0,
    }


# ---------------------------------------------------------------------------
# 1. Figure-generation helpers: valid data produces a file
# ---------------------------------------------------------------------------

def test_fig_stability_comparison_written_with_data(tmp_path):
    summary = pd.DataFrame([
        _hard_stability_row("karate", "lpa", mean_omega=0.9),
        _hard_stability_row("karate", "flpa", mean_omega=0.7),
    ])
    out = tmp_path / "fig01.png"
    assert fg.fig_stability_comparison(summary, out) is True
    assert out.exists() and out.stat().st_size > 0


def test_fig_quality_comparison_written_with_data(tmp_path):
    qs = pd.DataFrame([
        _qs_row("karate", "lpa", mean_q=0.4),
        _qs_row("karate", "slpa", mean_q=np.nan, mean_eq=0.35),
    ])
    out = tmp_path / "fig02.png"
    assert fg.fig_quality_comparison(qs, out) is True
    assert out.exists()


def test_fig_stability_quality_relationship_written_with_data(tmp_path):
    qs = pd.DataFrame([_qs_row("karate", "lpa"), _qs_row("dolphins", "lpa", dataset="dolphins")])
    out = tmp_path / "fig03.png"
    assert fg.fig_stability_quality_relationship(qs, out) is True
    assert out.exists()


def test_fig_lfr_mu_stability_uses_empirical_mu_when_present(tmp_path):
    qs = pd.DataFrame([
        _qs_row("lfr_mu10_i00", "lpa", dataset="lfr", mu_nominal=0.1, mu_empirical=0.11, mean_omega=0.9),
        _qs_row("lfr_mu20_i00", "lpa", dataset="lfr", mu_nominal=0.2, mu_empirical=0.21, mean_omega=0.6),
    ])
    out = tmp_path / "fig04.png"
    assert fg.fig_lfr_mu_stability(qs, out) is True
    assert out.exists()


def test_fig_lfr_mu_quality_written_with_data(tmp_path):
    qs = pd.DataFrame([
        _qs_row("lfr_mu10_i00", "lpa", dataset="lfr", mu_nominal=0.1, mu_empirical=0.11, mean_q=0.5),
        _qs_row("lfr_mu20_i00", "lpa", dataset="lfr", mu_nominal=0.2, mu_empirical=0.21, mean_q=0.4),
    ])
    out = tmp_path / "fig05.png"
    assert fg.fig_lfr_mu_quality(qs, out) is True
    assert out.exists()


def test_fig_community_structure_written_with_data(tmp_path):
    qs = pd.DataFrame([_qs_row("karate", "lpa", mean_community_count=3.0),
                       _qs_row("dolphins", "lpa", dataset="dolphins", mean_community_count=5.0)])
    out = tmp_path / "fig06.png"
    assert fg.fig_community_structure(qs, out) is True
    assert out.exists()


def test_fig_real_network_comparison_excludes_lfr(tmp_path):
    qs = pd.DataFrame([
        _qs_row("karate", "lpa", dataset="karate", mean_omega=0.9),
        _qs_row("lfr_mu10_i00", "lpa", dataset="lfr", mu_nominal=0.1, mean_omega=0.1),
    ])
    out = tmp_path / "fig07.png"
    assert fg.fig_real_network_comparison(qs, out) is True
    assert out.exists()


def test_fig_experiment_completeness_always_meaningful_even_when_empty(tmp_path):
    """The completeness figure is the one figure that must still be written when there is
    literally no experimental data at all (0 of N runs stored)."""
    empty_runs = fg.load_runs(tmp_path)  # no results dir written yet -> empty schema
    out = tmp_path / "fig08.png"
    assert fg.fig_experiment_completeness(empty_runs, TINY_CFG, out) is True
    assert out.exists()


# ---------------------------------------------------------------------------
# 7. Missing/empty input handling: never raises, never fabricates a plot
# ---------------------------------------------------------------------------

def test_figures_skip_gracefully_on_empty_data(tmp_path):
    empty = pd.DataFrame()
    assert fg.fig_stability_comparison(empty, tmp_path / "a.png") is False
    assert fg.fig_quality_comparison(empty, tmp_path / "b.png") is False
    assert fg.fig_stability_quality_relationship(empty, tmp_path / "c.png") is False
    assert fg.fig_lfr_mu_stability(empty, tmp_path / "d.png") is False
    assert fg.fig_lfr_mu_quality(empty, tmp_path / "e.png") is False
    assert fg.fig_community_structure(empty, tmp_path / "f.png") is False
    assert fg.fig_real_network_comparison(empty, tmp_path / "g.png") is False
    for p in ["a", "b", "c", "d", "e", "f", "g"]:
        assert not (tmp_path / f"{p}.png").exists()


# ---------------------------------------------------------------------------
# 6. Hard vs overlapping metric handling
# ---------------------------------------------------------------------------

def test_quality_col_respects_algorithm_scope():
    assert fg._quality_col("lpa") == "mean_Q"
    assert fg._quality_col("semi_sync_lpa") == "mean_Q"
    assert fg._quality_col("flpa") == "mean_Q"
    assert fg._quality_col("slpa") == "mean_EQ"


def test_table_quality_statistics_never_mixes_q_and_eq(tmp_path):
    qs = pd.DataFrame([
        _qs_row("karate", "lpa", mean_q=0.4, mean_eq=np.nan),
        _qs_row("karate", "slpa", mean_q=np.nan, mean_eq=0.3),
    ])
    table = fg.table_quality_statistics(qs)
    lpa_row = table[table["algorithm"] == "lpa"].iloc[0]
    slpa_row = table[table["algorithm"] == "slpa"].iloc[0]
    assert lpa_row["quality_metric"] == "Q"
    assert slpa_row["quality_metric"] == "EQ"


def test_table_stability_statistics_leaves_out_of_scope_metrics_absent(tmp_path):
    summary = pd.DataFrame([
        _hard_stability_row("karate", "lpa"),
        _cover_stability_row("karate", "slpa"),
    ])
    table = fg.table_stability_statistics(summary)
    slpa_metrics = set(table[table["algorithm"] == "slpa"]["metric"])
    lpa_metrics = set(table[table["algorithm"] == "lpa"]["metric"])
    assert "VI" not in slpa_metrics and "NMI" not in slpa_metrics  # never fabricated for SLPA
    assert "ONMI" in slpa_metrics
    assert "VI" in lpa_metrics and "ONMI" not in lpa_metrics


# ---------------------------------------------------------------------------
# 5. Real vs LFR separation
# ---------------------------------------------------------------------------

def test_lfr_mu_figures_ignore_real_network_rows(tmp_path):
    qs = pd.DataFrame([_qs_row("karate", "lpa", dataset="karate", mu_nominal=np.nan)])
    out = tmp_path / "fig04.png"
    # only a real-network row is present -> no LFR mu data -> must skip, not crash
    assert fg.fig_lfr_mu_stability(qs, out) is False


def test_table_lfr_mu_analysis_only_uses_lfr_rows(tmp_path):
    qs = pd.DataFrame([
        _qs_row("karate", "lpa", dataset="karate", mean_q=0.9),
        _qs_row("lfr_mu10_i00", "lpa", dataset="lfr", mu_nominal=0.1, mu_empirical=0.11, mean_q=0.4),
    ])
    table = fg.table_lfr_mu_analysis(qs)
    assert set(table["mu_nominal"]) == {0.1}
    assert 0.9 not in set(table["mean_quality"])  # the karate (real) row must not leak in


# ---------------------------------------------------------------------------
# 2/3/4. Table-generation helpers: correct output paths + column names
# ---------------------------------------------------------------------------

def test_table_dataset_characteristics_columns_and_real_plus_lfr(tmp_path):
    results_dir = tmp_path / "results"
    _write_preprocessing_log(results_dir, {
        "karate": {"final_n": 34, "final_m": 78, "ground_truth": "networkx karate club"},
        "dolphins": {"final_n": 62, "final_m": 159, "ground_truth": "none"},
    })
    _write_lfr_instances(results_dir, [
        {"graph_id": "lfr_mu10_i00", "mu_nominal": 0.1, "mu_empirical": 0.11, "n": 1000, "m": 5000},
        {"graph_id": "lfr_mu10_i01", "mu_nominal": 0.1, "mu_empirical": 0.12, "n": 1000, "m": 5010},
    ])
    table = fg.table_dataset_characteristics(TINY_CFG, results_dir)
    assert list(table.columns) == ["graph_id", "family", "n", "m", "mu_nominal",
                                    "mu_empirical_mean", "n_instances", "ground_truth"]
    assert set(table["family"]) == {"real", "lfr"}
    assert "karate" in set(table["graph_id"])


def test_table_algorithm_characteristics_covers_all_four(tmp_path):
    table = fg.table_algorithm_characteristics(TINY_CFG)
    assert set(table["algorithm"]) == {"lpa", "semi_sync_lpa", "flpa", "slpa"}
    slpa_row = table[table["algorithm"] == "slpa"].iloc[0]
    assert slpa_row["partition_type"] == "overlapping"
    assert slpa_row["config_T"] == 20
    lpa_row = table[table["algorithm"] == "lpa"].iloc[0]
    assert lpa_row["partition_type"] == "hard"
    assert pd.isna(lpa_row["config_T"])  # SLPA-only knob never fabricated for LPA


def test_table_statistical_significance_concatenates_with_analysis_column(tmp_path):
    friedman = pd.DataFrame([{"metric": "Omega", "statistic": 4.2, "p_value": 0.03, "n_blocks": 10,
                               "algorithms_tested": "lpa;flpa", "test_type": "Friedman", "notes": "ok"}])
    posthoc = pd.DataFrame([{"metric": "Omega", "algorithm_a": "lpa", "algorithm_b": "flpa",
                              "n_blocks": 10, "wilcoxon_statistic": 3.0, "p_value_raw": 0.02,
                              "p_value_corrected": 0.02, "correction_method": "Holm",
                              "effect_size": 0.4, "effect_size_type": "rank_biserial"}])
    table = fg.table_statistical_significance(friedman, posthoc)
    assert set(table["analysis"]) == {"friedman_omnibus", "wilcoxon_posthoc_holm"}
    assert len(table) == 2


def test_table_statistical_significance_empty_when_no_stage9_output():
    table = fg.table_statistical_significance(pd.DataFrame(), pd.DataFrame())
    assert table.empty


# ---------------------------------------------------------------------------
# 7 (table side): experiment completeness always has a row, even at zero
# ---------------------------------------------------------------------------

def test_table_experiment_completeness_all_zero_when_no_runs(tmp_path):
    empty_runs = fg.load_runs(tmp_path)
    table = fg.table_experiment_completeness(empty_runs, TINY_CFG)
    assert (table["runs_completed"] == 0).all()
    assert (table["fraction_complete"] == 0).all()
    assert set(table["algorithm"]) == {"lpa", "semi_sync_lpa", "flpa", "slpa"}


def test_table_experiment_completeness_never_ranks_algorithms(tmp_path):
    """No 'winner'/'rank' column should ever be produced -- per the no-fabrication rule."""
    empty_runs = fg.load_runs(tmp_path)
    table = fg.table_experiment_completeness(empty_runs, TINY_CFG)
    assert "rank" not in table.columns and "winner" not in table.columns


# ---------------------------------------------------------------------------
# 8/9. Empty-data top-level behavior + deterministic output
# ---------------------------------------------------------------------------

def test_make_final_tables_on_completely_empty_results_only_writes_static_and_completeness(tmp_path):
    """table02 (algorithm characteristics) is static config metadata, not an experimental
    result, so it is always available; table07 (completeness) is always meaningful too.
    Every other table needs an actual Stage 5-9 output and must be skipped here."""
    results_dir = tmp_path / "results"
    written = fg.make_final_tables(TINY_CFG, results_dir)
    names = {p.name for p in written}
    assert names == {"table02_algorithm_characteristics.csv", "table07_experiment_completeness.csv"}
    assert (results_dir / "tables" / "final" / "table07_experiment_completeness.csv").exists()


def test_make_final_figures_on_completely_empty_results_only_writes_completeness(tmp_path):
    results_dir = tmp_path / "results"
    written = fg.make_final_figures(TINY_CFG, results_dir)
    names = {p.name for p in written}
    assert names == {"fig08_experiment_completeness.png"}


def test_table_generation_is_deterministic(tmp_path):
    results_dir = tmp_path / "results"
    _write_stability_summary(results_dir, [_hard_stability_row("karate", "lpa")])
    _write_qs_summary(results_dir, [_qs_row("karate", "lpa")])
    written1 = fg.make_final_tables(TINY_CFG, results_dir)
    contents1 = {p.name: p.read_text() for p in written1}
    written2 = fg.make_final_tables(TINY_CFG, results_dir)
    contents2 = {p.name: p.read_text() for p in written2}
    assert contents1 == contents2


# ---------------------------------------------------------------------------
# 3. Correct output paths (custom figures_dir / tables_dir honored)
# ---------------------------------------------------------------------------

def test_make_final_figures_respects_custom_dir(tmp_path):
    results_dir = tmp_path / "results"
    custom_dir = tmp_path / "somewhere_else"
    _write_stability_summary(results_dir, [_hard_stability_row("karate", "lpa"),
                                            _hard_stability_row("karate", "flpa", mean_omega=0.5)])
    written = fg.make_final_figures(TINY_CFG, results_dir, figures_dir=custom_dir)
    assert all(p.parent == custom_dir for p in written)
    assert any(p.name == "fig01_stability_comparison.png" for p in written)


def test_make_final_tables_respects_custom_dir(tmp_path):
    results_dir = tmp_path / "results"
    custom_dir = tmp_path / "tables_elsewhere"
    written = fg.make_final_tables(TINY_CFG, results_dir, tables_dir=custom_dir)
    assert all(p.parent == custom_dir for p in written)


# ---------------------------------------------------------------------------
# 10/11/12. Read-only behavior: never touches results/raw/, never runs an
# algorithm, never regenerates an LFR graph
# ---------------------------------------------------------------------------

def test_run_figure_generation_never_writes_under_raw_or_data(tmp_path):
    results_dir = tmp_path / "results"
    _write_runs(results_dir, [{
        "graph_id": "karate", "dataset": "karate", "mu_nominal": np.nan, "mu_empirical": np.nan,
        "instance": np.nan, "algorithm": "lpa", "run": 0, "seed": 0,
        "number_of_communities": 3, "quality": 0.4, "quality_metric": "Q", "nmi": np.nan, "onmi": np.nan,
        "f1": np.nan, "runtime_seconds": 0.01, "largest_community_size": 10, "degenerate": False,
        "converged": True, "iterations": 3, "partition_file": "partitions/karate/lpa/seed000.csv",
        "partition_kind": "hard",
    }])
    raw_before = (results_dir / "raw" / "runs.csv").read_text()

    cfg = {**TINY_CFG, "paths": {**TINY_CFG["paths"], "results_dir": str(results_dir)}}
    fg.run_figure_generation(cfg, results_dir=results_dir, verbose=False)

    raw_after = (results_dir / "raw" / "runs.csv").read_text()
    assert raw_before == raw_after                      # untouched
    assert not (tmp_path / "data").exists()              # never touched/created


def test_module_never_imports_algorithms_or_lfr_generation():
    """Static check that this module has no path to invoking an algorithm or regenerating
    an LFR graph -- it should only ever read already-persisted CSV/JSON. (HARD_ALGORITHMS /
    OVERLAPPING_ALGORITHMS are just algorithm-name-scope constants, not the runnable
    registry, so they are deliberately excluded from this check.)"""
    src = Path(fg.__file__).read_text(encoding="utf-8")
    for forbidden in ("run_lpa", "run_flpa", "run_slpa", "run_semi_sync_lpa",
                      "from ..algorithms import", "from src.algorithms import",
                      "generate_all", "LFRGenerator", "run_experiments"):
        assert forbidden not in src, f"figure_generation.py must never reference {forbidden!r}"


# ---------------------------------------------------------------------------
# 13. CLI integration
# ---------------------------------------------------------------------------

def test_cli_generate_figures_help(capsys):
    import main as main_module
    with pytest.raises(SystemExit) as exc:
        main_module.main(["generate-figures", "--help"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "generate-figures" in out or "stage" in out.lower()


def test_cli_generate_figures_runs_end_to_end(tmp_path, monkeypatch):
    import main as main_module

    results_dir = tmp_path / "results"
    data_dir = tmp_path / "data"
    (data_dir / "raw").mkdir(parents=True)
    (data_dir / "lfr").mkdir(parents=True)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(f"""
runs:
  n_runs: 5
  seed_start: 0
paths:
  data_dir: {data_dir}
  raw_dir: {data_dir / 'raw'}
  lfr_dir: {data_dir / 'lfr'}
  results_dir: {results_dir}
real_datasets: [karate, dolphins]
lfr:
  n_instances: 2
  mu_levels: [0.1, 0.2]
slpa:
  T: 20
  r: 0.1
""", encoding="utf-8")

    rc = main_module.main(["generate-figures", "--config", str(config_path)])
    assert rc == 0
    assert (results_dir / "figures" / "final" / "fig08_experiment_completeness.png").exists()
    assert (results_dir / "tables" / "final" / "table07_experiment_completeness.csv").exists()


# ---------------------------------------------------------------------------
# Loaders: missing files return empty-but-typed frames, never raise
# ---------------------------------------------------------------------------

def test_loaders_return_empty_frames_when_files_absent(tmp_path):
    assert fg.load_stability_summary(tmp_path).empty
    assert fg.load_quality_stability_summary(tmp_path).empty
    assert fg.load_correlations(tmp_path).empty
    assert fg.load_lfr_instances(tmp_path).empty
    assert fg.load_statistical_tests(tmp_path).empty
    assert fg.load_posthoc_tests(tmp_path).empty
    assert fg.load_descriptive_statistics(tmp_path).empty
    assert fg.load_preprocessing_log(tmp_path) == {}
    assert fg.load_runs(tmp_path).empty
    assert list(fg.load_runs(tmp_path).columns) == list(fg.RUN_COLUMNS)
