"""Tests for Stage 9: statistical analysis layer."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.experiments.statistical_analysis import (
    _friedman_test,
    _holm_correction,
    _rank_biserial_effect_size,
    _wilcoxon_test,
    _construct_algorithm_metric_matrix,
    run_lfr_statistical_analysis,
    run_real_network_descriptive_analysis,
    run_statistical_analysis
)
from src.experiments.storage import atomic_write_text


def test_friedman_test_basic():
    """Test Friedman test with known values."""
    # Three related samples with clear differences
    sample_a = [1, 2, 3, 4, 5]
    sample_b = [2, 3, 4, 5, 6]
    sample_c = [3, 4, 5, 6, 7]

    statistic, p_value = _friedman_test(sample_a, sample_b, sample_c)

    # Should detect significant difference
    assert isinstance(statistic, float)
    assert 0 <= p_value <= 1
    assert p_value < 0.05  # Should be significant


def test_wilcoxon_test_basic():
    """Test Wilcoxon signed-rank test."""
    # Two related samples
    sample_a = [1, 2, 3, 4, 5]
    sample_b = [1, 2, 3, 4, 6]  # Only last element different

    statistic, p_value = _wilcoxon_test(sample_a, sample_b)

    assert isinstance(statistic, float)
    assert 0 <= p_value <= 1


def test_holm_correction_basic():
    """Test Holm step-down correction."""
    p_values = [0.01, 0.04, 0.03, 0.005]
    corrected = _holm_correction(p_values)

    # Should be same length
    assert len(corrected) == len(p_values)

    # All corrected p-values should be >= original (conservative)
    for orig, corr in zip(p_values, corrected):
        assert corr >= orig or corr == 1.0  # Holm can be less conservative for larger p-values

    # First (smallest) p-value should be multiplied by number of tests
    assert abs(corrected[3] - 0.005 * 4) < 1e-10  # 0.005 is smallest, should be * 4


def test_rank_biserial_effect_size():
    """Test rank-biserial effect size calculation."""
    # Perfect separation: all pairs in sample_a > sample_b
    # Wilcoxon statistic should be 0 (all negative ranks)
    effect_size = _rank_biserial_effect_size(0.0, 5)
    assert effect_size == -1.0  # Maximum negative effect

    # Perfect separation: all pairs in sample_b > sample_a
    # Wilcoxon statistic should be n*(n+1)/2 = 15 for n=5
    effect_size = _rank_biserial_effect_size(15.0, 5)
    assert effect_size == 1.0  # Maximum positive effect

    # No difference: statistic should be n*(n+1)/4 = 7.5 for n=5
    effect_size = _rank_biserial_effect_size(7.5, 5)
    assert abs(effect_size) < 1e-10  # Should be near zero


def test_construct_algorithm_metric_matrix():
    """Test construction of algorithm metric matrix."""
    # Create test data
    df = pd.DataFrame({
        'graph_id': ['g1', 'g1', 'g2', 'g2', 'g3', 'g3'],
        'algorithm': ['lpa', 'flpa', 'lpa', 'flpa', 'lpa', 'flpa'],
        'mean_VI': [0.5, 0.3, 0.4, 0.2, 0.6, 0.4]
    })

    result = _construct_algorithm_metric_matrix(df, 'mean_VI', ['lpa', 'flpa'])

    assert 'lpa' in result
    assert 'flpa' in result
    assert len(result['lpa']) == 3  # Three graph instances
    assert len(result['flpa']) == 3
    assert set(result['lpa']) == {0.5, 0.4, 0.6}
    assert set(result['flpa']) == {0.3, 0.2, 0.4}


def test_run_statistical_analysis_no_data():
    """Test statistical analysis with no input data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)

        # Create minimal config
        cfg = {
            "paths": {"results_dir": str(results_dir)},
            "real_datasets": {"karate": {"file": "karate.txt"}},
            "lfr": {"mu_values": [0.1, 0.2]}
        }

        friedman, posthoc, descriptive = run_statistical_analysis(cfg)

        # Should return empty DataFrames with correct structure
        assert isinstance(friedman, pd.DataFrame)
        assert isinstance(posthoc, pd.DataFrame)
        assert isinstance(descriptive, pd.DataFrame)
        assert len(friedman) == 0
        assert len(posthoc) == 0
        assert len(descriptive) == 0


def test_run_lfr_statistical_analysis_with_test_fixture():
    """Test LFR statistical analysis with a small deterministic fixture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        processed_dir = results_dir / "processed"
        processed_dir.mkdir()

        # Create a mock qs_summary.csv with test data
        # Three graph instances, three algorithms, with clear differences
        test_data = []
        graph_ids = ['lfr_mu10_i00', 'lfr_mu10_i01', 'lfr_mu10_i02']
        algorithms = ['lpa', 'flpa', 'slpa']

        # Create data where flpa has lowest VI (most stable), lpa medium, slpa highest (least stable)
        vi_values = {
            'lpa': [0.8, 0.7, 0.9],    # Medium stability
            'flpa': [0.2, 0.1, 0.3],   # High stability (low VI)
            'slpa': [1.2, 1.1, 1.3]    # Low stability (high VI)
        }

        for graph_id in graph_ids:
            for alg in algorithms:
                test_data.append({
                    'graph_id': graph_id,
                    'dataset': 'lfr',
                    'algorithm': alg,
                    'mean_VI': vi_values[alg][graph_ids.index(graph_id)],
                    'mean_Omega': [0.2, 0.1, 0.15][algorithms.index(alg)],  # Placeholder
                    'mean_modularity': [0.3, 0.5, 0.4][algorithms.index(alg)]  # Placeholder
                })

        df_test = pd.DataFrame(test_data)
        atomic_write_text(processed_dir / "qs_summary.csv", df_test.to_csv(index=False))

        cfg = {
            "paths": {"results_dir": str(results_dir)},
            "real_datasets": {"karate": {"file": "karate.txt"}},
            "lfr": {"mu_values": [0.1, 0.2]}
        }

        friedman, posthoc = run_lfr_statistical_analysis(cfg)

        # Should have results for VI metric
        assert len(friedman) >= 1
        vi_results = friedman[friedman['metric'] == 'VI']
        assert len(vi_results) == 1

        # Should be significant (p < 0.05) due to clear differences
        p_value = vi_results.iloc[0]['p_value']
        assert p_value < 0.05

        # Should have post-hoc results
        assert len(posthoc) >= 3  # 3 choose 2 = 3 pairs

        # Check that effect sizes are calculated
        assert not posthoc['effect_size'].isna().all()


def test_run_real_network_descriptive_analysis_with_test_fixture():
    """Test real network descriptive analysis with test fixture."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        processed_dir = results_dir / "processed"
        processed_dir.mkdir()

        # Create mock qs_summary.csv with real network data
        test_data = []
        graph_ids = ['karate', 'dolphins']
        algorithms = ['lpa', 'flpa']

        for graph_id in graph_ids:
            for alg in algorithms:
                test_data.append({
                    'graph_id': graph_id,
                    'dataset': 'real',  # Mark as real network
                    'algorithm': alg,
                    'mean_VI': [0.5, 0.6][graph_ids.index(graph_id)] if alg == 'lpa' else [0.3, 0.4][graph_ids.index(graph_id)],
                    'mean_Omega': [0.6, 0.5][graph_ids.index(graph_id)] if alg == 'lpa' else [0.7, 0.6][graph_ids.index(graph_id)],
                    'mean_modularity': [0.4, 0.35][graph_ids.index(graph_id)] if alg == 'lpa' else [0.45, 0.4][graph_ids.index(graph_id)]
                })

        df_test = pd.DataFrame(test_data)
        atomic_write_text(processed_dir / "qs_summary.csv", df_test.to_csv(index=False))

        cfg = {
            "paths": {"results_dir": str(results_dir)},
            "real_datasets": {"karate": {"file": "karate.txt"}, "dolphins": {"file": "dolphins.txt"}},
            "lfr": {"mu_values": [0.1, 0.2]}
        }

        descriptive = run_real_network_descriptive_analysis(cfg)

        # Should have descriptive statistics
        assert len(descriptive) >= 1

        # Should have rows for each graph-algorithm-metric combination
        karate_lpa_vi = descriptive[
            (descriptive['graph_id'] == 'karate') &
            (descriptive['algorithm'] == 'lpa') &
            (descriptive['metric'] == 'VI')
        ]
        assert len(karate_lpa_vi) == 1
        assert karate_lpa_vi.iloc[0]['mean'] == 0.5
        assert karate_lpa_vi.iloc[0]['n_observations'] == 1


def test_statistical_analysis_read_only():
    """Test that statistical analysis doesn't modify input data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        processed_dir = results_dir / "processed"
        processed_dir.mkdir()

        # Create test data
        original_data = pd.DataFrame({
            'graph_id': ['g1', 'g1', 'g2', 'g2'],
            'algorithm': ['lpa', 'flpa', 'lpa', 'flpa'],
            'dataset': ['lfr', 'lfr', 'lfr', 'lfr'],
            'mean_VI': [0.5, 0.3, 0.4, 0.2],
            'mean_Omega': [0.6, 0.8, 0.7, 0.9],
            'mean_modularity': [0.4, 0.5, 0.45, 0.55]
        })
        atomic_write_text(processed_dir / "qs_summary.csv", original_data.to_csv(index=False))

        # Store original file content
        original_content = (processed_dir / "qs_summary.csv").read_text()

        cfg = {
            "paths": {"results_dir": str(results_dir)},
            "real_datasets": {"karate": {"file": "karate.txt"}},
            "lfr": {"mu_values": [0.1, 0.2]}
        }

        # Run analysis
        run_statistical_analysis(cfg)

        # Check that file wasn't modified
        new_content = (processed_dir / "qs_summary.csv").read_text()
        assert original_content == new_content


def test_statistical_analysis_determinitive():
    """Test that statistical analysis is deterministic."""
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = Path(tmpdir)
        processed_dir = results_dir / "processed"
        processed_dir.mkdir()

        # Create test data
        test_data = pd.DataFrame({
            'graph_id': ['g1', 'g1', 'g2', 'g2', 'g3', 'g3'],
            'algorithm': ['lpa', 'flpa', 'lpa', 'flpa', 'lpa', 'flpa'],
            'dataset': ['lfr', 'lfr', 'lfr', 'lfr', 'lfr', 'lfr'],
            'mean_VI': [0.5, 0.3, 0.4, 0.2, 0.6, 0.4],
            'mean_Omega': [0.6, 0.8, 0.7, 0.9, 0.5, 0.7],
            'mean_modularity': [0.4, 0.5, 0.45, 0.55, 0.4, 0.45]
        })
        atomic_write_text(processed_dir / "qs_summary.csv", test_data.to_csv(index=False))

        cfg = {
            "paths": {"results_dir": str(results_dir)},
            "real_datasets": {"karate": {"file": "karate.txt"}},
            "lfr": {"mu_values": [0.1, 0.2]}
        }

        # Run analysis twice
        friedman1, posthoc1, desc1 = run_statistical_analysis(cfg)
        friedman2, posthoc2, desc2 = run_statistical_analysis(cfg)

        # Should be identical
        pd.testing.assert_frame_equal(friedman1, friedman2)
        pd.testing.assert_frame_equal(posthoc1, posthoc2)
        pd.testing.assert_frame_equal(desc1, desc2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])