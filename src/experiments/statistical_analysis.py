"""Stage 9: statistical analysis layer.

Implements inferential statistics for LFR networks (Friedman test with
Holm-corrected Wilcoxon post-hoc) and descriptive statistics for real
networks. Reads existing Stage 5+6 outputs; never runs algorithms or
regenerates graphs.

RESEARCH DESIGN CONSIDERATIONS:
- LFR graph instance is the repeated-measures block for inferential testing
- Individual stochastic runs and pairwise comparisons are NOT independent units
- Real networks (n=4) receive descriptive analysis only
- Proper handling of metric scopes (hard vs overlapping algorithms)
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .quality_stability import GRAPH_META_COLUMNS, HARD_ALGORITHMS, OVERLAPPING_ALGORITHMS, load_runs
from .stability import _applicable_metrics, _summary_stats_from_pairwise, SUMMARY_METRIC_STATS

# Columns identifying a graph, carried through from runs.csv unchanged
GRAPH_META_COLUMNS = ["graph_id", "dataset", "mu_nominal", "mu_empirical", "instance"]

# Output column names for statistical tests
FRIEDMAN_TEST_COLUMNS = [
    "metric", "statistic", "p_value", "n_blocks", "algorithms_tested",
    "test_type", "notes"
]

POSTHOC_TEST_COLUMNS = [
    "metric", "algorithm_a", "algorithm_b", "n_blocks",
    "wilcoxon_statistic", "p_value_raw", "p_value_corrected",
    "correction_method", "effect_size", "effect_size_type"
]

DESCRIPTIVE_STATS_COLUMNS = [
    "graph_id", "dataset", "algorithm", "metric",
    "n_observations", "mean", "median", "std", "min", "max"
]


def _is_hard_partition_algorithm(algorithm: str) -> bool:
    """Check if algorithm uses hard partitions (VI/NMI applicable)."""
    return algorithm in HARD_ALGORITHMS


def _is_overlapping_algorithm(algorithm: str) -> bool:
    """Check if algorithm uses overlapping covers (VI/NMI not applicable)."""
    return algorithm in OVERLAPPING_ALGORITHMS


def _load_stability_summary(results_dir: Path) -> pd.DataFrame:
    """Load Stage 6 stability summary from results/processed/stability_summary.csv."""
    path = results_dir / "processed" / "stability_summary.csv"
    if not path.exists():
        return pd.DataFrame(columns=GRAPH_META_COLUMNS + ["algorithm"] +
                           [f"{stat}_{met}" for met in ["VI", "normalized_VI", "NMI", "Omega", "ONMI"]
                            for stat in ["mean", "median", "std"]])
    return pd.read_csv(path)


def _load_quality_summary(results_dir: Path) -> pd.DataFrame:
    """Load Stage 7 quality summary from results/processed/quality_summary.csv."""
    path = results_dir / "processed" / "quality_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_joined_summary(results_dir: Path) -> pd.DataFrame:
    """Load Stage 7 quality-stability joined data from results/processed/qs_summary.csv."""
    # Stage 7 (quality_stability.py) writes quality_stability_summary.csv. The legacy name
    # qs_summary.csv is kept as a fallback (the Stage 9 unit tests write that name).
    proc = results_dir / "processed"
    path = proc / "quality_stability_summary.csv"
    if not path.exists():
        path = proc / "qs_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _get_applicable_stability_metrics(algorithm: str) -> list[str]:
    """Get stability metrics applicable to an algorithm."""
    if _is_hard_partition_algorithm(algorithm):
        return ["VI", "normalized_VI", "NMI", "Omega"]
    if _is_overlapping_algorithm(algorithm):
        return ["Omega", "ONMI"]
    raise ValueError(f"unknown algorithm scope for {algorithm!r}")


def _get_applicable_quality_metrics(algorithm: str) -> list[str]:
    """Get quality metrics applicable to an algorithm."""
    if _is_hard_partition_algorithm(algorithm):
        return ["modularity"]
    if _is_overlapping_algorithm(algorithm):
        return ["extended_modularity"]
    raise ValueError(f"unknown algorithm scope for {algorithm!r}")


def _friedman_test(*samples: list[float]) -> tuple[float, float]:
    """Perform Friedman test on k related samples.

    Returns:
        (statistic, p_value)
    """
    # Use scipy's friedmanchisquare which expects arrays as separate arguments
    statistic, p_value = stats.friedmanchisquare(*samples)
    return float(statistic), float(p_value)


def _wilcoxon_test(sample_a: list[float], sample_b: list[float]) -> tuple[float, float]:
    """Perform Wilcoxon signed-rank test on two related samples.

    Returns:
        (statistic, p_value)
    """
    statistic, p_value = stats.wilcoxon(sample_a, sample_b)
    return float(statistic), float(p_value)


def _holm_correction(p_values: list[float]) -> list[float]:
    """Apply Holm step-down correction for multiple testing.

    Args:
        p_values: List of raw p-values

    Returns:
        List of corrected p-values in same order as input
    """
    if not p_values:
        return []

    # Create list of (index, p_value) pairs
    indexed_p_values = list(enumerate(p_values))
    # Sort by p-value ascending
    indexed_p_values.sort(key=lambda x: x[1])

    corrected = [0.0] * len(p_values)
    n_tests = len(p_values)

    for i, (original_index, p_val) in enumerate(indexed_p_values):
        # Holm-Bonferroni: p_corrected = p_raw * (n_tests - i)
        corrected_p = min(p_val * (n_tests - i), 1.0)
        corrected[original_index] = corrected_p

    return corrected


def _rank_biserial_effect_size(wilcoxon_statistic: float, n_pairs: int) -> float:
    """Calculate rank-biserial effect size from Wilcoxon statistic.

    Args:
        wilcoxon_statistic: Wilcoxon signed-rank test statistic
        n_pairs: Number of pairs in the test

    Returns:
        Rank-biserial correlation in [-1, 1]
    """
    # For Wilcoxon signed-rank, rank-biserial = (2W)/(n*(n+1)) - 1
    # where W is the Wilcoxon statistic (sum of ranks of positive differences)
    # This gives:
    #   W = 0 (all differences negative) -> r = -1
    #   W = n*(n+1)/2 (all differences positive) -> r = +1
    #   W = n*(n+1)/4 (no net difference) -> r = 0
    if n_pairs <= 0:
        return 0.0

    max_statistic = n_pairs * (n_pairs + 1) / 2
    if max_statistic == 0:
        return 0.0

    rbc = (2 * wilcoxon_statistic) / max_statistic - 1
    # Clamp to [-1, 1] for numerical stability
    return max(-1.0, min(1.0, rbc))


def _construct_algorithm_metric_matrix(
    df: pd.DataFrame,
    metric_col: str,
    algorithm_list: list[str]
) -> dict[str, list[float]]:
    """Construct a matrix of algorithm observations across blocks (graph instances).

    Args:
        df: DataFrame with columns [graph_id, algorithm, metric_col]
        metric_col: Name of the metric column to analyze
        algorithm_list: List of algorithms to include

    Returns:
        Dictionary mapping algorithm -> list of metric values (one per block)
    """
    # Pivot to get algorithms as columns, graph instances as rows
    pivot_df = df.pivot_table(
        index="graph_id",
        columns="algorithm",
        values=metric_col,
        aggfunc="mean"  # In case of duplicates, take mean
    )

    # Extract values for each algorithm in the requested order
    result = {}
    for algorithm in algorithm_list:
        if algorithm in pivot_df.columns:
            # Drop NaN values (missing blocks for this algorithm)
            values = pivot_df[algorithm].dropna().tolist()
            result[algorithm] = values
        else:
            result[algorithm] = []

    return result


def run_lfr_statistical_analysis(
    cfg: dict,
    graph_ids: list[str] | None = None,
    algorithms: list[str] | None = None,
    results_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 9 entry point for LFR inferential analysis.

    Performs Friedman omnibus test followed by Holm-corrected Wilcoxon
    post-hoc tests for each applicable metric, using graph instance as
    the repeated-measures block.

    Args:
        cfg: Configuration dictionary
        graph_ids: Optional list of graph IDs to filter
        algorithms: Optional list of algorithms to filter
        results_dir: Optional results directory path

    Returns:
        (friedman_results_df, posthoc_results_df)
    """
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir")

    # Load the joined quality-stability summary (Stage 7 output)
    df_joined = _load_joined_summary(results_dir)

    if df_joined.empty:
        # Return empty DataFrames with proper structure
        friedman_df = pd.DataFrame(columns=FRIEDMAN_TEST_COLUMNS)
        posthoc_df = pd.DataFrame(columns=POSTHOC_TEST_COLUMNS)
        return friedman_df, posthoc_df

    # Apply filters if provided
    if graph_ids is not None:
        df_joined = df_joined[df_joined["graph_id"].isin(graph_ids)]
    if algorithms is not None:
        df_joined = df_joined[df_joined["algorithm"].isin(algorithms)]

    # Restrict to LFR graphs only (exclude real networks for inferential test)
    df_lfr = df_joined[df_joined["dataset"] == "lfr"].copy()

    if df_lfr.empty:
        friedman_df = pd.DataFrame(columns=FRIEDMAN_TEST_COLUMNS)
        posthoc_df = pd.DataFrame(columns=POSTHOC_TEST_COLUMNS)
        return friedman_df, posthoc_df

    # Get unique algorithms and metrics present in the data
    available_algorithms = sorted(df_lfr["algorithm"].unique())
    if algorithms is not None:
        available_algorithms = [alg for alg in available_algorithms if alg in algorithms]

    # Determine which metrics to test based on available algorithms
    stability_metrics_to_test = set()
    quality_metrics_to_test = set()

    for algorithm in available_algorithms:
        if _is_hard_partition_algorithm(algorithm):
            stability_metrics_to_test.update(["VI", "normalized_VI", "NMI", "Omega"])
            quality_metrics_to_test.update(["modularity"])
        elif _is_overlapping_algorithm(algorithm):
            stability_metrics_to_test.update(["Omega", "ONMI"])
            quality_metrics_to_test.update(["extended_modularity"])

    stability_metrics_to_test = sorted(stability_metrics_to_test)
    quality_metrics_to_test = sorted(quality_metrics_to_test)

    # Prepare results containers
    friedman_rows = []
    posthoc_rows = []

    # Analyze each stability metric
    for metric in stability_metrics_to_test:
        # Find the corresponding column in the joined summary
        metric_col = f"mean_{metric}"  # Using mean as the summary statistic
        if metric_col not in df_lfr.columns and f"mean_stability_{metric}" in df_lfr.columns:
            # Stage 7's joined summary names the Stage 6 VI/normalized_VI/NMI means
            # mean_stability_<metric>; same quantity, different column name.
            metric_col = f"mean_stability_{metric}"
        if metric_col not in df_lfr.columns:
            # Try to find any column with this metric
            metric_cols = [col for col in df_lfr.columns if metric in col.lower()]
            if not metric_cols:
                continue
            metric_col = metric_cols[0]  # Take the first match

        # Construct algorithm matrix for this metric
        algorithm_data = _construct_algorithm_metric_matrix(
            df_lfr[["graph_id", "algorithm", metric_col]].dropna(),
            metric_col,
            available_algorithms
        )

        # Check if we have enough data for Friedman test (≥3 algorithms, ≥3 blocks)
        algorithms_with_data = [alg for alg, vals in algorithm_data.items() if len(vals) >= 3]
        if len(algorithms_with_data) < 3:
            # Not enough data for meaningful test
            friedman_rows.append({
                "metric": metric,
                "statistic": np.nan,
                "p_value": np.nan,
                "n_blocks": 0,
                "algorithms_typed": ";".join(available_algorithms),
                "test_type": "Friedman",
                "notes": "Insufficient data for Friedman test (need ≥3 algorithms with ≥3 blocks each)"
            })
            continue

        # Prepare samples for Friedman test (one list per algorithm)
        samples = [algorithm_data[alg] for alg in algorithms_with_data]

        # Check if all algorithms have the same number of blocks
        block_counts = [len(vals) for vals in samples]
        if len(set(block_counts)) > 1:
            # Different numbers of blocks - this shouldn't happen with proper data
            # but handle gracefully by using the minimum count
            min_blocks = min(block_counts)
            samples = [vals[:min_blocks] for vals in samples]
            note = f"Truncated to {min_blocks} blocks per algorithm due to uneven replication"
        else:
            note = f"{block_counts[0]} blocks per algorithm"

        # Perform Friedman test
        try:
            statistic, p_value = _friedman_test(*samples)
            friedman_rows.append({
                "metric": metric,
                "statistic": statistic,
                "p_value": p_value,
                "n_blocks": len(samples[0]) if samples else 0,
                "algorithms_typed": ";".join(algorithms_with_data),
                "test_type": "Friedman",
                "notes": note
            })

            # If significant, perform post-hoc tests
            if p_value < 0.05:  # Standard alpha level
                # Generate all unique pairs
                algorithm_pairs = list(itertools.combinations(algorithms_with_data, 2))

                # Collect raw p-values for Holm correction
                raw_p_values = []
                wilcoxon_results = []

                for alg_a, alg_b in algorithm_pairs:
                    sample_a = algorithm_data[alg_a]
                    sample_b = algorithm_data[alg_b]

                    # Ensure equal length for pairwise comparison
                    min_len = min(len(sample_a), len(sample_b))
                    if min_len < 2:
                        # Not enough data for Wilcoxon test
                        wilcoxon_results.append({
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "n_pairs": min_len
                        })
                        raw_p_values.append(1.0)  # Will get corrected to 1.0
                        continue

                    # Truncate to equal length
                    sample_a_trunc = sample_a[:min_len]
                    sample_b_trunc = sample_b[:min_len]

                    try:
                        statistic, p_value = _wilcoxon_test(sample_a_trunc, sample_b_trunc)
                        wilcoxon_results.append({
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "statistic": statistic,
                            "p_value": p_value,
                            "n_pairs": min_len
                        })
                        raw_p_values.append(p_value)
                    except Exception:
                        wilcoxon_results.append({
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "n_pairs": min_len
                        })
                        raw_p_values.append(1.0)

                # Apply Holm correction
                if raw_p_values:
                    corrected_p_values = _holm_correction(raw_p_values)

                    # Build post-hoc rows
                    for i, (pair, raw_p, corr_p) in enumerate(zip(wilcoxon_results, raw_p_values, corrected_p_values)):
                        # Calculate effect size
                        alg_a = pair["algorithm_a"]
                        alg_b = pair["algorithm_b"]
                        sample_a = algorithm_data[alg_a][:pair["n_pairs"]]
                        sample_b = algorithm_data[alg_b][:pair["n_pairs"]]

                        if len(sample_a) >= 2 and len(sample_b) >= 2:
                            try:
                                # Re-run Wilcoxon to get statistic for effect size
                                stat, _ = _wilcoxon_test(sample_a, sample_b)
                                effect_size = _rank_biserial_effect_size(stat, len(sample_a))
                            except Exception:
                                effect_size = np.nan
                        else:
                            effect_size = np.nan

                        posthoc_rows.append({
                            "metric": metric,
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "n_blocks": pair["n_pairs"],
                            "wilcoxon_statistic": pair["statistic"],
                            "p_value_raw": raw_p,
                            "p_value_corrected": corr_p,
                            "correction_method": "Holm",
                            "effect_size": effect_size,
                            "effect_size_type": "rank_biserial"
                        })

        except Exception as e:
            friedman_rows.append({
                "metric": metric,
                "statistic": np.nan,
                "p_value": np.nan,
                "n_blocks": 0,
                "algorithms_typed": ";".join(available_algorithms),
                "test_type": "Friedman",
                "notes": f"Error during Friedman test: {str(e)}"
            })

    # Analyze each quality metric (similar pattern)
    for metric in quality_metrics_to_test:
        metric_col = f"mean_{metric}"
        if metric_col not in df_lfr.columns:
            metric_cols = [col for col in df_lfr.columns if metric in col.lower()]
            if not metric_cols:
                continue
            metric_col = metric_cols[0]

        algorithm_data = _construct_algorithm_metric_matrix(
            df_lfr[["graph_id", "algorithm", metric_col]].dropna(),
            metric_col,
            available_algorithms
        )

        algorithms_with_data = [alg for alg, vals in algorithm_data.items() if len(vals) >= 3]
        if len(algorithms_with_data) < 3:
            friedman_rows.append({
                "metric": metric,
                "statistic": np.nan,
                "p_value": np.nan,
                "n_blocks": 0,
                "algorithms_typed": ";".join(available_algorithms),
                "test_type": "Friedman",
                "notes": "Insufficient data for Friedman test (need ≥3 algorithms with ≥3 blocks each)"
            })
            continue

        samples = [algorithm_data[alg] for alg in algorithms_with_data]
        block_counts = [len(vals) for vals in samples]
        if len(set(block_counts)) > 1:
            min_blocks = min(block_counts)
            samples = [vals[:min_blocks] for vals in samples]
            note = f"Truncated to {min_blocks} blocks per algorithm due to uneven replication"
        else:
            note = f"{block_counts[0]} blocks per algorithm"

        try:
            statistic, p_value = _friedman_test(*samples)
            friedman_rows.append({
                "metric": metric,
                "statistic": statistic,
                "p_value": p_value,
                "n_blocks": len(samples[0]) if samples else 0,
                "algorithms_typed": ";".join(algorithms_with_data),
                "test_type": "Friedman",
                "notes": note
            })

            # Post-hoc tests for quality metrics (same pattern as stability)
            if p_value < 0.05:
                algorithm_pairs = list(itertools.combinations(algorithms_with_data, 2))

                raw_p_values = []
                wilcoxon_results = []

                for alg_a, alg_b in algorithm_pairs:
                    sample_a = algorithm_data[alg_a]
                    sample_b = algorithm_data[alg_b]

                    min_len = min(len(sample_a), len(sample_b))
                    if min_len < 2:
                        wilcoxon_results.append({
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "n_pairs": min_len
                        })
                        raw_p_values.append(1.0)
                        continue

                    sample_a_trunc = sample_a[:min_len]
                    sample_b_trunc = sample_b[:min_len]

                    try:
                        statistic, p_value = _wilcoxon_test(sample_a_trunc, sample_b_trunc)
                        wilcoxon_results.append({
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "statistic": statistic,
                            "p_value": p_value,
                            "n_pairs": min_len
                        })
                        raw_p_values.append(p_value)
                    except Exception:
                        wilcoxon_results.append({
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "statistic": np.nan,
                            "p_value": np.nan,
                            "n_pairs": min_len
                        })
                        raw_p_values.append(1.0)

                if raw_p_values:
                    corrected_p_values = _holm_correction(raw_p_values)

                    for i, (pair, raw_p, corr_p) in enumerate(zip(wilcoxon_results, raw_p_values, corrected_p_values)):
                        alg_a = pair["algorithm_a"]
                        alg_b = pair["algorithm_b"]
                        sample_a = algorithm_data[alg_a][:pair["n_pairs"]]
                        sample_b = algorithm_data[alg_b][:pair["n_pairs"]]

                        if len(sample_a) >= 2 and len(sample_b) >= 2:
                            try:
                                stat, _ = _wilcoxon_test(sample_a, sample_b)
                                effect_size = _rank_biserial_effect_size(stat, len(sample_a))
                            except Exception:
                                effect_size = np.nan
                        else:
                            effect_size = np.nan

                        posthoc_rows.append({
                            "metric": metric,
                            "algorithm_a": alg_a,
                            "algorithm_b": alg_b,
                            "n_blocks": pair["n_pairs"],
                            "wilcoxon_statistic": pair["statistic"],
                            "p_value_raw": raw_p,
                            "p_value_corrected": corr_p,
                            "correction_method": "Holm",
                            "effect_size": effect_size,
                            "effect_size_type": "rank_biserial"
                        })

        except Exception as e:
            friedman_rows.append({
                "metric": metric,
                "statistic": np.nan,
                "p_value": np.nan,
                "n_blocks": 0,
                "algorithms_typed": ";".join(available_algorithms),
                "test_type": "Friedman",
                "notes": f"Error during Friedman test: {str(e)}"
            })

    # Create DataFrames
    friedman_df = pd.DataFrame(friedman_rows, columns=FRIEDMAN_TEST_COLUMNS)
    posthoc_df = pd.DataFrame(posthoc_rows, columns=POSTHOC_TEST_COLUMNS)

    return friedman_df, posthoc_df


def run_real_network_descriptive_analysis(
    cfg: dict,
    graph_ids: list[str] | None = None,
    algorithms: list[str] | None = None,
    results_dir: Path | None = None
) -> pd.DataFrame:
    """Stage 9 entry point for real-network descriptive analysis.

    Computes descriptive statistics (mean, median, std, min, max) for
    each graph-analgorithm-metric combination in the real networks.

    Args:
        cfg: Configuration dictionary
        graph_ids: Optional list of graph IDs to filter
        algorithms: Optional list of algorithms to filter
        results_dir: Optional results directory path

    Returns:
        DataFrame with descriptive statistics
    """
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir")

    # Load the joined quality-stability summary
    df_joined = _load_joined_summary(results_dir)

    if df_joined.empty:
        return pd.DataFrame(columns=DESCRIPTIVE_STATS_COLUMNS)

    # Apply filters if provided
    if graph_ids is not None:
        df_joined = df_joined[df_joined["graph_id"].isin(graph_ids)]
    if algorithms is not None:
        df_joined = df_joined[df_joined["algorithm"].isin(algorithms)]

    # Restrict to real networks only (exclude LFR for descriptive analysis)
    df_real = df_joined[df_joined["dataset"] != "lfr"].copy()

    if df_real.empty:
        return pd.DataFrame(columns=DESCRIPTIVE_STATS_COLUMNS)

    # Get unique combinations
    combinations = df_real[["graph_id", "dataset", "algorithm"]].drop_duplicates()

    # Stage 7's joined summary names the Stage 6 VI/normalized_VI/NMI means
    # mean_stability_<metric>; alias them to the mean_<metric> names used below.
    df_real = df_real.copy()
    for _m in ("VI", "normalized_VI", "NMI"):
        if f"mean_{_m}" not in df_real.columns and f"mean_stability_{_m}" in df_real.columns:
            df_real[f"mean_{_m}"] = df_real[f"mean_stability_{_m}"]

    # Determine which metrics to analyze based on available algorithms
    metric_columns = []
    metric_names = []

    for _, row in combinations.iterrows():
        algorithm = row["algorithm"]
        if _is_hard_partition_algorithm(algorithm):
            metric_columns.extend(["mean_modularity", "mean_VI", "mean_normalized_VI", "mean_NMI", "mean_Omega"])
            metric_names.extend(["modularity", "VI", "normalized_VI", "NMI", "Omega"])
        elif _is_overlapping_algorithm(algorithm):
            metric_columns.extend(["mean_modularity", "mean_Omega", "mean_ONMI"])
            metric_names.extend(["modularity", "Omega", "ONMI"])

    # Remove duplicates while preserving order
    seen = set()
    unique_metric_columns = []
    unique_metric_names = []
    for col, name in zip(metric_columns, metric_names):
        if col not in seen:
            seen.add(col)
            unique_metric_columns.append(col)
            unique_metric_names.append(name)

    # Filter to only columns that actually exist
    available_metric_columns = [col for col in unique_metric_columns if col in df_real.columns]
    available_metric_names = [name for col, name in zip(unique_metric_columns, unique_metric_names)
                             if col in df_real.columns]

    if not available_metric_columns:
        return pd.DataFrame(columns=DESCRIPTIVE_STATS_COLUMNS)

    # Compute descriptive statistics for each combination
    descriptive_rows = []

    for _, row in combinations.iterrows():
        graph_id = row["graph_id"]
        dataset = row["dataset"]
        algorithm = row["algorithm"]

        # Filter data for this combination
        mask = (df_real["graph_id"] == graph_id) & (df_real["algorithm"] == algorithm)
        df_subset = df_real[mask]

        if df_subset.empty:
            continue

        # Compute statistics for each available metric
        for metric_col, metric_name in zip(available_metric_columns, available_metric_names):
            if metric_col not in df_subset.columns:
                continue

            values = df_subset[metric_col].dropna()

            if len(values) == 0:
                continue

            descriptive_rows.append({
                "graph_id": graph_id,
                "dataset": dataset,
                "algorithm": algorithm,
                "metric": metric_name,
                "n_observations": len(values),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max())
            })

    # Create DataFrame with proper column order
    descriptive_df = pd.DataFrame(descriptive_rows, columns=DESCRIPTIVE_STATS_COLUMNS)

    return descriptive_df


def run_statistical_analysis(
    cfg: dict,
    graph_ids: list[str] | None = None,
    algorithms: list[str] | None = None,
    results_dir: Path | None = None,
    analysis_type: str = "both"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main entry point for Stage 9 statistical analysis.

    Args:
        cfg: Configuration dictionary
        graph_ids: Optional list of graph IDs to filter
        algorithms: Optional list of algorithms to filter
        results_dir: Optional results directory path
        analysis_type: "lfr" (inferential only), "real" (descriptive only), or "both"

    Returns:
        (friedman_results, posthoc_results, descriptive_results)
    """
    friedman_df = pd.DataFrame()
    posthoc_df = pd.DataFrame()
    descriptive_df = pd.DataFrame()

    if analysis_type in ["lfr", "both"]:
        friedman_df, posthoc_df = run_lfr_statistical_analysis(
            cfg, graph_ids=graph_ids, algorithms=algorithms, results_dir=results_dir
        )

    if analysis_type in ["real", "both"]:
        descriptive_df = run_real_network_descriptive_analysis(
            cfg, graph_ids=graph_ids, algorithms=algorithms, results_dir=results_dir
        )

    return friedman_df, posthoc_df, descriptive_df