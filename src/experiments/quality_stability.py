"""Stage 7: quality x stability analysis.

Central question: "Does improved stability come at a cost to community detection
quality?" This module NEVER runs an algorithm and NEVER regenerates LFR graphs -- it only
reads what Stage 5 (results/raw/runs.csv) and Stage 6 (results/processed/stability_summary.csv)
already computed and persisted, joins them one row per (graph_id, algorithm), and computes
descriptive Spearman correlations between quality and stability at that unit of observation.

UNIT OF ANALYSIS (see "CORRELATION OUTPUT" in the project handoff):
Every row fed into a correlation is one (graph_id, algorithm) aggregate -- e.g. one LFR graph
instance run through one algorithm, or one real dataset run through one algorithm. This is
NOT the same thing as a pairwise run comparison (C(R,2) per group, 435 for R=30): those are
already collapsed into the per-group means Stage 6 wrote to stability_summary.csv, and are
never treated as additional independent observations here. Correlating across pairwise values
directly would both violate independence and mix within-group noise into a between-graph
relationship -- see stability.py's own module docstring for the parallel warning about
bootstrapping the 435 pairs as if independent.

ALGORITHM SCOPE (mirrors stability.py -- never force a metric outside its native scope):
  - Hard partitions (lpa, semi_sync_lpa, flpa): quality = Q; stability = VI/normalized_VI/NMI/Omega.
  - Overlapping (slpa): quality = EQ; stability = Omega/ONMI (VI/NMI stay NaN, inherited from
    Stage 6 -- never fabricated here).

STABILITY DIRECTION (see project handoff "STABILITY DIRECTION"): VI and normalized_VI are
LOWER-is-more-stable; NMI, Omega, ONMI are HIGHER-is-more-stable. This module never inverts a
metric silently -- correlation rows report the metric's native direction, and callers /
figures must read the metric name to know which way "more stable" points.

DEGENERATE RUNS: degenerate_run_count / degenerate_pair_count are carried through from Stage 6
unchanged (never recomputed, never used to filter out rows) so a collapsed-to-one-community
group that happens to look "stable" is visible in the same row as its stability numbers,
never silently interpreted as evidence of good structure.

No significance-testing framework (Friedman/Holm-Wilcoxon) lives here -- that is Stage 9. The
p-values produced here are plain Spearman correlation-test p-values for a single pair of
variables, not part of any cross-algorithm comparison, and are labelled as such in the output.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import storage
from .stability import GRAPH_META_COLUMNS, HARD_ALGORITHMS, OVERLAPPING_ALGORITHMS, load_runs

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

QUALITY_STABILITY_COLUMNS = GRAPH_META_COLUMNS + [
    "algorithm",
    "mean_Q", "std_Q",
    "mean_EQ", "std_EQ",
    "mean_ground_truth_NMI", "std_ground_truth_NMI",
    "mean_ground_truth_ONMI", "std_ground_truth_ONMI",
    "n_quality_runs",
    "mean_stability_VI", "mean_stability_normalized_VI", "mean_stability_NMI",
    "mean_Omega", "mean_ONMI",
    "stability_VI_ci_low", "stability_VI_ci_high",
    "stability_normalized_VI_ci_low", "stability_normalized_VI_ci_high",
    "stability_NMI_ci_low", "stability_NMI_ci_high",
    "stability_Omega_ci_low", "stability_Omega_ci_high",
    "stability_ONMI_ci_low", "stability_ONMI_ci_high",
    "mean_community_count", "std_community_count",
    "number_of_stability_runs", "number_of_stability_pairs",
    "degenerate_run_count", "degenerate_pair_count",
]

CORRELATION_COLUMNS = [
    "subset", "algorithm", "quality_metric", "stability_metric",
    "n_graphs", "spearman_rho", "p_value", "confidence_interval_if_implemented",
]


# ---------------------------------------------------------------------------
# Loading Stage 5 / Stage 6 outputs (read-only)
# ---------------------------------------------------------------------------

def load_stability_summary(results_dir: Path) -> pd.DataFrame:
    """Read results/processed/stability_summary.csv as Stage 6 left it. Never recomputes
    it -- if it does not exist yet (Stage 6 has not been run for these graphs), returns an
    empty frame rather than fabricating one."""
    path = Path(results_dir) / "processed" / "stability_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Stage 5 quality aggregation (runs.csv -> one row per graph_id x algorithm)
# ---------------------------------------------------------------------------

def aggregate_quality(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse Stage 5's per-run rows (one row per graph x algorithm x seed) into one row
    per graph x algorithm: mean/std of quality (split into Q for hard partitions / EQ for
    SLPA, per each run's own `quality_metric` -- never mixed), mean/std of ground-truth
    NMI/ONMI where present (missing ground truth stays NaN, never 0), and community-count
    stats. A metric with zero non-null observations in a group is left NaN, never 0."""
    if runs.empty:
        return pd.DataFrame(columns=GRAPH_META_COLUMNS + [
            "algorithm", "mean_Q", "std_Q", "mean_EQ", "std_EQ",
            "mean_ground_truth_NMI", "std_ground_truth_NMI",
            "mean_ground_truth_ONMI", "std_ground_truth_ONMI",
            "n_quality_runs",
        ])

    rows = []
    group_cols = GRAPH_META_COLUMNS + ["algorithm"]
    for keys, g in runs.groupby(group_cols, dropna=False, sort=True):
        meta = dict(zip(group_cols, keys))
        q_vals = g.loc[g["quality_metric"] == "Q", "quality"]
        eq_vals = g.loc[g["quality_metric"] == "EQ", "quality"]
        nmi_vals = g["nmi"].dropna()
        onmi_vals = g["onmi"].dropna()
        rows.append({
            **meta,
            "mean_Q": float(q_vals.mean()) if len(q_vals) else np.nan,
            "std_Q": float(q_vals.std()) if len(q_vals) > 1 else np.nan,
            "mean_EQ": float(eq_vals.mean()) if len(eq_vals) else np.nan,
            "std_EQ": float(eq_vals.std()) if len(eq_vals) > 1 else np.nan,
            "mean_ground_truth_NMI": float(nmi_vals.mean()) if len(nmi_vals) else np.nan,
            "std_ground_truth_NMI": float(nmi_vals.std()) if len(nmi_vals) > 1 else np.nan,
            "mean_ground_truth_ONMI": float(onmi_vals.mean()) if len(onmi_vals) else np.nan,
            "std_ground_truth_ONMI": float(onmi_vals.std()) if len(onmi_vals) > 1 else np.nan,
            "n_quality_runs": int(len(g)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Join Stage 5 quality aggregate with Stage 6 stability summary
# ---------------------------------------------------------------------------

def build_quality_stability_summary(results_dir: Path) -> pd.DataFrame:
    """One row per (graph_id, algorithm): Stage 5 quality aggregate joined with the Stage 6
    stability summary. An outer join -- a group present in only one of the two sources still
    appears, with the other source's columns left NaN (never fabricated), so a partially-run
    experiment matrix is visible rather than silently dropped."""
    results_dir = Path(results_dir)
    runs = load_runs(results_dir)
    stab = load_stability_summary(results_dir)
    quality = aggregate_quality(runs)

    if quality.empty and stab.empty:
        return pd.DataFrame(columns=QUALITY_STABILITY_COLUMNS)

    join_keys = ["graph_id", "algorithm"]
    meta_from_quality = [c for c in GRAPH_META_COLUMNS if c not in join_keys]

    if stab.empty:
        merged = quality.copy()
        for col in ["number_of_runs", "number_of_pairs", "degenerate_run_count",
                    "degenerate_pair_count", "mean_number_of_communities",
                    "std_number_of_communities"]:
            merged[col] = np.nan
        for metric in ["VI", "normalized_VI", "NMI", "Omega", "ONMI"]:
            merged[f"mean_{metric}"] = np.nan
            merged[f"mean_{metric}_ci_low"] = np.nan
            merged[f"mean_{metric}_ci_high"] = np.nan
    elif quality.empty:
        merged = stab.copy()
        for col in ["mean_Q", "std_Q", "mean_EQ", "std_EQ",
                    "mean_ground_truth_NMI", "std_ground_truth_NMI",
                    "mean_ground_truth_ONMI", "std_ground_truth_ONMI", "n_quality_runs"]:
            merged[col] = np.nan
    else:
        merged = quality.merge(stab, on=join_keys, how="outer", suffixes=("_q", "_s"))
        # Prefer Stage 6's copy of the shared graph-identity columns (dataset/mu_nominal/
        # mu_empirical/instance); Stage 5's copy is used only to fill rows Stage 6 doesn't
        # have yet (e.g. quality computed but Stage 6 not re-run for that group).
        for col in meta_from_quality:
            qc, sc = f"{col}_q", f"{col}_s"
            if qc in merged.columns and sc in merged.columns:
                merged[col] = merged[sc].where(merged[sc].notna(), merged[qc])
                merged = merged.drop(columns=[qc, sc])

    out = pd.DataFrame({
        **{c: merged.get(c) for c in GRAPH_META_COLUMNS + ["algorithm"]},
        "mean_Q": merged.get("mean_Q"), "std_Q": merged.get("std_Q"),
        "mean_EQ": merged.get("mean_EQ"), "std_EQ": merged.get("std_EQ"),
        "mean_ground_truth_NMI": merged.get("mean_ground_truth_NMI"),
        "std_ground_truth_NMI": merged.get("std_ground_truth_NMI"),
        "mean_ground_truth_ONMI": merged.get("mean_ground_truth_ONMI"),
        "std_ground_truth_ONMI": merged.get("std_ground_truth_ONMI"),
        "n_quality_runs": merged.get("n_quality_runs"),
        "mean_stability_VI": merged.get("mean_VI"),
        "mean_stability_normalized_VI": merged.get("mean_normalized_VI"),
        "mean_stability_NMI": merged.get("mean_NMI"),
        "mean_Omega": merged.get("mean_Omega"),
        "mean_ONMI": merged.get("mean_ONMI"),
        "stability_VI_ci_low": merged.get("mean_VI_ci_low"),
        "stability_VI_ci_high": merged.get("mean_VI_ci_high"),
        "stability_normalized_VI_ci_low": merged.get("mean_normalized_VI_ci_low"),
        "stability_normalized_VI_ci_high": merged.get("mean_normalized_VI_ci_high"),
        "stability_NMI_ci_low": merged.get("mean_NMI_ci_low"),
        "stability_NMI_ci_high": merged.get("mean_NMI_ci_high"),
        "stability_Omega_ci_low": merged.get("mean_Omega_ci_low"),
        "stability_Omega_ci_high": merged.get("mean_Omega_ci_high"),
        "stability_ONMI_ci_low": merged.get("mean_ONMI_ci_low"),
        "stability_ONMI_ci_high": merged.get("mean_ONMI_ci_high"),
        "mean_community_count": merged.get("mean_number_of_communities"),
        "std_community_count": merged.get("std_number_of_communities"),
        "number_of_stability_runs": merged.get("number_of_runs"),
        "number_of_stability_pairs": merged.get("number_of_pairs"),
        "degenerate_run_count": merged.get("degenerate_run_count"),
        "degenerate_pair_count": merged.get("degenerate_pair_count"),
    })
    if len(out):
        out = out.sort_values(["dataset", "graph_id", "algorithm"]).reset_index(drop=True)
    return out[QUALITY_STABILITY_COLUMNS]


# ---------------------------------------------------------------------------
# Correlation analysis (graph x algorithm is the unit of observation)
# ---------------------------------------------------------------------------

def _quality_metric_for(algorithm: str) -> str:
    return "mean_EQ" if algorithm in OVERLAPPING_ALGORITHMS else "mean_Q"


def _stability_metrics_for(algorithm: str) -> list[str]:
    if algorithm in OVERLAPPING_ALGORITHMS:
        return ["mean_Omega", "mean_ONMI"]
    return ["mean_stability_VI", "mean_stability_normalized_VI", "mean_stability_NMI", "mean_Omega"]


def _safe_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Spearman rho/p-value over the paired, non-null, finite values of x and y. Returns
    (nan, nan, n) when there are fewer than 3 usable pairs or either variable is constant
    (zero variance -> correlation undefined, not fabricated as 0)."""
    df = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(df)
    if n < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return np.nan, np.nan, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = spearmanr(df["x"], df["y"])
    return float(res.statistic), float(res.pvalue), n


def _subset_frame(summary: pd.DataFrame, subset: str) -> pd.DataFrame:
    if subset == "real":
        return summary[summary["dataset"] != "lfr"]
    if subset == "lfr":
        return summary[summary["dataset"] == "lfr"]
    return summary  # "all"


def compute_correlations(summary: pd.DataFrame, subsets: list[str] | None = None) -> pd.DataFrame:
    """Spearman correlation between quality and stability metrics, with one observation per
    (graph_id, algorithm) row of `summary` -- i.e. exactly the unit `build_quality_stability_
    summary` already produced (see module docstring: never the 435 pairwise values). Computed
    separately per algorithm (never mixing Q and EQ scales) and per `subsets` grouping,
    additionally pooling the three hard algorithms together (algorithm='hard_algorithms
    _pooled') so a hard-vs-overlapping difference in the quality-stability relationship is
    visible without ranking either group."""
    if subsets is None:
        subsets = ["all", "real", "lfr"]
    if summary.empty:
        return pd.DataFrame(columns=CORRELATION_COLUMNS)

    rows = []

    def _add(subset_name, algo_name, frame, quality_col, stability_col):
        rho, p, n = _safe_spearman(frame[quality_col], frame[stability_col])
        rows.append({
            "subset": subset_name, "algorithm": algo_name,
            "quality_metric": quality_col, "stability_metric": stability_col,
            "n_graphs": n, "spearman_rho": rho, "p_value": p,
            "confidence_interval_if_implemented": None,  # see module docstring: not implemented
        })

    for subset in subsets:
        sub = _subset_frame(summary, subset)

        for algorithm, g in sub.groupby("algorithm"):
            qcol = _quality_metric_for(algorithm)
            for scol in _stability_metrics_for(algorithm):
                _add(subset, algorithm, g, qcol, scol)
            # community-count variability vs stability (question 5)
            for scol in _stability_metrics_for(algorithm):
                if scol == "mean_Omega" or algorithm in OVERLAPPING_ALGORITHMS:
                    _add(subset, algorithm, g, "std_community_count", scol)
                    break
            # stability vs ground-truth NMI/ONMI, only where ground truth exists
            truth_col = "mean_ground_truth_ONMI" if algorithm in OVERLAPPING_ALGORITHMS else "mean_ground_truth_NMI"
            has_truth = g[truth_col].notna().sum()
            if has_truth >= 3:
                for scol in _stability_metrics_for(algorithm):
                    if scol != truth_col:
                        _add(subset, algorithm, g[g[truth_col].notna()], truth_col, scol)

        # hard-algorithms-pooled view (question 4: hard vs overlapping)
        hard_sub = sub[sub["algorithm"].isin(HARD_ALGORITHMS)]
        if len(hard_sub):
            for scol in ["mean_stability_VI", "mean_stability_normalized_VI",
                         "mean_stability_NMI", "mean_Omega"]:
                _add(subset, "hard_algorithms_pooled", hard_sub, "mean_Q", scol)

    return pd.DataFrame(rows, columns=CORRELATION_COLUMNS)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_quality_stability_analysis(cfg: dict, graph_ids: list[str] | None = None,
                                    algorithms: list[str] | None = None,
                                    results_dir: Path | None = None,
                                    verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 7 entry point. Reads results/raw/runs.csv (Stage 5) and
    results/processed/stability_summary.csv (Stage 6); NEVER calls an algorithm, NEVER
    regenerates LFR graphs, NEVER re-runs Stage 5/6. `graph_ids`/`algorithms` optionally
    restrict which rows of the already-joined summary are analyzed/written (default: every
    group present). Writes results/processed/quality_stability_summary.csv and
    results/processed/quality_stability_correlations.csv; returns (summary_df, correlations_df).
    """
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir")
    proc_dir = Path(results_dir) / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    summary = build_quality_stability_summary(results_dir)
    if graph_ids is not None and len(summary):
        summary = summary[summary["graph_id"].isin(graph_ids)]
    if algorithms is not None and len(summary):
        summary = summary[summary["algorithm"].isin(algorithms)]
    summary = summary.reset_index(drop=True)
    correlations = compute_correlations(summary)

    storage.atomic_write_text(proc_dir / "quality_stability_summary.csv", summary.to_csv(index=False))
    storage.atomic_write_text(proc_dir / "quality_stability_correlations.csv", correlations.to_csv(index=False))
    if verbose:
        print(f"wrote {proc_dir / 'quality_stability_summary.csv'} ({len(summary)} rows) and "
              f"{proc_dir / 'quality_stability_correlations.csv'} ({len(correlations)} rows)")
    return summary, correlations
