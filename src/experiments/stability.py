"""Stage 6: run-to-run stability analysis.

Answers "how stable is the detected community structure across independent runs of the
SAME algorithm on the SAME graph?" -- purely from what Stage 5 already persisted
(results/raw/runs.csv + results/raw/partitions/). This module never calls an algorithm,
never generates or regenerates LFR graphs, and never re-executes anything Stage 5 already
did; it is safe to delete results/processed/ and regenerate it entirely by re-running this
stage (see module-level ``run_stability_analysis``).

ALGORITHM -> METRIC SCOPE (do not force hard-partition metrics onto SLPA, per the project
handoff; see src/metrics/stability.py's own module docstring for why):
  - Hard partitions (lpa, semi_sync_lpa, flpa): VI, normalized VI, NMI, Omega.
    ONMI is left NaN -- it is not the metric used for hard partitions (plain NMI already
    covers that role; ONMI is reported only where SLPA needs an overlap-aware NMI).
  - Overlapping (slpa): Omega, ONMI. VI/NMI are left NaN -- they are hard-partition-only
    formulas (VI needs single-label entropy; sklearn's NMI expects one label per node).
  - Omega is therefore the only metric present for every algorithm (usable for the
    "does improved stability cost quality" cross-algorithm comparison in later stages).

PAIRWISE vs AGGREGATE, AND WHY BOOTSTRAP IS AT THE RUN LEVEL:
For a graph x algorithm group with R stored runs there are C(R, 2) pairwise comparisons
(435 for R=30). These are NOT independent observations -- every pair reuses one of only R
underlying partitions, so treating the 435 values as 435 i.i.d. samples (e.g. via a naive
percentile CI over the raw pairwise values) would understate uncertainty. Stage 6 therefore
bootstraps at the RUN level: resample the R runs with replacement, form ALL pairs among the
resampled runs (including a run paired with itself, which contributes a perfect-agreement
pair -- an expected and correct consequence of resampling identical items, not a bug),
recompute the summary statistic, repeat, and take percentiles. See
``_bootstrap_group_summary`` for the implementation and ``docs/methodology.md`` /
STAGE 6 handoff for the "do not bootstrap the 435 pairwise values as if independent" warning.

No statistical significance testing (Friedman/Wilcoxon) happens here -- that is Stage 9.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics.stability import nmi as nmi_score
from ..metrics.stability import omega_index, onmi_lfk, variation_of_information
from . import storage

# Algorithms that store a single-label-per-node hard partition (VI/NMI apply).
HARD_ALGORITHMS = {"lpa", "semi_sync_lpa", "flpa"}
# Algorithms that store an overlapping cover (VI/NMI do not apply; Omega/ONMI do).
OVERLAPPING_ALGORITHMS = {"slpa"}

# Columns identifying a graph, carried through from runs.csv unchanged (never recomputed).
GRAPH_META_COLUMNS = ["graph_id", "dataset", "mu_nominal", "mu_empirical", "instance"]

PAIRWISE_COLUMNS = GRAPH_META_COLUMNS + [
    "algorithm", "run_i", "run_j", "seed_i", "seed_j",
    "VI", "normalized_VI", "NMI", "Omega", "ONMI",
    "degenerate_i", "degenerate_j",
]

SUMMARY_METRIC_STATS = {
    # metric_name -> which aggregate columns to produce
    "VI": ["mean", "median", "std"],
    "normalized_VI": ["mean"],
    "NMI": ["mean", "std"],
    "Omega": ["mean", "std"],
    "ONMI": ["mean", "std"],
}

DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_BOOTSTRAP_SEED = 20240607  # arbitrary but fixed -- see "REPRODUCIBILITY" in the handoff


# ---------------------------------------------------------------------------
# Loading Stage 5 output (read-only: this module never writes to results/raw/)
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> pd.DataFrame:
    """Read results/raw/runs.csv as Stage 5 left it. Does NOT rebuild it -- Stage 6 only
    reads Stage 5's output, never re-executes algorithms or mutates raw results."""
    path = Path(results_dir) / "raw" / "runs.csv"
    if not path.exists():
        return pd.DataFrame(columns=storage.RUN_COLUMNS)
    return pd.read_csv(path)


def _load_group_partitions(raw_dir: Path, rows: pd.DataFrame) -> list[dict]:
    """Load every partition for one (graph_id, algorithm) group. Rows whose partition file
    no longer loads (deleted/corrupted since Stage 5 ran) are skipped with a note printed,
    exactly like Stage 5's own corruption handling -- Stage 6 never fabricates a value for
    a run it cannot actually load."""
    loaded = []
    for _, row in rows.sort_values("run").iterrows():
        ppath = Path(raw_dir) / row["partition_file"]
        kind = row["partition_kind"]
        try:
            partition = storage.load_partition(ppath, kind)
        except Exception as e:  # noqa: BLE001 -- any load failure means "skip this run"
            print(f"  [stability] skipping unreadable partition {ppath} ({e})")
            continue
        loaded.append({
            "run": int(row["run"]),
            "seed": int(row["seed"]),
            "kind": kind,
            "labels": partition if kind == "hard" else None,
            "cover": partition if kind == "cover" else None,
            "degenerate": bool(row["degenerate"]),
            "n": (len(partition) if kind == "hard"
                  else (max((v for c in partition for v in c), default=-1) + 1)),
        })
    return loaded


# ---------------------------------------------------------------------------
# Pairwise comparison of two stored runs
# ---------------------------------------------------------------------------

def _compare_runs(run_a: dict, run_b: dict, algorithm: str) -> dict:
    """Compute the metrics valid for `algorithm` between two loaded runs of the same graph.
    Unsupported metrics for this algorithm are left as NaN, never fabricated."""
    n = run_a["n"]
    out = {"VI": np.nan, "normalized_VI": np.nan, "NMI": np.nan, "Omega": np.nan, "ONMI": np.nan}

    if algorithm in HARD_ALGORITHMS:
        a, b = run_a["labels"], run_b["labels"]
        out["VI"] = variation_of_information(a, b, normalize=False)
        out["normalized_VI"] = variation_of_information(a, b, normalize=True)
        out["NMI"] = nmi_score(a, b)
        out["Omega"] = omega_index(n, labels1=a, labels2=b)
    elif algorithm in OVERLAPPING_ALGORITHMS:
        ca, cb = run_a["cover"], run_b["cover"]
        out["Omega"] = omega_index(n, cover1=ca, cover2=cb)
        out["ONMI"] = onmi_lfk(n, ca, cb)
    else:
        raise ValueError(f"unknown algorithm scope for {algorithm!r} "
                          f"(not in HARD_ALGORITHMS or OVERLAPPING_ALGORITHMS)")
    return out


def pairwise_stability(loaded_runs: list[dict], algorithm: str, meta: dict) -> pd.DataFrame:
    """All C(R, 2) unique pairwise comparisons among `loaded_runs` (already-loaded, valid
    partitions for one graph x algorithm group). meta carries the graph-identity columns
    (graph_id, dataset, mu_nominal, mu_empirical, instance) through unchanged."""
    rows = []
    for ra, rb in itertools.combinations(loaded_runs, 2):
        metrics = _compare_runs(ra, rb, algorithm)
        rows.append({
            **meta, "algorithm": algorithm,
            "run_i": ra["run"], "run_j": rb["run"],
            "seed_i": ra["seed"], "seed_j": rb["seed"],
            **metrics,
            "degenerate_i": ra["degenerate"], "degenerate_j": rb["degenerate"],
        })
    return pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)


# ---------------------------------------------------------------------------
# Aggregation (one graph x algorithm group -> one summary row, no bootstrap)
# ---------------------------------------------------------------------------

def _applicable_metrics(algorithm: str) -> list[str]:
    if algorithm in HARD_ALGORITHMS:
        return ["VI", "normalized_VI", "NMI", "Omega"]
    if algorithm in OVERLAPPING_ALGORITHMS:
        return ["Omega", "ONMI"]
    raise ValueError(f"unknown algorithm scope for {algorithm!r}")


def _summary_stats_from_pairwise(pw: pd.DataFrame, algorithm: str) -> dict:
    """Mean/median/std for each metric valid for `algorithm`; metrics not applicable to
    this algorithm are left as NaN (never fabricated) rather than omitted, so every summary
    row has the same column set."""
    out = {}
    applicable = set(_applicable_metrics(algorithm))
    for metric, stats in SUMMARY_METRIC_STATS.items():
        for stat in stats:
            col = f"{stat}_{metric}"
            if metric not in applicable or pw.empty:
                out[col] = np.nan
                continue
            series = pw[metric]
            out[col] = float(getattr(series, stat)())
    return out


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals (run-level resampling; see module docstring)
# ---------------------------------------------------------------------------

def _pairwise_matrix(loaded_runs: list[dict], algorithm: str) -> dict:
    """Precompute every metric for every UNORDERED pair of DISTINCT stored runs once,
    keyed by (i, j) with i<j (positional index into loaded_runs, not run/seed id). A
    bootstrap resample only changes which positions get drawn (with replacement) -- the
    comparison of position i vs position j never changes, so recomputing it per bootstrap
    replicate would be redundant work producing the exact same number every time. Self-pairs
    (a position matched with itself, which happens whenever a run is drawn more than once in
    one resample) are defined here too: identical partition vs itself = perfect agreement
    (VI=0, normalized_VI=0, NMI=1, Omega=1, ONMI=1 when applicable), which is the correct
    value, not a placeholder."""
    r = len(loaded_runs)
    applicable = set(_applicable_metrics(algorithm))
    self_val = {"VI": 0.0, "normalized_VI": 0.0, "NMI": 1.0, "Omega": 1.0, "ONMI": 1.0}
    M = {}
    for i in range(r):
        M[(i, i)] = {k: (self_val[k] if k in applicable else np.nan) for k in self_val}
    for i, j in itertools.combinations(range(r), 2):
        vals = _compare_runs(loaded_runs[i], loaded_runs[j], algorithm)
        M[(i, j)] = vals
        M[(j, i)] = vals
    return M


def _summary_from_matrix(M: dict, indices: np.ndarray, algorithm: str) -> dict:
    """Same aggregate statistics as `_summary_stats_from_pairwise`, but computed from the
    precomputed pairwise matrix for one (possibly-resampled, possibly-repeated) list of
    run positions -- used by both the point estimate and every bootstrap replicate."""
    applicable = set(_applicable_metrics(algorithm))
    pairs = list(itertools.combinations(range(len(indices)), 2))
    out = {}
    for metric, stats in SUMMARY_METRIC_STATS.items():
        if metric not in applicable or not pairs:
            for stat in stats:
                out[f"{stat}_{metric}"] = np.nan
            continue
        vals = np.array([M[(indices[a], indices[b])][metric] for a, b in pairs], dtype=float)
        for stat in stats:
            out[f"{stat}_{metric}"] = float(getattr(np, f"nan{stat}")(vals)) if len(vals) else np.nan
    return out


def _bootstrap_group_summary(loaded_runs: list[dict], algorithm: str, n_bootstrap: int,
                              seed_sequence: np.random.SeedSequence) -> dict:
    """Resample the R stored runs WITH REPLACEMENT `n_bootstrap` times (run-level, not
    pairwise-level -- see module docstring), recompute the pairwise-summary statistic each
    time from the precomputed comparison matrix, and return 2.5/97.5-percentile CIs for
    every mean_* statistic. Deterministic for a fixed seed_sequence."""
    r = len(loaded_runs)
    ci_cols = {}
    applicable = set(_applicable_metrics(algorithm))
    mean_metrics = [m for m in SUMMARY_METRIC_STATS if "mean" in SUMMARY_METRIC_STATS[m] and m in applicable]
    if r < 2:
        for metric in SUMMARY_METRIC_STATS:
            ci_cols[f"mean_{metric}_ci_low"] = np.nan
            ci_cols[f"mean_{metric}_ci_high"] = np.nan
        return ci_cols

    M = _pairwise_matrix(loaded_runs, algorithm)
    rng = np.random.default_rng(seed_sequence)
    replicates = {m: np.empty(n_bootstrap) for m in mean_metrics}
    for b in range(n_bootstrap):
        idx = rng.integers(0, r, size=r)
        summ = _summary_from_matrix(M, idx, algorithm)
        for m in mean_metrics:
            replicates[m][b] = summ[f"mean_{m}"]

    for metric in SUMMARY_METRIC_STATS:
        if metric in mean_metrics:
            lo, hi = np.nanpercentile(replicates[metric], [2.5, 97.5])
            ci_cols[f"mean_{metric}_ci_low"] = float(lo)
            ci_cols[f"mean_{metric}_ci_high"] = float(hi)
        else:
            ci_cols[f"mean_{metric}_ci_low"] = np.nan
            ci_cols[f"mean_{metric}_ci_high"] = np.nan
    return ci_cols


# ---------------------------------------------------------------------------
# Per-group driver + top-level entry point
# ---------------------------------------------------------------------------

def analyze_group(raw_dir: Path, group_rows: pd.DataFrame, graph_id: str, algorithm: str,
                   n_bootstrap: int, seed_sequence: np.random.SeedSequence,
                   verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """Analyze one (graph_id, algorithm) group: load its stored partitions, compute every
    pairwise comparison, and aggregate (with bootstrap CIs) into one summary row."""
    meta = {c: group_rows.iloc[0][c] for c in GRAPH_META_COLUMNS}
    loaded = _load_group_partitions(raw_dir, group_rows)

    pw = pairwise_stability(loaded, algorithm, meta)

    n_comm = group_rows["number_of_communities"].astype(float)
    summary = {
        **meta, "algorithm": algorithm,
        "number_of_runs": len(loaded),
        "number_of_pairs": len(pw),
        **_summary_stats_from_pairwise(pw, algorithm),
        "degenerate_run_count": int(group_rows["degenerate"].sum()),
        "degenerate_pair_count": int(((pw["degenerate_i"]) | (pw["degenerate_j"])).sum()) if len(pw) else 0,
        "mean_number_of_communities": float(n_comm.mean()) if len(n_comm) else np.nan,
        "std_number_of_communities": float(n_comm.std()) if len(n_comm) else np.nan,
        "n_bootstrap": n_bootstrap,
    }
    summary.update(_bootstrap_group_summary(loaded, algorithm, n_bootstrap, seed_sequence))

    if verbose:
        print(f"  stability  {graph_id:20s} {algorithm:14s} "
              f"runs={len(loaded):2d} pairs={len(pw):3d} "
              f"degenerate={summary['degenerate_run_count']}")
    return pw, summary


def run_stability_analysis(cfg: dict, graph_ids: list[str] | None = None,
                            algorithms: list[str] | None = None,
                            results_dir: Path | None = None,
                            n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                            bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
                            verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 6 entry point. Reads results/raw/runs.csv (written by Stage 5) and the
    partition files it points to; NEVER calls an algorithm or LFR generation. Writes
    results/processed/pairwise.csv and results/processed/stability_summary.csv, and
    returns (pairwise_df, summary_df).

    graph_ids / algorithms: optional filters (default: every group present in runs.csv).
    Deterministic: the same stored runs + the same (n_bootstrap, bootstrap_seed) always
    produce identical output -- each (graph_id, algorithm) group gets its own independent,
    reproducible bootstrap stream spawned from `bootstrap_seed` via numpy's SeedSequence.
    """
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir")
    raw_dir = Path(results_dir) / "raw"
    proc_dir = Path(results_dir) / "processed"

    runs = load_runs(results_dir)
    if graph_ids is not None:
        runs = runs[runs["graph_id"].isin(graph_ids)]
    if algorithms is not None:
        runs = runs[runs["algorithm"].isin(algorithms)]

    groups = list(runs.groupby(["graph_id", "algorithm"], sort=True))
    root_ss = np.random.SeedSequence(bootstrap_seed)
    child_seeds = root_ss.spawn(len(groups))  # one independent, reproducible stream per group

    pairwise_frames, summary_rows = [], []
    for (graph_id, algorithm), group_rows in groups:
        if len(group_rows) < 2:
            if verbose:
                print(f"  stability  {graph_id:20s} {algorithm:14s} "
                      f"skipped (only {len(group_rows)} stored run(s), need >= 2)")
            continue
        ss = child_seeds[len(pairwise_frames)]  # positional index into groups -> deterministic seed stream
        pw, summary = analyze_group(raw_dir, group_rows, graph_id, algorithm,
                                     n_bootstrap, ss, verbose=verbose)
        pairwise_frames.append(pw)
        summary_rows.append(summary)

    pairwise_df = (pd.concat(pairwise_frames, ignore_index=True) if pairwise_frames
                   else pd.DataFrame(columns=PAIRWISE_COLUMNS))
    summary_df = pd.DataFrame(summary_rows)
    if len(summary_df):
        summary_df = summary_df.sort_values(["dataset", "graph_id", "algorithm"]).reset_index(drop=True)

    proc_dir.mkdir(parents=True, exist_ok=True)
    storage.atomic_write_text(proc_dir / "pairwise.csv", pairwise_df.to_csv(index=False))
    storage.atomic_write_text(proc_dir / "stability_summary.csv", summary_df.to_csv(index=False))
    if verbose:
        print(f"\nwrote {proc_dir/'pairwise.csv'} ({len(pairwise_df)} rows) and "
              f"{proc_dir/'stability_summary.csv'} ({len(summary_df)} rows) "
              f"[bootstrap: n={n_bootstrap}, seed={bootstrap_seed}]")
    return pairwise_df, summary_df
