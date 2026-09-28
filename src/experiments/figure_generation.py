"""Stage 10: final figure and table generation.

Read-only reporting layer built entirely from what earlier stages already computed and
persisted under results/processed/ (Stage 5's runs.csv, Stage 6's stability_summary.csv /
pairwise.csv, Stage 7's quality_stability_summary.csv / quality_stability_correlations.csv,
Stage 8's lfr_instances.csv, Stage 9's statistical_tests.csv / posthoc_tests.csv /
descriptive_statistics.csv) plus results/processed/preprocessing_log.json (Stage 1) for
real-dataset characteristics.

This module NEVER:
  - calls a community-detection algorithm,
  - generates or regenerates an LFR graph,
  - re-runs Stage 5 experiments, Stage 6 stability bootstraps, or Stage 9 statistical tests,
  - writes to results/raw/ or data/,
  - fabricates a mean, p-value, effect size, correlation, or ranking that the loaded data
    does not actually support.

Every figure/table function is guarded against missing/empty input: if the data a figure or
table needs does not exist yet (most commonly because the full 8,880-run experiment has not
been executed), the function prints a one-line note and returns without writing anything,
rather than raising or emitting a misleading empty/placeholder plot or table. The one
exception is the completeness table/figure (table07 / fig08), which is always meaningful --
including "0 of 8,880 runs complete" -- because it reports on the experiment's progress
itself rather than on results the experiment would produce.

Output layout:
    results/figures/final/fig01_stability_comparison.png ... fig08_*
    results/tables/final/table01_dataset_characteristics.csv ... table07_*
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stability import OVERLAPPING_ALGORITHMS
from .storage import RUN_COLUMNS, atomic_write_text

ALGO_ORDER = ["lpa", "semi_sync_lpa", "flpa", "slpa"]
ALGO_MARKERS = {"lpa": "o", "semi_sync_lpa": "s", "flpa": "^", "slpa": "D"}

# Static, non-experimental algorithm metadata (citations / partition type / config knobs).
# This describes the algorithms themselves, not any experimental result -- never fabricated.
ALGORITHM_INFO = {
    "lpa": {
        "display_name": "LPA",
        "partition_type": "hard",
        "citation": "Raghavan, Albert & Kumara (2007), Physical Review E 76, 036106",
        "update_scheme": "asynchronous",
    },
    "semi_sync_lpa": {
        "display_name": "Semi-Synchronous LPA",
        "partition_type": "hard",
        "citation": "Cordasco & Gargano (2010), IEEE BASNA workshop",
        "update_scheme": "semi-synchronous",
    },
    "flpa": {
        "display_name": "FLPA",
        "partition_type": "hard",
        "citation": "Traag & Subelj (2023), Scientific Reports 13, 2701",
        "update_scheme": "queue-based asynchronous",
    },
    "slpa": {
        "display_name": "SLPA",
        "partition_type": "overlapping",
        "citation": "Xie, Szymanski & Liu (2011), ICDM Workshops",
        "update_scheme": "speaker-listener, memory-based",
    },
}


# ---------------------------------------------------------------------------
# Loaders (read-only; every one returns an empty-but-correctly-columned frame
# rather than raising when the upstream file does not exist yet)
# ---------------------------------------------------------------------------

def load_runs(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "raw" / "runs.csv"
    if not path.exists():
        return pd.DataFrame(columns=RUN_COLUMNS)
    return pd.read_csv(path)


def load_stability_summary(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "stability_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_quality_stability_summary(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "quality_stability_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_correlations(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "quality_stability_correlations.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_lfr_instances(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "lfr_instances.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_preprocessing_log(results_dir: Path) -> dict:
    path = Path(results_dir) / "processed" / "preprocessing_log.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def load_statistical_tests(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "statistical_tests.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_posthoc_tests(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "posthoc_tests.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_descriptive_statistics(results_dir: Path) -> pd.DataFrame:
    path = Path(results_dir) / "processed" / "descriptive_statistics.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _quality_col(algorithm: str) -> str:
    return "mean_EQ" if algorithm in OVERLAPPING_ALGORITHMS else "mean_Q"


def _present_algorithms(df: pd.DataFrame, col: str = "algorithm") -> list[str]:
    if df.empty or col not in df.columns:
        return []
    present = set(df[col].dropna())
    return [a for a in ALGO_ORDER if a in present]


def _skip(out_path: Path, reason: str) -> bool:
    print(f"  [stage10 figures] skipping {out_path.name}: {reason}")
    return False


def _mu_col(df: pd.DataFrame) -> str | None:
    """Prefer empirical mu (per project convention); fall back to nominal mu."""
    if "mu_empirical" in df.columns and df["mu_empirical"].notna().any():
        return "mu_empirical"
    if "mu_nominal" in df.columns and df["mu_nominal"].notna().any():
        return "mu_nominal"
    return None


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig_stability_comparison(stability_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 1: stability (mean Omega -- the one metric common to every algorithm; HIGHER =
    more stable) compared across algorithms, one box per algorithm, pooling every graph
    (real + LFR) that has a stored stability summary row."""
    algos = _present_algorithms(stability_summary)
    if not algos or "mean_Omega" not in stability_summary.columns \
            or stability_summary["mean_Omega"].notna().sum() == 0:
        return _skip(out_path, "no stability_summary.csv data yet")

    data = [stability_summary.loc[stability_summary["algorithm"] == a, "mean_Omega"].dropna().values
            for a in algos]
    fig, ax = plt.subplots(figsize=(1.6 * len(algos) + 2, 4.5))
    ax.boxplot(data, tick_labels=[ALGORITHM_INFO.get(a, {}).get("display_name", a) for a in algos])
    for i, vals in enumerate(data, start=1):
        if len(vals):
            ax.scatter(np.full(len(vals), i) + np.random.default_rng(0).normal(0, 0.04, len(vals)),
                       vals, alpha=0.5, s=14, color="black")
    ax.set_ylabel("mean Omega (run-to-run stability, higher = more stable)")
    ax.set_title("Stability comparison across algorithms\n(one point per graph x algorithm group)")
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_quality_comparison(qs_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 2: community-detection quality (Q for hard algorithms, EQ for SLPA -- never
    mixed onto a shared scale beyond sitting on the same categorical axis) compared across
    algorithms."""
    algos = _present_algorithms(qs_summary)
    if not algos:
        return _skip(out_path, "no quality_stability_summary.csv data yet")

    data, labels = [], []
    for a in algos:
        qcol = _quality_col(a)
        vals = qs_summary.loc[qs_summary["algorithm"] == a, qcol].dropna().values
        if len(vals):
            data.append(vals)
            labels.append(f"{ALGORITHM_INFO.get(a, {}).get('display_name', a)}\n({qcol.replace('mean_', '')})")
    if not data:
        return _skip(out_path, "no non-null quality values yet")

    fig, ax = plt.subplots(figsize=(1.6 * len(data) + 2, 4.5))
    ax.boxplot(data, tick_labels=labels)
    ax.set_ylabel("community quality (Q or EQ)")
    ax.set_title("Quality comparison across algorithms")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_stability_quality_relationship(qs_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 3: stability (mean Omega) vs quality (Q/EQ), one subplot per algorithm."""
    algos = _present_algorithms(qs_summary)
    if not algos or "mean_Omega" not in qs_summary.columns \
            or qs_summary["mean_Omega"].notna().sum() == 0:
        return _skip(out_path, "no stability/quality data yet")

    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 4), squeeze=False)
    any_plotted = False
    for ax, algo in zip(axes[0], algos):
        sub = qs_summary[qs_summary["algorithm"] == algo]
        qcol = _quality_col(algo)
        plotted = sub.dropna(subset=["mean_Omega", qcol])
        if len(plotted):
            any_plotted = True
            for dataset, g in plotted.groupby("dataset"):
                ax.scatter(g["mean_Omega"], g[qcol], label=dataset, alpha=0.7, s=28)
            ax.legend(fontsize=7)
        ax.set_xlabel("mean Omega (stability)")
        ax.set_ylabel(qcol.replace("mean_", ""))
        ax.set_title(ALGORITHM_INFO.get(algo, {}).get("display_name", algo))
        ax.set_xlim(-0.05, 1.05)
    if not any_plotted:
        plt.close(fig)
        return _skip(out_path, "no graph has both a stability and a quality value yet")
    fig.suptitle("Stability vs quality, by algorithm")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_lfr_mu_stability(qs_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 4: LFR mixing parameter mu (empirical preferred, nominal fallback) vs stability
    (mean Omega), one line/marker series per algorithm."""
    lfr = qs_summary[qs_summary.get("dataset") == "lfr"] if not qs_summary.empty else qs_summary
    mu_col = _mu_col(lfr) if not lfr.empty else None
    if lfr.empty or mu_col is None or "mean_Omega" not in lfr.columns \
            or lfr["mean_Omega"].notna().sum() == 0:
        return _skip(out_path, "no LFR stability data yet")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for algo in _present_algorithms(lfr):
        sub = lfr[lfr["algorithm"] == algo].dropna(subset=[mu_col, "mean_Omega"]).sort_values(mu_col)
        if len(sub):
            ax.scatter(sub[mu_col], sub["mean_Omega"], marker=ALGO_MARKERS.get(algo, "o"),
                       label=ALGORITHM_INFO.get(algo, {}).get("display_name", algo), alpha=0.75, s=30)
    ax.set_xlabel(f"LFR mixing parameter ({mu_col})")
    ax.set_ylabel("mean Omega (stability, higher = more stable)")
    ax.set_title("LFR mixing parameter vs stability")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_lfr_mu_quality(qs_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 5: LFR mixing parameter mu vs quality (Q/EQ), one series per algorithm."""
    lfr = qs_summary[qs_summary.get("dataset") == "lfr"] if not qs_summary.empty else qs_summary
    mu_col = _mu_col(lfr) if not lfr.empty else None
    if lfr.empty or mu_col is None:
        return _skip(out_path, "no LFR quality data yet")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    any_plotted = False
    for algo in _present_algorithms(lfr):
        qcol = _quality_col(algo)
        sub = lfr[lfr["algorithm"] == algo].dropna(subset=[mu_col, qcol]).sort_values(mu_col)
        if len(sub):
            any_plotted = True
            ax.scatter(sub[mu_col], sub[qcol], marker=ALGO_MARKERS.get(algo, "o"),
                       label=ALGORITHM_INFO.get(algo, {}).get("display_name", algo), alpha=0.75, s=30)
    if not any_plotted:
        plt.close(fig)
        return _skip(out_path, "no LFR graph has a quality value yet")
    ax.set_xlabel(f"LFR mixing parameter ({mu_col})")
    ax.set_ylabel("quality (Q or EQ)")
    ax.set_title("LFR mixing parameter vs quality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_community_structure(qs_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 6: community-count behavior -- mean community count (with std as an errorbar)
    across algorithms, pooling every graph with a stored value."""
    algos = _present_algorithms(qs_summary)
    if not algos or "mean_community_count" not in qs_summary.columns \
            or qs_summary["mean_community_count"].notna().sum() == 0:
        return _skip(out_path, "no community-count data yet")

    fig, ax = plt.subplots(figsize=(1.6 * len(algos) + 2, 4.5))
    xs, means, stds = [], [], []
    for i, algo in enumerate(algos):
        sub = qs_summary[qs_summary["algorithm"] == algo]
        vals = sub["mean_community_count"].dropna()
        if len(vals):
            xs.append(i)
            means.append(float(vals.mean()))
            stds.append(float(vals.std()) if len(vals) > 1 else 0.0)
    if not xs:
        plt.close(fig)
        return _skip(out_path, "no non-null community-count values yet")
    ax.errorbar(xs, means, yerr=stds, fmt="o", capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels([ALGORITHM_INFO.get(algos[i], {}).get("display_name", algos[i]) for i in xs])
    ax.set_ylabel("mean number of communities (+/- std across graphs)")
    ax.set_title("Community-count behavior across algorithms")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_real_network_comparison(qs_summary: pd.DataFrame, out_path: Path) -> bool:
    """Fig 7: per-real-dataset, per-algorithm stability (mean Omega) grouped bar chart."""
    real = qs_summary[qs_summary.get("dataset") != "lfr"] if not qs_summary.empty else qs_summary
    real = real.dropna(subset=["mean_Omega"]) if "mean_Omega" in real.columns else real
    if real.empty:
        return _skip(out_path, "no real-network stability data yet")

    datasets = sorted(real["graph_id"].dropna().unique())
    algos = _present_algorithms(real)
    if not datasets or not algos:
        return _skip(out_path, "no real-network stability data yet")

    width = 0.8 / max(len(algos), 1)
    fig, ax = plt.subplots(figsize=(1.4 * len(datasets) + 2, 4.5))
    for j, algo in enumerate(algos):
        heights = []
        for ds in datasets:
            row = real[(real["graph_id"] == ds) & (real["algorithm"] == algo)]
            heights.append(float(row["mean_Omega"].iloc[0]) if len(row) else np.nan)
        xs = np.arange(len(datasets)) + j * width
        ax.bar(xs, heights, width=width, label=ALGORITHM_INFO.get(algo, {}).get("display_name", algo))
    ax.set_xticks(np.arange(len(datasets)) + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("mean Omega (stability)")
    ax.set_title("Real-network comparison: stability by dataset and algorithm")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_experiment_completeness(runs: pd.DataFrame, cfg: dict, out_path: Path) -> bool:
    """Fig 8: data-availability / completeness summary -- planned vs completed
    (graph_id, algorithm) run counts. Always meaningful (including all-zero), since it
    reports on the experiment's progress rather than on a result the experiment would
    produce; this is the figure to consult before trusting any of figs 1-7."""
    real_datasets = list(cfg.get("real_datasets", []))
    n_lfr = int(cfg.get("lfr", {}).get("n_instances", 0)) * len(cfg.get("lfr", {}).get("mu_levels", []))
    n_graphs = len(real_datasets) + n_lfr
    n_runs_planned = int(cfg.get("runs", {}).get("n_runs", 0))
    algorithms = ALGO_ORDER

    if n_graphs == 0 or n_runs_planned == 0:
        return _skip(out_path, "config has no planned graphs/runs to report completeness against")

    completed = np.zeros((len(algorithms), 1))
    if not runs.empty:
        counts = runs.groupby("algorithm").size()
        for i, algo in enumerate(algorithms):
            completed[i, 0] = int(counts.get(algo, 0))
    planned_per_algo = n_graphs * n_runs_planned

    fig, ax = plt.subplots(figsize=(7.5, 4))
    fractions = (completed[:, 0] / planned_per_algo) if planned_per_algo else np.zeros(len(algorithms))
    ax.barh(range(len(algorithms)), fractions, color="steelblue")
    for i, (done, frac) in enumerate(zip(completed[:, 0], fractions)):
        ax.text(min(frac, 1.0) + 0.01, i, f"{int(done)}/{planned_per_algo}", va="center", fontsize=8)
    ax.set_yticks(range(len(algorithms)))
    ax.set_yticklabels([ALGORITHM_INFO.get(a, {}).get("display_name", a) for a in algorithms])
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("fraction of planned runs completed")
    total_done = int(completed.sum())
    total_planned = planned_per_algo * len(algorithms)
    ax.set_title(f"Experiment completeness: {total_done} / {total_planned} planned runs stored\n"
                 "(the full experiment has NOT been executed unless this reads 100% everywhere)",
                 fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


FIGURE_JOBS = [
    ("fig01_stability_comparison.png", fig_stability_comparison, "stability_summary"),
    ("fig02_quality_comparison.png", fig_quality_comparison, "qs_summary"),
    ("fig03_stability_quality_relationship.png", fig_stability_quality_relationship, "qs_summary"),
    ("fig04_lfr_mu_stability.png", fig_lfr_mu_stability, "qs_summary"),
    ("fig05_lfr_mu_quality.png", fig_lfr_mu_quality, "qs_summary"),
    ("fig06_community_structure.png", fig_community_structure, "qs_summary"),
    ("fig07_real_network_comparison.png", fig_real_network_comparison, "qs_summary"),
    ("fig08_experiment_completeness.png", fig_experiment_completeness, "runs_cfg"),
]


def make_final_figures(cfg: dict, results_dir: Path, figures_dir: Path | None = None) -> list[Path]:
    """Generate every Stage 10 figure into results/figures/final/ (or `figures_dir` if
    given). Read-only: only reads already-persisted Stage 5-9 outputs. Returns the paths
    actually written -- a figure with no supporting data is skipped, not written as a
    misleading empty/placeholder plot."""
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir) if figures_dir is not None else results_dir / "figures" / "final"

    stability_summary = load_stability_summary(results_dir)
    qs_summary = load_quality_stability_summary(results_dir)
    runs = load_runs(results_dir)

    written = []
    for filename, fn, needs in FIGURE_JOBS:
        out_path = figures_dir / filename
        if needs == "stability_summary":
            ok = fn(stability_summary, out_path)
        elif needs == "qs_summary":
            ok = fn(qs_summary, out_path)
        else:  # "runs_cfg"
            ok = fn(runs, cfg, out_path)
        if ok:
            written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def table_dataset_characteristics(cfg: dict, results_dir: Path) -> pd.DataFrame:
    """Table 1: one row per graph -- real datasets from the Stage 1 preprocessing log
    (n, m, ground truth availability), LFR graphs summarized per nominal mu level from
    Stage 8's lfr_instances.csv (never per-instance -- 70 rows would swamp the table;
    per-instance detail already lives in lfr_instances.csv itself)."""
    rows = []
    log = load_preprocessing_log(results_dir)
    for name in cfg.get("real_datasets", []):
        entry = log.get(name)
        if entry is None:
            continue
        rows.append({
            "graph_id": name, "family": "real", "n": entry.get("final_n"), "m": entry.get("final_m"),
            "mu_nominal": np.nan, "mu_empirical_mean": np.nan, "n_instances": 1,
            "ground_truth": entry.get("ground_truth"),
        })

    lfr = load_lfr_instances(results_dir)
    if not lfr.empty:
        for mu_nominal, g in lfr.groupby("mu_nominal", sort=True):
            rows.append({
                "graph_id": f"lfr_mu{mu_nominal}", "family": "lfr",
                "n": int(g["n"].iloc[0]) if "n" in g.columns and len(g) else np.nan,
                "m": float(g["m"].mean()) if "m" in g.columns else np.nan,
                "mu_nominal": mu_nominal,
                "mu_empirical_mean": float(g["mu_empirical"].mean()) if "mu_empirical" in g.columns else np.nan,
                "n_instances": int(len(g)),
                "ground_truth": "planted (LFR community assignment)",
            })
    return pd.DataFrame(rows, columns=["graph_id", "family", "n", "m", "mu_nominal",
                                        "mu_empirical_mean", "n_instances", "ground_truth"])


def table_algorithm_characteristics(cfg: dict) -> pd.DataFrame:
    """Table 2: static algorithm metadata (partition type, update scheme, citation, and
    the configured SLPA T/r knobs from config.yaml) -- describes the algorithms under
    comparison, not any experimental result."""
    rows = []
    slpa_cfg = cfg.get("slpa", {})
    for algo in ALGO_ORDER:
        info = ALGORITHM_INFO.get(algo, {})
        row = {
            "algorithm": algo, "display_name": info.get("display_name", algo),
            "partition_type": info.get("partition_type"), "update_scheme": info.get("update_scheme"),
            "citation": info.get("citation"),
            "config_T": slpa_cfg.get("T") if algo == "slpa" else np.nan,
            "config_r": slpa_cfg.get("r") if algo == "slpa" else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=["algorithm", "display_name", "partition_type",
                                        "update_scheme", "citation", "config_T", "config_r"])


def table_stability_statistics(stability_summary: pd.DataFrame) -> pd.DataFrame:
    """Table 3: descriptive stability statistics per algorithm, pooling every graph with a
    stored stability_summary row (each applicable metric only, per algorithm scope --
    never fabricated for a metric outside that algorithm's native scope)."""
    if stability_summary.empty:
        return pd.DataFrame()
    metrics = ["VI", "normalized_VI", "NMI", "Omega", "ONMI"]
    rows = []
    for algo, g in stability_summary.groupby("algorithm", sort=True):
        for metric in metrics:
            col = f"mean_{metric}"
            if col not in g.columns:
                continue
            vals = g[col].dropna()
            if not len(vals):
                continue
            rows.append({
                "algorithm": algo, "metric": metric, "n_graphs": int(len(vals)),
                "mean": float(vals.mean()), "median": float(vals.median()),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
                "min": float(vals.min()), "max": float(vals.max()),
            })
    return pd.DataFrame(rows, columns=["algorithm", "metric", "n_graphs", "mean", "median",
                                        "std", "min", "max"])


def table_quality_statistics(qs_summary: pd.DataFrame) -> pd.DataFrame:
    """Table 4: descriptive quality statistics per algorithm (Q for hard algorithms, EQ for
    SLPA -- read from each algorithm's own native column, never mixed)."""
    if qs_summary.empty:
        return pd.DataFrame()
    rows = []
    for algo, g in qs_summary.groupby("algorithm", sort=True):
        qcol = _quality_col(algo)
        if qcol not in g.columns:
            continue
        vals = g[qcol].dropna()
        if not len(vals):
            continue
        rows.append({
            "algorithm": algo, "quality_metric": qcol.replace("mean_", ""), "n_graphs": int(len(vals)),
            "mean": float(vals.mean()), "median": float(vals.median()),
            "std": float(vals.std()) if len(vals) > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()),
        })
    return pd.DataFrame(rows, columns=["algorithm", "quality_metric", "n_graphs", "mean",
                                        "median", "std", "min", "max"])


def table_lfr_mu_analysis(qs_summary: pd.DataFrame) -> pd.DataFrame:
    """Table 5: mean stability (Omega) and quality (Q/EQ) per algorithm x nominal-mu level,
    with the mean empirical mu actually observed at that level, computed from stored LFR
    rows only."""
    if qs_summary.empty:
        return pd.DataFrame()
    lfr = qs_summary[qs_summary.get("dataset") == "lfr"]
    if lfr.empty:
        return pd.DataFrame()
    rows = []
    for (algo, mu_nom), g in lfr.groupby(["algorithm", "mu_nominal"], sort=True, dropna=False):
        qcol = _quality_col(algo)
        omega = g["mean_Omega"].dropna() if "mean_Omega" in g.columns else pd.Series(dtype=float)
        quality = g[qcol].dropna() if qcol in g.columns else pd.Series(dtype=float)
        mu_emp = g["mu_empirical"].dropna() if "mu_empirical" in g.columns else pd.Series(dtype=float)
        rows.append({
            "algorithm": algo, "mu_nominal": mu_nom,
            "mu_empirical_mean": float(mu_emp.mean()) if len(mu_emp) else np.nan,
            "n_graph_instances": int(g["graph_id"].nunique()),
            "mean_stability_Omega": float(omega.mean()) if len(omega) else np.nan,
            "mean_quality": float(quality.mean()) if len(quality) else np.nan,
        })
    return pd.DataFrame(rows, columns=["algorithm", "mu_nominal", "mu_empirical_mean",
                                        "n_graph_instances", "mean_stability_Omega", "mean_quality"])


def table_statistical_significance(statistical_tests: pd.DataFrame, posthoc_tests: pd.DataFrame) -> pd.DataFrame:
    """Table 6: Friedman omnibus results and Holm-corrected Wilcoxon post-hoc results from
    Stage 9, concatenated (never recomputed) with an `analysis` column distinguishing the
    two, and every column either source produces (missing ones left NaN for that half)."""
    if statistical_tests.empty and posthoc_tests.empty:
        return pd.DataFrame()
    parts = []
    if not statistical_tests.empty:
        f = statistical_tests.copy()
        f["analysis"] = "friedman_omnibus"
        parts.append(f)
    if not posthoc_tests.empty:
        p = posthoc_tests.copy()
        p["analysis"] = "wilcoxon_posthoc_holm"
        parts.append(p)
    combined = pd.concat(parts, ignore_index=True, sort=False)
    cols = ["analysis"] + [c for c in combined.columns if c != "analysis"]
    return combined[cols]


def table_experiment_completeness(runs: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Table 7: summary of final experimental findings, WHERE they exist -- and otherwise
    (as here, before the 8,880-run experiment has been executed) a completeness report
    instead of a fabricated 'findings' table. Never creates an overall algorithm ranking or
    winner. Always meaningful, including all-zero rows."""
    real_datasets = list(cfg.get("real_datasets", []))
    n_lfr_graphs = int(cfg.get("lfr", {}).get("n_instances", 0)) * len(cfg.get("lfr", {}).get("mu_levels", []))
    n_runs_planned = int(cfg.get("runs", {}).get("n_runs", 0))
    n_graphs = len(real_datasets) + n_lfr_graphs

    rows = []
    for algo in ALGO_ORDER:
        planned = n_graphs * n_runs_planned
        completed = int(runs[runs["algorithm"] == algo].shape[0]) if not runs.empty else 0
        rows.append({
            "algorithm": algo, "graphs_planned": n_graphs, "runs_per_graph_planned": n_runs_planned,
            "runs_planned_total": planned, "runs_completed": completed,
            "fraction_complete": (completed / planned) if planned else np.nan,
        })
    return pd.DataFrame(rows, columns=["algorithm", "graphs_planned", "runs_per_graph_planned",
                                        "runs_planned_total", "runs_completed", "fraction_complete"])


TABLE_JOBS = [
    "table01_dataset_characteristics.csv",
    "table02_algorithm_characteristics.csv",
    "table03_stability_statistics.csv",
    "table04_quality_statistics.csv",
    "table05_lfr_mu_analysis.csv",
    "table06_statistical_significance.csv",
    "table07_experiment_completeness.csv",
]


def make_final_tables(cfg: dict, results_dir: Path, tables_dir: Path | None = None) -> list[Path]:
    """Generate every Stage 10 table into results/tables/final/ (or `tables_dir` if given).
    Read-only: only reads already-persisted Stage 1-9 outputs and config.yaml. A table with
    no supporting data (other than the always-available completeness table) is skipped, not
    written as an empty/misleading file. Returns the paths actually written."""
    results_dir = Path(results_dir)
    tables_dir = Path(tables_dir) if tables_dir is not None else results_dir / "tables" / "final"
    tables_dir.mkdir(parents=True, exist_ok=True)

    stability_summary = load_stability_summary(results_dir)
    qs_summary = load_quality_stability_summary(results_dir)
    statistical_tests = load_statistical_tests(results_dir)
    posthoc_tests = load_posthoc_tests(results_dir)
    runs = load_runs(results_dir)

    builders = {
        "table01_dataset_characteristics.csv": lambda: table_dataset_characteristics(cfg, results_dir),
        "table02_algorithm_characteristics.csv": lambda: table_algorithm_characteristics(cfg),
        "table03_stability_statistics.csv": lambda: table_stability_statistics(stability_summary),
        "table04_quality_statistics.csv": lambda: table_quality_statistics(qs_summary),
        "table05_lfr_mu_analysis.csv": lambda: table_lfr_mu_analysis(qs_summary),
        "table06_statistical_significance.csv": lambda: table_statistical_significance(
            statistical_tests, posthoc_tests),
        "table07_experiment_completeness.csv": lambda: table_experiment_completeness(runs, cfg),
    }

    written = []
    for filename in TABLE_JOBS:
        df = builders[filename]()
        out_path = tables_dir / filename
        if df is None or df.empty:
            print(f"  [stage10 tables] skipping {filename}: no supporting data yet")
            continue
        atomic_write_text(out_path, df.to_csv(index=False))
        written.append(out_path)
    return written


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def run_figure_generation(cfg: dict, results_dir: Path | None = None,
                           figures_dir: Path | None = None, tables_dir: Path | None = None,
                           verbose: bool = True) -> tuple[list[Path], list[Path]]:
    """Stage 10 entry point. Read-only with respect to results/raw/ and data/: reads
    whatever Stages 1-9 have already persisted under results/processed/ and writes final
    figures to results/figures/final/ and final tables to results/tables/final/. Returns
    (figure_paths_written, table_paths_written)."""
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir")

    figures = make_final_figures(cfg, results_dir, figures_dir)
    tables = make_final_tables(cfg, results_dir, tables_dir)

    if verbose:
        print(f"\nwrote {len(figures)} figure(s) under "
              f"{Path(figures_dir) if figures_dir is not None else results_dir / 'figures' / 'final'}")
        print(f"wrote {len(tables)} table(s) under "
              f"{Path(tables_dir) if tables_dir is not None else results_dir / 'tables' / 'final'}")
    return figures, tables
