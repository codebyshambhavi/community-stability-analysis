"""Stage 7 figures: read-only visualizations built from
results/processed/quality_stability_summary.csv (never recomputes quality or stability;
never touches results/raw/). Each figure function is guarded against empty/missing data --
if nothing is available yet (e.g. the 8,880-run experiment has not been executed in this
environment), it prints a note and skips rather than raising or fabricating a plot.

Kept deliberately separate from `src/visualization.py`, which is reserved for the Stage 10
figures/tables stage and is not implemented yet.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .stability import OVERLAPPING_ALGORITHMS

ALGO_ORDER = ["lpa", "semi_sync_lpa", "flpa", "slpa"]
ALGO_MARKERS = {"lpa": "o", "semi_sync_lpa": "s", "flpa": "^", "slpa": "D"}


def _quality_col(algorithm: str) -> str:
    return "mean_EQ" if algorithm in OVERLAPPING_ALGORITHMS else "mean_Q"


def _present_algorithms(df: pd.DataFrame) -> list[str]:
    present = set(df["algorithm"].dropna())
    return [a for a in ALGO_ORDER if a in present]


def fig_stability_vs_quality(df: pd.DataFrame, out_path: Path) -> bool:
    """Stability (mean Omega -- the one metric common to every algorithm; HIGHER = more
    stable, see stability.py's module docstring) vs quality (Q for hard algorithms, EQ for
    SLPA), one subplot per algorithm."""
    algos = _present_algorithms(df)
    if not algos or df["mean_Omega"].notna().sum() == 0:
        print(f"  [stage7 figures] skipping {out_path.name}: no stability/quality data yet")
        return False

    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 4), squeeze=False)
    for ax, algo in zip(axes[0], algos):
        sub = df[df["algorithm"] == algo]
        qcol = _quality_col(algo)
        plotted = sub.dropna(subset=["mean_Omega", qcol])
        for dataset, g in plotted.groupby("dataset"):
            ax.scatter(g["mean_Omega"], g[qcol], label=dataset, alpha=0.7, s=28)
        ax.set_xlabel("mean Omega (stability, higher = more stable)")
        ax.set_ylabel(qcol.replace("mean_", "mean "))
        ax.set_title(algo)
        ax.set_xlim(-0.05, 1.05)
        if len(plotted):
            ax.legend(fontsize=7)
    fig.suptitle("Run-to-run stability (Omega) vs community quality, by algorithm")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_stability_vs_ground_truth(df: pd.DataFrame, out_path: Path) -> bool:
    """Stability (Omega) vs stability's own agreement with GROUND TRUTH (mean_ground_truth_NMI
    for hard algorithms, mean_ground_truth_ONMI for SLPA) -- only rows where ground truth
    actually exists (karate, lfr_*; never dolphins/polbooks/polblogs, per the project's
    ground-truth policy)."""
    algos = _present_algorithms(df)
    rows = []
    for algo in algos:
        truth_col = "mean_ground_truth_ONMI" if algo in OVERLAPPING_ALGORITHMS else "mean_ground_truth_NMI"
        sub = df[df["algorithm"] == algo].dropna(subset=["mean_Omega", truth_col])
        if len(sub):
            rows.append((algo, truth_col, sub))
    if not rows:
        print(f"  [stage7 figures] skipping {out_path.name}: no ground-truth-bearing graphs in the data yet")
        return False

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for algo, truth_col, sub in rows:
        ax.scatter(sub["mean_Omega"], sub[truth_col], label=f"{algo} ({truth_col.split('_')[-1]})",
                   marker=ALGO_MARKERS.get(algo, "o"), alpha=0.7, s=32)
    ax.set_xlabel("mean Omega (run-to-run stability, higher = more stable)")
    ax.set_ylabel("mean ground-truth NMI / ONMI (higher = closer to ground truth)")
    ax.set_title("Stability vs agreement with ground truth\n(only graphs with verified ground truth)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_community_count_variability_vs_stability(df: pd.DataFrame, out_path: Path) -> bool:
    """Community-count variability (std_community_count across runs) vs stability (Omega),
    one subplot per algorithm. High std with high Omega would flag exactly the "degenerate
    collapse looks falsely stable" case the project handoff warns about -- degenerate_run_count
    is annotated on points where it is nonzero."""
    algos = _present_algorithms(df)
    plotted_any = False
    fig, axes = plt.subplots(1, len(algos), figsize=(4.2 * len(algos), 4), squeeze=False) if algos else (None, None)
    for ax, algo in zip(axes[0], algos) if algos else []:
        sub = df[df["algorithm"] == algo].dropna(subset=["std_community_count", "mean_Omega"])
        if len(sub):
            plotted_any = True
            degenerate = sub["degenerate_run_count"].fillna(0) > 0
            ax.scatter(sub.loc[~degenerate, "std_community_count"], sub.loc[~degenerate, "mean_Omega"],
                       alpha=0.7, s=28, label="no degenerate runs")
            ax.scatter(sub.loc[degenerate, "std_community_count"], sub.loc[degenerate, "mean_Omega"],
                       alpha=0.9, s=40, marker="x", color="red", label="has degenerate run(s)")
            ax.legend(fontsize=7)
        ax.set_xlabel("std(number of communities) across runs")
        ax.set_ylabel("mean Omega (stability)")
        ax.set_title(algo)
    if not plotted_any:
        print(f"  [stage7 figures] skipping {out_path.name}: no community-count data yet")
        if fig is not None:
            plt.close(fig)
        return False
    fig.suptitle("Community-count variability vs stability, by algorithm\n"
                  "(red x = group includes at least one degenerate/trivial run)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def fig_quality_vs_stability_by_dataset(df: pd.DataFrame, out_path: Path) -> bool:
    """Quality vs stability (Omega), one subplot per dataset (real datasets each get one
    point per algorithm; LFR is faceted by empirical mu via color, per the project handoff's
    instruction to preserve mu rather than immediately collapsing the 70 LFR graphs)."""
    datasets = [d for d in df["dataset"].dropna().unique()]
    if not datasets or df["mean_Omega"].notna().sum() == 0:
        print(f"  [stage7 figures] skipping {out_path.name}: no data yet")
        return False
    datasets = sorted(datasets, key=lambda d: (d != "lfr", d))  # real datasets first, lfr last

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4), squeeze=False)
    for ax, dataset in zip(axes[0], datasets):
        sub = df[df["dataset"] == dataset]
        for algo in _present_algorithms(sub):
            g = sub[sub["algorithm"] == algo]
            qcol = _quality_col(algo)
            plotted = g.dropna(subset=["mean_Omega", qcol])
            if not len(plotted):
                continue
            if dataset == "lfr" and plotted["mu_empirical"].notna().any():
                sc = ax.scatter(plotted["mean_Omega"], plotted[qcol], c=plotted["mu_empirical"],
                                 cmap="viridis", marker=ALGO_MARKERS.get(algo, "o"),
                                 label=algo, alpha=0.75, s=26)
            else:
                ax.scatter(plotted["mean_Omega"], plotted[qcol], marker=ALGO_MARKERS.get(algo, "o"),
                           label=algo, alpha=0.75, s=32)
        ax.set_xlabel("mean Omega (stability)")
        ax.set_ylabel("Q / EQ")
        ax.set_title(dataset)
        ax.legend(fontsize=7)
    fig.suptitle("Quality vs stability by dataset (LFR points colored by empirical mu)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def make_stage7_figures(summary: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Generate every Stage 7 figure into `figures_dir`. Returns the paths actually written
    (a figure that has no supporting data is skipped, not written as an empty/misleading
    plot)."""
    figures_dir = Path(figures_dir)
    jobs = [
        (fig_stability_vs_quality, figures_dir / "stability_vs_quality.png"),
        (fig_stability_vs_ground_truth, figures_dir / "stability_vs_ground_truth.png"),
        (fig_community_count_variability_vs_stability, figures_dir / "community_count_variability_vs_stability.png"),
        (fig_quality_vs_stability_by_dataset, figures_dir / "quality_vs_stability_by_dataset.png"),
    ]
    written = []
    for fn, path in jobs:
        if fn(summary, path):
            written.append(path)
    return written
