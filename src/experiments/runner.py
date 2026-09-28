"""Stage 5 experiment runner: executes (graph, algorithm, seed) triples, persists a
partition + a metrics record for each (atomically, via storage.py), and rebuilds
results/raw/runs.csv from those records.

Resumability / idempotence: before running anything, `storage.is_run_complete` is
checked for that (graph_id, algorithm, seed); if already complete it is skipped (unless
force=True). This means the SAME command can be interrupted and re-run any number of
times without duplicating work or CSV rows -- it just finishes what's missing.

This module does NOT compute pairwise/cross-run stability (VI, NMI-across-runs, Omega,
ONMI-across-runs) -- that is Stage 6, computed later from the stored partitions without
re-running any algorithm. It also does not run statistical tests. See docs/methodology.md
("STAGE 5 MUST NOT DO").
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..algorithms import ALGORITHMS
from ..data_loader import load_graph
from ..graph import GraphBundle
from ..metrics.partitions import community_size_stats, cover_from_labels, is_trivial, n_communities
from ..metrics.quality import extended_modularity
from ..metrics.stability import nmi as nmi_score
from ..metrics.stability import onmi_lfk
from . import storage


def algorithm_kwargs(algorithm: str, cfg: dict) -> dict:
    """Extra keyword arguments beyond (graph, seed) for one algorithm, from config.yaml.
    Only SLPA has configurable extras (T, r); the other three use their own defaults."""
    if algorithm == "slpa":
        slpa_cfg = cfg.get("slpa", {})
        kwargs = {}
        if "T" in slpa_cfg:
            kwargs["T"] = int(slpa_cfg["T"])
        if "r" in slpa_cfg:
            kwargs["r"] = float(slpa_cfg["r"])
        return kwargs
    return {}


def graph_fields(graph: GraphBundle) -> dict:
    """The dataset/mu_nominal/mu_empirical/instance columns, derived from the graph
    itself -- real datasets carry none of the LFR fields; LFR graphs carry them in
    `meta` (written by src/lfr.py at generation time)."""
    if graph.family == "lfr":
        return {
            "dataset": "lfr",
            "mu_nominal": graph.meta.get("mu_nominal"),
            "mu_empirical": graph.meta.get("mu_empirical"),
            "instance": graph.meta.get("instance"),
        }
    return {"dataset": graph.family, "mu_nominal": None, "mu_empirical": None, "instance": None}


def compute_quality_and_truth_metrics(graph: GraphBundle, labels, cover) -> dict:
    """Q/EQ (whichever applies), plus NMI/ONMI vs. ground truth where truth exists.
    NMI is hard-partition-only (per stability.py's own scope note); ONMI is computed
    whenever truth exists, for both hard and overlapping results, via LFK's own
    definition (which does not claim to reduce exactly to NMI on hard partitions --
    see metrics/stability.py's module docstring). F1 is left as None: not implemented
    in src/metrics yet (see "RAW RESULT SCHEMA": "F1 if already implemented")."""
    n = graph.n
    quality = extended_modularity(graph.adj, labels=labels, cover=cover)
    quality_metric = "Q" if labels is not None else "EQ"

    nmi_val = None
    onmi_val = None
    if graph.truth is not None:
        truth_cover = cover_from_labels(graph.truth)
        if labels is not None:
            nmi_val = nmi_score(labels, graph.truth)
            result_cover = cover_from_labels(labels)
        else:
            result_cover = cover
        onmi_val = onmi_lfk(n, result_cover, truth_cover)

    stats = community_size_stats(labels=labels, cover=cover)
    return {
        "quality": quality,
        "quality_metric": quality_metric,
        "nmi": nmi_val,
        "onmi": onmi_val,
        "f1": None,
        "number_of_communities": n_communities(labels=labels, cover=cover),
        "largest_community_size": stats["size_max"],
        "degenerate": bool(is_trivial(labels=labels, cover=cover, n=n)),
    }


def run_single(results_dir: Path, cfg: dict, graph: GraphBundle, algorithm: str, seed: int,
               run_index: int, force: bool = False) -> dict | None:
    """Execute exactly one (graph, algorithm, seed) run and persist it atomically.
    Returns the record dict, or None if it was already complete and force=False
    (the "skip" case that makes reruns idempotent)."""
    graph_id = graph.graph_id
    if not force and storage.is_run_complete(results_dir, graph_id, algorithm, seed):
        return None

    fn = ALGORITHMS[algorithm]
    kwargs = algorithm_kwargs(algorithm, cfg)

    t0 = time.perf_counter()
    result = fn(graph, seed=seed, **kwargs)
    runtime = time.perf_counter() - t0            # algorithm execution only -- no I/O inside this window

    metrics = compute_quality_and_truth_metrics(graph, result.labels, result.cover)

    ppath = storage.partition_path(results_dir, graph_id, algorithm, seed)
    if result.labels is not None:
        storage.save_hard_partition(ppath, result.labels)
        partition_kind = "hard"
    else:
        storage.save_cover_partition(ppath, result.cover)
        partition_kind = "cover"

    record = {
        "graph_id": graph_id,
        **graph_fields(graph),
        "algorithm": algorithm,
        "run": run_index,
        "seed": seed,
        **metrics,
        "runtime_seconds": runtime,
        "converged": result.converged,
        "iterations": result.iterations,
        "partition_file": str(ppath.relative_to(Path(results_dir))),
        "partition_kind": partition_kind,
    }
    storage.save_record(storage.record_path(results_dir, graph_id, algorithm, seed), record)
    return record


def run_experiments(cfg: dict, graph_ids: list, algorithms: list, seeds: list,
                     results_dir: Path | None = None, force: bool = False,
                     verbose: bool = True):
    """Run every (graph, algorithm, seed) combination, skipping ones already complete
    (unless force=True), then rebuild results/raw/runs.csv from all persisted records.
    Returns (runs_df, n_executed, n_skipped).

    Each entry of `graph_ids` is normally a graph_id string (dispatched through
    data_loader.load_graph, i.e. a real dataset name or an 'lfr_muXX_iYY' id). A
    GraphBundle can be passed directly instead -- used by the Stage 5 tests, which run
    against a tiny synthetic graph that was never generated/cached as a real dataset."""
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir") / "raw"

    n_executed = 0
    n_skipped = 0
    for graph_or_id in graph_ids:
        graph = graph_or_id if isinstance(graph_or_id, GraphBundle) else load_graph(graph_or_id, cfg)
        graph_id = graph.graph_id
        for algorithm in algorithms:
            for run_index, seed in enumerate(seeds):
                rec = run_single(results_dir, cfg, graph, algorithm, seed, run_index, force=force)
                if rec is None:
                    n_skipped += 1
                    if verbose:
                        print(f"  skip  {storage.run_key(graph_id, algorithm, seed)} (already complete)")
                else:
                    n_executed += 1
                    if verbose:
                        q = rec["quality"]
                        print(f"  ran   {storage.run_key(graph_id, algorithm, seed)}  "
                              f"k={rec['number_of_communities']:<4d} Q/EQ={q:.4f}  "
                              f"{rec['runtime_seconds']:.3f}s")

    df = storage.rebuild_runs_csv(results_dir)
    if verbose:
        print(f"\n{n_executed} run, {n_skipped} skipped (already complete). "
              f"results/raw/runs.csv now has {len(df)} rows.")
    return df, n_executed, n_skipped
