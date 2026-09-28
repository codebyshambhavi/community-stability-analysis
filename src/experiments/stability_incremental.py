"""Stage 6 (incremental / checkpointed driver).

Wraps the EXISTING Stage 6 methodology in src/experiments/stability.py (metric definitions,
bootstrap count, seeds, aggregation are all reused unchanged -- `analyze_group` is called as-is).
Only the orchestration differs: every (graph_id, algorithm) group is persisted to
results/processed/stability_chunks/ as soon as it finishes, so an interrupted session resumes
from the first incomplete group without repeating completed work.

Determinism: identical to `run_stability_analysis` -- one SeedSequence(bootstrap_seed) is spawned into
one child per group, indexed by the group's position in the sorted (graph_id, algorithm) list.
The final assembly writes pairwise.csv / stability_summary.csv with the same ordering and format.

Per group, two files: <graph>__<algo>.pairwise.csv and <graph>__<algo>.summary.json. The summary
JSON is written LAST (atomically); a group is COMPLETE iff its summary JSON parses and its pairwise
CSV has the expected number of rows. progress.json is a human-readable manifest.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import storage
from .stability import (DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP, PAIRWISE_COLUMNS,
                        analyze_group, load_runs)


def _paths(chunk_dir: Path, graph_id: str, algorithm: str):
    stem = f"{graph_id}__{algorithm}"
    return chunk_dir / f"{stem}.pairwise.csv", chunk_dir / f"{stem}.summary.json"


def _is_complete(chunk_dir: Path, graph_id: str, algorithm: str, n_bootstrap: int) -> bool:
    pw_path, sm_path = _paths(chunk_dir, graph_id, algorithm)
    if not (pw_path.exists() and sm_path.exists()):
        return False
    try:
        summ = json.loads(sm_path.read_text(encoding="utf-8"))
        if summ.get("n_bootstrap") != n_bootstrap:
            return False
        pw = pd.read_csv(pw_path)
        return len(pw) == summ["number_of_pairs"]
    except Exception:
        return False


def _write_progress(chunk_dir: Path, total: int, done: list, t_start: float, extra: dict) -> None:
    payload = {"groups_total": total, "groups_completed": len(done),
               "fraction": round(len(done) / total, 4) if total else 0.0,
               "elapsed_seconds_this_session": round(time.time() - t_start, 1),
               "completed_groups": done, **extra}
    storage.atomic_write_text(chunk_dir / "progress.json", json.dumps(payload, indent=1))


def run_stability_incremental(cfg: dict, results_dir: Path | None = None,
                              n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
                              bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
                              max_groups: int | None = None, verbose: bool = True,
                              assemble: bool = True):
    from ..config import resolve
    results_dir = Path(results_dir) if results_dir is not None else resolve(cfg, "results_dir")
    raw_dir, proc_dir = results_dir / "raw", results_dir / "processed"
    chunk_dir = proc_dir / "stability_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(results_dir)
    groups = list(runs.groupby(["graph_id", "algorithm"], sort=True))
    child_seeds = np.random.SeedSequence(bootstrap_seed).spawn(len(groups))
    total = len(groups)
    t_start = time.time()
    done = [f"{g}__{a}" for (g, a), _ in groups if _is_complete(chunk_dir, g, a, n_bootstrap)]
    extra = {"n_bootstrap": n_bootstrap, "bootstrap_seed": bootstrap_seed}
    if verbose:
        print(f"stability-incremental: {len(done)}/{total} groups already checkpointed")

    n_new = 0
    for pos, ((graph_id, algorithm), rows) in enumerate(groups):
        if _is_complete(chunk_dir, graph_id, algorithm, n_bootstrap):
            continue
        if max_groups is not None and n_new >= max_groups:
            break
        t0 = time.time()
        pw, summary = analyze_group(raw_dir, rows, graph_id, algorithm, n_bootstrap,
                                    child_seeds[pos], verbose=False)
        pw_path, sm_path = _paths(chunk_dir, graph_id, algorithm)
        storage.atomic_write_text(pw_path, pw.to_csv(index=False))
        storage.atomic_write_text(sm_path, json.dumps(summary, default=_jd))   # written LAST
        done.append(f"{graph_id}__{algorithm}")
        n_new += 1
        _write_progress(chunk_dir, total, done, t_start, extra)
        if verbose:
            print(f"  [{len(done):3d}/{total}] {graph_id:16s} {algorithm:14s} "
                  f"pairs={len(pw)} {time.time()-t0:.1f}s", flush=True)
    _write_progress(chunk_dir, total, done, t_start, extra)

    if assemble and len(done) == total:
        return assemble_stability(groups, chunk_dir, proc_dir, n_bootstrap, bootstrap_seed, verbose)
    return None


def _jd(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def assemble_stability(groups, chunk_dir: Path, proc_dir: Path, n_bootstrap: int,
                       bootstrap_seed: int, verbose: bool = True):
    """Concatenate chunks in the same order as run_stability_analysis and write the same files."""
    frames, rows = [], []
    for (graph_id, algorithm), _ in groups:
        pw_path, sm_path = _paths(chunk_dir, graph_id, algorithm)
        frames.append(pd.read_csv(pw_path))
        rows.append(json.loads(sm_path.read_text(encoding="utf-8")))
    pairwise_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=PAIRWISE_COLUMNS)
    summary_df = pd.DataFrame(rows)
    summary_df = summary_df.sort_values(["dataset", "graph_id", "algorithm"]).reset_index(drop=True)
    storage.atomic_write_text(proc_dir / "pairwise.csv", pairwise_df.to_csv(index=False))
    storage.atomic_write_text(proc_dir / "stability_summary.csv", summary_df.to_csv(index=False))
    if verbose:
        print(f"assembled pairwise.csv ({len(pairwise_df)} rows), stability_summary.csv ({len(summary_df)} rows)")
    return pairwise_df, summary_df
