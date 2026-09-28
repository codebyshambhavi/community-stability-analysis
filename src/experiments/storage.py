"""Atomic persistence for Stage 5 (the experiment runner).

Two artefacts are written per (graph_id, algorithm, seed) run, both via atomic
write-then-rename so a killed process can never leave a *partially written* file
that later code would mistake for a complete one:

  - a PARTITION file  (results/raw/partitions/<graph_id>/<algorithm>/seed<SSS>.csv)
    the raw community structure: node->label (hard) or community->node (cover).
  - a RECORD file     (results/raw/records/<graph_id>/<algorithm>/seed<SSS>.json)
    everything else the CSV schema (RUN_COLUMNS) needs for that run, plus a pointer
    to the partition file and which representation it uses.

`results/raw/runs.csv` is never appended to directly (a crash mid-append could corrupt
or truncate it). Instead it is fully REBUILT from the record files by `rebuild_runs_csv`,
itself written atomically -- so rebuilding is idempotent and safe to call after every run
or once at the end of a batch. A run counts as COMPLETE only if both its record file AND
its partition file exist and parse successfully; anything else (missing, truncated,
mismatched) is treated as not-yet-done and will be re-executed.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Canonical column order for results/raw/runs.csv (matches the handoff's "RAW RESULT SCHEMA").
RUN_COLUMNS = [
    "graph_id", "dataset", "mu_nominal", "mu_empirical", "instance",
    "algorithm", "run", "seed",
    "number_of_communities", "quality", "quality_metric",
    "nmi", "onmi", "f1",
    "runtime_seconds", "largest_community_size", "degenerate",
    "converged", "iterations",
    "partition_file", "partition_kind",
]


def run_key(graph_id: str, algorithm: str, seed: int) -> str:
    return f"{graph_id}__{algorithm}__seed{seed:03d}"


def partition_path(results_dir: Path, graph_id: str, algorithm: str, seed: int) -> Path:
    return Path(results_dir) / "partitions" / graph_id / algorithm / f"seed{seed:03d}.csv"


def record_path(results_dir: Path, graph_id: str, algorithm: str, seed: int) -> Path:
    return Path(results_dir) / "records" / graph_id / algorithm / f"seed{seed:03d}.json"


# ---------------------------------------------------------------------------
# Atomic write primitive
# ---------------------------------------------------------------------------

def atomic_write_text(path: Path, text: str) -> None:
    """Write `text` to `path` atomically: write to a sibling temp file, fsync, then
    os.replace (atomic rename on the same filesystem). A reader can never observe a
    partially-written file at `path` -- it's either the old content or the new content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# Partition (hard / cover) persistence
# ---------------------------------------------------------------------------

def save_hard_partition(path: Path, labels: np.ndarray) -> None:
    lines = ["node,label"] + [f"{i},{int(lbl)}" for i, lbl in enumerate(labels)]
    atomic_write_text(path, "\n".join(lines) + "\n")


def load_hard_partition(path: Path) -> np.ndarray:
    """Raises on any malformed/truncated file -- callers use this as the validity check."""
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["node", "label"]:
            raise ValueError(f"unexpected header in {path}: {header}")
        rows = [(int(r[0]), int(r[1])) for r in reader]
    if not rows:
        raise ValueError(f"empty partition file: {path}")
    nodes = [r[0] for r in rows]
    n = max(nodes) + 1
    if sorted(nodes) != list(range(n)):
        raise ValueError(f"partition file {path} does not cover nodes 0..{n - 1} exactly once")
    labels = np.zeros(n, dtype=np.int64)
    for node, lbl in rows:
        labels[node] = lbl
    return labels


def save_cover_partition(path: Path, cover: list) -> None:
    lines = ["community,node"]
    for cid, community in enumerate(cover):
        for v in sorted(community):
            lines.append(f"{cid},{v}")
    atomic_write_text(path, "\n".join(lines) + "\n")


def load_cover_partition(path: Path) -> list:
    """Raises on any malformed/truncated file -- callers use this as the validity check."""
    communities: dict[int, set] = defaultdict(set)
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["community", "node"]:
            raise ValueError(f"unexpected header in {path}: {header}")
        for row in reader:
            if not row:
                continue
            cid, v = int(row[0]), int(row[1])
            communities[cid].add(v)
    if not communities:
        raise ValueError(f"empty cover partition file: {path}")
    return [communities[c] for c in sorted(communities)]


def load_partition(path: Path, kind: str):
    if kind == "hard":
        return load_hard_partition(path)
    if kind == "cover":
        return load_cover_partition(path)
    raise ValueError(f"unknown partition kind: {kind!r}")


# ---------------------------------------------------------------------------
# Record persistence + completion detection
# ---------------------------------------------------------------------------

def save_record(path: Path, record: dict) -> None:
    atomic_write_text(path, json.dumps(record, indent=2, sort_keys=True, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def load_record(path: Path) -> dict | None:
    """None for missing OR corrupt/unparseable -- both mean 'treat as incomplete'."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def is_run_complete(results_dir: Path, graph_id: str, algorithm: str, seed: int) -> bool:
    """A run is complete iff its record parses AND its partition file parses AND the
    record's own partition_file field matches where we'd look. Anything else (missing
    record, missing partition, truncated/corrupt file, mismatched pointer) is incomplete
    and will be (re-)executed."""
    rec = load_record(record_path(results_dir, graph_id, algorithm, seed))
    if rec is None:
        return False
    ppath = partition_path(results_dir, graph_id, algorithm, seed)
    expected_rel = str(ppath.relative_to(Path(results_dir)))
    if rec.get("partition_file") != expected_rel:
        return False
    if not ppath.exists():
        return False
    try:
        load_partition(ppath, rec.get("partition_kind", ""))
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# runs.csv assembly (always a full, atomic rebuild -- never an append)
# ---------------------------------------------------------------------------

def rebuild_runs_csv(results_dir: Path) -> pd.DataFrame:
    """Scan every record under results/raw/records/, keep only ones whose partition
    file still loads successfully, and (re)write results/raw/runs.csv from scratch.
    Idempotent and safe to call repeatedly (e.g. after every run, or once at the end)."""
    results_dir = Path(results_dir)
    records_dir = results_dir / "records"
    rows = []
    if records_dir.exists():
        for rp in sorted(records_dir.rglob("*.json")):
            rec = load_record(rp)
            if rec is None:
                continue
            ppath = results_dir / rec.get("partition_file", "")
            try:
                load_partition(ppath, rec.get("partition_kind", ""))
            except Exception:
                continue
            rows.append({col: rec.get(col) for col in RUN_COLUMNS})

    df = pd.DataFrame(rows, columns=RUN_COLUMNS)
    if len(df):
        df = df.sort_values(["dataset", "graph_id", "algorithm", "seed"]).reset_index(drop=True)
    atomic_write_text(results_dir / "runs.csv", df.to_csv(index=False))
    return df
