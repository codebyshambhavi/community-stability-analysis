"""Shared partition/cover representations used by every metric.

Two representations are used throughout the project:
  - HARD partition: labels, an (n,) int array, canonical 0..k-1 (one community per node).
    Produced by LPA, semi-sync LPA, FLPA.
  - COVER (overlapping): a list of node-id sets, one set per community. A node may appear
    in zero, one, or several sets. Produced by SLPA.

Every stability/quality metric that needs the (n x k) binary membership matrix goes through
`membership_matrix()` so hard and overlapping partitions are handled by the exact same code path.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def canonicalize_labels(labels: np.ndarray) -> np.ndarray:
    """Relabel to 0..k-1 by ascending original value (np.unique's inverse). The specific
    relabeling is arbitrary and never affects VI/NMI/Omega, which are permutation-invariant."""
    _, inv = np.unique(np.asarray(labels), return_inverse=True)
    return inv.astype(np.int64)


def cover_from_labels(labels: np.ndarray) -> list[set[int]]:
    """Hard partition -> cover representation (one community per label)."""
    labels = np.asarray(labels)
    k = int(labels.max()) + 1 if len(labels) else 0
    return [set(np.flatnonzero(labels == c).tolist()) for c in range(k)]


def membership_matrix(n: int, labels: np.ndarray | None = None,
                       cover: list[set[int]] | None = None) -> sp.csr_matrix:
    """Build the (n, k) binary membership matrix from either representation. Exactly one of
    labels/cover must be given. A cover community may be empty; empty communities are dropped."""
    if (labels is None) == (cover is None):
        raise ValueError("pass exactly one of labels= or cover=")
    if labels is not None:
        labels = canonicalize_labels(np.asarray(labels))
        if len(labels) != n:
            raise ValueError(f"labels length {len(labels)} != n {n}")
        k = int(labels.max()) + 1 if n else 0
        rows, cols = np.arange(n), labels
        data = np.ones(n, dtype=np.int64)
        return sp.csr_matrix((data, (rows, cols)), shape=(n, k))
    cover = [c for c in cover if c]                      # drop empty communities
    k = len(cover)
    rows, cols = [], []
    for cid, community in enumerate(cover):
        for v in community:
            if not (0 <= v < n):
                raise ValueError(f"node id {v} out of range [0, {n})")
            rows.append(v)
            cols.append(cid)
    data = np.ones(len(rows), dtype=np.int64)
    return sp.csr_matrix((data, (rows, cols)), shape=(n, max(k, 1)) if k == 0 else (n, k))


def n_communities(labels: np.ndarray | None = None, cover: list[set[int]] | None = None) -> int:
    if labels is not None:
        return int(np.asarray(labels).max()) + 1 if len(labels) else 0
    return sum(1 for c in cover if c)


def community_size_stats(labels: np.ndarray | None = None, cover: list[set[int]] | None = None) -> dict:
    """Size distribution of communities (works for hard or overlapping). Node counts, not fractions."""
    if labels is not None:
        sizes = np.bincount(canonicalize_labels(np.asarray(labels)))
    else:
        sizes = np.array([len(c) for c in cover if c], dtype=np.int64)
    if len(sizes) == 0:
        return {"n_communities": 0, "size_mean": np.nan, "size_std": np.nan,
                "size_min": np.nan, "size_max": np.nan, "largest_fraction": np.nan}
    total = sizes.sum()
    return {
        "n_communities": int(len(sizes)),
        "size_mean": float(sizes.mean()), "size_std": float(sizes.std()),
        "size_min": int(sizes.min()), "size_max": int(sizes.max()),
        "largest_fraction": float(sizes.max() / total) if total else np.nan,
    }


def is_trivial(labels: np.ndarray | None = None, cover: list[set[int]] | None = None, n: int | None = None) -> bool:
    """A degenerate partition: everyone in one community, or every community a singleton.
    Trivial partitions make several stability metrics artificially perfect, so runs are flagged."""
    k = n_communities(labels, cover)
    nn = len(labels) if labels is not None else n
    return k <= 1 or (nn is not None and k >= nn)
