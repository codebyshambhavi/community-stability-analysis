"""Shared plumbing for the four LPA-family algorithm implementations.

`AlgorithmResult` is the single return type every algorithm in this package
produces, so the (future) experiment runner has one interface regardless of
whether the algorithm produced a hard partition (LPA, semi-sync LPA, FLPA) or
an overlapping cover (SLPA).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from ..metrics.partitions import canonicalize_labels


def rng_from_seed(seed: int) -> random.Random:
    """Every algorithm gets its OWN fresh random.Random(seed) -- never a shared/global
    RNG -- so runs are independent of call order and reproducible in isolation."""
    return random.Random(seed)


def labels_from_communities(communities, n: int) -> np.ndarray:
    """Hard partition: an iterable of node-id sets/iterables (e.g. from NetworkX) -> a
    canonical (n,) label array. Every node must appear in exactly one community."""
    labels = np.full(n, -1, dtype=np.int64)
    for cid, nodes in enumerate(communities):
        for v in nodes:
            labels[int(v)] = cid
    if (labels < 0).any():
        missing = np.flatnonzero(labels < 0).tolist()
        raise ValueError(f"nodes missing from communities: {missing[:10]}{'...' if len(missing) > 10 else ''}")
    return canonicalize_labels(labels)


@dataclass(frozen=True)
class AlgorithmResult:
    """Exactly one of labels/cover is set (mirrors the labels=/cover= convention used
    throughout src/metrics). `converged`/`iterations` are None for algorithms (FLPA) whose
    stopping rule isn't a fixed-point sweep count; `meta` carries algorithm-specific extras
    (e.g. SLPA's per-node label-frequency memory, used for r-sensitivity analysis later)."""
    labels: np.ndarray | None = None
    cover: list | None = None
    converged: bool | None = None
    iterations: int | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if (self.labels is None) == (self.cover is None):
            raise ValueError("AlgorithmResult: pass exactly one of labels= or cover=")
