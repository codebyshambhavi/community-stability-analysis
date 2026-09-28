"""Semi-synchronous Label Propagation Algorithm.

Cordasco & Gargano (2010), "Community detection via semi-synchronous label propagation
algorithms", IEEE BASNA workshop.

Why "semi-synchronous": fully asynchronous LPA (one node updated at a time, using the
freshest neighbour labels) never oscillates but is inherently sequential; fully
synchronous LPA (every node updates simultaneously from the previous round) is
embarrassingly parallel but can flip-flop between two labelings forever on bipartite-like
structure. Cordasco & Gargano's fix: properly COLOUR the graph (no two adjacent nodes
share a colour), then update one colour class at a time, synchronously *within* a class.
Because same-colour nodes are never neighbours, updating them "simultaneously" is
race-free -- concretely, each node can just be updated in place; its same-colour peers
never read its label during this pass, since they never depend on it.

TIE-BREAKING: Prec-Max (Cordasco & Gargano's own rule, reproduced here exactly as
NetworkX's semi-synchronous implementation documents it): among the tied
maximum-frequency neighbour labels, the node's CURRENT label is kept if it is one of
them; otherwise the tied label with the numerically LARGEST value is chosen. This is why
step (1) below assigns RANDOM label values, not just a random node order -- Prec-Max's
outcome depends on the actual numbers, so randomizing which node holds which numeric
label is what makes the tie-breaking (and hence the whole run) stochastic.

WHY THIS IS "PAPER-FAITHFUL" AND NOT NetworkX's `label_propagation_communities`:
NetworkX's built-in version is DETERMINISTIC -- it assigns labels via plain node
enumeration order (labels[v] = index of v) and colours the graph once with
`nx.coloring.greedy_color`'s default (non-random) strategy, so it always returns the
same partition for a given graph, with no seed argument. We instead randomize BOTH the
initial label values and the node order fed to greedy colouring, each from `seed`, so
repeated runs of the *same* graph explore different colourings/label orderings, matching
the paper's description of the algorithm as randomized. See README "Decisions fixed so
far" and tests/test_algorithms.py::test_semi_sync_matches_networkx_when_derandomized for
a proof that, with randomization disabled, our implementation reduces exactly to
NetworkX's reference behaviour on the same coloring.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..graph import GraphBundle
from ..metrics.partitions import canonicalize_labels
from .common import AlgorithmResult, rng_from_seed


def greedy_coloring(adj: list[list[int]], order: list[int]) -> dict[int, int]:
    """Sequential greedy colouring in the given node order: colour(v) = smallest
    non-negative integer not used by any already-coloured neighbour of v. Produces a
    PROPER colouring (no two adjacent nodes share a colour) for any order."""
    color: dict[int, int] = {}
    for v in order:
        used = {color[u] for u in adj[v] if u in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    return color


def _most_frequent_labels(v: int, labels: list[int], adj: list[list[int]]) -> list[int]:
    if not adj[v]:
        return [labels[v]]                  # isolated node: trivially "complete"
    counts = Counter(labels[u] for u in adj[v])
    max_freq = max(counts.values())
    return [lbl for lbl, c in counts.items() if c == max_freq]


def _converged(labels: list[int], adj: list[list[int]]) -> bool:
    return all(labels[v] in _most_frequent_labels(v, labels, adj)
               for v in range(len(labels)) if adj[v])


def _run_core(adj: list[list[int]], n: int, initial_labels: list[int],
              color_classes: list[list[int]], max_iter: int) -> tuple[list[int], bool, int]:
    """The deterministic update dynamics, factored out so tests can feed it NetworkX's
    own (unrandomized) initial labeling + coloring and check for an exact match."""
    labels = list(initial_labels)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        for cls in color_classes:           # colour classes processed in a fixed order each round
            for v in cls:                   # simultaneous in principle; race-free since
                high = _most_frequent_labels(v, labels, adj)   # same-colour nodes aren't neighbours
                if labels[v] not in high:
                    labels[v] = max(high)   # Prec-Max
        if _converged(labels, adj):
            converged = True
            break
    return labels, converged, it


def run_semi_sync_lpa(graph: GraphBundle, seed: int, max_iter: int = 100) -> AlgorithmResult:
    rng = rng_from_seed(seed)
    n = graph.n
    adj = graph.adjacency_lists()

    # 1. Randomized initial label VALUES (a permutation of 0..n-1 assigned to nodes) --
    #    matters for Prec-Max, see module docstring.
    perm = list(range(n))
    rng.shuffle(perm)

    # 2. Random node order fed to greedy colouring -> a randomized (but always proper) colouring.
    color_order = list(range(n))
    rng.shuffle(color_order)
    color = greedy_coloring(adj, color_order)
    classes_by_id: dict[int, list[int]] = {}
    for v, c in color.items():
        classes_by_id.setdefault(c, []).append(v)
    color_classes = [classes_by_id[c] for c in sorted(classes_by_id)]

    labels, converged, it = _run_core(adj, n, perm, color_classes, max_iter)
    final = canonicalize_labels(np.array(labels, dtype=np.int64))
    return AlgorithmResult(labels=final, converged=converged, iterations=it,
                            meta={"tie_break": "prec-max", "n_colors": len(color_classes), "max_iter": max_iter})
