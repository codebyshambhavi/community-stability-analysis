"""Plain asynchronous Label Propagation Algorithm (LPA).

Raghavan, Albert & Kumara (2007), "Near linear time algorithm to detect community
structures in large-scale networks", Physical Review E 76, 036106.

ALGORITHM
Each node starts with its own unique label. Repeatedly, in a fresh random node order
each sweep, every node adopts the label held by the largest number of its neighbours
(unweighted graphs here, so this is just a majority vote by neighbour count).

TIE-BREAKING (the "convergence concern" flagged in the project brief)
When several labels are tied for the neighbourhood majority, fully random tie-breaking
(reroll a new label every time, even if the current one is already tied for best) can
made two adjacent regions of a large graph flip back and forth between two labels
forever -- this is a documented failure mode on graphs like PolBlogs that have a strong,
near-symmetric two-community split. We use the standard fix (also used internally by
NetworkX's asyn_lpa_communities): a node's CURRENT label is retained whenever it is
already among the tied maximum-frequency labels; a new label is drawn uniformly at
random from the tied set only when the current label is NOT among them. This makes the
stopping criterion ("every node already holds a majority label") reachable rather than
an oscillating target, while still leaving real ties (where the current label truly
isn't competitive) randomly resolved -- so the algorithm stays stochastic.

STOPPING RULE
Converged when a full sweep makes zero forced label changes (every node already held a
max-frequency label). Capped at max_iter sweeps as a fallback for pathological cases;
`AlgorithmResult.converged` reports whether the cap was hit.

Isolated nodes (no neighbours) keep their own initial label -- they are, correctly,
singleton communities.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..graph import GraphBundle
from ..metrics.partitions import canonicalize_labels
from .common import AlgorithmResult, rng_from_seed


def run_lpa(graph: GraphBundle, seed: int, max_iter: int = 100) -> AlgorithmResult:
    rng = rng_from_seed(seed)
    n = graph.n
    adj = graph.adjacency_lists()          # plain Python lists: fast for this loop
    labels = list(range(n))                # each node its own label initially

    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        changed = False
        order = list(range(n))
        rng.shuffle(order)                 # fresh randomized order every sweep
        for v in order:
            neighbors = adj[v]
            if not neighbors:
                continue                    # isolated node: nothing to vote on
            counts = Counter(labels[u] for u in neighbors)
            max_freq = max(counts.values())
            best = [lbl for lbl, c in counts.items() if c == max_freq]
            if labels[v] not in best:
                labels[v] = rng.choice(best)
                changed = True
        if not changed:
            converged = True
            break

    final = canonicalize_labels(np.array(labels, dtype=np.int64))
    return AlgorithmResult(labels=final, converged=converged, iterations=it,
                            meta={"tie_break": "retain-current-else-random", "max_iter": max_iter})
