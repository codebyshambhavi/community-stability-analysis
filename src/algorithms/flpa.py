"""Fast Label Propagation (FLPA).

Traag & Subelj (2023), "Large network community detection by fast label propagation",
Scientific Reports 13, 2701.

We use NetworkX 3.6.1's own `networkx.community.fast_label_propagation_communities`
(added in NetworkX 3.3) rather than a from-scratch reimplementation: it is a direct,
seed-controlled implementation of the paper's algorithm (queue-based: only nodes whose
neighbourhood label frequencies could plausibly have changed are re-queued, instead of
resweeping every node every round), and re-implementing it ourselves would risk
introducing exactly the kind of "implementation artifact vs. algorithmic behaviour"
confound the project's own methodology (README/methodology.md) warns against. See
requirements.txt for the pinned `networkx>=3.4` floor and README for the exact version
this project was developed against (3.6.1).
"""
from __future__ import annotations

import networkx as nx

from ..graph import GraphBundle
from .common import AlgorithmResult, labels_from_communities


def run_flpa(graph: GraphBundle, seed: int) -> AlgorithmResult:
    G = graph.to_networkx()                 # nodes 0..n-1, matching graph.adj row order
    communities = list(nx.community.fast_label_propagation_communities(G, weight=None, seed=seed))
    labels = labels_from_communities(communities, graph.n)
    return AlgorithmResult(labels=labels, converged=True, iterations=None,
                            meta={"n_communities": len(communities),
                                  "networkx_version": nx.__version__})
