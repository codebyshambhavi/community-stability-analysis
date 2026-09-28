"""GraphBundle: the single graph container every downstream module consumes.

Invariants (enforced on construction): the adjacency is a canonical CSR matrix of a
SIMPLE UNDIRECTED UNWEIGHTED graph on nodes 0..n-1 (symmetric, binary, zero diagonal,
sorted indices, no duplicates). Algorithms never see directed/weighted/multi graphs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


def validate_simple_undirected(A: sp.spmatrix) -> None:
    """Raise ValueError unless A is a canonical simple undirected unweighted graph."""
    if not sp.isspmatrix_csr(A):
        raise ValueError("adjacency must be scipy CSR")
    n, m_ = A.shape
    if n != m_:
        raise ValueError(f"adjacency not square: {A.shape}")
    if not A.has_canonical_format:
        raise ValueError("CSR not canonical (unsorted indices or duplicate entries)")
    if A.nnz and not np.all(A.data == 1):
        raise ValueError("adjacency must be binary (all stored values == 1)")
    if A.diagonal().any():
        raise ValueError("self-loops present")
    if (A != A.T).nnz != 0:
        raise ValueError("adjacency not symmetric (directed edges present)")


@dataclass(frozen=True)
class GraphBundle:
    graph_id: str                       # unique key, e.g. "karate", "lfr_mu30_i04"
    family: str                         # "karate" | "dolphins" | "polbooks" | "polblogs" | "lfr"
    adj: sp.csr_matrix
    truth: np.ndarray | None = None     # ground-truth membership (int, length n) or None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        validate_simple_undirected(self.adj)
        if self.truth is not None and len(self.truth) != self.adj.shape[0]:
            raise ValueError("truth length != number of nodes")

    @property
    def n(self) -> int:
        return int(self.adj.shape[0])

    @property
    def m(self) -> int:
        return int(self.adj.nnz // 2)

    @property
    def degrees(self) -> np.ndarray:
        return np.diff(self.adj.indptr)

    def neighbors(self, v: int) -> np.ndarray:
        return self.adj.indices[self.adj.indptr[v]:self.adj.indptr[v + 1]]

    def adjacency_lists(self) -> list[list[int]]:
        """Plain Python neighbour lists (fast for the pure-Python label-propagation loops)."""
        ind, ptr = self.adj.indices, self.adj.indptr
        return [ind[ptr[v]:ptr[v + 1]].tolist() for v in range(self.n)]

    def edges(self) -> np.ndarray:
        """(m, 2) array with u < v, sorted."""
        coo = sp.triu(self.adj, k=1).tocoo()
        e = np.stack([coo.row, coo.col], axis=1).astype(np.int64)
        return e[np.lexsort((e[:, 1], e[:, 0]))]

    def to_networkx(self):
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(self.n))
        G.add_edges_from(map(tuple, self.edges().tolist()))
        return G
