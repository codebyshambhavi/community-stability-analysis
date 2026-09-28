"""Community-quality metrics: modularity Q (hard partitions) and extended modularity EQ
(overlapping covers, Shen, Yang & Cheng 2009 "Detect overlapping and hierarchical community
structure in networks", Physica A). EQ is defined so that it reduces EXACTLY to Q when every
node belongs to exactly one community (tested in tests/test_metrics_quality.py).

    EQ = (1/2m) * sum_c sum_{i,j in c} (1/(O_i O_j)) * (A_ij - k_i k_j / 2m)

where O_i is the number of communities node i belongs to. All formulas are vectorized over
the (n x k) sparse membership matrix -- no per-community Python loop -- since this is called
many thousands of times across the full run matrix (74 graphs x 4 algorithms x 30 runs).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .partitions import membership_matrix


def _weighted_membership(S: sp.csr_matrix) -> tuple[sp.csr_matrix, np.ndarray]:
    """Row-normalize S by each node's community count O_i (1/O_i per entry); O_i=0 rows stay 0."""
    O = np.asarray(S.sum(axis=1)).ravel()
    inv_O = np.divide(1.0, O, out=np.zeros_like(O, dtype=np.float64), where=O > 0)
    W = S.multiply(inv_O[:, None]).tocsr()
    return W, O


def extended_modularity(adj: sp.csr_matrix, labels: np.ndarray | None = None,
                         cover: list | None = None) -> float:
    """EQ for a hard partition (labels=) or an overlapping cover (cover=). Isolated / uncovered
    nodes (O_i=0) contribute nothing, matching the definition (they are in no community)."""
    n = adj.shape[0]
    m = adj.nnz / 2.0
    if m == 0:
        return 0.0
    S = membership_matrix(n, labels=labels, cover=cover)
    W, _O = _weighted_membership(S)
    deg = np.asarray(adj.sum(axis=1)).ravel()

    AW = adj @ W                                          # (n, k)
    internal = float(W.multiply(AW).sum())                # sum_c trace_c(W^T A W) = sum_c sum_{i,j in c} A_ij/(O_iO_j)
    dw = np.asarray(W.T @ deg).ravel()                     # dw_c = sum_{i in c} k_i / O_i
    null_model = float(np.square(dw).sum()) / (2 * m)
    return (internal - null_model) / (2 * m)


def modularity(adj: sp.csr_matrix, labels: np.ndarray) -> float:
    """Standard Newman modularity Q for a hard partition. Implemented as a special case of
    extended_modularity (O_i == 1 for every node) -- kept as its own function since Q is the
    metric actually reported for the three hard-partition algorithms."""
    return extended_modularity(adj, labels=labels)
