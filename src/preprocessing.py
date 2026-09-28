"""Explicit, logged preprocessing. Nothing is altered silently: every step records
what it removed/changed so it can be reported in the methodology section."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .graph import validate_simple_undirected


def symmetrize_and_clean(M: sp.spmatrix, log: list[dict]) -> sp.csr_matrix:
    """Directed / weighted / multi adjacency -> simple undirected unweighted CSR.

    Rule: an undirected edge {i,j} exists iff an arc i->j OR j->i exists (i != j).
    Edge weights / multiplicities are discarded. Self-loops are dropped.
    """
    M = sp.csr_matrix(M)
    M.eliminate_zeros()
    n = M.shape[0]
    n_arcs = int(M.nnz)
    n_loops = int((M.diagonal() != 0).sum())
    is_symmetric = (M != M.T).nnz == 0
    weight_sum = float(M.sum())

    B = M.copy()
    B.data = np.ones_like(B.data, dtype=np.int32)
    U = sp.triu(B + B.T, k=1).tocsr()          # strict upper triangle: no loops
    U.data[:] = 1
    A = (U + U.T).tocsr()
    A.sort_indices()
    A.sum_duplicates()
    A = A.astype(np.int32)

    m_after = int(A.nnz // 2)
    log.append({
        "step": "symmetrize_and_clean",
        "n": n,
        "stored_entries_before": n_arcs,
        "weight_sum_before": weight_sum,
        "input_was_symmetric": bool(is_symmetric),
        "self_loops_dropped": n_loops,
        "undirected_edges_after": m_after,
        "isolated_nodes_after": int((np.diff(A.indptr) == 0).sum()),
        "rule": "edge iff arc in either direction; weights/multiplicity discarded; loops dropped",
    })
    return A


def largest_connected_component(A: sp.csr_matrix, log: list[dict]) -> tuple[sp.csr_matrix, np.ndarray]:
    """Restrict to the largest connected component. Returns (A_lcc, orig_ids) where
    orig_ids[k] is the index (in A) of node k of A_lcc (ascending order preserved)."""
    n_comp, labels = connected_components(A, directed=False)
    sizes = np.bincount(labels)
    big = int(np.argmax(sizes))                 # ties -> lowest component label (deterministic)
    orig_ids = np.flatnonzero(labels == big)
    A_l = A[orig_ids][:, orig_ids].tocsr()
    A_l.sort_indices()
    log.append({
        "step": "largest_connected_component",
        "n_before": int(A.shape[0]),
        "m_before": int(A.nnz // 2),
        "n_components_before": int(n_comp),
        "n_after": int(A_l.shape[0]),
        "m_after": int(A_l.nnz // 2),
        "nodes_removed": int(A.shape[0] - A_l.shape[0]),
    })
    validate_simple_undirected(A_l)
    return A_l, orig_ids


def n_connected_components(A: sp.csr_matrix) -> int:
    return int(connected_components(A, directed=False)[0])
