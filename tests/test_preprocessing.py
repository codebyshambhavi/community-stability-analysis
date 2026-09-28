import numpy as np
import pytest
import scipy.sparse as sp

from src.graph import GraphBundle, validate_simple_undirected
from src.preprocessing import largest_connected_component, symmetrize_and_clean


def _directed_multigraph():
    # 6 nodes. arcs: 0->1 (weight 2 = duplicate), 1->0 (reciprocal), 1->2, 3->3 (loop), 4->3; node 5 isolated
    rows = [0, 1, 1, 3, 4]
    cols = [1, 0, 2, 3, 3]
    vals = [2, 1, 1, 1, 1]
    return sp.csr_matrix((vals, (rows, cols)), shape=(6, 6))


def test_symmetrize_rule_loops_weights():
    log = []
    A = symmetrize_and_clean(_directed_multigraph(), log)
    validate_simple_undirected(A)
    edges = {frozenset(e) for e in zip(*sp.triu(A, 1).nonzero())}
    assert edges == {frozenset({0, 1}), frozenset({1, 2}), frozenset({3, 4})}   # edge iff arc either way
    assert log[0]["self_loops_dropped"] == 1
    assert log[0]["isolated_nodes_after"] == 1
    assert log[0]["input_was_symmetric"] is False
    assert np.all(A.data == 1)                                                    # weights discarded


def test_lcc_and_mapping():
    log = []
    A = symmetrize_and_clean(_directed_multigraph(), log)
    A_l, orig = largest_connected_component(A, log)
    assert list(orig) == [0, 1, 2]                    # component {0,1,2} is largest (size 3)
    assert A_l.shape == (3, 3) and A_l.nnz // 2 == 2
    assert log[-1]["nodes_removed"] == 3


def test_validate_rejects_bad_graphs():
    loop = sp.csr_matrix(np.array([[1, 1], [1, 0]]), dtype=np.int32)
    with pytest.raises(ValueError, match="self-loops"):
        validate_simple_undirected(loop)
    directed = sp.csr_matrix(np.array([[0, 1], [0, 0]]), dtype=np.int32)
    with pytest.raises(ValueError, match="symmetric"):
        validate_simple_undirected(directed)
    weighted = sp.csr_matrix(np.array([[0, 2], [2, 0]]), dtype=np.int32)
    with pytest.raises(ValueError, match="binary"):
        validate_simple_undirected(weighted)


def test_graphbundle_helpers():
    A = symmetrize_and_clean(_directed_multigraph(), [])
    g = GraphBundle("t", "t", A)
    assert (g.n, g.m) == (6, 3)
    assert g.degrees.tolist() == [1, 2, 1, 1, 1, 0]
    assert sorted(g.neighbors(1).tolist()) == [0, 2]
    assert g.edges().tolist() == [[0, 1], [1, 2], [3, 4]]
    assert g.adjacency_lists()[1] == [0, 2]
    assert g.to_networkx().number_of_edges() == 3
    with pytest.raises(ValueError):
        GraphBundle("t", "t", A, truth=np.zeros(3, dtype=int))
