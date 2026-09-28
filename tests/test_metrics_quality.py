import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp

from src.metrics.quality import extended_modularity, modularity


def karate_adj():
    # nx.karate_club_graph() carries an edge 'weight' attribute (Zachary's original interaction
    # counts); our pipeline always works on UNWEIGHTED graphs (preprocessing.symmetrize_and_clean
    # explicitly discards weights), so weight=None here to match what the project actually loads.
    G = nx.karate_club_graph()
    return sp.csr_matrix(nx.to_scipy_sparse_array(G, nodelist=range(34), weight=None))


def test_modularity_matches_networkx_reference():
    A = karate_adj()
    G_unweighted = nx.Graph(nx.karate_club_graph().edges())   # same edges, weight attr dropped
    G = nx.karate_club_graph()
    labels = np.array([0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(34)])
    ours = modularity(A, labels)
    ref = nx.community.modularity(G_unweighted, [set(np.flatnonzero(labels == c)) for c in (0, 1)])
    assert ours == pytest.approx(ref, abs=1e-9)


def test_modularity_singleton_partition_matches_networkx():
    A = karate_adj()
    G_unweighted = nx.Graph(nx.karate_club_graph().edges())
    labels = np.arange(34)
    ours = modularity(A, labels)
    ref = nx.community.modularity(G_unweighted, [{i} for i in range(34)])
    assert ours == pytest.approx(ref, abs=1e-9)


def test_eq_reduces_to_q_for_disjoint_cover():
    A = karate_adj()
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1] + [0, 1] * 13)
    q = modularity(A, labels)
    cover = [set(np.flatnonzero(labels == c)) for c in (0, 1)]
    eq = extended_modularity(A, cover=cover)
    assert eq == pytest.approx(q, abs=1e-9)


def test_eq_overlapping_differs_from_hard_and_is_bounded():
    A = karate_adj()
    hard_cover = [set(range(0, 17)), set(range(17, 34))]
    # deliberately asymmetric overlap (the 0-18/16-34 split used earlier turned out to be a
    # coincidental symmetry of this particular split of the karate graph and gave an equal EQ)
    overlap_cover = [set(range(0, 20)), set(range(10, 34))]
    eq_hard = extended_modularity(A, cover=hard_cover)
    eq_overlap = extended_modularity(A, cover=overlap_cover)
    assert eq_hard != pytest.approx(eq_overlap)
    assert -1.0 <= eq_overlap <= 1.0 and -1.0 <= eq_hard <= 1.0


def test_uncovered_node_contributes_nothing():
    A = karate_adj()
    cover_all = [set(range(0, 17)), set(range(17, 34))]
    cover_missing = [set(range(0, 17)), set(range(17, 33))]   # node 33 in no community
    eq_all = extended_modularity(A, cover=cover_all)
    eq_missing = extended_modularity(A, cover=cover_missing)
    assert eq_all != pytest.approx(eq_missing)                # dropping a node changes the score
