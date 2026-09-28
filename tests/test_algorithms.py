import networkx as nx
import numpy as np
import pytest

from src.algorithms import run_flpa, run_lpa, run_semi_sync_lpa, run_slpa
from src.algorithms.semi_sync import greedy_coloring
from src.config import load_config, resolve
from src.data_loader import load_real
from src.graph import GraphBundle
from src.metrics.partitions import canonicalize_labels, community_size_stats, membership_matrix, n_communities

CFG = load_config()
RAW = resolve(CFG, "raw_dir")

HARD_ALGOS = {"lpa": run_lpa, "semi_sync_lpa": run_semi_sync_lpa, "flpa": run_flpa}

# Loaded once at module scope (mirrors CFG/RAW in the other test files) rather than as
# pytest fixtures, since these graphs are read-only and reused across every test below.
karate = load_real("karate", RAW)
polbooks = load_real("polbooks", RAW)
polblogs = load_real("polblogs", RAW)   # the graph the "convergence concern" note was about


def _bridge_graph():
    """Two dense triangleish clusters {0..4} and {5..9} joined by a bridge node shared
    by both -- a small graph designed to make genuine overlap plausible for SLPA."""
    import scipy.sparse as sp
    edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 4),
             (5, 6), (5, 7), (6, 7), (6, 8), (7, 9), (8, 9),
             (4, 5), (3, 5), (4, 6)]                            # extra links straddling the bridge
    n = 10
    rows, cols = zip(*edges)
    data = [1] * len(edges)
    U = sp.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    A = (U + U.T).tocsr()
    A.data[:] = 1
    A.sort_indices()
    A.sum_duplicates()
    return GraphBundle(graph_id="bridge", family="bridge", adj=A)


# ---------- same seed -> same result / different seed -> can differ ----------

@pytest.mark.parametrize("name", list(HARD_ALGOS))
def test_hard_algo_same_seed_is_deterministic(name):
    fn = HARD_ALGOS[name]
    a = fn(polbooks, seed=3)
    b = fn(polbooks, seed=3)
    assert np.array_equal(a.labels, b.labels)


@pytest.mark.parametrize("name", list(HARD_ALGOS))
def test_hard_algo_different_seeds_can_differ(name):
    fn = HARD_ALGOS[name]
    results = [fn(polbooks, seed=s).labels for s in range(6)]
    # not asserting every pair differs (stochastic algorithms CAN coincide) -- just that
    # the seed isn't silently ignored across the whole batch.
    assert any(not np.array_equal(results[0], r) for r in results[1:])


def test_slpa_same_seed_is_deterministic():
    a = run_slpa(polbooks, seed=3, T=15, r=0.1)
    b = run_slpa(polbooks, seed=3, T=15, r=0.1)
    a_sets = sorted(frozenset(c) for c in a.cover)
    b_sets = sorted(frozenset(c) for c in b.cover)
    assert a_sets == b_sets


def test_slpa_different_seeds_can_differ():
    covers = [tuple(sorted(frozenset(c) for c in run_slpa(polbooks, seed=s, T=15, r=0.1).cover))
              for s in range(4)]
    assert any(covers[0] != c for c in covers[1:])


# ---------- input graph is not mutated ----------

@pytest.mark.parametrize("name", list(HARD_ALGOS))
def test_hard_algo_does_not_mutate_input_graph(name):
    fn = HARD_ALGOS[name]
    before_nnz, before_sum = karate.adj.nnz, karate.adj.sum()
    before = karate.adj.toarray().copy()
    fn(karate, seed=0)
    assert karate.adj.nnz == before_nnz and karate.adj.sum() == before_sum
    assert np.array_equal(karate.adj.toarray(), before)


def test_slpa_does_not_mutate_input_graph():
    before = karate.adj.toarray().copy()
    run_slpa(karate, seed=0, T=10, r=0.1)
    assert np.array_equal(karate.adj.toarray(), before)


# ---------- partitions/covers are valid ----------

@pytest.mark.parametrize("name", list(HARD_ALGOS))
def test_hard_algo_produces_valid_canonical_partition(name):
    fn = HARD_ALGOS[name]
    res = fn(karate, seed=1)
    assert len(res.labels) == karate.n
    assert np.array_equal(canonicalize_labels(res.labels), res.labels)   # already canonical
    stats = community_size_stats(labels=res.labels)
    assert stats["n_communities"] == n_communities(labels=res.labels)
    assert sum(1 for _ in res.labels) == karate.n
    # membership_matrix must accept it without complaint (round-trips through the same
    # code path every downstream metric uses)
    S = membership_matrix(karate.n, labels=res.labels)
    assert S.shape[0] == karate.n


def test_slpa_produces_valid_cover():
    res = run_slpa(karate, seed=1, T=20, r=0.1)
    covered = set()
    for community in res.cover:
        assert community                          # empty communities are never emitted
        covered |= community
    assert covered == set(range(karate.n))         # every node kept in >=1 community (fallback rule)
    S = membership_matrix(karate.n, cover=res.cover)   # must not raise
    assert S.shape[0] == karate.n
    for freqs in res.meta["label_freq"]:
        assert abs(sum(freqs.values()) - 1.0) < 1e-9   # a node's memory frequencies sum to 1


def test_slpa_can_produce_genuine_overlap():
    g = _bridge_graph()
    overlap_seen = False
    for seed in range(20):
        res = run_slpa(g, seed=seed, T=100, r=0.15)
        node_membership_counts = {v: 0 for v in range(g.n)}
        for community in res.cover:
            for v in community:
                node_membership_counts[v] += 1
        if any(c > 1 for c in node_membership_counts.values()):
            overlap_seen = True
            break
    assert overlap_seen, "expected at least one seed to place a node in >1 community on a bridge graph"


# ---------- Semi-sync: structural validation against NetworkX ----------
# NetworkX's label_propagation_communities is DETERMINISTIC (fixed node-enumeration
# labeling + nx.coloring.greedy_color's default, non-random strategy; no seed argument),
# so it is NOT the same stochastic protocol as our paper-faithful, randomized
# implementation (see module docstring in src/algorithms/semi_sync.py) -- we therefore
# validate STRUCTURALLY (coloring validity, partition validity, convergence), not via
# exact output equality, which would be comparing two different (if related) algorithms.

def test_greedy_coloring_is_proper():
    for g in (karate, polbooks):
        adj = g.adjacency_lists()
        order = list(range(g.n))
        color = greedy_coloring(adj, order)
        for v in range(g.n):
            for u in adj[v]:
                assert color[v] != color[u]


def test_semi_sync_converges_and_matches_networkx_structurally():
    res = run_semi_sync_lpa(karate, seed=0, max_iter=200)
    assert res.converged
    # NetworkX's own (deterministic) semi-sync reference also produces a valid partition
    # of the same graph -- a sanity cross-check, not a claim of identical output.
    ref_communities = list(nx.community.label_propagation_communities(karate.to_networkx()))
    covered = set().union(*ref_communities)
    assert covered == set(range(karate.n))
    assert sum(len(c) for c in ref_communities) == karate.n


@pytest.mark.parametrize("seed", range(5))
def test_semi_sync_converges_on_polblogs_scale(seed):
    res = run_semi_sync_lpa(polblogs, seed=seed, max_iter=200)
    assert res.converged


# ---------- FLPA: validated directly against NetworkX 3.6.1 (it IS the same call) ----------

def test_flpa_matches_direct_networkx_call():
    res = run_flpa(karate, seed=7)
    G = karate.to_networkx()
    direct = list(nx.community.fast_label_propagation_communities(G, weight=None, seed=7))
    ours_sets = {frozenset(np.flatnonzero(res.labels == c).tolist()) for c in range(res.labels.max() + 1)}
    direct_sets = {frozenset(c) for c in direct}
    assert ours_sets == direct_sets


def test_flpa_same_seed_deterministic_different_seed_can_differ():
    a = run_flpa(polbooks, seed=1).labels
    b = run_flpa(polbooks, seed=1).labels
    assert np.array_equal(a, b)
    others = [run_flpa(polbooks, seed=s).labels for s in range(2, 6)]
    assert any(not np.array_equal(a, o) for o in others)


# ---------- LPA: convergence-safe tie-breaking behaves as documented ----------

@pytest.mark.parametrize("seed", range(5))
def test_lpa_converges_on_polblogs_scale(seed):
    res = run_lpa(polblogs, seed=seed, max_iter=200)
    assert res.converged
