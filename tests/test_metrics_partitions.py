import numpy as np
import pytest

from src.metrics.partitions import (canonicalize_labels, community_size_stats, cover_from_labels,
                                    is_trivial, membership_matrix, n_communities)


def test_canonicalize_relabels_by_ascending_value():
    # np.unique's inverse sorts by VALUE, not order of first appearance (2 < 5 < 9).
    # This is fine: every metric here is invariant to which relabeling is used.
    assert canonicalize_labels(np.array([5, 5, 2, 2, 9])).tolist() == [1, 1, 0, 0, 2]


def test_cover_from_labels_roundtrip():
    labels = np.array([0, 0, 1, 1, 1])
    cover = cover_from_labels(labels)
    assert cover == [{0, 1}, {2, 3, 4}]


def test_membership_matrix_hard():
    S = membership_matrix(5, labels=np.array([0, 0, 1, 1, 1]))
    assert S.shape == (5, 2)
    assert S.toarray().tolist() == [[1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]


def test_membership_matrix_overlapping_and_empty_dropped():
    S = membership_matrix(4, cover=[{0, 1}, set(), {1, 2, 3}])
    assert S.shape == (4, 2)                          # empty community dropped
    assert S.toarray()[1].tolist() == [1, 1]           # node 1 in both


def test_membership_matrix_rejects_both_or_neither():
    with pytest.raises(ValueError):
        membership_matrix(3)
    with pytest.raises(ValueError):
        membership_matrix(3, labels=np.zeros(3, dtype=int), cover=[{0}])


def test_membership_matrix_rejects_out_of_range_node():
    with pytest.raises(ValueError):
        membership_matrix(3, cover=[{0, 5}])


def test_n_communities_and_size_stats():
    labels = np.array([0, 0, 1, 2, 2, 2])
    assert n_communities(labels=labels) == 3
    stats = community_size_stats(labels=labels)
    assert stats["n_communities"] == 3 and stats["size_max"] == 3 and stats["size_min"] == 1
    assert stats["largest_fraction"] == pytest.approx(3 / 6)
    cover = [{0, 1}, {2, 3, 4, 5}, set()]
    assert n_communities(cover=cover) == 2               # empty ignored
    assert community_size_stats(cover=cover)["n_communities"] == 2


def test_size_stats_empty_partition():
    stats = community_size_stats(cover=[])
    assert stats["n_communities"] == 0 and np.isnan(stats["size_mean"])


def test_is_trivial():
    assert is_trivial(labels=np.zeros(10, dtype=int))                 # one giant community
    assert is_trivial(labels=np.arange(10))                            # all singletons
    assert not is_trivial(labels=np.array([0, 0, 0, 1, 1, 1, 2, 2]))
