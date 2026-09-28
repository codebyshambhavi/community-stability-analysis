import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.metrics.partitions import cover_from_labels
from src.metrics.stability import nmi, omega_index, onmi_lfk, variation_of_information

RNG = np.random.default_rng(0)


def random_partition(n, k, seed):
    return np.random.default_rng(seed).integers(0, k, size=n)


# ---------- VI ----------

def test_vi_zero_for_identical_and_relabeled():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([5, 5, 9, 9, 1, 1])            # same partition, different label names
    assert variation_of_information(a, b) == pytest.approx(0.0, abs=1e-12)


def test_vi_symmetric_and_nonnegative():
    a, b = random_partition(40, 4, 1), random_partition(40, 5, 2)
    assert variation_of_information(a, b) == pytest.approx(variation_of_information(b, a))
    assert variation_of_information(a, b) >= 0


def test_vi_normalized_in_unit_range():
    a, b = random_partition(50, 3, 1), np.arange(50)   # totally different structure
    v = variation_of_information(a, b, normalize=True)
    assert 0.0 <= v <= 1.0 + 1e-9


def test_vi_rejects_length_mismatch():
    with pytest.raises(ValueError):
        variation_of_information(np.zeros(3, dtype=int), np.zeros(4, dtype=int))


# ---------- NMI ----------

def test_nmi_matches_sklearn():
    a, b = random_partition(60, 5, 3), random_partition(60, 4, 4)
    assert nmi(a, b) == pytest.approx(normalized_mutual_info_score(a, b, average_method="arithmetic"))


def test_nmi_identical_is_one():
    a = random_partition(30, 6, 7)
    assert nmi(a, a) == pytest.approx(1.0)


# ---------- Omega vs ARI (the core proof: Omega reduces to ARI on hard partitions) ----------

@pytest.mark.parametrize("seed", range(8))
def test_omega_equals_ari_on_hard_partitions(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(15, 60)
    a = rng.integers(0, rng.integers(1, 6), size=n)
    b = rng.integers(0, rng.integers(1, 6), size=n)
    w = omega_index(n, labels1=a, labels2=b)
    ref = adjusted_rand_score(a, b)
    assert w == pytest.approx(ref, abs=1e-9)


def test_omega_identical_partition_is_one():
    a = random_partition(25, 4, 11)
    assert omega_index(25, labels1=a, labels2=a) == pytest.approx(1.0)


def test_omega_accepts_mixed_hard_and_cover_input():
    labels = np.array([0, 0, 1, 1, 1])
    cover = cover_from_labels(labels)
    w = omega_index(5, labels1=labels, cover2=cover)
    assert w == pytest.approx(1.0)                     # same partition, different representation


def test_omega_detects_true_overlap_difference():
    n = 10
    hard = [set(range(0, 5)), set(range(5, 10))]
    overlap = [set(range(0, 6)), set(range(4, 10))]     # nodes 4,5 now shared
    w_self = omega_index(n, cover1=hard, cover2=hard)
    w_diff = omega_index(n, cover1=hard, cover2=overlap)
    assert w_self == pytest.approx(1.0)
    assert w_diff < 1.0


def test_omega_bounded_above_by_one():
    for _ in range(20):
        a, b = random_partition(30, 5, RNG.integers(1 << 30)), random_partition(30, 5, RNG.integers(1 << 30))
        assert omega_index(30, labels1=a, labels2=b) <= 1.0 + 1e-9


# ---------- ONMI (LFK) ----------

def test_onmi_identical_cover_is_one():
    cover = [{0, 1, 2}, {3, 4, 5, 6}, {7, 8, 9}]
    assert onmi_lfk(10, cover, cover) == pytest.approx(1.0)


def test_onmi_close_to_but_not_asserted_exact_vs_nmi_on_hard_partitions():
    """LFK's ONMI is KNOWN in the literature not to reduce exactly to ordinary NMI on hard
    partitions (Hric, Darst & Fortunato 2016, "Community detection in networks: A user guide",
    arXiv:1608.00163: "neither the definition by Lancichinetti, Fortunato and Kertesz nor the one
    by McDaid, Greene and Hurley are proper extensions of the NMI ... [but] the differences are
    typically small"). This test checks that documented property -- close, not identical -- so a
    future change to onmi_lfk that makes them diverge sharply on an easy case is still caught."""
    a = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    b = np.array([0, 0, 1, 1, 1, 2, 2, 0, 2])
    cov_a, cov_b = cover_from_labels(a), cover_from_labels(b)
    assert abs(onmi_lfk(9, cov_a, cov_b) - nmi(a, b)) < 0.2


def test_onmi_in_unit_range_for_true_overlap():
    n = 20
    c1 = [set(range(0, 12)), set(range(8, 20))]
    c2 = [set(range(0, 10)), set(range(6, 20))]
    v = onmi_lfk(n, c1, c2)
    assert -1e-9 <= v <= 1.0 + 1e-9


def test_onmi_empty_cover_handled():
    assert onmi_lfk(5, [], [{0, 1}]) == pytest.approx(0.0)
