"""Run-to-run stability metrics.

Hard partitions (LPA, semi-sync LPA, FLPA): Variation of Information (VI, Meila 2003) and
Normalized Mutual Information (NMI, arithmetic-mean normalization, via scikit-learn).

All four algorithms (including overlapping SLPA): the Omega Index (Collins & Dent 1988),
which generalizes pairwise cluster agreement to overlapping covers and is PROVEN below (and
checked in tests/test_metrics_stability.py) to reduce exactly to the Adjusted Rand Index when
both inputs are hard partitions -- so it is the one metric usable across all four algorithms.

Overlapping-specific NMI: ONMI (Lancichinetti, Fortunato & Kertesz 2009, New J. Phys. 11,
033015, "Detecting the overlapping and hierarchical community structure in complex networks").
This is their original LFK definition, NOT the later "corrected" McDaid, Greene & Hurley (2011)
variant -- stated explicitly because the two give different numbers on the same input.

KNOWN LIMITATION (stated here rather than discovered by a surprised reader of the results): the
LFK measure is a "best-match, per-community-normalized" construction, not a direct generalization
of joint/conditional entropy, so it does NOT reduce exactly to ordinary NMI when both covers happen
to be hard partitions -- this is documented in the literature (e.g. Hric, Darst & Fortunato's
"Community detection in networks: A user guide", and Lutov's OvpNMI notes; the discrepancy is
usually small). Omega is therefore the metric we use for cross-algorithm comparison; ONMI_LFK is
reported alongside it for SLPA specifically, not treated as interchangeable with plain NMI.

Derivation of Omega == ARI on hard partitions (why this is not a coincidence):
For a hard partition, the number of communities shared by a pair (i,j) is 0 or 1 (same
cluster or not), so Omega's "observed agreement" term is exactly the unadjusted Rand
Index, and Omega's "expected agreement" term is exactly the Rand Index's own chance
baseline E[RI] under the generalized hypergeometric model -- so
    Omega = (RI - E[RI]) / (1 - E[RI]) = (RI - E[RI]) / (max(RI) - E[RI]) = ARI.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import normalized_mutual_info_score

from .partitions import canonicalize_labels, membership_matrix

_LOG2 = np.log(2.0)


def _entropy_bits(p: np.ndarray) -> float:
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / _LOG2)


def variation_of_information(labels1: np.ndarray, labels2: np.ndarray, normalize: bool = False) -> float:
    """VI(X,Y) = H(X|Y) + H(Y|X) = 2 H(X,Y) - H(X) - H(Y), in bits. 0 = identical partitions
    (up to relabeling); higher = more different. normalize=True divides by log2(n) (in [0,1])."""
    a, b = canonicalize_labels(labels1), canonicalize_labels(labels2)
    n = len(a)
    if n != len(b):
        raise ValueError("labels must have the same length")
    ka, kb = a.max() + 1, b.max() + 1
    contingency = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(contingency, (a, b), 1)
    p_joint = contingency / n
    p_a, p_b = p_joint.sum(axis=1), p_joint.sum(axis=0)
    h_joint = _entropy_bits(p_joint.ravel())
    vi = 2 * h_joint - _entropy_bits(p_a) - _entropy_bits(p_b)
    vi = max(vi, 0.0)                                       # clip tiny negative floating-point noise
    if normalize:
        return vi / np.log2(n) if n > 1 else 0.0
    return vi


def nmi(labels1: np.ndarray, labels2: np.ndarray) -> float:
    """Normalized Mutual Information, arithmetic-mean normalization (sklearn default). 1 =
    identical partitions up to relabeling, 0 = independent. sklearn returns 1.0 for two
    identical single-cluster (trivial) partitions and 0.0 if only one side is trivial."""
    return float(normalized_mutual_info_score(labels1, labels2, average_method="arithmetic"))


def _pair_histogram(T: np.ndarray) -> np.ndarray:
    """Upper-triangle (i<j) histogram of an (n,n) integer 'shared communities' matrix,
    as a count-by-value array: hist[j] = number of pairs with exactly j communities in common."""
    n = T.shape[0]
    iu = np.triu_indices(n, k=1)
    return np.bincount(T[iu])


def omega_index(n: int, labels1: np.ndarray | None = None, cover1: list | None = None,
                 labels2: np.ndarray | None = None, cover2: list | None = None) -> float:
    """Omega Index (Collins & Dent 1988) between two partitions/covers of the same n nodes.
    Each side is given as labels= (hard) or cover= (overlapping); mix freely. 1 = identical,
    0 = chance-level agreement, can be slightly negative (like ARI)."""
    S1 = membership_matrix(n, labels=labels1, cover=cover1)
    S2 = membership_matrix(n, labels=labels2, cover=cover2)
    T1 = (S1 @ S1.T).toarray()
    T2 = (S2 @ S2.T).toarray()
    h1, h2 = _pair_histogram(T1), _pair_histogram(T2)

    total = n * (n - 1) // 2
    if total == 0:
        return 1.0
    max_j = max(len(h1), len(h2))
    h1 = np.pad(h1, (0, max_j - len(h1)))
    h2 = np.pad(h2, (0, max_j - len(h2)))

    iu = np.triu_indices(n, k=1)
    observed = float(np.mean(T1[iu] == T2[iu]))
    expected = float(np.dot(h1, h2)) / (total ** 2)
    if expected >= 1.0:
        return 1.0                                          # both trivial and identical in every pairing
    return (observed - expected) / (1 - expected)


def _lfk_conditional_entropy_bits(a: np.ndarray, b: np.ndarray, n: int) -> float:
    """H(a|b) in bits for two binary membership vectors (one community each), with the LFK
    correction: if a and b are negatively associated (matching would be worse than nothing),
    b is treated as uninformative about a (contributes H(a), i.e. "no reduction")."""
    n11 = int(np.sum(a & b)); n10 = int(np.sum(a & ~b))
    n01 = int(np.sum(~a & b)); n00 = n - n11 - n10 - n01
    h_a = _entropy_bits(np.array([n11 + n10, n00 + n01], dtype=float) / n)
    if h_a == 0.0:
        return 0.0
    if n11 * n00 < n10 * n01:                                # LFK correction (Lancichinetti et al. 2009)
        return h_a
    counts = np.array([n11, n10, n01, n00], dtype=float)
    h_joint = _entropy_bits(counts / n)
    h_b = _entropy_bits(np.array([n11 + n01, n10 + n00], dtype=float) / n)
    return max(h_joint - h_b, 0.0)


def onmi_lfk(n: int, cover1: list, cover2: list) -> float:
    """Overlapping NMI (Lancichinetti, Fortunato & Kertesz 2009). Works for hard partitions
    too (passed as one-node-per-community-per-label covers via partitions.cover_from_labels),
    but Omega is the metric used for cross-algorithm comparison; this is the SLPA-specific one.

    Convention (not specified by the original paper, which assumes non-empty covers): if either
    side has zero communities -- e.g. an SLPA run where the r threshold removes every label --
    ONMI is defined as 0.0 (no shared structure), matching Omega's and NMI's treatment of that case."""
    if not cover1 or not cover2:
        return 0.0

    def half(cover_x, cover_y):
        total = 0.0
        for cx in cover_x:
            a = np.zeros(n, dtype=bool); a[list(cx)] = True
            h_a = _entropy_bits(np.array([a.sum(), n - a.sum()], dtype=float) / n)
            if h_a == 0.0:
                continue
            best = h_a
            for cy in cover_y:
                b = np.zeros(n, dtype=bool); b[list(cy)] = True
                best = min(best, _lfk_conditional_entropy_bits(a, b, n))
            total += best / h_a
        return total / len(cover_x)

    return 1.0 - 0.5 * (half(cover1, cover2) + half(cover2, cover1))
