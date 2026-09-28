"""Speaker-listener Label Propagation Algorithm (SLPA).

Xie, Szymanski & Liu (2011), "SLPA: Uncovering overlapping communities in social networks
via a speaker-listener interaction dynamic process", ICDM Workshops.

Unlike LPA/semi-sync LPA/FLPA, SLPA is memory-based and produces OVERLAPPING
communities: every node accumulates a MEMORY (multiset) of labels received over T
iterations, and a node's final community membership is every label that stayed
"popular enough" (relative frequency > r) in its memory -- so a node can end up in
several communities at once. Output is therefore a COVER (list of node-id sets), not a
label array; see src/metrics/partitions.py's cover representation, used identically by
the stability/quality metrics (Omega, ONMI, EQ).

ALGORITHM (one iteration, over ALL nodes, in a fresh random order -- "listener order
must be seed-controlled" per the project brief):
  for each node v, taken as LISTENER, in the (shuffled) order:
    - every neighbour u of v acts as SPEAKER: u emits ONE label drawn at random from
      u's own memory, with probability proportional to that label's current frequency
      in u's memory (implemented for free: memory is stored as a flat list, so a
      uniform random draw from the list IS the frequency-weighted draw the paper
      specifies -- more frequent labels simply appear more times in the list).
    - v, as LISTENER, picks the label that arrived from the largest number of speakers
      (ties broken uniformly at random) and APPENDS it to its own memory.
  Because nodes are processed sequentially within an iteration and memory updates are
  applied immediately, a node speaking later in the same sweep can already reflect a
  label it received earlier in that same sweep -- this asynchrony is intentional and is
  what the seed-controlled order actually controls (mirrors the asynchronous character
  of LPA/semi-sync LPA, applied here to speaking rather than labeling).

POST-PROCESSING (after all T iterations): for each node, keep every label whose
relative frequency in that node's final memory exceeds threshold r. FALLBACK (a project
convention, not specified by the original paper, and stated explicitly here rather than
left implicit): if thresholding at r would leave a node in ZERO communities -- both
possible in principle when r is high relative to a very spread-out memory, and the
overwhelmingly common case at the initial iteration boundary where a node's own seed
label is still under-represented -- that node instead keeps its single most frequent
label, so every node always belongs to at least one community and cover_from_labels-style
downstream code never has to special-case an empty row.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from ..graph import GraphBundle
from .common import AlgorithmResult, rng_from_seed


def run_slpa(graph: GraphBundle, seed: int, T: int = 100, r: float = 0.1) -> AlgorithmResult:
    rng = rng_from_seed(seed)
    n = graph.n
    adj = graph.adjacency_lists()
    memory: list[list[int]] = [[v] for v in range(n)]      # each node starts with its own label

    for _ in range(T):
        order = list(range(n))
        rng.shuffle(order)                                  # seed-controlled listener order
        for listener in order:
            neighbors = adj[listener]
            if not neighbors:
                continue                                     # isolated node: nothing to listen to
            received = []
            for speaker in neighbors:
                mem = memory[speaker]
                received.append(mem[rng.randrange(len(mem))])  # frequency-weighted via list draw
            counts = Counter(received)
            max_freq = max(counts.values())
            candidates = [lbl for lbl, c in counts.items() if c == max_freq]
            chosen = candidates[0] if len(candidates) == 1 else rng.choice(candidates)
            memory[listener].append(chosen)

    label_freq: list[dict[int, float]] = []
    label_to_nodes: dict[int, set[int]] = defaultdict(set)
    for v in range(n):
        counts = Counter(memory[v])
        total = len(memory[v])
        freqs = {lbl: c / total for lbl, c in counts.items()}
        label_freq.append(freqs)
        kept = {lbl for lbl, f in freqs.items() if f > r}
        if not kept:                                          # fallback: documented above
            kept = {max(freqs, key=freqs.get)}
        for lbl in kept:
            label_to_nodes[lbl].add(v)

    cover = [nodes for nodes in label_to_nodes.values() if nodes]
    return AlgorithmResult(cover=cover, converged=None, iterations=T,
                            meta={"T": T, "r": r, "label_freq": label_freq, "n_communities": len(cover)})
