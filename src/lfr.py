"""LFR benchmark graphs: generation (NetworKit) and cached loading.

Why NetworKit and not networkx.LFR_benchmark_graph: the NetworkX generator's realised mixing
drifts far above the requested mu (e.g. nominal 0.1 -> ~0.19, 0.5 -> ~0.68) and emits self-loops,
because its wiring loop counts degree contributed by earlier nodes and rounds deg*(1-mu) on
small degrees. NetworKit tracks nominal mu much more closely (still slightly above it), so we
ALWAYS record the empirical mu per graph and use that for analysis.

NOTE: NetworKit's generator is a re-implementation, not Lancichinetti's original binary. State
this in the report and cite the reference NetworKit's documentation gives for it.

Generation is a separate, run-once stage; generated graphs are cached under data/lfr/ with
checksums, so the experiments themselves do not need NetworKit installed.
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .graph import GraphBundle
from .manifest import sha256_file
from .preprocessing import n_connected_components


def graph_id(mu_nominal: float, instance: int) -> str:
    return f"lfr_mu{int(round(mu_nominal * 100)):02d}_i{instance:02d}"


def instance_seed(lfr_cfg: dict, mu_index: int, instance: int) -> int:
    """Distinct seed per (mu level, instance) -> the 70 graphs are independent draws."""
    return int(lfr_cfg["base_seed"]) + 100 * int(mu_index) + int(instance)


def empirical_mu(edges: np.ndarray, membership: np.ndarray) -> float:
    """Fraction of edges joining nodes of different ground-truth communities."""
    return float(np.mean(membership[edges[:, 0]] != membership[edges[:, 1]]))


def _canonical_edges(raw_edges: list[tuple[int, int]], n: int, gid: str) -> np.ndarray:
    e = np.asarray(raw_edges, dtype=np.int64).reshape(-1, 2)
    if (e[:, 0] == e[:, 1]).any():
        raise ValueError(f"[{gid}] generator produced self-loops")
    e = np.sort(e, axis=1)
    if e.max() >= n or e.min() < 0:
        raise ValueError(f"[{gid}] node id out of range")
    uniq = np.unique(e, axis=0)                 # sorted lexicographically
    if len(uniq) != len(e):
        raise ValueError(f"[{gid}] generator produced duplicate edges")
    return uniq


def generate_instance(lfr_cfg: dict, mu: float, seed: int, gid: str = "lfr") -> tuple[np.ndarray, np.ndarray]:
    """Generate one LFR graph with NetworKit. Returns (edges (m,2) u<v sorted, membership (n,))."""
    import networkit as nk  # lazy: only required for generation

    n = int(lfr_cfg["n"])
    nk.setSeed(int(seed), False)
    nk.setNumberOfThreads(1)                     # single thread -> reproducible
    g = nk.generators.LFRGenerator(n)
    g.generatePowerlawDegreeSequence(lfr_cfg["average_degree"], lfr_cfg["max_degree"], -lfr_cfg["tau1"])
    g.generatePowerlawCommunitySizeSequence(lfr_cfg["min_community"], lfr_cfg["max_community"], -lfr_cfg["tau2"])
    g.setMu(float(mu))
    g.run()
    G, P = g.getGraph(), g.getPartition()
    if G.numberOfNodes() != n:
        raise ValueError(f"[{gid}] expected {n} nodes, got {G.numberOfNodes()}")
    edges = _canonical_edges(list(G.iterEdges()), n, gid)
    raw_membership = np.asarray([P[v] for v in range(n)], dtype=np.int64)
    _, membership = np.unique(raw_membership, return_inverse=True)   # canonical 0..k-1
    return edges, membership.astype(np.int64)


def _adjacency(edges: np.ndarray, n: int) -> sp.csr_matrix:
    data = np.ones(len(edges), dtype=np.int32)
    U = sp.coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    A = (U + U.T).tocsr()
    A.sort_indices()
    return A


def write_instance(out_dir: Path, gid: str, edges: np.ndarray, membership: np.ndarray, meta: dict) -> dict:
    d = out_dir / gid
    d.mkdir(parents=True, exist_ok=True)
    np.savetxt(d / "edges.csv", edges, fmt="%d", delimiter=",", header="u,v", comments="")
    np.savetxt(d / "membership.csv", np.column_stack([np.arange(len(membership)), membership]),
               fmt="%d", delimiter=",", header="node,community", comments="")
    meta = dict(meta)
    meta["sha256_edges"] = sha256_file(d / "edges.csv")
    meta["sha256_membership"] = sha256_file(d / "membership.csv")
    (d / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def load_lfr(gid: str, lfr_dir: str | Path) -> GraphBundle:
    """Load a cached LFR graph; verifies checksums and recomputes empirical mu."""
    d = Path(lfr_dir) / gid
    if not (d / "meta.json").exists():
        raise FileNotFoundError(f"{gid} not generated yet (run: python main.py generate-lfr)")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for key, fname in (("sha256_edges", "edges.csv"), ("sha256_membership", "membership.csv")):
        if sha256_file(d / fname) != meta[key]:
            raise RuntimeError(f"[{gid}] checksum mismatch for {fname}; cached graph was modified")
    edges = np.loadtxt(d / "edges.csv", delimiter=",", skiprows=1, dtype=np.int64).reshape(-1, 2)
    mem = np.loadtxt(d / "membership.csv", delimiter=",", skiprows=1, dtype=np.int64)
    membership = mem[np.argsort(mem[:, 0]), 1]
    n = int(meta["n"])
    A = _adjacency(edges, n)
    emp = empirical_mu(edges, membership)
    if abs(emp - meta["mu_empirical"]) > 1e-12:
        raise RuntimeError(f"[{gid}] empirical mu mismatch on reload")
    return GraphBundle(graph_id=gid, family="lfr", adj=A, truth=membership, meta=meta)


def generate_all(cfg: dict, force: bool = False, verbose: bool = True) -> list[dict]:
    """Generate every (mu level x instance) graph. Idempotent: existing graphs are kept unless
    force=True. Errors are NOT swallowed: a failing seed is reported with its graph id."""
    import numpy
    import scipy

    lf = cfg["lfr"]
    out_dir = Path(cfg["_repo_root"]) / cfg["paths"]["lfr_dir"]
    try:
        import networkit as nk
        nk_version = nk.__version__
    except ImportError:
        nk_version = None
    rows = []
    for mi, mu in enumerate(lf["mu_levels"]):
        for inst in range(lf["n_instances"]):
            gid = graph_id(mu, inst)
            if (out_dir / gid / "meta.json").exists() and not force:
                meta = json.loads((out_dir / gid / "meta.json").read_text(encoding="utf-8"))
                rows.append(meta)
                continue
            if nk_version is None:
                raise ImportError("networkit is required to generate LFR graphs: pip install -r requirements-generate.txt")
            seed = instance_seed(lf, mi, inst)
            edges, membership = generate_instance(lf, mu, seed, gid)
            n = int(lf["n"])
            A = _adjacency(edges, n)
            deg = np.diff(A.indptr)
            sizes = np.bincount(membership)
            meta = {
                "graph_id": gid, "generator": "networkit.LFRGenerator", "networkit_version": nk_version,
                "numpy_version": numpy.__version__, "scipy_version": scipy.__version__,
                "python_version": platform.python_version(),
                "params": {k: lf[k] for k in ("n", "tau1", "tau2", "average_degree", "max_degree",
                                              "min_community", "max_community")},
                "mu_nominal": float(mu), "mu_index": mi, "instance": inst, "seed": seed,
                "n": n, "m": int(len(edges)), "n_communities": int(len(sizes)),
                "community_size_min": int(sizes.min()), "community_size_max": int(sizes.max()),
                "degree_mean": float(deg.mean()), "degree_max": int(deg.max()),
                "n_components": n_connected_components(A),
                "mu_empirical": empirical_mu(edges, membership),
            }
            meta = write_instance(out_dir, gid, edges, membership, meta)
            rows.append(meta)
            if verbose:
                print(f"  {gid}: seed={seed} m={meta['m']} k={meta['n_communities']} "
                      f"comps={meta['n_components']} mu_nominal={mu:.2f} mu_emp={meta['mu_empirical']:.3f}")
    return rows
