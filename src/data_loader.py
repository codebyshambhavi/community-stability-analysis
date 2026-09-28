"""Loading of the real datasets and dispatch to LFR graphs.

Real graphs are always loaded from the .mtx files (the source of truth). The .graphml
files produced by convert_datasets.py are derived artefacts (for Gephi) and are not used:
they lose PolBlogs' edge weights semantics and mark it directed.

Every dataset carries hard-coded EXPECTED sizes that were measured during data inspection.
If a file ever changes, loading fails loudly instead of silently altering the study.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io
import scipy.sparse as sp

from . import lfr as lfr_mod
from .graph import GraphBundle, validate_simple_undirected
from .manifest import read_manifest, verify_raw_file
from .preprocessing import largest_connected_component, symmetrize_and_clean

# Expected values were measured on the provided files (see README "Datasets").
REAL_DATASETS: dict[str, dict] = {
    "karate": {
        "file": "karate/soc-karate.mtx", "lcc": False,
        "expect": {"raw_n": 34, "raw_entries": 156, "sym_m": 78, "sym_isolates": 0, "final_n": 34, "final_m": 78},
    },
    "dolphins": {
        "file": "dolphins/dolphins.mtx", "lcc": False,
        "expect": {"raw_n": 62, "raw_entries": 318, "sym_m": 159, "sym_isolates": 0, "final_n": 62, "final_m": 159},
    },
    "polbooks": {
        "file": "polbooks/polbooks.mtx", "lcc": False,
        "expect": {"raw_n": 105, "raw_entries": 882, "sym_m": 441, "sym_isolates": 0, "final_n": 105, "final_m": 441},
    },
    "polblogs": {
        # directed multigraph collapsed to weights: 19025 unique arcs, weights sum to 19090
        "file": "polblogs/polblogs.mtx", "lcc": True,
        "expect": {"raw_n": 1490, "raw_entries": 19025, "raw_weight_sum": 19090, "self_loops": 3,
                   "sym_m": 16715, "sym_isolates": 266, "final_n": 1222, "final_m": 16714},
    },
}


class DatasetMismatchError(RuntimeError):
    pass


def _expect(name: str, what: str, got, want) -> None:
    if got != want:
        raise DatasetMismatchError(
            f"[{name}] {what}: got {got}, expected {want}. The dataset file differs from the one "
            f"the study documented; refusing to continue silently.")


def load_real(name: str, raw_dir: str | Path, manifest_path: str | Path | None = None) -> GraphBundle:
    """Load + preprocess one real dataset into a GraphBundle (with a preprocessing log in meta)."""
    if name not in REAL_DATASETS:
        raise KeyError(f"unknown real dataset '{name}'. Known: {sorted(REAL_DATASETS)}")
    spec = REAL_DATASETS[name]
    raw_dir = Path(raw_dir)
    path = raw_dir / spec["file"]

    manifest = read_manifest(manifest_path) if manifest_path else None
    ok = verify_raw_file(manifest, spec["file"], path)
    if ok is False:
        raise DatasetMismatchError(f"[{name}] checksum mismatch for {spec['file']} vs data/MANIFEST.json")

    M = scipy.io.mmread(str(path))
    M = sp.csr_matrix(M)
    exp = spec["expect"]
    _expect(name, "raw n", M.shape[0], exp["raw_n"])
    _expect(name, "raw stored entries", int(M.nnz), exp["raw_entries"])
    if "raw_weight_sum" in exp:
        _expect(name, "raw weight sum", int(M.sum()), exp["raw_weight_sum"])
    if "self_loops" in exp:
        _expect(name, "raw self-loop entries", int((M.diagonal() != 0).sum()), exp["self_loops"])

    log: list[dict] = []
    A = symmetrize_and_clean(M, log)
    _expect(name, "undirected edges after symmetrize", int(A.nnz // 2), exp["sym_m"])
    _expect(name, "isolated nodes after symmetrize", log[-1]["isolated_nodes_after"], exp["sym_isolates"])

    orig_ids = np.arange(A.shape[0])
    if spec["lcc"]:
        A, orig_ids = largest_connected_component(A, log)
    validate_simple_undirected(A)
    _expect(name, "final n", int(A.shape[0]), exp["final_n"])
    _expect(name, "final m", int(A.nnz // 2), exp["final_m"])

    truth, truth_note = load_truth(name, A, orig_ids)
    meta = {
        "source_file": spec["file"],
        "orig_ids": orig_ids,                  # index in the raw .mtx (0-based) of each final node
        "preprocessing": log,
        "truth_source": truth_note,
        "checksum_verified": ok,               # True / None (not tracked yet)
    }
    return GraphBundle(graph_id=name, family=name, adj=A, truth=truth, meta=meta)


def load_truth(name: str, A: sp.csr_matrix, orig_ids: np.ndarray) -> tuple[np.ndarray | None, str]:
    """Ground truth ONLY where it is verified against the loaded graph; otherwise None.

    - karate: NetworkX's Zachary graph. Verified here to have an identical edge set to the
      .mtx (0-indexed); labels = 'club' (Mr. Hi -> 0, Officer -> 1).
    - dolphins: no agreed ground truth -> None (by design; modularity + stability only).
    - polbooks / polblogs: label files are not in the provided data. Ground-truth evaluation
      is deferred until they are obtained AND alignment is verified -> None for now.
    """
    if name == "karate":
        import networkx as nx
        K = nx.karate_club_graph()
        G_edges = {frozenset(e) for e in zip(*sp.triu(A, k=1).nonzero())}
        K_edges = {frozenset(e) for e in K.edges()}
        if G_edges != K_edges:
            raise DatasetMismatchError("[karate] .mtx edge set != networkx karate_club_graph; refusing to attach labels")
        club = np.array([0 if K.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(K.number_of_nodes())])
        return club, "networkx.karate_club_graph 'club' (edge set verified identical to .mtx)"
    if name == "dolphins":
        return None, "none: no agreed ground truth (deliberately not invented)"
    return None, "none yet: label file not obtained/verified (deferred)"


def load_graph(graph_id: str, cfg: dict) -> GraphBundle:
    """Dispatch by id: real dataset name, or 'lfr_mu{XX}_i{YY}'."""
    root = Path(cfg["_repo_root"])
    if graph_id.startswith("lfr_"):
        return lfr_mod.load_lfr(graph_id, root / cfg["paths"]["lfr_dir"])
    return load_real(graph_id, root / cfg["paths"]["raw_dir"], root / cfg["paths"]["data_dir"] / "MANIFEST.json")


def list_lfr_ids(cfg: dict) -> list[str]:
    lf = cfg["lfr"]
    return [lfr_mod.graph_id(mu, i) for mu in lf["mu_levels"] for i in range(lf["n_instances"])]


def all_graph_ids(cfg: dict) -> list[str]:
    return list(cfg["real_datasets"]) + list_lfr_ids(cfg)
