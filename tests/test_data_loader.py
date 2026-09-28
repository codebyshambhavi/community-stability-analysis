import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.config import load_config, resolve
from src.data_loader import REAL_DATASETS, DatasetMismatchError, load_graph, load_real
from src.manifest import sha256_file, write_manifest

CFG = load_config()
RAW = resolve(CFG, "raw_dir")


@pytest.mark.parametrize("name,n,m", [("karate", 34, 78), ("dolphins", 62, 159),
                                      ("polbooks", 105, 441), ("polblogs", 1222, 16714)])
def test_real_dataset_sizes(name, n, m):
    g = load_real(name, RAW)
    assert (g.n, g.m) == (n, m)
    assert g.family == name


def test_polblogs_preprocessing_facts_are_logged():
    g = load_real("polblogs", RAW)
    sym, lcc = g.meta["preprocessing"]
    assert sym["stored_entries_before"] == 19025 and sym["weight_sum_before"] == 19090
    assert sym["self_loops_dropped"] == 3 and sym["input_was_symmetric"] is False
    assert sym["undirected_edges_after"] == 16715 and sym["isolated_nodes_after"] == 266
    assert (lcc["n_after"], lcc["m_after"], lcc["nodes_removed"]) == (1222, 16714, 268)
    assert len(g.meta["orig_ids"]) == 1222 and np.all(np.diff(g.meta["orig_ids"]) > 0)


def test_ground_truth_policy():
    k = load_real("karate", RAW)
    assert sorted(np.bincount(k.truth).tolist()) == [17, 17]
    for name in ("dolphins", "polbooks", "polblogs"):
        assert load_real(name, RAW).truth is None          # never invented; polbooks/polblogs deferred


def test_loading_is_deterministic():
    a, b = load_real("polbooks", RAW), load_real("polbooks", RAW)
    assert (a.adj != b.adj).nnz == 0


def test_loader_refuses_modified_dataset(tmp_path):
    """A well-formed but altered file (one edge removed, header updated) must be rejected."""
    raw = tmp_path / "raw"
    shutil.copytree(RAW / "dolphins", raw / "dolphins")
    f = raw / "dolphins" / "dolphins.mtx"
    lines = f.read_text().splitlines()
    hdr = next(i for i, l in enumerate(lines) if not l.startswith("%"))   # size line "62 62 159"
    r, c, nnz = lines[hdr].split()
    lines[hdr] = f"{r} {c} {int(nnz) - 1}"
    f.write_text("\n".join(lines[:-1]) + "\n")                            # drop last edge, keep file valid
    with pytest.raises(DatasetMismatchError):
        load_real("dolphins", raw)


def test_manifest_checksum_detects_tampering(tmp_path):
    raw = tmp_path / "raw"
    shutil.copytree(RAW / "karate", raw / "karate")
    f = raw / "karate" / "soc-karate.mtx"
    mpath = tmp_path / "MANIFEST.json"
    write_manifest(mpath, {"karate/soc-karate.mtx": sha256_file(f)}, {})
    assert load_real("karate", raw, mpath).meta["checksum_verified"] is True
    f.write_text(f.read_text() + "% tampered\n")          # harmless comment, but bytes changed
    with pytest.raises(DatasetMismatchError, match="checksum"):
        load_real("karate", raw, mpath)


def test_repo_manifest_matches_raw_files():
    manifest = json.loads((Path(CFG["_repo_root"]) / "data" / "MANIFEST.json").read_text())
    for name, spec in REAL_DATASETS.items():
        assert manifest["raw"][spec["file"]] == sha256_file(RAW / spec["file"])


def test_load_graph_dispatch():
    assert load_graph("karate", CFG).n == 34
