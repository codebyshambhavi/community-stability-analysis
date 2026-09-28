import json

import numpy as np
import pytest

from src import lfr
from src.config import load_config, resolve
from src.data_loader import all_graph_ids, list_lfr_ids, load_graph

CFG = load_config()
LFR_DIR = resolve(CFG, "lfr_dir")


def test_graph_ids_and_counts():
    ids = list_lfr_ids(CFG)
    assert len(ids) == 70 and len(set(ids)) == 70
    assert ids[0] == "lfr_mu10_i00" and ids[-1] == "lfr_mu70_i09"
    assert len(all_graph_ids(CFG)) == 74


def test_instance_seeds_are_distinct():
    lf = CFG["lfr"]
    seeds = [lfr.instance_seed(lf, mi, i) for mi in range(len(lf["mu_levels"])) for i in range(lf["n_instances"])]
    assert len(set(seeds)) == len(seeds)


def test_empirical_mu_definition():
    edges = np.array([[0, 1], [1, 2], [2, 3], [0, 3]])
    membership = np.array([0, 0, 1, 1])
    assert lfr.empirical_mu(edges, membership) == 0.5      # edges (1,2) and (0,3) cross


@pytest.mark.parametrize("gid", ["lfr_mu10_i00", "lfr_mu40_i05", "lfr_mu70_i09"])
def test_cached_graphs_load_and_are_consistent(gid):
    g = load_graph(gid, CFG)
    assert g.n == CFG["lfr"]["n"] and g.truth is not None and g.family == "lfr"
    assert g.m == g.meta["m"]
    assert g.meta["n_components"] == 1
    assert set(g.truth.tolist()) == set(range(g.meta["n_communities"]))


def test_all_70_cached_and_empirical_mu_monotone_in_nominal():
    means = []
    for mi, mu in enumerate(CFG["lfr"]["mu_levels"]):
        vals = [load_graph(lfr.graph_id(mu, i), CFG).meta["mu_empirical"] for i in range(CFG["lfr"]["n_instances"])]
        assert max(vals) - min(vals) < 0.02          # tight within a level
        means.append(np.mean(vals))
    assert all(a < b for a, b in zip(means, means[1:]))


def test_cache_tamper_detected(tmp_path):
    src = LFR_DIR / "lfr_mu30_i01"
    dst = tmp_path / "lfr_mu30_i01"
    dst.mkdir()
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    lfr.load_lfr("lfr_mu30_i01", tmp_path)               # intact: fine
    (dst / "edges.csv").write_text((dst / "edges.csv").read_text() + "0,999\n")
    with pytest.raises(RuntimeError, match="checksum"):
        lfr.load_lfr("lfr_mu30_i01", tmp_path)


def test_generation_reproducible_same_seed_different_across_seeds():
    pytest.importorskip("networkit")
    lf = dict(CFG["lfr"], n=300)
    e1, m1 = lfr.generate_instance(lf, 0.3, 7)
    e2, m2 = lfr.generate_instance(lf, 0.3, 7)
    e3, _ = lfr.generate_instance(lf, 0.3, 8)
    assert np.array_equal(e1, e2) and np.array_equal(m1, m2)
    assert e1.shape != e3.shape or not np.array_equal(e1, e3)
    assert (e1[:, 0] < e1[:, 1]).all()                   # canonical, no loops, no duplicates


def test_cached_graph_equals_fresh_regeneration():
    pytest.importorskip("networkit")
    lf = CFG["lfr"]
    gid = lfr.graph_id(0.3, 2)
    seed = lfr.instance_seed(lf, lf["mu_levels"].index(0.3), 2)
    edges, membership = lfr.generate_instance(lf, 0.3, seed, gid)
    g = load_graph(gid, CFG)
    assert np.array_equal(g.edges(), edges) and np.array_equal(g.truth, membership)
