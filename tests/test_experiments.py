"""Stage 5 tests: atomic partition/record persistence, resumability, idempotent CSV
rebuilds, and the runner end-to-end. Uses tiny fixtures only -- never the full 8,880-run
matrix (see docs/methodology.md "STAGE 5 MUST NOT DO" / "TEST REQUIREMENTS")."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from src.experiments import run_experiments, run_single, storage
from src.graph import GraphBundle


def _tiny_graph(graph_id="tiny", seed_truth=True):
    """Two triangles {0,1,2} and {3,4,5} joined by a single bridge edge -- small enough
    to run every algorithm instantly, structured enough that LPA-family algorithms find
    a real (non-trivial) two-community split most of the time."""
    edges = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]
    n = 6
    rows, cols = zip(*edges)
    U = sp.coo_matrix(([1] * len(edges), (rows, cols)), shape=(n, n)).tocsr()
    A = (U + U.T).tocsr()
    A.data[:] = 1
    A.sort_indices()
    A.sum_duplicates()
    truth = np.array([0, 0, 0, 1, 1, 1]) if seed_truth else None
    return GraphBundle(graph_id=graph_id, family="tiny", adj=A, truth=truth)


TINY_CFG = {
    "runs": {"n_runs": 5, "seed_start": 0},
    "paths": {"data_dir": "data", "raw_dir": "data/raw", "lfr_dir": "data/lfr", "results_dir": "results"},
    "real_datasets": ["tiny"],
    "slpa": {"T": 20, "r": 0.1},          # small T: keeps the SLPA tests fast
}


@pytest.fixture
def graph():
    return _tiny_graph()


# ---------- 1. tiny graph + one algorithm + few seeds works ----------

def test_run_single_hard_algorithm_works(tmp_path, graph):
    rec = run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    assert rec is not None
    assert rec["graph_id"] == "tiny" and rec["algorithm"] == "lpa" and rec["seed"] == 0
    assert rec["partition_kind"] == "hard"
    assert rec["quality_metric"] == "Q"
    assert rec["number_of_communities"] >= 1


def test_run_single_slpa_produces_cover(tmp_path, graph):
    rec = run_single(tmp_path, TINY_CFG, graph, "slpa", seed=0, run_index=0)
    assert rec["partition_kind"] == "cover"
    assert rec["quality_metric"] == "EQ"


# ---------- 2. results CSV schema is correct ----------

def test_results_csv_schema(tmp_path, graph):
    df, _, _ = run_experiments(TINY_CFG, [graph], ["lpa", "slpa"], [0, 1],
                                results_dir=tmp_path, verbose=False)
    assert list(df.columns) == storage.RUN_COLUMNS
    assert len(df) == 4                                     # 1 graph x 2 algos x 2 seeds
    assert set(df["algorithm"]) == {"lpa", "slpa"}
    assert (tmp_path / "runs.csv").exists()


# ---------- 3 & 4. partition files are created and can be loaded back ----------

def test_partition_files_created_and_loadable(tmp_path, graph):
    rec = run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    ppath = tmp_path / rec["partition_file"]
    assert ppath.exists()
    labels = storage.load_hard_partition(ppath)
    assert len(labels) == graph.n

    rec2 = run_single(tmp_path, TINY_CFG, graph, "slpa", seed=0, run_index=0)
    cover = storage.load_cover_partition(tmp_path / rec2["partition_file"])
    covered = set().union(*cover) if cover else set()
    assert covered == set(range(graph.n))                    # every node in >=1 community


# ---------- 5 & 6. rerunning does not duplicate rows / completed experiments skipped ----------

def test_rerun_does_not_duplicate_and_skips_completed(tmp_path, graph):
    df1, n_exec1, n_skip1 = run_experiments(TINY_CFG, [graph], ["lpa"], [0, 1, 2],
                                             results_dir=tmp_path, verbose=False)
    assert n_exec1 == 3 and n_skip1 == 0 and len(df1) == 3

    df2, n_exec2, n_skip2 = run_experiments(TINY_CFG, [graph], ["lpa"], [0, 1, 2],
                                             results_dir=tmp_path, verbose=False)
    assert n_exec2 == 0 and n_skip2 == 3                      # nothing re-executed
    assert len(df2) == 3                                      # still exactly 3 rows, no duplicates
    assert sorted(df2["seed"]) == [0, 1, 2]


# ---------- 7. missing experiments run ----------

def test_missing_experiments_are_filled_in(tmp_path, graph):
    run_experiments(TINY_CFG, [graph], ["lpa"], [0, 1], results_dir=tmp_path, verbose=False)
    df, n_exec, n_skip = run_experiments(TINY_CFG, [graph], ["lpa"], [0, 1, 2, 3],
                                          results_dir=tmp_path, verbose=False)
    assert n_exec == 2 and n_skip == 2                        # only seeds 2,3 were missing
    assert sorted(df["seed"]) == [0, 1, 2, 3]


# ---------- 8. partial/corrupt artifacts are not treated as complete ----------

def test_corrupt_partition_file_is_not_complete(tmp_path, graph):
    rec = run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    ppath = tmp_path / rec["partition_file"]
    assert storage.is_run_complete(tmp_path, "tiny", "lpa", 0)

    ppath.write_text("not,a,valid,partition,file\ngarbage\n")
    assert not storage.is_run_complete(tmp_path, "tiny", "lpa", 0)

    # rerunning the batch must repair it rather than silently leaving it corrupt
    df, n_exec, n_skip = run_experiments(TINY_CFG, [graph], ["lpa"], [0], results_dir=tmp_path, verbose=False)
    assert n_exec == 1
    assert storage.is_run_complete(tmp_path, "tiny", "lpa", 0)


def test_missing_record_is_not_complete(tmp_path, graph):
    run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    rp = storage.record_path(tmp_path, "tiny", "lpa", 0)
    rp.unlink()
    assert not storage.is_run_complete(tmp_path, "tiny", "lpa", 0)


def test_truncated_record_json_is_not_complete(tmp_path, graph):
    run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    rp = storage.record_path(tmp_path, "tiny", "lpa", 0)
    rp.write_text('{"graph_id": "tiny", "algorithm": "lpa"')      # truncated JSON
    assert not storage.is_run_complete(tmp_path, "tiny", "lpa", 0)


# ---------- 9. runtime is recorded ----------

def test_runtime_is_recorded(tmp_path, graph):
    rec = run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    assert isinstance(rec["runtime_seconds"], float)
    assert rec["runtime_seconds"] >= 0.0


# ---------- 10. seeds are preserved ----------

def test_seeds_are_preserved_in_records_and_csv(tmp_path, graph):
    df, _, _ = run_experiments(TINY_CFG, [graph], ["lpa"], [7, 8, 9], results_dir=tmp_path, verbose=False)
    assert sorted(df["seed"].tolist()) == [7, 8, 9]
    # same seed -> same partition for a deterministic-given-seed hard algorithm
    a = run_single(tmp_path, TINY_CFG, graph, "lpa", seed=42, run_index=0, force=True)
    b = run_single(tmp_path, TINY_CFG, graph, "lpa", seed=42, run_index=0, force=True)
    labels_a = storage.load_hard_partition(tmp_path / a["partition_file"])
    labels_b = storage.load_hard_partition(tmp_path / b["partition_file"])
    assert np.array_equal(labels_a, labels_b)


# ---------- 11. graph_id/algorithm/seed completion key works ----------

def test_completion_key_is_graph_algorithm_seed(tmp_path, graph):
    run_single(tmp_path, TINY_CFG, graph, "lpa", seed=0, run_index=0)
    assert storage.is_run_complete(tmp_path, "tiny", "lpa", 0)
    # different seed, different algorithm -> independently incomplete
    assert not storage.is_run_complete(tmp_path, "tiny", "lpa", 1)
    assert not storage.is_run_complete(tmp_path, "tiny", "flpa", 0)
    assert not storage.is_run_complete(tmp_path, "other_graph", "lpa", 0)


# ---------- 12. SLPA overlapping storage works ----------

def test_slpa_overlapping_cover_roundtrip(tmp_path):
    cover = [{0, 1, 2}, {2, 3, 4}, {4, 5}]        # node 2 and node 4 are genuinely overlapping
    ppath = tmp_path / "cover.csv"
    storage.save_cover_partition(ppath, cover)
    loaded = storage.load_cover_partition(ppath)
    assert loaded == cover
    # node 2 must appear in exactly the communities it was placed in
    membership_of_2 = [i for i, c in enumerate(loaded) if 2 in c]
    assert len(membership_of_2) == 2


def test_slpa_run_persists_overlap_not_hard_conversion(tmp_path, graph):
    rec = run_single(tmp_path, TINY_CFG, graph, "slpa", seed=0, run_index=0)
    cover = storage.load_cover_partition(tmp_path / rec["partition_file"])
    # a cover is a list of sets, never collapsed into a single-label-per-node array
    assert isinstance(cover, list) and all(isinstance(c, set) for c in cover)


# ---------- atomic-write sanity: a killed write must never leave a half-written file ----------

def test_atomic_write_leaves_no_partial_file_on_failure(tmp_path):
    class Boom(Exception):
        pass

    target = tmp_path / "sub" / "file.txt"
    storage.atomic_write_text(target, "first version\n")

    import src.experiments.storage as storage_mod
    real_replace = storage_mod.os.replace

    def failing_replace(*a, **k):
        raise Boom("simulated crash before rename")

    storage_mod.os.replace = failing_replace
    try:
        with pytest.raises(Boom):
            storage.atomic_write_text(target, "second version (should not land)\n")
    finally:
        storage_mod.os.replace = real_replace

    # original content survives untouched, and no stray .tmp files are left behind
    assert target.read_text() == "first version\n"
    leftovers = list(target.parent.glob("*.tmp"))
    assert leftovers == []


# ---------- existing Stage 1-4 baseline is unaffected (smoke check, not a re-run of those suites) ----------

def test_algorithms_module_still_importable_and_registered():
    from src.algorithms import ALGORITHMS
    assert set(ALGORITHMS) == {"lpa", "semi_sync_lpa", "flpa", "slpa"}
