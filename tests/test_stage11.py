"""Stage 11 tests: checkpointed Stage 6 driver (equivalence with the original
run_stability_analysis, resume without recomputation) and the Stage 9 loader / column-alias fixes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.experiments import run_experiments
from src.experiments.stability import run_stability_analysis
from src.experiments.stability_incremental import run_stability_incremental
from src.experiments import statistical_analysis as sa
from tests.test_experiments import TINY_CFG, _tiny_graph


def _setup(tmp_path):
    raw = tmp_path / "raw"
    run_experiments(TINY_CFG, [_tiny_graph()], ["lpa", "flpa", "slpa"], list(range(5)),
                    results_dir=raw, verbose=False)
    return tmp_path


def test_incremental_matches_original(tmp_path):
    res = _setup(tmp_path)
    pw0, sm0 = run_stability_analysis(TINY_CFG, results_dir=res, n_bootstrap=50, verbose=False)
    pw1, sm1 = run_stability_incremental(TINY_CFG, results_dir=res, n_bootstrap=50, verbose=False)
    num = [c for c in sm0.columns if c.startswith(("mean_", "std_", "median_"))]
    np.testing.assert_allclose(sm0[num].astype(float).values, sm1[num].astype(float).values,
                               atol=1e-12, equal_nan=True)
    assert len(pw0) == len(pw1) == 3 * 10


def test_incremental_resumes_without_recomputing(tmp_path):
    res = _setup(tmp_path)
    run_stability_incremental(TINY_CFG, results_dir=res, n_bootstrap=20, max_groups=1, verbose=False)
    chunks = res / "processed" / "stability_chunks"
    first = sorted(chunks.glob("*.summary.json"))
    assert len(first) == 1
    mtime = first[0].stat().st_mtime_ns
    out = run_stability_incremental(TINY_CFG, results_dir=res, n_bootstrap=20, verbose=False)
    assert out is not None and len(list(chunks.glob("*.summary.json"))) == 3
    assert first[0].stat().st_mtime_ns == mtime              # completed checkpoint untouched
    prog = json.loads((chunks / "progress.json").read_text())
    assert prog["groups_completed"] == prog["groups_total"] == 3


def test_stage9_reads_stage7_filename_and_stability_alias(tmp_path):
    proc = tmp_path / "processed"
    proc.mkdir()
    rows = []
    rng = np.random.default_rng(0)
    for i in range(6):
        for a in ["lpa", "semi_sync_lpa", "flpa"]:
            rows.append({"graph_id": f"g{i}", "dataset": "lfr", "algorithm": a,
                         "mean_stability_VI": rng.random() + (a == "flpa"),
                         "mean_Omega": rng.random()})
    pd.DataFrame(rows).to_csv(proc / "quality_stability_summary.csv", index=False)
    fr, ph = sa.run_lfr_statistical_analysis({"paths": {"results_dir": str(tmp_path)}},
                                             results_dir=tmp_path)
    assert {"VI", "Omega"} <= set(fr["metric"])
    assert (fr.loc[fr.metric == "VI", "n_blocks"] == 6).all()
