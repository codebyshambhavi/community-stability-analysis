"""Command-line entry point. Stages are separate so each can be (re)run independently.

    python main.py prepare        # verify raw datasets, log preprocessing, write data/MANIFEST.json
    python main.py generate-lfr   # generate + cache LFR graphs (needs networkit)
    python main.py run            # execute (graph, algorithm, seed) runs; resumable/idempotent
        --graphs karate,dolphins      comma-separated graph ids, or 'real' / 'lfr' / 'all'
                                       (default: 'real' -- the 4 real datasets only)
        --algorithms lpa,flpa         comma-separated algorithm names, or 'all' (default: all)
        --seeds 0-4                   'a-b' range or comma-separated ints (default: all configured seeds)
        --force                       re-run and overwrite even already-complete runs

    python main.py quality-stability   # join Stage 5 quality + Stage 6 stability, correlate, plot
        --graphs / --algorithms       same filters as 'run' (default: every group present in
                                       results/raw/runs.csv and results/processed/stability_summary.csv)

    python main.py statistical-analysis   # Stage 9: statistical analysis (Friedman/Wilcoxon with Holm correction)
        --graphs / --algorithms       same filters as 'run' (default: every group present in
                                       results/raw/runs.csv and results/processed/stability_summary.csv)
        --type                        lfr, real, or both (default: both)
                                       lfr: inferential statistics only (Friedman test)
                                       real: descriptive statistics only
                                       both: both inferential and descriptive

    python main.py generate-figures   # Stage 10: final figures + tables (read-only;
                                       # never re-runs Stages 5/6/9, never regenerates LFR)
                                       # writes results/figures/final/ and results/tables/final/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.config import load_config, resolve
from src.data_loader import REAL_DATASETS, load_real
from src.experiments.storage import atomic_write_text
from src.manifest import sha256_file, write_manifest


def cmd_prepare(cfg: dict) -> None:
    raw_dir, data_dir = resolve(cfg, "raw_dir"), resolve(cfg, "data_dir")
    proc_dir = resolve(cfg, "results_dir") / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    log, raw_hashes = {}, {}
    for name in cfg["real_datasets"]:
        g = load_real(name, raw_dir)                    # raises DatasetMismatchError on any surprise
        log[name] = {
            "source_file": g.meta["source_file"],
            "final_n": g.n, "final_m": g.m,
            "ground_truth": g.meta["truth_source"],
            "steps": g.meta["preprocessing"],
        }
        raw_hashes[REAL_DATASETS[name]["file"]] = sha256_file(raw_dir / REAL_DATASETS[name]["file"])
        if len(g.meta["orig_ids"]) != g.meta["preprocessing"][0]["n"]:      # LCC restriction happened
            np.savetxt(proc_dir / f"orig_ids_{name}.csv", g.meta["orig_ids"], fmt="%d",
                       header="index_in_raw_mtx_0based", comments="")
        print(f"  {name:9s} n={g.n:5d} m={g.m:6d}  truth: {g.meta['truth_source']}")

    (proc_dir / "preprocessing_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")

    lfr_dir = resolve(cfg, "lfr_dir")
    lfr_hashes = {}
    for meta_path in sorted(lfr_dir.glob("lfr_*/meta.json")):
        m = json.loads(meta_path.read_text(encoding="utf-8"))
        lfr_hashes[m["graph_id"]] = {k: m[k] for k in ("sha256_edges", "sha256_membership", "seed", "mu_empirical")}
    write_manifest(data_dir / "MANIFEST.json", raw_hashes, lfr_hashes)
    print(f"wrote {proc_dir/'preprocessing_log.json'} and {data_dir/'MANIFEST.json'} "
          f"({len(raw_hashes)} raw files, {len(lfr_hashes)} LFR graphs)")


def cmd_generate_lfr(cfg: dict, force: bool) -> None:
    from src.lfr import generate_all
    rows = generate_all(cfg, force=force)
    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (df.groupby("mu_nominal")
                 .agg(n_graphs=("graph_id", "count"), mu_emp_mean=("mu_empirical", "mean"),
                      mu_emp_min=("mu_empirical", "min"), mu_emp_max=("mu_empirical", "max"),
                      k_mean=("n_communities", "mean"), comps_max=("n_components", "max")))
    print("\nnominal vs empirical mu (per nominal level):")
    print(summary.round(3).to_string())
    proc = resolve(cfg, "results_dir") / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["params"]).to_csv(proc / "lfr_instances.csv", index=False)
    print("\nnow run: python main.py prepare   (refreshes data/MANIFEST.json)")


def _parse_graphs(spec: str | None, cfg: dict) -> list[str]:
    from src.data_loader import list_lfr_ids
    if spec is None or spec == "real":
        return list(cfg["real_datasets"])
    if spec == "lfr":
        return list_lfr_ids(cfg)
    if spec == "all":
        return list(cfg["real_datasets"]) + list_lfr_ids(cfg)
    return [g.strip() for g in spec.split(",") if g.strip()]


def _parse_algorithms(spec: str | None) -> list[str]:
    from src.algorithms import ALGORITHMS
    if spec is None or spec == "all":
        return list(ALGORITHMS)
    algos = [a.strip() for a in spec.split(",") if a.strip()]
    unknown = [a for a in algos if a not in ALGORITHMS]
    if unknown:
        raise ValueError(f"unknown algorithm(s) {unknown}; known: {sorted(ALGORITHMS)}")
    return algos


def _parse_seeds(spec: str | None, cfg: dict) -> list[int]:
    from src.config import run_seeds
    if spec is None:
        return run_seeds(cfg)
    if "-" in spec and "," not in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(s.strip()) for s in spec.split(",") if s.strip()]


def cmd_run(cfg: dict, spec_graphs: str | None, spec_algos: str | None, spec_seeds: str | None,
            force: bool) -> None:
    from src.experiments.runner import run_experiments

    graph_ids = _parse_graphs(spec_graphs, cfg)
    algorithms = _parse_algorithms(spec_algos)
    seeds = _parse_seeds(spec_seeds, cfg)

    n_total = len(graph_ids) * len(algorithms) * len(seeds)
    print(f"run: {len(graph_ids)} graph(s) x {len(algorithms)} algorithm(s) x {len(seeds)} seed(s) "
          f"= {n_total} run(s) to check (already-complete ones are skipped)")
    run_experiments(cfg, graph_ids, algorithms, seeds, force=force)


def cmd_stability(cfg: dict, spec_graphs: str | None, spec_algos: str | None,
                   n_bootstrap: int, bootstrap_seed: int) -> None:
    from src.experiments.stability import run_stability_analysis

    graph_ids = _parse_graphs(spec_graphs, cfg) if spec_graphs is not None else None
    algorithms = _parse_algorithms(spec_algos) if spec_algos is not None else None
    run_stability_analysis(cfg, graph_ids=graph_ids, algorithms=algorithms,
                            n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed)


def cmd_quality_stability(cfg: dict, spec_graphs: str | None, spec_algos: str | None) -> None:
    from src.experiments.quality_stability import run_quality_stability_analysis
    from src.experiments.quality_stability_figures import make_stage7_figures

    graph_ids = _parse_graphs(spec_graphs, cfg) if spec_graphs is not None else None
    algorithms = _parse_algorithms(spec_algos) if spec_algos is not None else None
    summary, correlations = run_quality_stability_analysis(cfg, graph_ids=graph_ids, algorithms=algorithms)
    figures_dir = resolve(cfg, "results_dir") / "figures" / "stage7"
    written = make_stage7_figures(summary, figures_dir)
    if written:
        print(f"wrote {len(written)} figure(s) under {figures_dir}")


def cmd_statistical_analysis(cfg: dict, spec_graphs: str | None, spec_algos: str | None,
                             analysis_type: str) -> None:
    """Stage 9: statistical analysis."""
    from src.experiments.statistical_analysis import run_statistical_analysis

    graph_ids = _parse_graphs(spec_graphs, cfg) if spec_graphs is not None else None
    algorithms = _parse_algorithms(spec_algos) if spec_algos is not None else None

    friedman_df, posthoc_df, descriptive_df = run_statistical_analysis(
        cfg,
        graph_ids=graph_ids,
        algorithms=algorithms,
        analysis_type=analysis_type
    )

    from src.config import resolve
    results_dir = resolve(cfg, "results_dir")
    proc_dir = results_dir / "processed"
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Write results
    if not friedman_df.empty:
        friedman_path = proc_dir / "statistical_tests.csv"
        atomic_write_text(friedman_path, friedman_df.to_csv(index=False))
        print(f"wrote {friedman_path} ({len(friedman_df)} rows)")

    if not posthoc_df.empty:
        posthoc_path = proc_dir / "posthoc_tests.csv"
        atomic_write_text(posthoc_path, posthoc_df.to_csv(index=False))
        print(f"wrote {posthoc_path} ({len(posthoc_df)} rows)")

    if not descriptive_df.empty:
        descriptive_path = proc_dir / "descriptive_statistics.csv"
        atomic_write_text(descriptive_path, descriptive_df.to_csv(index=False))
        print(f"wrote {descriptive_path} ({len(descriptive_df)} rows)")

    total_rows = len(friedman_df) + len(posthoc_df) + len(descriptive_df)
    if total_rows == 0:
        print("no statistical analysis results to write (insufficient data or wrong filters)")


def cmd_generate_figures(cfg: dict) -> None:
    """Stage 10: final figures + tables. Read-only -- reads whatever Stages 1-9 have
    already persisted under results/processed/; never re-runs an algorithm, never
    regenerates an LFR graph, never re-executes Stage 5/6/9."""
    from src.experiments.figure_generation import run_figure_generation
    run_figure_generation(cfg)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["prepare", "generate-lfr", "run", "stability", "stability-incremental",
                                      "quality-stability", "statistical-analysis", "generate-figures"])
    p.add_argument("--config", default=None)
    p.add_argument("--force", action="store_true",
                    help="regenerate existing LFR graphs / re-run and overwrite already-complete runs")
    p.add_argument("--graphs", default=None,
                    help="run stage: comma-separated graph ids, or 'real'/'lfr'/'all' (default: 'real')")
    p.add_argument("--algorithms", default=None,
                    help="run stage: comma-separated algorithm names, or 'all' (default: 'all')")
    p.add_argument("--seeds", default=None,
                    help="run stage: 'a-b' range or comma-separated ints (default: all configured seeds)")
    p.add_argument("--n-bootstrap", type=int, default=None,
                    help="stability stage: bootstrap replicates per graph x algorithm group "
                         "(default: 2000)")
    p.add_argument("--bootstrap-seed", type=int, default=None,
                    help="stability stage: base seed for the (deterministic) bootstrap "
                         "resampling (default: fixed project seed)")
    p.add_argument("--type", default=None,
                    help="statistical-analysis stage: lfr, real, or both (default: both)")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    if a.stage == "prepare":
        cmd_prepare(cfg)
    elif a.stage == "generate-lfr":
        cmd_generate_lfr(cfg, a.force)
    elif a.stage == "run":
        cmd_run(cfg, a.graphs, a.algorithms, a.seeds, a.force)
    elif a.stage == "stability":
        from src.experiments.stability import DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP
        cmd_stability(cfg, a.graphs, a.algorithms,
                      a.n_bootstrap if a.n_bootstrap is not None else DEFAULT_N_BOOTSTRAP,
                      a.bootstrap_seed if a.bootstrap_seed is not None else DEFAULT_BOOTSTRAP_SEED)
    elif a.stage == "stability-incremental":
        from src.experiments.stability import DEFAULT_BOOTSTRAP_SEED, DEFAULT_N_BOOTSTRAP
        from src.experiments.stability_incremental import run_stability_incremental
        run_stability_incremental(cfg,
            n_bootstrap=a.n_bootstrap if a.n_bootstrap is not None else DEFAULT_N_BOOTSTRAP,
            bootstrap_seed=a.bootstrap_seed if a.bootstrap_seed is not None else DEFAULT_BOOTSTRAP_SEED)
    elif a.stage == "quality-stability":
        cmd_quality_stability(cfg, a.graphs, a.algorithms)
    elif a.stage == "statistical-analysis":
        analysis_type = a.type if a.type is not None else "both"
        cmd_statistical_analysis(cfg, a.graphs, a.algorithms, analysis_type)
    elif a.stage == "generate-figures":
        cmd_generate_figures(cfg)
    else:
        print(f"stage '{a.stage}' is not built yet (build order: data → metrics → algorithms → runner → stats → figures)")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())