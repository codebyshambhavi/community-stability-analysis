# Stage 11 Summary

Fresh execution from the Stage 10 baseline. All 8,880 runs, Stage 6-10 outputs were regenerated here.

## Executed
| Step | Result |
|---|---|
| Raw experiments (`python main.py run --graphs all`) | 8,880/8,880 (8,400 LFR + 480 real); validated (30 seeds 0-29 per group, no duplicates, all partitions load) |
| Stage 6 (`python main.py stability-incremental`) | 296/296 groups, 128,760 pairwise rows, 2000 bootstrap, seed 20240607; checkpoints in `results/processed/stability_chunks/` |
| Stage 7 (`quality-stability`) | 296-row summary, 94 correlation rows, 4 figures (`results/figures/stage7`) |
| Stage 8 (LFR mu analysis) | No separate command exists; carried by `lfr_instances.csv`, Stage 7 mu-coloured outputs and Stage 10 fig04/fig05 + table05 |
| Stage 9 (`statistical-analysis`) | Friedman: Omega, VI, normalized_VI, NMI (70 blocks); ONMI untestable (SLPA only); 15 Holm-corrected Wilcoxon rows; 56 descriptive rows |
| Stage 10 (`generate-figures`) | 8 figures, 7 tables |

## Code changes (no metric / seed / bootstrap / statistical-method changes)
1. NEW `src/experiments/stability_incremental.py` + `stability-incremental` command: per-group checkpointing/resume around the unchanged `analyze_group`. Verified equal to the original `run_stability_analysis` (max abs diff ~4e-16 incl. bootstrap CIs at identical seed positions).
2. `statistical_analysis.py` (bug fixes, baseline Stage 9 had never been run on real Stage 7 output):
   - loader now reads `quality_stability_summary.csv` (Stage 7's real filename; `qs_summary.csv` kept as fallback for the old tests);
   - VI / normalized_VI / NMI columns resolved via `mean_stability_<metric>` (previously silently skipped);
   - same alias for the real-network descriptive table.
3. NEW `tests/test_stage11.py` (3 tests).

## Known limitations / caveats
- **Quality-metric Friedman tests were NOT produced.** Stage 9's quality loop looks for `mean_modularity`/`mean_extended_modularity`, which do not exist (Stage 7 has `mean_Q`/`mean_EQ`), and it would mix Q (hard) with EQ (SLPA) in one test. Left unchanged pending a design decision. Real-network descriptive table likewise has no modularity rows.
- **Degenerate runs**: 926 FLPA, 854 LPA, 904 semi-sync, 3 SLPA runs collapsed to trivial partitions (mostly mu >= 0.5). Their Omega near 1.0 reflects collapse, not meaningful stability (see fig04).
- **Bootstrap CI bias**: existing run-level bootstrap counts self-pairs as perfect agreement, so 28/296 Omega and 34/74 ONMI CIs lie slightly above the point estimate (max gap 0.019).
- **Tests**: 189 passed, 0 failed, 2 skipped (networkit not installed -> LFR regeneration tests). Real pytest is not installable offline; the suite was executed with a minimal stand-in runner (shipped in `tools/pytest_standin/`; run `python tools/pytest_standin/run_tests.py` from the project root).
