# A Comparative Stability Analysis of Label Propagation Algorithms for Community Detection in Social Networks

**Experimental Methodology**
Social Networks and Network Mining (SNM) — Community Detection Project

---

## 1. Research Question

To what extent do algorithmic modifications to the Label Propagation Algorithm (semi-synchronous updating, queue-based fast propagation, and speaker-listener multi-labeling) improve the run-to-run stability of detected community structures, and does improved stability come at a cost to community detection quality?

---

## 2. Objectives

1. Implement/run LPA, Semi-synchronous LPA, FLPA, and SLPA on a common set of benchmark networks.
2. Quantify **run-to-run stability** for each algorithm using a common, formal metric (not just visual inspection).
3. Quantify **community quality** for each algorithm using standard metrics.
4. Determine whether there is a **stability–quality trade-off**, or whether some algorithm dominates on both axes.
5. Determine whether **network structure** (real vs. synthetic; clear vs. ambiguous community structure) affects the stability ranking of these algorithms.
6. Produce a reproducible experimental pipeline and a clear, evidence-based comparison — not a claim of a new "best" algorithm.

---

## 3. Hypotheses

These are stated as things to be *tested*, not assumed. A null or contrary result is a valid, reportable outcome.

- **H1:** Plain LPA will show the highest run-to-run instability (largest self-VI / lowest pairwise NMI-across-runs) among the four algorithms, due to unconstrained random tie-breaking and update order.
- **H2:** Semi-synchronous LPA and FLPA will show measurably lower instability than plain LPA, since both were explicitly designed to reduce oscillation/randomness — but FLPA is not designed *for* stability (it's designed for speed), so this needs empirical testing rather than assumption.
- **H3:** SLPA will show different stability behavior than the three non-overlapping methods, since its memory-based accumulation process is structurally different — direction (more or less stable) is left open, not assumed.
- **H4:** Instability will be more pronounced on networks with ambiguous/weak community structure (e.g., PolBlogs, LFR with high mixing parameter μ) than on networks with strong, well-separated communities (e.g., Karate Club, low-μ LFR).
- **H5 (quality):** No algorithm will dominate on quality (modularity/NMI) across all datasets — quality differences will be small and dataset-dependent, since all four are fundamentally the same LPA-family mechanism.

---

## 4. Dataset Selection

| Dataset | Nodes/Edges (approx.) | Ground truth for NMI? | Role in study |
|---|---|---|---|
| **Karate Club** | 34 / 78 | Yes — Zachary's 2-faction split | Small, clean, strong community structure — sanity check / baseline |
| **Dolphins** | 62 / 159 | **No official ground truth** | Real-world, moderate size — quality reported via modularity only; NMI excluded or reported with an explicit caveat if using an informal 2-group split |
| **PolBooks** | 105 / 441 | Yes (informal) — liberal/neutral/conservative labels commonly used as ground truth | Medium-size, 3-class structure — good for NMI comparison |
| **PolBlogs** | 1,490 / 19,090 (largest connected component ~1,222 nodes) | Yes — left/right political leaning | Largest, noisiest real network — tests behavior on ambiguous structure; must remove self-loops and use only the largest connected component before running |
| **LFR synthetic** | Configurable | Yes — always known by construction | Only dataset type where community structure difficulty (mixing parameter μ) can be systematically controlled — essential for testing H4 |

### LFR synthetic network parameters

- Generate networks at **3–4 mixing-parameter (μ) levels**, e.g., μ = 0.1 (clear structure), 0.3 (moderate), 0.5 (ambiguous), 0.7 (very ambiguous) — this directly tests H4.
- Keep node count moderate (e.g., n = 500–1,000) so all four algorithms run quickly and repeatedly.
- Generate **3–5 independent LFR graph instances per μ level** (not just one), so stability results aren't an artifact of one particular random graph.
- Use NetworkX's built-in LFR generator (`networkx.generators.community.LFR_benchmark_graph`) — avoids needing external tools.

**Note on Dolphins:** Since it lacks agreed-upon ground truth, NMI/F-score results for Dolphins (if reported at all) should use an informally-accepted split and be read with more caution than Karate/PolBooks/PolBlogs.

---

## 5. Experimental Procedure

Step-by-step pipeline, per (dataset, algorithm) pair:

1. **Load and preprocess** the dataset (undirected, no self-loops, largest connected component only where applicable).
2. **Run the algorithm N times** (see §6), each with a different random seed, recording the full community partition from each run.
3. For each run, compute:
   - All **quality metrics** (§8) — using ground truth where available.
   - Store the **raw partition** (node → community label mapping) for later pairwise stability comparison.
4. After all N runs are complete for that (dataset, algorithm) pair, compute:
   - All **stability metrics** (§7), using the N stored partitions.
5. Repeat for all 4 algorithms × 8 datasets (4 real + 4 LFR μ-levels) = **32 (dataset, algorithm) combinations**, each run N times.
6. Aggregate results into the master results table (§9) and generate visualizations (§10).
7. Run statistical significance tests (§11) across algorithms.

**Controlled variables** to fix across all runs of a given dataset: same graph (same LFR instance, same preprocessing), same evaluation metric implementations, same convergence criteria (e.g., max iterations) applied consistently per algorithm.

**Practical note:** For LFR, run this whole pipeline once per (μ level × graph instance), then average results across the 3–5 instances at each μ level — this separates "randomness from the algorithm" (what you're studying) from "randomness from which particular graph was generated" (a nuisance variable you want to average out).

---

## 6. Number of Runs

- **N = 30 runs per (dataset, algorithm) pair** is a reasonable, defensible default — large enough for stable statistics (mean/std of stability and quality metrics), small enough to be feasible on Karate/Dolphins/PolBooks quickly and on PolBlogs/LFR within reasonable compute time.
- If PolBlogs or larger LFR graphs make 30 runs too slow for FLPA/SLPA in practice, N = 20 is an acceptable fallback — keep N **consistent across all algorithms and datasets** so comparisons are fair. State the chosen N and justify it briefly in the report.
- Use a **fixed, documented random seed sequence** (e.g., seeds 0–29) reused identically across algorithms, so each algorithm faces the same set of random initializations — this makes the comparison fairer than fully independent randomness per algorithm.

---

## 7. Stability Metrics

Use **at least two** complementary metrics, since each captures a different aspect of stability:

1. **Self-VI (Variation of Information across runs)** — following the Traag & Šubelj (2023) precedent. For each pair of runs on the same graph, compute VI between the two partitions; average across all pairs (or a random sample of pairs if N is large). **Lower self-VI = more stable.**
2. **Pairwise NMI-across-runs** — same idea using Normalized Mutual Information between pairs of runs instead of VI (**higher = more stable**). Reporting both VI and NMI cross-checks that the stability conclusion isn't an artifact of one metric's quirks.
3. **(Overlapping-specific) Pairwise ONMI or Omega Index for SLPA** — since SLPA produces overlapping communities, standard VI/NMI (built for hard partitions) don't directly apply. Use **Overlapping NMI (ONMI)** or the **Omega Index** for SLPA's pairwise run comparisons, noting this methodological difference clearly in the report.
4. **(Optional) Standard deviation of modularity across the N runs** — easy to compute and explain; a useful first-pass "does this algorithm even give consistent quality" check, though it captures quality-consistency rather than partition-identity.

---

## 8. Community-Quality Metrics

- **Modularity (Q)** — computable for all datasets and all four algorithms (including SLPA, using its overlapping-modularity variant, e.g., extended/EQ modularity), no ground truth needed. Report mean ± std across the N runs.
- **NMI vs. ground truth** — for Karate, PolBooks, PolBlogs (and all LFR graphs). Report mean ± std across N runs. Excluded or caveated for Dolphins.
- **NF1 / F1-score vs. ground truth** — complement to NMI; report **Omega Index** or **F1 for overlapping communities** specifically for SLPA, alongside ONMI.
- **Number of communities detected** — report mean ± std per algorithm per dataset. Large variance here is itself a stability signal.

---

## 9. What Results to Record

For **every single run**, log:
- Dataset name, algorithm name, run index, random seed used
- Full partition (node → community ID, or node → set of community IDs for SLPA)
- Modularity, NMI (if ground truth exists), F1/Omega (if applicable), number of communities detected
- Runtime (seconds)

After all runs, compute **per (dataset, algorithm) aggregates**:
- Mean ± std of each quality metric across the N runs
- Self-VI and pairwise NMI-across-runs (or ONMI/Omega for SLPA)
- Min/max number of communities detected across runs

Keep the raw per-run log (e.g., one CSV row per run) for the statistical tests in §11 and for generating plots in §10.

---

## 10. Visualizations

1. **Box plots or violin plots of modularity (and NMI) per algorithm**, one panel per dataset.
2. **Bar chart of self-VI (or pairwise NMI-across-runs) per algorithm**, grouped by dataset — the primary "stability" figure.
3. **Line plot: stability metric vs. LFR mixing parameter μ**, one line per algorithm — directly visualizes H4.
4. **Heatmap: algorithm × dataset, colored by self-VI or NMI-across-runs** — compact summary view.
5. **Community-count variability plot** (box plot of "number of communities detected" per algorithm per dataset).
6. **(Optional) Example partition visualizations** — Gephi or NetworkX/Matplotlib renderings of 2–3 example runs per algorithm on Karate Club, to qualitatively show instability on a small, interpretable graph.

---

## 11. Statistical Analysis

- **Friedman test** — non-parametric test comparing the 4 algorithms across multiple datasets (each dataset as a "block"); tests whether there's a statistically significant overall difference in ranks among the 4 algorithms on a given metric.
- **Post-hoc Nemenyi test** (or pairwise Wilcoxon signed-rank tests with Bonferroni/Holm correction) — if the Friedman test is significant, identifies which specific pairs of algorithms differ significantly.
- **Paired Wilcoxon signed-rank test** — for targeted comparisons (e.g., "is FLPA significantly more stable than LPA specifically?"), using the N per-run values as paired samples.
- Report **effect sizes**, not just p-values (e.g., rank-biserial correlation for Wilcoxon).
- Set the significance threshold (e.g., α = 0.05) upfront; correct for multiple comparisons when running several pairwise tests.

---

## 12. Final Comparison Tables

**Table A — Quality summary (Modularity)**
*Rows = algorithm, columns = dataset, cell = mean ± std modularity*

| Algorithm | Karate | Dolphins | PolBooks | PolBlogs | LFR μ=0.1 | LFR μ=0.3 | LFR μ=0.5 | LFR μ=0.7 |
|---|---|---|---|---|---|---|---|---|
| LPA | | | | | | | | |
| Semi-sync LPA | | | | | | | | |
| FLPA | | | | | | | | |
| SLPA | | | | | | | | |

**Table B — Quality summary (NMI)**
*Same structure as Table A, excluding/caveating Dolphins*

**Table C — Stability summary**
*Same structure as Table A, cell = self-VI or pairwise NMI-across-runs*

**Table D — Statistical significance summary**

| Algorithm Pair | Metric | Test | p-value | Effect size | Significant at α=0.05? |
|---|---|---|---|---|---|
| LPA vs. Semi-sync LPA | Self-VI | Wilcoxon | | | |
| LPA vs. FLPA | Self-VI | Wilcoxon | | | |
| LPA vs. SLPA | ONMI | Wilcoxon | | | |
| ... | | | | | |

**Table E — Overall verdict**

| Algorithm | Quality | Stability | Best suited for |
|---|---|---|---|
| LPA | | | |
| Semi-sync LPA | | | |
| FLPA | | | |
| SLPA | | | |

---

## Notes Before Implementation

- SLPA's overlapping nature is treated as a **documented methodological difference** throughout (separate stability/quality metrics: ONMI, Omega, EQ) rather than forced into the same pipeline as the three non-overlapping methods — this is a deliberate, defensible choice, not an inconsistency.
- Decide and document early whether LPA/Semi-sync LPA are implemented from scratch or run via NetworkX's built-in versions, and FLPA via `python-igraph` — either is defensible, but the choice and its justification (e.g., "reference implementations used to ensure results reflect algorithmic behavior, not implementation artifacts") should be stated in the report's methodology section.
