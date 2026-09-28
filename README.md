# Community Stability Analysis

## A Comparative Stability Analysis of Label Propagation Algorithms for Community Detection in Social Networks

> **Research project:** Empirical comparison of the run-to-run stability and community quality of four Label Propagation Algorithm (LPA) variants on real-world and synthetic social-network benchmarks.

---

## Abstract

Community detection algorithms are often evaluated by the quality of the communities they produce in a single execution. However, many community-detection methods are stochastic: running the same algorithm multiple times on the same network can produce different community structures.

This project studies that **run-to-run stability** for four label-propagation-based community detection algorithms:

- **LPA** — Label Propagation Algorithm
- **Semi-synchronous LPA**
- **FLPA** — Fast Label Propagation Algorithm
- **SLPA** — Speaker-Listener Label Propagation Algorithm

The study repeatedly runs each algorithm on the same network, compares the resulting community structures pairwise, and relates stability to community quality and network structure.

The project is **not proposing a new community-detection algorithm**. It is an empirical and reproducible comparison of existing methods. The original project README explicitly framed the work as reporting the algorithms' observed behavior rather than introducing a new method. fileciteturn17file0L3-L4

---

# Research Question

> **To what extent do algorithmic modifications to the Label Propagation Algorithm — semi-synchronous updating, queue-based fast propagation, and speaker-listener multi-labeling — affect the run-to-run stability of detected community structures in social networks, and does improved stability come at a cost to community detection quality?**

### Objectives

1. Compare LPA, Semi-sync LPA, FLPA, and SLPA on common benchmark networks.
2. Quantify run-to-run stability of detected community structures.
3. Quantify community detection quality.
4. Examine the relationship between stability and quality.
5. Study how network structure affects stability.
6. Provide a reproducible experimental pipeline with stored raw runs, partitions, metrics, figures, and tables.

---

# Why Stability?

For a stochastic algorithm, a single run does not tell the whole story.

Consider the same graph:

```text
             Same graph
                 |
       +---------+---------+
       |         |         |
     seed 0    seed 1    seed 2
       |         |         |
       v         v         v
   partition  partition  partition
       \         |         /
        \        |        /
         +-------+-------+
                 |
          Compare outputs
                 |
                 v
             Stability
```

If repeated runs produce very similar community structures, the algorithm is more reproducible on that network.

If the detected structures vary substantially across seeds, the algorithm exhibits greater run-to-run variability.

**Stability is therefore treated as a property separate from community quality.**

---

# Algorithms

| Algorithm | Output type | Main characteristic | Reference |
|---|---|---|---|
| **LPA** | Hard partition | Asynchronous label propagation | Raghavan, Albert & Kumara (2007) |
| **Semi-sync LPA** | Hard partition | Randomized initial labels + randomized greedy coloring + Prec-Max updates | Cordasco & Gargano (2010) |
| **FLPA** | Hard partition | Fast queue-based label propagation | Traag & Šubelj (2023) |
| **SLPA** | Overlapping cover | Speaker-listener propagation with label memory | Xie, Szymanski & Liu (2011) |

### Implementation decisions

The project deliberately distinguishes the algorithm implementations rather than treating every LPA variant as the same procedure.

- **LPA:** randomized sweep order and random tie-breaking, with the current label retained when it is already among the maximally frequent labels.
- **Semi-sync LPA:** paper-faithful randomized initial labels, randomized greedy coloring, and Prec-Max tie-breaking. It is **not** substituted with NetworkX's deterministic label-propagation implementation.
- **FLPA:** implemented as a wrapper around NetworkX's `fast_label_propagation_communities(seed=...)`.
- **SLPA:** overlapping, memory-based propagation with primary parameters `T=100` and `r=0.1`. Seed-controlled listener order is used.

The original methodology records these implementation decisions explicitly. fileciteturn17file0L55-L66

---

# Datasets

The experiment uses **4 real-world networks** and **70 synthetic LFR networks**.

## Real-world networks

| Dataset | Nodes | Edges | Preprocessing | Ground truth |
|---|---:|---:|---|---|
| **Karate** | 34 | 78 | None | Zachary's two-faction ground truth |
| **Dolphins** | 62 | 159 | None | Not used |
| **PolBooks** | 105 | 441 | None | Not currently verified |
| **PolBlogs** | 1,222 | 16,714 | Symmetrize, remove self-loops/weights, keep LCC | Not currently verified |

The Karate ground truth was verified against the NetworkX `club` labels and edge set. Dolphins has no agreed ground truth in the project. PolBooks and PolBlogs ground-truth evaluation is only used where label data can be obtained and alignment verified. fileciteturn17file0L31-L45

### PolBlogs preprocessing

The raw PolBlogs network is directed and weighted. The preprocessing pipeline:

1. Symmetrizes the graph.
2. Removes self-loops.
3. Removes edge weights.
4. Extracts the largest connected component.
5. Preserves an original-node-ID mapping.

The resulting analysis graph contains:

```text
1,222 nodes
16,714 edges
```

The preprocessing details and original-ID mapping are retained in the project outputs. fileciteturn17file0L40-L42

---

# LFR Synthetic Benchmark

The project contains **70 LFR graphs**:

```text
7 nominal μ levels
×
10 graph instances per level
=
70 graphs
```

### Parameters

| Parameter | Value |
|---|---:|
| Nodes | 1,000 |
| τ1 | 3 |
| τ2 | 1.5 |
| Average degree | 10 |
| Maximum degree | 50 |
| Community size | 20–100 |
| Nominal μ | 0.1–0.7 |
| Instances per μ | 10 |

Each graph has its own deterministic generation seed.

### Nominal vs empirical μ

The requested LFR mixing parameter and the actually observed mixing parameter are both retained.

**Empirical μ is used for the main structural analysis**, because the generated graph may differ slightly from the nominal parameter.

The project records empirical μ in:

```text
data/lfr/*/meta.json
results/processed/lfr_instances.csv
```

The project also documents that its NetworKit generator is a reimplementation rather than the original Lancichinetti binary. fileciteturn17file0L47-L53

---

# Experimental Design

Every graph–algorithm combination is executed **30 independent times** using seeds `0–29`.

### Real networks

```text
4 graphs × 4 algorithms × 30 runs
= 480 runs
```

### LFR networks

```text
70 graphs × 4 algorithms × 30 runs
= 8,400 runs
```

### Complete experiment

```text
480 + 8,400
= 8,880 raw runs
```

For each graph–algorithm group, all unique pairs of the 30 runs are compared:

\[
\binom{30}{2}=435
\]

There are:

```text
74 graphs × 4 algorithms
= 296 graph–algorithm groups
```

Therefore:

```text
296 × 435
= 128,760 pairwise comparisons
```

---

# Metrics

## Stability metrics

### Variation of Information — VI

Measures the information-theoretic difference between two partitions.

```text
Lower VI → greater agreement
VI = 0   → identical hard partitions
```

### Normalized Variation of Information — NVI

Normalized form of VI.

```text
Lower NVI → greater agreement
```

### Normalized Mutual Information — NMI

Measures agreement between community assignments.

```text
Higher NMI → greater agreement
NMI = 1    → identical hard partitions
```

### Omega

Used as a common stability metric across the four algorithm outputs.

```text
Higher Omega → greater agreement
Omega = 1    → identical covers/partitions
```

### Overlapping NMI — ONMI

Used for overlapping community structures produced by SLPA.

The project does **not** force hard-partition metrics onto overlapping covers. The metric selection is representation-aware. fileciteturn17file0L81-L104

### Metric applicability

| Algorithm | VI | NVI | NMI | Omega | ONMI |
|---|---:|---:|---:|---:|---:|
| LPA | ✓ | ✓ | ✓ | ✓ | — |
| Semi-sync LPA | ✓ | ✓ | ✓ | ✓ | — |
| FLPA | ✓ | ✓ | ✓ | ✓ | — |
| SLPA | — | — | — | ✓ | ✓ |

---

# Community Quality

Stability does **not** imply quality.

The project therefore evaluates the detected communities separately using:

- **Modularity (Q)** for hard partitions.
- **Extended Modularity (EQ)** for overlapping communities.
- **Ground-truth NMI/ONMI** where verified ground truth is available.

This creates two separate questions:

```text
STABILITY
"Does the algorithm repeatedly produce
similar community structures?"

QUALITY
"How good / meaningful are those
community structures according to
the selected quality criterion?"
```

The project then studies their relationship rather than treating one as a substitute for the other. fileciteturn17file0L108-L129

---

# Experimental Pipeline

```text
                    ┌──────────────────┐
                    │      DATASETS    │
                    │ Real + LFR       │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │ Load + Validate  │
                    │ + Preprocess     │
                    └────────┬─────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │   Community Detection        │
              │                              │
              │ LPA | Semi-sync | FLPA | SLPA│
              └──────────────┬───────────────┘
                             │
                    30 runs / group
                             │
                             ▼
                  ┌────────────────────┐
                  │ Raw partitions +   │
                  │ per-run metadata   │
                  └─────────┬──────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
    ┌─────────────────┐          ┌─────────────────┐
    │ Stability       │          │ Community       │
    │ VI/NVI/NMI/     │          │ Quality        │
    │ Omega/ONMI      │          │ Q/EQ/GT        │
    └────────┬────────┘          └────────┬────────┘
             │                            │
             └─────────────┬──────────────┘
                           ▼
                 ┌────────────────────┐
                 │ Stability × Quality│
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ LFR μ Analysis     │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Statistical Tests  │
                 └─────────┬──────────┘
                           │
                           ▼
                 ┌────────────────────┐
                 │ Figures + Tables   │
                 └────────────────────┘
```

---

# Reproducible Experiment Pipeline

The implementation is organized as a sequence of stages.

| Stage | Component | Status |
|---|---|---|
| 1 | Repository skeleton and configuration | ✅ |
| 2 | Dataset layer, preprocessing, LFR generation/cache | ✅ |
| 3 | Metrics and validation | ✅ |
| 4 | Algorithm implementations and validation | ✅ |
| 5 | Experiment runner and storage | ✅ |
| 6 | Pairwise stability analysis + bootstrap CIs | ✅ |
| 7 | Quality × stability analysis | ✅ |
| 8 | LFR μ analysis | ✅ |
| 9 | Statistical analysis | ✅ |
| 10 | Figures and tables | ✅ |
| 11 | Final experiment assembly and validation | ✅ |

The original Claude README described the early development stages as a build-status table; this version updates that structure to reflect the **completed final project** rather than leaving stages 8–11 marked as pending. fileciteturn17file0L6-L18

---

# Stage 5 — Experiment Runner

The runner executes every:

```text
(graph, algorithm, seed)
```

combination.

For each run it stores:

- the detected partition/cover,
- a JSON run record,
- quality information,
- runtime,
- seed,
- number of communities,
- convergence information,
- degeneracy information.

The storage layer uses atomic writes and can skip already-complete runs, allowing interrupted experiments to be resumed. `runs.csv` is rebuilt deterministically from the stored run records rather than treated as an append-only log. fileciteturn17file0L73-L78

---

# Stage 6 — Stability Analysis

The stability stage operates on **stored experiment results** rather than rerunning the algorithms.

For each `(graph, algorithm)` group:

```text
30 stored runs
       ↓
435 unique run pairs
       ↓
VI / NVI / NMI / Omega / ONMI
       ↓
mean / median / std
       ↓
bootstrap confidence intervals
```

The main outputs are:

```text
results/processed/pairwise.csv
results/processed/stability_summary.csv
```

Bootstrap resampling is performed at the **run level**, rather than treating the 435 pairwise comparisons as independent observations. A deterministic seed stream makes the analysis reproducible. fileciteturn17file0L81-L104

---

# Stage 7 — Quality × Stability

The quality–stability stage joins:

```text
Stage 5
per-run quality
        +
Stage 6
stability summary
```

to produce one graph–algorithm-level record.

Outputs:

```text
results/processed/quality_stability_summary.csv
results/processed/quality_stability_correlations.csv
results/figures/stage7/
```

Spearman correlations are calculated using the graph–algorithm group as the unit of observation rather than individual pairwise comparisons. The analysis is descriptive and does not produce an overall algorithm ranking. fileciteturn17file0L108-L139

---

# Stage 8 — LFR μ Analysis

The LFR analysis examines how stability and quality vary with network mixing.

The primary structural variable is:

```text
empirical μ
```

rather than nominal μ.

This allows the analysis to use the actual generated network structure rather than assuming every generated graph exactly matches its requested parameter.

---

# Stage 9 — Statistical Analysis

For the LFR benchmark, algorithms are evaluated on the same graph instances, enabling paired/repeated-measures analysis.

The statistical pipeline uses:

1. **Friedman test** for overall repeated-measures differences.
2. **Wilcoxon signed-rank tests** for paired post-hoc comparisons when appropriate.
3. **Holm correction** for multiple comparisons.
4. **Rank-biserial effect size** for paired comparisons.

Current stored outputs include:

```text
5 Friedman test rows
15 post-hoc comparison rows
```

ONMI is not included in the same four-algorithm statistical comparison because it is only available for the overlapping SLPA output in the current design.

---

# Stage 10 — Figures and Tables

The final artifact contains **8 final figures**, **4 Stage 7 figures**, and **7 final tables**.

### Final figures

```text
fig01_stability_comparison.png
fig02_quality_comparison.png
fig03_stability_quality_relationship.png
fig04_lfr_mu_stability.png
fig05_lfr_mu_quality.png
fig06_community_structure.png
fig07_real_network_comparison.png
fig08_experiment_completeness.png
```

### Stage 7 figures

```text
community_count_variability_vs_stability.png
quality_vs_stability_by_dataset.png
stability_vs_ground_truth.png
stability_vs_quality.png
```

### Final tables

```text
table01_dataset_characteristics.csv
table02_algorithm_characteristics.csv
table03_stability_statistics.csv
table04_quality_statistics.csv
table05_lfr_mu_analysis.csv
table06_statistical_significance.csv
table07_experiment_completeness.csv
```

---

# Repository Structure

```text
community-stability-analysis/
│
├── README.md
├── main.py
├── config.yaml
├── requirements.txt
├── pytest.ini
│
├── data/
│   ├── raw/
│   └── lfr/
│
├── docs/
│
├── notebooks/
│
├── results/
│   ├── raw/
│   │   ├── runs.csv
│   │   └── partitions/
│   │
│   ├── processed/
│   │   ├── pairwise.csv
│   │   ├── stability_summary.csv
│   │   ├── quality_stability_summary.csv
│   │   ├── quality_stability_correlations.csv
│   │   ├── statistical_tests.csv
│   │   ├── posthoc_tests.csv
│   │   ├── lfr_instances.csv
│   │   └── stability_chunks/
│   │
│   ├── figures/
│   │   ├── final/
│   │   └── stage7/
│   │
│   └── tables/
│       └── final/
│
├── src/
│   ├── algorithms/
│   ├── data/
│   ├── experiments/
│   ├── metrics/
│   └── ...
│
├── tests/
│
└── tools/
```

---

# Quick Start

## 1. Create a virtual environment

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Check the CLI

```bash
python main.py --help
```

## 4. Run a small example

For example, run one LPA seed on Karate:

```bash
python main.py run --graphs karate --algorithms lpa --seeds 0
```

This executes one stochastic community-detection run and records its partition, quality, runtime, and metadata.

## 5. Analyze the 30 stored Karate/LPA runs

```bash
python main.py stability --graphs karate --algorithms lpa
```

For the complete stored experiment, the repository already contains the generated raw and processed results, so the full 8,880-run experiment does not need to be rerun simply to inspect the research outputs.

---

# Testing

The final validation run completed with:

```text
189 passed
2 skipped
```

The two skipped tests require the optional `networkit` dependency for fresh LFR regeneration. The stored LFR benchmark graphs remain available in the project.

The tests cover:

- dataset validation,
- graph preprocessing,
- metric correctness,
- algorithm behavior,
- deterministic seed behavior,
- partition/cover validity,
- experiment storage,
- resumability,
- stability calculations,
- quality–stability joins,
- statistical analysis,
- figure/table generation.

---

# Reproducibility and Integrity

The project records seeds, dataset metadata, algorithm identity, graph identity, run metadata, quality metrics, runtime, degeneracy/convergence information, and partition references.

Dataset integrity is additionally supported through SHA-256 manifests and preprocessing metadata. The original project design explicitly uses manifest checksums to detect byte-level dataset changes. fileciteturn17file0L42-L45

The completed experiment contains:

| Artifact | Count |
|---|---:|
| Raw experiment runs | **8,880** |
| Graph–algorithm groups | **296** |
| Pairwise stability comparisons | **128,760** |
| LFR graphs | **70** |
| Final figures | **8** |
| Stage 7 figures | **4** |
| Final tables | **7** |

---

# Interpretation Notes

## Stability is not the same as quality

Repeatedly producing the same communities does not automatically mean those communities are structurally meaningful.

The project therefore keeps stability and quality as separate quantities and studies their relationship.

## High-μ LFR cases

At higher LFR mixing levels, some LPA-family runs can collapse into trivial community structures.

A highly consistent trivial output should therefore not be interpreted as meaningful community-detection success without considering:

- degeneracy,
- quality,
- number of communities,
- and the underlying network structure.

## SLPA is structurally different

SLPA produces overlapping covers, whereas LPA, Semi-sync LPA, and FLPA produce hard partitions.

Consequently:

- hard-partition metrics are not forced onto SLPA,
- overlapping metrics are used where appropriate,
- ONMI is treated separately in analyses where it is applicable.

---

# Limitations

- Not every real-world network has verified ground-truth communities.
- SLPA's overlapping output makes some direct metric comparisons different from those for hard-partition algorithms.
- High LFR mixing can produce degenerate community structures.
- Bootstrap confidence intervals follow the implemented run-level resampling procedure.
- The selected datasets and algorithms do not represent every community-detection method or every network type.
- The project is an empirical comparison and should not be interpreted as establishing a universal ranking of community-detection algorithms.

---

# References

### Core algorithms

**LPA**

Raghavan, U. N., Albert, R., & Kumara, S. (2007). *Near linear time algorithm to detect community structures in large-scale networks*. Physical Review E, 76, 036106.

**Semi-synchronous LPA**

Cordasco, G., & Gargano, L. (2010). *Community detection via semi-synchronous label propagation algorithms*.

**FLPA**

Traag, V. A., & Šubelj, L. (2023). *Fast label propagation for community detection*. Scientific Reports, 13, 2701.

**SLPA**

Xie, J., Szymanski, B. K., & Liu, X. (2011). *SLPA: Uncovering overlapping communities in social networks via a speaker-listener interaction dynamic process*.

### Related work

- Meena, S. S. et al. (2025). *Graph embedding based label propagation for community detection in social networks*. Scientific Reports.
- Wu, X. et al. (2026). *Label acceptance based label propagation algorithm for community detection*. Information Processing & Management.
- Yu, J. et al. (2025). *A framework for overlapping and non-overlapping communities detection based on seed extension and label propagation*. Physica A.
- Teng, M. et al. (2026). *Multi-scale graph contrastive learning for community detection in dynamic graphs*. Information Processing & Management.
- Paoletti, G. et al. (2025). *CoDÆN: Benchmarks and Comparison of Evolutionary Community Detection Algorithms for Dynamic Networks*. ACM Transactions on the Web.

---

# Project Status

## ✅ Completed

The final project contains:

- complete algorithm implementations,
- real and synthetic benchmark networks,
- 8,880 stored experimental runs,
- 128,760 pairwise stability comparisons,
- stability summaries and bootstrap confidence intervals,
- quality–stability analysis,
- LFR μ analysis,
- statistical analysis,
- final figures and tables,
- automated tests,
- reproducibility metadata.

**This repository is intended to document the complete experimental study, not just the source code.**
