# Ground Truth Key Facts

## Mission Facts
- I4: 4 civilian crew, 3 days LEO (~585 km), September 2021, SpaceX Dragon
- NASA Twins: 340 days ISS, Scott Kelly vs Mark Kelly (twin control), 2015-2016
- JAXA CFE: 6 astronauts, >120 days ISS, cell-free RNA epigenome

## Benchmark Statistics
- 21 ML tasks: 19 main + 2 supplementary (E2, E3)
- 9 modality categories, 11 scoring categories (Microbiome split into bodysite/phase/source)
- Tiers: Calibration=1, Standard=5, Advanced=9, Frontier=6
- 7 baseline models: Random, Majority, LogReg, RF, MLP, XGBoost, LightGBM

## Task Metadata
| Task | N | Classes | Features | Metric | Eval | Tier |
|------|---|---------|----------|--------|------|------|
| A1 | 28 | 3 (pre/post/recovery) | 39 (20 CBC + 19 CMP) | macro_f1 | LOCO | Standard |
| A2 | 28 | 3 (pre/post/recovery) | 71 cytokines | macro_f1 | LOCO | Standard |
| B1 | 26845 | 2 (DRR vs non-DRR) | 29 per gene | AUPRC | Feature 80/20 | Advanced |
| B2 | 466 | 16 clusters | 466 (correlation vector) | micro_f1 | Feature 80/20 | Advanced |
| C1 | 21 | 3 (pre/post/recovery) | 2845 → PCA 10 | macro_f1 | LOCO | Standard |
| C2 | 380 | 2 (DE concordance) | DE features | AUROC | Feature 80/20 | Frontier |
| D1 | 433 | 2 (responsive) | chemical properties | AUROC | Feature 80/20 | Advanced |
| E1 | 18677 | 2 (cross-layer DE) | spatial features | AUPRC | Feature 80/20 | Advanced |
| E4 | 18677 | 2 (cross-layer DE) | spatial features | AUPRC | Feature 80/20 | Advanced |
| F1 | 275 | 10 body sites | taxonomy CPM | macro_f1 | LOCO | Standard |
| F2 | 275 | 4 (pre/in/post/recovery) | taxonomy CPM | macro_f1 | LOCO | Frontier |
| F3 | 314 | 2 (human/environmental) | taxonomy CPM | AUROC | LOTO | Calibration |
| F4 | 275 | 10 body sites | pathway CPM | macro_f1 | LOCO | Standard |
| F5 | 275 | 4 (pre/in/post/recovery) | pathway CPM | macro_f1 | LOCO | Frontier |
| G1 | 21 | 3 (pre/post/recovery) | ~56 fused (clinical + PCA 8 prot + PCA 8 met) | macro_f1 | LOCO | Advanced |
| H1 | 731 | 2 (conserved DE) | tissue DE features | AUPRC | Feature 80/20 | Advanced |
| I1 | 26845 | 2 (hemoglobin gene) | 3 fold-change features | AUPRC | Feature 80/20 | Frontier |
| I2 | 452 | 2 (conserved pathway) | 8 aggregated features | AUROC | Feature 80/20 | Advanced |
| I3 | 15540 | 2 (conserved DE gene) | 9 aggregated features | AUPRC | Feature 80/20 | Advanced |

Note: A1/A2 use N=28 (4 crew × 7 timepoints). C1/G1 use N=21 (4 crew × ~5 timepoints, plasma only). F2/F5 use 4-class phase (including in_flight). A1/A2/C1/G1 use 3-class (no in_flight clinical samples). G1 PCA uses 8 components per modality; C1 PCA uses 10 components.

## Baseline Results (Best per Task)
| Task | Tier | Best Model | Score | Random |
|------|------|-----------|-------|--------|
| A1 | Standard | LogReg | 0.546 | 0.214 |
| A2 | Standard | LogReg | 0.493 | 0.214 |
| B1 | Advanced | LightGBM | 0.922 | 0.020 |
| B2 | Advanced | LogReg | 0.154 | 0.083 |
| C1 | Standard | MLP | 0.517 | 0.170 |
| C2 | Frontier | LightGBM | 0.565 | 0.529 |
| D1 | Advanced | RF | 0.676 | 0.481 |
| E1 | Advanced | LogReg | 0.017 | 0.008 |
| E4 | Advanced | LogReg | 0.022 | 0.003 |
| F1 | Standard | LightGBM | 0.200 | 0.112 |
| F2 | Frontier | LightGBM | 0.280 | 0.205 |
| F3 | Calibration | RF | 0.841 | 0.402 |
| F4 | Standard | LogReg | 0.163 | 0.112 |
| F5 | Frontier | LightGBM | 0.304 | 0.205 |
| G1 | Advanced | LogReg | 0.517 | 0.253 |
| H1 | Advanced | LightGBM | 0.284 | 0.060 |
| I1 | Frontier | LightGBM | 0.006 | 0.003 |
| I2 | Advanced | LightGBM | 0.735 | 0.504 |
| I3 | Advanced | MLP | 0.090 | 0.059 |

## Composite Scores (Normalized, 11-category average)
Formula: `normalized = max(0, (score - random) / (1 - random))`, then average within each of 11 categories, then average across categories equally.

- RF: 0.258 (best overall)
- XGBoost: 0.250
- LightGBM: 0.238
- LogReg: 0.201
- MLP: 0.133

## Cross-Mission Facts
- 146/452 pathways conserved between I4 and Twins (32.3%)
- 814/15,540 genes show conserved DE (5.2%)
- 57 hemoglobin/erythropoiesis genes in the dataset
- HBB shows ~40% post-flight expression increase

## Key Limitations
- N=4 crew (I4), N=1 treatment (Twins) — extremely small sample sizes
- Clinical tasks: N=28 (4×7 timepoints), Proteomics: N=21 (4×~5 timepoints)
- 3-day mission may not capture chronic adaptation effects
- LOCO evaluation stringent but low-N means high variance
- Cross-mission comparisons confounded by different platforms, crews, and durations
- LightGBM/XGBoost collapse to majority baseline on LOCO tasks with N≤28
