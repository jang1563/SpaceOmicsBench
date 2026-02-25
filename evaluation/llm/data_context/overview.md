# SpaceOmicsBench — Mission and Benchmark Overview

## Benchmark Summary
SpaceOmicsBench is a multi-omics AI benchmark for spaceflight biomedical data, featuring 21 ML tasks across 9 modalities from three human spaceflight studies.

## Missions

### Inspiration4 (I4)
- **Type**: SpaceX commercial crew mission, first all-civilian orbital spaceflight
- **Date**: September 15-18, 2021
- **Duration**: ~3 days in low Earth orbit (LEO, ~585 km altitude)
- **Crew**: 4 civilians (C001-C004), no prior spaceflight experience
- **Sample collection**: 7 timepoints — L-92, L-44, L-3 (pre-flight), R+1 (post-flight), R+45, R+82, R+194 (recovery)
- **Modalities collected**: CBC, CMP, cytokines, cfRNA, proteomics (plasma + EVP), metabolomics, spatial transcriptomics (skin), microbiome (10 body sites + ISS environmental)
- **Data source**: NASA OSDR (OSD-569 to OSD-687)

### JAXA Cell-Free Epigenome (CFE) Study
- **Type**: ISS long-duration mission study
- **Duration**: >120 days aboard ISS
- **Subjects**: 6 astronauts
- **Focus**: Cell-free RNA epigenome analysis
- **Data source**: NASA OSDR (OSD-530)

### NASA Twins Study
- **Type**: Year-long ISS mission with identical twin control
- **Date**: March 2015 - March 2016
- **Duration**: 340 days aboard ISS
- **Subjects**: Scott Kelly (space, N=1) vs Mark Kelly (ground control, N=1)
- **Blood cell analysis**: Single-cell RNA-seq of CD4, CD8, CD19, PBMC, LD (lymphocyte-depleted)
- **Contrasts analyzed**: 21 different coefficients (In-flight vs Pre-flight, Post-flight vs Pre-flight, etc.)
- **Total DEG entries**: 115,493 across all cell types and contrasts (19,446 unique genes)

## Benchmark Structure

| Property | Value |
|----------|-------|
| Total tasks | 21 (19 main + 2 supplementary) |
| Categories | A (Clinical), B (cfRNA), C (Proteomics), D (Metabolomics), E (Spatial), F (Microbiome), G (Multi-modal), H (Cross-tissue), I (Cross-mission) |
| Difficulty tiers | Calibration (1), Standard (5), Advanced (9), Frontier (6) |
| Evaluation strategies | LOCO, LOTO, Feature 80/20 (see below) |
| ML baselines | Random, Majority, LogReg, RF, MLP, XGBoost, LightGBM |

### Evaluation Strategy Mapping
- **LOCO** (Leave-One-Crew-Out, 4 folds): A1, A2, C1, F1, F2, F4, F5, G1
- **LOTO** (Leave-One-Timepoint-Out, 7 folds): F3
- **Feature 80/20** (stratified, 5 reps): B1, B2, C2, D1, E1, E2, E3, E4, H1, I1, I2, I3

## Flight Phases
- **pre_flight**: Samples collected before launch (L-92, L-44, L-3)
- **in_flight**: During spaceflight (I4 only ~3 days; Twins ~340 days)
- **post_flight**: Immediately after return (R+1)
- **recovery**: Extended post-flight period (R+45, R+82, R+194)

## Key Challenges
- Extremely small sample sizes: N=4 crew (I4), N=1 treatment (Twins)
- High dimensionality: 2,845 proteins, 26,845 genes, 433 metabolites
- Class imbalance: some tasks have <0.2% positive rate
- Cross-mission comparisons: 3-day vs 340-day missions with different measurement platforms
