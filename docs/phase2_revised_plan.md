# Phase 2 Task Design — SpaceOmicsBench v2 (Final)

**Date:** 2025-02-25 (verified)
**Status:** All tasks verified against actual data

---

## 17 Tasks in 7 Categories

### Category A: Clinical Biomarkers (2 tasks)

**A1: Phase Classification from Blood Panel**
- Input: clinical_cbc (20 features) + clinical_cmp (19 features) = 39 features
- Target: 3 phases (pre_flight=12, post_flight=4, recovery=12)
- N: 28 samples
- Split: LOCO (4 folds)
- Metrics: Macro F1, 95% CI
- Caveat: N=28, post_flight has only 4 samples

**A2: Phase Classification from Immune Markers**
- Input: clinical_cytokines_eve (71 features)
- Target: 3 phases (same as A1)
- N: 28 samples (same sample set as A1, different features)
- Split: LOCO (4 folds)
- Value: Tests immune panel vs standard blood panel

---

### Category B: cfRNA Transcriptomics (2 tasks)

**B1: Spaceflight-Responsive Gene Ranking**
- Input: cfrna_3group_de statistics (38 features per gene: ANOVA, edgeR pairwise FC/pval, group means)
- Target: DRR status (binary: 466 positive / 26,379 negative)
- N: 26,845 genes
- Split: 80/20 stratified, 5 reps (seed=42)
- Metrics: AUROC, AUPRC, NDCG@100
- Note: 1.7% positive rate → AUPRC is primary

**B2: Coregulated Gene Cluster Prediction**
- Input: gt_cfrna_correlation.csv (466×466 gene-gene correlation matrix)
- Target: gt_cfrna_cluster_labels.csv (466 genes × 16 binary cluster labels)
- N: 466 genes (226 with ≥1 cluster, 240 unassigned)
- Split: 80/20, 5 reps
- Metrics: Micro F1, Hamming loss, per-cluster AUROC
- Data fix applied: "Y" → 1, NaN → 0

---

### Category C: Proteomics (2 tasks)

**C1: Protein DE from Expression Profile**
- Input: proteomics_plasma_matrix expression values (21 sample measurements per protein)
- Target: DE status from proteomics_plasma_de_clean (adj_pval < 0.05)
- N: 1,686 proteins (57 DE = 3.4%)
- Split: 80/20 stratified, 5 reps
- Metrics: AUROC, AUPRC
- Design: Tests whether ML can learn what limma detects from expression profiles

**C2: Cross-Biofluid Protein Concordance**
- Input: plasma DE statistics (logFC, AveExpr, t, B from proteomics_plasma_de_clean)
- Target: EVP DE status for overlapping proteins (adj_pval < 0.05 in proteomics_evp_de_clean)
- N: 380 overlapping proteins (124 EVP-DE, 256 not)
- Split: 80/20 stratified, 5 reps
- Metrics: AUROC, AUPRC, direction concordance
- Baseline: 64.2% direction concordance, 67.4% DE concordance
- Value: Biologically meaningful — does plasma response predict exosome response?

---

### Category D: Metabolomics (1 task)

**D1: Metabolite Super-Pathway Classification**
- Input: metabolomics_anppos_matrix (24 sample expression values) + chemical annotations (Mass, RT, Formula) + DE statistics from metabolomics_de (4 contrasts)
- Target: SuperPathway (multi-class)
- N: 451 metabolites with labels
- Classes: 12 (after merging 6 rare classes <10 into "Other": 17 metabolites merged)
  - Lipid (232), Amino acid (106), Nucleotide (24), Cofactors (17), Other (17), Carbohydrate (14), Energy (14), Peptide (12), Xenobiotics (10), ...
- Split: Stratified 80/20, 5 reps
- Metrics: Macro F1

---

### Category E: Spatial Cross-Layer DE Prediction (4 tasks)

**Design: From global skin DE → predict layer-specific DE**
- Input: gt_spatial_de_all_skin.csv features (baseMean, log2FoldChange, lfcSE) for 18,677 genes
- Target: adj_pval < 0.05 in each specific layer
- Note: all_skin ≡ vasculature (r=1.000) → vasculature task removed (redundant)
- Cross-layer log2FC correlation: r=0.06~0.14 (non-trivial prediction)

| ID | Target Layer | DE Genes | Positive Rate |
|----|-------------|----------|---------------|
| **E1** | outer_epidermis | 35 | 0.19% |
| **E2** | inner_epidermis | 11 | 0.06% |
| **E3** | outer_dermis | 18 | 0.10% |
| **E4** | epidermis (OE+IE) | 40 | 0.21% |

- N: 18,677 genes per task
- Split: 80/20 stratified, 5 reps
- Metrics: AUROC, AUPRC (primary: AUPRC due to extreme imbalance)
- Input features: baseMean, log2FoldChange, lfcSE from all_skin (NOT stat/pval to avoid leakage)

---

### Category F: Microbiome (5 tasks — strongest ML tasks)

**F1: Body Site Classification (Taxonomy)**
- Input: microbiome_human_taxonomy_cpm (16,172 taxa per sample)
- Target: 10 body sites (ARM/EAR/GLU/NAC/NAP/ORC/PIT/TZO/UMB/WEB)
- N: 275 samples (balanced: 26-28 per class)
- Split: LOCO (4 folds)
- Metrics: Macro F1, per-class F1

**F2: Flight Phase Detection (Taxonomy)**
- Input: microbiome_human_taxonomy_cpm
- Target: 4 phases (pre_flight=78, in_flight=78, post_flight=40, recovery=79)
- N: 275 samples
- Split: LOCO (4 folds)
- Metrics: Macro F1

**F3: Source Classification (Human vs Environmental)**
- Input: microbiome taxonomy (taxid-aligned intersection: ~5,830 shared taxa)
- Target: binary (human=275, environmental=39)
- N: 314 samples
- Split: Leave-One-Timepoint-Out
- Metrics: AUROC, F1

**F4: Body Site Classification (Pathways)**
- Input: microbiome_human_pathways_cpm (567 pathways per sample)
- Target: 10 body sites
- N: 275 samples (drop 2 extra pathway-only samples)
- Split: LOCO (4 folds)
- Metrics: Macro F1
- Value: Taxonomy (F1) vs functional pathways (F4) comparison

**F5: Flight Phase Detection (Pathways)**
- Input: microbiome_human_pathways_cpm
- Target: 4 phases
- N: 275 samples
- Split: LOCO (4 folds)
- Metrics: Macro F1
- Value: Taxonomy (F2) vs pathways (F5) comparison

---

### Category G: Cross-Study & Multi-Modal (1 task)

**G1: Multi-Modal Phase Classification**
- Input: clinical_cbc + clinical_cmp (39 features) + proteomics_plasma_matrix (2,838 proteins) + metabolomics_anppos_matrix (454 metabolites)
- Target: 3 phases
- N: 21 matched samples (via crew + timepoint_days join)
- Split: LOCO (4 folds)
- Metrics: Macro F1, 95% CI
- Caveat: N=21, but uniquely tests multi-omic fusion
- Value: Comparison with A1 (clinical-only) shows added value of omics

---

## Tasks Removed (with reasons)

| Original | Why Removed |
|----------|-----------|
| E5 (vasculature) | ≡ all_skin (r=1.000), completely redundant |
| E6 (all_skin self-prediction) | Trivial: stat → adj_pval is mathematical identity |
| G1 old (Conserved DEG) | 57 columns but extremely sparse (JAXA: 36/806, NASA: 4-8/806) |
| G2 (Pathway conservation) | Pathway sets don't align across studies; all pathways already significant |

---

## Summary Table

| ID | Task | Type | N | Difficulty |
|----|------|------|---|-----------|
| A1 | Blood panel phase | sample-clf (3-class) | 28 | Medium |
| A2 | Immune marker phase | sample-clf (3-class) | 28 | Medium |
| B1 | cfRNA DEG ranking | feature-binary | 26,845 | Hard |
| B2 | Gene cluster prediction | feature-multilabel | 466 | Hard |
| C1 | Expression → protein DE | feature-binary | 1,686 | Medium |
| C2 | Cross-biofluid concordance | feature-binary | 380 | Hard |
| D1 | Metabolite pathway | feature-multiclass | 451 | Hard |
| E1 | Cross-layer: outer epidermis | feature-binary | 18,677 | Hard |
| E2 | Cross-layer: inner epidermis | feature-binary | 18,677 | Expert |
| E3 | Cross-layer: outer dermis | feature-binary | 18,677 | Expert |
| E4 | Cross-layer: epidermis | feature-binary | 18,677 | Hard |
| F1 | Body site (taxonomy) | sample-clf (10-class) | 275 | Easy |
| F2 | Phase (taxonomy) | sample-clf (4-class) | 275 | Hard |
| F3 | Human vs env | sample-clf (binary) | 314 | Easy |
| F4 | Body site (pathways) | sample-clf (10-class) | 275 | Medium |
| F5 | Phase (pathways) | sample-clf (4-class) | 275 | Hard |
| G1 | Multi-modal phase | sample-clf (3-class) | 21 | Hard |

**Total: 17 ML tasks** across 7 omics modalities.

---

## Evaluation Framework

| Task Type | Primary | Secondary |
|-----------|---------|-----------|
| Sample-level classification | Macro F1 | Accuracy, per-class P/R, 95% CI |
| Feature-level binary (balanced) | AUROC | AUPRC, F1 |
| Feature-level binary (imbalanced) | AUPRC | AUROC, NDCG@100, precision@k |
| Feature-level multi-class | Macro F1 | Per-class F1 |
| Feature-level multi-label | Micro F1 | Hamming loss, per-cluster AUROC |

All results with 95% CI via bootstrap (1,000 iterations).

---

## Split Strategies

| Strategy | Tasks | Folds |
|----------|-------|-------|
| LOCO (Leave-One-Crew-Out) | A1, A2, F1, F2, F4, F5, G1 | 4 |
| LOTO (Leave-One-Timepoint-Out) | F3 | 4-7 |
| Feature-level stratified 80/20 | B1, B2, C1, C2, D1, E1-E4 | 5 reps |

Seed: 42 (deterministic).
