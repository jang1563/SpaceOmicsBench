# Phase 1 Preprocessing Review Report

**Date:** 2025-02-25 (updated 2025-02-25)
**Reviewer:** Automated audit + cross-check against raw source data

---

## Executive Summary

All 7 preprocessing scripts + 1 post-processing script ran successfully, producing **59 output files** in `data/processed/`. Cross-checks confirm processed values match raw source data faithfully.

**Resolved issues (from initial review):**
- C003_L-92 CBC column-shift: **CORRECTED** using Quest Diagnostics original lab reports
- Microbiome H20/OAC water controls: **EXCLUDED** (284 -> 275 human samples)
- Log-transform inconsistency: **RESOLVED** with `_log10` versions for EVP + metabolomics
- EVP zero convention: **RESOLVED** with `_nan` version (zeros -> NaN)
- Proteomics DE contaminants: **RESOLVED** with `_clean` versions (CON_/PEPCAL/isoforms removed)
- Microbiome pathway files broken: **FIXED** (`_Abundance-CPM` suffix stripped, gut 8 samples + env 39 samples restored)

**Remaining known issues:** 3 missing plasma proteomics samples (never collected), plasma 28.4% MNAR missingness, systematic R+45 hypoglycemia (verified real), sample_id format inconsistency across modalities (use crew + timepoint_days for cross-modal joins).

---

## Critical Issues

### 1. C003_L-92 CBC data corrupted in source -- RESOLVED

**File:** `clinical_cbc.csv` (row C003_L-92)
**Source:** `LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv`
**Root cause:** MCH row was dropped during GeneLab's CSV merging step, causing all subsequent values to shift down by one row.
**Verified against:** Original Quest Diagnostics lab reports at `CBC_Quest/2022_05_10_Eliah/Processed_Tables/C003_2021-06-16_*.csv`

| Analyte | GeneLab (corrupted) | Quest Original (correct) |
|---------|-------------------|------------------------|
| mch | (blank) | 29.6 |
| mchc | 29.6 | 32.4 |
| rdw | 32.4 | 12.9 |
| platelet_count | 12.9 | 330.0 |
| mpv | 330.0 | 10.1 |
| absolute_neutrophils | 10.1 | 6826.0 |
| absolute_lymphocytes | 6826 | 1997.0 |
| absolute_monocytes | 1997 | 614.0 |
| absolute_eosinophils | 614 | 125.0 |
| absolute_basophils | 125 | 38.0 |

**Status:** CORRECTED in `preprocess_clinical.py`. All corrected values are within normal reference ranges.

### 2. H20 body site in microbiome = water controls

**File:** `microbiome_metadata.csv`, `microbiome_human_taxonomy_cpm.csv`
**Samples:** 6 samples (C002/C003/C004 × R+45/R+82) labeled `source=human`

"H20" is almost certainly "H2O" (water controls), not a body site. Evidence:
- Only appears at recovery timepoints, not during standard collection windows
- Only 3 of 4 crew members
- No human-readable body site name in BODY_SITES mapping

**Action needed:** Exclude H20 from human sample count and analysis. Consider also investigating OAC (3 samples, likely oral anterior commissure).

---

## High-Priority Issues

### 3. Three missing plasma proteomics samples (not one)

**File:** `proteomics_plasma_matrix.csv`

Missing samples confirmed in raw source (never collected):
- C001_Nov_Post (≈R+45)
- C002_Sept_Pre (≈L-3)
- C003_Nov_Post (≈R+45)

The previous documentation mentioned only C002_Sept_Pre. All 3 are genuinely absent from raw data.

### 4. Inconsistent transformation across modalities

| Matrix | Transformation | Scale |
|--------|---------------|-------|
| Plasma proteomics | Log10 | 0.7 - 7.5 |
| EVP proteomics | Raw intensity | 0 - 371B |
| Metabolomics (RPPOS/ANPPOS) | Raw intensity | 396 - 350M |
| Microbiome | CPM | 0 - 905K |
| Clinical CBC/CMP | Raw units | varies |

**Action needed:** Document transformations clearly. Consider whether downstream tasks need normalization harmonization.

### 5. Plasma proteomics: 28.4% missingness, non-random

Per-sample missingness ranges from 5.3% (C003_Dec_Post) to 42.4% (C003_Sept_Pre). Sept_Pre timepoints have systematically higher missingness. This is inherent to the mass-spec data but must be accounted for in task design.

### 6. EVP matrix uses zeros for missing values

EVP proteomics has 55.9% zeros representing undetected proteins (not true absence), whereas plasma proteomics uses NaN for missing values. This inconsistency must be documented.

### 7. Contaminants in proteomics DE results

30 CON_ (contaminant) entries and 4 PEPCAL (calibration) entries are in `proteomics_plasma_de.csv`. 79 DE entries total have no corresponding column in the matrix (1,765 DE vs 2,838 matrix proteins, overlap = 1,686).

---

## Medium/Low Issues

### 8. Duplicate metabolite in DE file

`FA (22:5) [M-H]-` appears twice per contrast in `metabolomics_de.csv` with different logFC values (656 rows per contrast instead of 655).

### 9. Metabolomics filename misleading

`metabolomics_anppos_matrix.csv` contains both ANPPOS (275) and ANPNEG (179) mode metabolites.

### 10. Conserved DEGs have 26 duplicate gene entries

`gt_conserved_degs.csv` has 806 rows but only 780 unique genes. 16 genes appear 2-4 times (e.g., S100A9, IFITM2, DYNLT1, KLRB1). These represent different transcript variants with different cross-study values.

### 11. Cluster ground truth is hybrid matrix

`gt_cfrna_clusters.csv` is NOT a pure 466×466 correlation matrix. It contains 466 gene correlation rows + 16 binary cluster/pathway annotation rows = 482 total rows. Must be separated for different downstream uses.

### 12. cfRNA fold change 29.1% missing

In `cfrna_3group_de.csv`, 7,823 of 26,845 genes have undefined fold change (likely zero expression in at least one group). EdgeR pairwise FCs have no missing values.

### 13. Systematic hypoglycemia at R+45

Three of four crews (C002=44, C003=36, C004=43 mg/dL) show very low glucose at R+45. Values confirmed in raw source. Likely fasting blood draw, but clinically notable.

### 14. Spatial DE filtered by raw p-value, not adjusted

GLDS-566 processed data uses pval < 0.05 filter (2,469 genes), while only 220 are significant by adj_pval < 0.05. The P06 ground truth has all 18,677 genes unfiltered.

### 15. Sporadic clinical NaN patterns

- `cbc_mchc`: 6/28 missing (C002 pre-flight + C003/C004 L-92)
- `cmp_bun_to_creatinine_ratio`: 21/28 missing (raw source "NA")
- `cbc_mcv`: C002_R+194 missing
- Metadata `sample_id` not unique (shared across tissues by design)

---

## Verified Correct

- All 28 clinical samples (4 crew × 7 timepoints) present in CBC/CMP/merged
- Clinical values cross-checked against raw source: exact match
- Plasma proteomics technical replicate averaging verified: exact match
- All 466 DRR genes have ANOVA FDR < 0.05
- P06 full genome (18,677 genes) is strict superset of GLDS-566 filtered (2,469 genes)
- Overlapping spatial DE values are IDENTICAL between GLDS-566 and P06
- Hemoglobin DE gene set (26,845) perfectly matches cfRNA 3-group DE gene set
- Microbiome CPM values all non-negative
- All phase assignments correct (pre_flight/post_flight/recovery)
- All timepoint_days mappings correct

---

## Sample Count Summary

| Modality | Expected | Actual | Notes |
|----------|----------|--------|-------|
| Clinical CBC | 28 | 28 | 4 crew × 7 timepoints |
| Clinical CMP | 28 | 28 | 4 crew × 7 timepoints |
| Clinical Eve Immune | 28 | 28 | |
| Clinical Alamar Serum | 27 | 27 | C003_L-44 missing |
| Clinical Alamar Urine | 22 | 22 | No R+194, C001 missing L-3/R+1 |
| cfRNA | N/A | N/A | Group-level summaries only |
| Plasma Proteomics | 24 | 21 | 3 samples never collected |
| EVP Proteomics | 24 | 24 | Complete |
| Metabolomics | 24 | 24 | Complete |
| Spatial DE | 8 | 8 | 4 crew × 2 timepoints |
| Microbiome Human | ~275 | 284 | Includes 6 H20 + 3 OAC |
| Microbiome Gut | 8 | 8 | C002/C004 only |
| Microbiome Env | 39 | 39 | Dragon capsule |

---

## Output File Inventory (53 files)

### Clinical (9 files)
- clinical_cbc.csv (28 × 27)
- clinical_cmp.csv (28 × 26)
- clinical_cytokines_eve.csv (28 × 78)
- clinical_cardiovascular_eve.csv (28 × 16)
- clinical_cytokines_alamar_serum.csv (27 × 210)
- clinical_cytokines_alamar_urine.csv (22 × 209)
- clinical_merged_serum.csv (28 × 126)
- clinical_metadata.csv (78 rows)

### cfRNA (5 files)
- cfrna_3group_de.csv (26,845 × 39)
- cfrna_466drr.csv (466 × 27)
- cfrna_9group_pairwise.csv (26,845 × 42)
- cfrna_11group_timecourse.csv (26,845 × 23)
- cfrna_cd36_enrichment.csv (22,475 × 9)

### Proteomics (5 files)
- proteomics_plasma_de.csv (1,765 × 7)
- proteomics_evp_de.csv (527 × 7)
- proteomics_plasma_matrix.csv (21 × 2,846)
- proteomics_evp_matrix.csv (24 × 1,451)
- proteomics_metadata.csv (45 rows)

### Metabolomics (4 files)
- metabolomics_de.csv (2,624 × 9)
- metabolomics_rppos_matrix.csv (24 × ~234)
- metabolomics_anppos_matrix.csv (24 × ~465)
- metabolomics_metadata.csv (24 rows)

### Spatial Transcriptomics (9 files)
- spatial_de_{layer}.csv × 6 (2,469 genes each)
- spatial_skin_taxonomy_cpm.csv (1,237 × 16)
- spatial_skin_pathways_cpm.csv (340 × 9)
- spatial_metadata.csv (8 rows)

### Microbiome (8 files)
- microbiome_human_taxonomy_cpm.csv (16,172 × 292)
- microbiome_human_pathways_cpm.csv (567 × ~)
- microbiome_human_metaphlan.csv (2,723 × ~)
- microbiome_gut_taxonomy_cpm.csv (4,094 × 16)
- microbiome_gut_pathways_cpm.csv (363 × ~)
- microbiome_env_taxonomy_cpm.csv (6,852 × 47)
- microbiome_env_pathways_cpm.csv (494 × ~)
- microbiome_metadata.csv (331 rows)

### Ground Truth (13 files)
- gt_spatial_de_{layer}.csv × 6 (18,677 genes each)
- gt_hemoglobin_de.csv (26,845 × 7)
- gt_hemoglobin_globin_genes.csv (59 genes)
- gt_hemoglobin_i4_expression.csv (58 × 5)
- gt_hemoglobin_crossmission.csv (1,950 rows)
- gt_cfrna_clusters.csv (482 × 467)
- gt_cfrna_tissue_enrichment.csv (34 × 8)
- gt_conserved_degs.csv (806 × 57)
- gt_conserved_pathways_{study}.csv × 5
