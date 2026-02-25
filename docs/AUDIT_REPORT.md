# Audit Report: SpaceOmicsBench v2 (Public)

Thorough verification of all claims, data sources, and scripts.
Conducted: 2026-02-24

---

## 1. OSD Dataset IDs -- ALL VERIFIED

Every OSD ID was verified against the live OSDR Biodata API.

| OSD ID | Verified Title | Files | Status |
|--------|---------------|-------|--------|
| OSD-569 | Whole Blood Measurements from the SpaceX Inspiration4 Mission (RNA-seq, epitranscriptome, WGS, clonal hematopoiesis, CBC) | 11 processed | CONFIRMED |
| OSD-570 | SpaceX Inspiration4 PBMC Profiling: snRNA-seq, snATAC-seq, and T-cell and B-cell repertoire profiling | 12 | CONFIRMED |
| OSD-571 | SpaceX Inspiration4 Plasma Profiling: Proteomics, EVPs, Metabolomics, cfDNA Sequencing and cfRNA Seq | 14 | CONFIRMED |
| OSD-572 | SpaceX Inspiration4 Oral, Nasal, and Skin Metagenomic and Metatranscriptomic Microbial Swabs | 5,404 | CONFIRMED |
| OSD-573 | SpaceX Inspiration4 Dragon Capsule Metagenomic and Metatranscriptomic Microbial Swabs | 659 | CONFIRMED |
| OSD-574 | SpaceX Inspiration4 Deltoid Skin and Microbiome Data: Spatial Transcriptomics (NanoString GeoMx)... | 164 | CONFIRMED |
| OSD-575 | SpaceX Inspiration4 Blood Serum Metabolic Panel and Immune/Cardiac Cytokine Arrays | 9 | CONFIRMED |
| OSD-630 | SpaceX Inspiration4 Stool Metagenome Profiling | 150 | CONFIRMED |
| OSD-656 | SpaceX Inspiration4 Urine Inflammation Panel (Multiplex, NULISAseq) | 4 | CONFIRMED |
| OSD-687 | SpaceX Inspiration4 T-Cell Histone Modification Profiling (H3K4me1, H3K4me3, H3K27ac) | 87 | CONFIRMED |
| OSD-530 | Cell-free RNA analysis of plasma samples collected from six astronauts in JAXA Cell-Free Epigenome (CFE) Study | 8 | CONFIRMED |
| OSD-532 | Space flight mouse plasma cell-free RNA sequencing in Mouse Habitat Unit-1 (MHU-1) | (checked via GEO) | CONFIRMED |

### Key Accession ID Mappings (IMPORTANT)

The GLDS (GeneLab Data System) IDs do NOT match OSD IDs. Actual file names use GLDS IDs:

| OSD ID | GLDS ID | LSDS ID |
|--------|---------|---------|
| OSD-569 | **GLDS-561** | LSDS-7 |
| OSD-570 | **GLDS-562** | - |
| OSD-571 | **GLDS-563** | - |
| OSD-530 | **GLDS-530** | - |

This means file names start with `GLDS-561_`, `GLDS-562_`, `GLDS-563_`, not `OSD-569_` etc.

---

## 2. Paper DOIs -- ALL VERIFIED

| Paper | DOI | HTTP Status |
|-------|-----|-------------|
| P1 SOMA | 10.1038/s41586-024-07639-y | 200 OK |
| P2 Secretome | 10.1038/s41467-024-48841-w | 200 OK |
| P3 Single-cell | 10.1038/s41467-024-49211-2 | 200 OK |
| P4 Epigenomics | 10.1038/s41467-024-48806-z | 200 OK |
| P5 Microbiome | 10.1038/s41564-024-01635-8 | 200 OK |
| P6 Spatial | 10.1038/s41467-024-48625-2 | 200 OK |
| P7 Hemoglobin | 10.1038/s41467-024-49289-8 | 200 OK |
| P8 JAXA CFE | 10.1038/s41467-023-41995-z | 200 OK |
| P9 TERRA | 10.1038/s42003-024-06014-x | 200 OK |
| P10 Twins | 10.1126/science.aau8650 | 403 (paywall, but DOI is valid) |

### Note on P10 (Twins Study)
Science.org returns 403 for direct curl (requires browser/cookies). The DOI is valid --
it redirects correctly to the Science article page. Supplementary data may require
browser download.

---

## 3. GEO Dataset -- VERIFIED

- **GSE213808**: CONFIRMED as public.
  - Title: "Space flight mouse plasma cell-free RNA sequencing in Mouse Habitat Unit-1 (MHU-1)"
  - Status: Public on Nov 16, 2022
  - Last updated: Jun 28, 2024
  - PMID: 38862469

---

## 4. API Endpoints -- ISSUES FOUND

### 4.1 Biodata Query API (`/v2/query/data/`) -- DOES NOT WORK FOR THESE DATASETS

**Problem**: The `file.data_type=normalized_counts` parameter returns errors for our datasets:
- OSD-530: `422 UNPROCESSABLE_ENTITY` -- "No data found matching constraints"
- OSD-569: `400 BAD_REQUEST` -- "requested data type is not available for selected studies"

**Root cause**: These datasets were NOT processed through GeneLab's standard bulk RNA-seq pipeline.
- OSD-530 (JAXA): Processed with CLC Genomics Workbench (custom pipeline)
- OSD-569 (I4): Mixed assay types (CBC, long-read RNA, WGS) -- not standard bulk RNA-seq

**Impact**: The fetch script's Method 1 (Query API for normalized_counts) will FAIL for all primary datasets.

### 4.2 OSDR File Listing API (`/osdr/data/osd/files/`) -- INCONSISTENT

**Problem**: Returns 0 files for OSD-569, OSD-571, but works for OSD-530.
- OSD-530: Returns 8 files correctly
- OSD-569: Returns 0 files (but has 11 via biodata API)
- OSD-571: Returns 0 files (but has 14 via biodata API)

**Root cause**: Likely pagination or caching issue with the OSDR API for larger/newer datasets.

### 4.3 Biodata REST File Listing (`/v2/dataset/{ID}/files/`) -- WORKS RELIABLY

This is the most reliable method. Returns correct file counts and names for all datasets.

### 4.4 GEODE Download (`/geode-py/ws/studies/`) -- WORKS

- Returns HTTP 302 redirect to S3 presigned URL
- Must follow redirects (`-L` flag)
- Downloaded OSD-530 file successfully (10.7 MB)

---

## 5. Actual Processed Files Available (Verified via API)

### OSD-569 (I4 Whole Blood) -- 11 files
```
LSDS-7_Complete_Blood_Count_CBC.upload_SUBMITTED.csv       <-- CBC raw
LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv            <-- CBC processed (standardized)
GLDS-561_long-readRNAseq_Direct_RNA_seq_Gene_Expression_Processed.xlsx  <-- ONT RNA-seq DE
GLDS-561_directm6Aseq_Direct_RNA_seq_m6A_Processed_Data.xlsx           <-- m6A methylation
GLDS-561_Glong-readRNAseq_raw_multiqc_GLlong-readRNAseq_report.zip    <-- QC
GLDS-561_Gwgs_raw_multiqc_GLwgs_report.zip                              <-- WGS QC
GLDS-561_rna_seq_raw_multiqc_GLbulkRNAseq_report.zip                    <-- RNA-seq QC
GLDS-561_GtargetSeq_raw_multiqc_GLtargetSeq_report.zip                  <-- Target seq QC
OSD-569_metadata_OSD-569-ISA.zip                                          <-- Metadata
+ 2 md5sum files
```

**NOTE**: No standard `*_Normalized_Counts_GLbulkRNAseq.csv` files!
The processed RNA-seq data is in `.xlsx` format, not the standard GeneLab CSV format.
This is because the long-read (ONT) RNA-seq used a custom pipeline (`pipeline-transcriptome-de`).

### OSD-530 (JAXA CFE) -- 8 files
```
GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized.xlsx (10.7 MB)
GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount.xlsx (2.4 MB)
GLDS-530_rna-seq_..._466genes.xlsx (138 KB)  <-- 466 DRR genes filtered
GLDS-530_rna-seq_..._9group_...pairwise_analysis_included.xlsx (11.5 MB)
GLDS-530_rna-seq_..._9group_...SEM.xlsx (5.0 MB)
GLDS-530_rna-seq_..._11group_...SEM.xlsx (4.5 MB)
GLDS-530_rna-seq_TGB_063_Input_vs_IP_totalcount_all0removed_scalingnormalized.xlsx (1.9 MB)
OSD-530_metadata_OSD-530-ISA.zip (129 KB)
```

**All processed data is in `.xlsx` format** (CLC Genomics Workbench output).

### OSD-571 (I4 Plasma Proteomics/Metabolomics) -- 14 files
```
GLDS-563_proteomics_Plasma_Proteomics_Processed_Data.xlsx
GLDS-563_proteomics_plasma_proteomics_preprocessed_data.tsv
GLDS-563_proteomics_plasma_metadata_all_samples_collapsed.csv
GLDS-563_proteomics_EVP_Proteomics_Processed_Data.xlsx
GLDS-563_proteomics_EVPs_proteomics_preprocessed_data.txt
GLDS-563_proteomics_EVPs_sample_metadata.csv
GLDS-563_metabolomics_Plasma_Metabolomics_Processed_Data.xlsx
GLDS-563_metabolomics_metabolomics_RPPOS-NEG_preprocessed_data.xlsx
GLDS-563_metabolomics_metabolomics_ANPPOS-NEG_preprocessed_data.xlsx
+ QC reports, metadata, md5sums
```

### OSD-570 (I4 PBMCs Single-Cell) -- 12 files
```
GLDS-562_snRNA-Seq_PBMC_Gene_Expression_snRNA-seq_Processed_Data.xlsx
GLDS-562_snATAC-Seq_PBMC_Chromatin_Accessibility_snATAC-seq_Processed_Data.xlsx
GLDS-562_scRNA-Seq_VDJ_Results.xlsx
+ QC reports, md5sums, metadata
```

---

## 6. CORRECTIONS NEEDED IN DATA_INVENTORY.md

### 6.1 CRITICAL: GeneLab file naming
**Current claim**: Files named like `*_Normalized_Counts_GLbulkRNAseq.csv`
**Reality**: Our datasets use GLDS IDs and `.xlsx` format:
- OSD-569 files start with `GLDS-561_` or `LSDS-7_`
- OSD-571 files start with `GLDS-563_`
- OSD-530 files start with `GLDS-530_`
- OSD-570 files start with `GLDS-562_`

**Section 2.3** describes standard GeneLab pipeline outputs, but our I4 and JAXA datasets
mostly used custom pipelines and provide `.xlsx` processed data, not the standard CSV format.

### 6.2 CRITICAL: Query API method does not work
**Current claim (Section 6.1)**: `file.data_type=normalized_counts` as primary fetch method
**Reality**: Returns 400/422 errors. Must use file listing API + GEODE download instead.

### 6.3 MINOR: OSD-569 CBC description
**Current**: "CBC, bulk RNA-seq (ONT+Illumina), WGS..."
**Verified**: The CBC data is under `LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv`.
The Illumina bulk RNA-seq was done on Ultima Genomics UG100, not standard Illumina.

### 6.4 MINOR: Protein/metabolite counts
These numbers came from the P02 paper and need verification against actual OSDR files.
The OSDR files may contain different filtered/unfiltered counts.

---

## 7. CORRECTIONS NEEDED IN FETCH SCRIPTS

### fetch_genelab.py

1. **Method 1 (Query API)** will fail for all our datasets. Should be demoted to fallback.
2. **Method 2 (OSDR file listing)** returns 0 for some datasets. Need to use biodata API instead.
3. **File patterns** in DATASETS config use wrong patterns (e.g., `Normalized_Counts` won't match `scalingnormalized`).
4. **GEODE download** works but needs `-L` (follow redirects) -- the script uses `requests.get()` which follows redirects by default, so this is OK.
5. **File names use GLDS IDs**, not OSD IDs -- patterns need updating.

### fetch_supplementary.py

1. Nature supplementary files require browser download (automated download blocked by Cloudflare/cookies).
2. GEO FTP URL pattern needs verification.

---

## 8. ITEMS THAT ARE UNVERIFIED (Could Not Confirm)

These items came from WebFetch of paper content and agent research. They are
**plausible but not independently verified** against the actual files:

| Claim | Source | Status |
|-------|--------|--------|
| P1 has 8 supplementary tables | Agent fetched paper | UNVERIFIED (need manual browser check) |
| P2 Source Data is 3 MB ZIP | Agent fetched paper | UNVERIFIED |
| P3 has 15 supplementary data files | Agent fetched paper | UNVERIFIED |
| P6 Source Data is 26.6 MB | Agent fetched paper | UNVERIFIED |
| P6 Supp Data 1 is 9.5 MB | Agent fetched paper | UNVERIFIED |
| P7 Source Data is 2.3 MB xlsx | Agent fetched paper | UNVERIFIED |
| P8 has 6 supplementary data files | Agent fetched paper | UNVERIFIED |
| P9 Supp Data sizes (647 KB, etc.) | Agent fetched paper | UNVERIFIED |
| Exact protein counts (2,992/1,765/1,443/527) | From P02 paper | UNVERIFIED against OSDR files |
| Exact metabolite count (1,135) | From P02 paper | UNVERIFIED against OSDR files |
| 18,676 genes in spatial data | From P06 paper | UNVERIFIED against OSDR files |
| SOMA Browser URLs accessible | Agent provided | UNVERIFIED (URLs plausible) |
| GitHub repos accessible | Agent provided | UNVERIFIED (URLs plausible) |

### What IS verified:
- All 12 OSD dataset IDs exist with correct titles
- All 10 paper DOIs resolve
- GEO GSE213808 exists and is public
- Actual processed file names and sizes for OSD-569, 530, 571, 570
- GEODE download works (tested with 10.7 MB file)
- Biodata REST API file listing works for all datasets

---

## 9. NO FABRICATED/SIMULATED DATA FOUND

After thorough review:
- **No fake dataset IDs** -- all 12 OSD IDs verified against live API
- **No fabricated DOIs** -- all 10 DOIs resolve to real papers
- **No invented file names** -- all file names came from actual API responses
- **No simulated numbers** -- protein/metabolite counts came from paper reports (unverified against files but from real papers)

The main risk is **inaccurate details** about supplementary table contents and sizes,
which were extracted by AI agents reading web pages. These should be verified during
actual download.

---

*Audit completed: 2026-02-24*
