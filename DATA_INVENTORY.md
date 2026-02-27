# SpaceOmicsBench v2 (Public) -- Data Inventory

> **Goal**: Build a fully public benchmark using only openly accessible processed data.
> No raw sequencing data. No IRB-restricted individual-level data.
> All sources are published papers (supplementary tables) + NASA OSDR processed outputs.
>
> **Tracks**:
> - **Open track**: Publicly accessible processed/summary data only.
> - **Controlled track (optional)**: Any restricted human sequence-level or individual data require approved access (e.g., OSDR DAR, dbGaP/LSDA).

---

## 1. Data Sources Overview

### 1.1 Missions / Studies Covered

| Mission | Duration | Crew | Key Datasets |
|---------|----------|------|--------------|
| **SpaceX Inspiration4 (I4)** | 3 days, 585 km orbit | 4 civilians | OSD-569 to OSD-687 (10 studies) |
| **JAXA Cell-Free Epigenome (CFE)** | >120 days ISS | 6 astronauts | OSD-530 |
| **NASA Twins Study** | 340 days ISS | 1 astronaut + 1 twin ground control | Supplementary tables only |

### 1.2 Papers and Their Processed Data

| # | Paper (DOI) | Short Title | Mission | Freely Available Processed Data |
|---|-------------|-------------|---------|-------------------------------|
| P1 | 10.1038/s41586-024-07639-y | SOMA Atlas | I4 | Supp Tables 1-8; OSD-569 to OSD-687 processed |
| P2 | 10.1038/s41467-024-48841-w | Secretome (Proteomics+Metabolomics) | I4 | Source Data ZIP (3 MB); OSD-571 processed |
| P3 | 10.1038/s41467-024-49211-2 | Single-cell Multi-ome & Immune | I4 | Supp Data 1-15; OSD-570 processed |
| P4 | 10.1038/s41467-024-48806-z | Epigenomics & Clonal Hematopoiesis | I4 | Source Data; OSD-569 processed |
| P5 | 10.1038/s41564-024-01635-8 | Microbiome | I4 | Supp Data files; OSD-572, OSD-573 processed |
| P6 | 10.1038/s41467-024-48625-2 | Spatial Skin Transcriptomics | I4 | Supp Data 1-3 + Source Data (26.6 MB); OSD-574 |
| P7 | 10.1038/s41467-024-49289-8 | Hemoglobin in Space | I4+JAXA+Twins | Source Data xlsx; OSD-530, OSD-570 |
| P8 | 10.1038/s41467-023-41995-z | JAXA CFE cfRNA (CD36/mtDNA) | JAXA CFE | Supp Data 1-6; OSD-530 processed; GEO GSE213808 (mouse) |
| P9 | 10.1038/s42003-024-06014-x | TERRA Telomeric RNA | I4+Twins+Everest | Supp Data 1-5; OSD-569/570 |
| P10 | 10.1126/science.aau8650 | NASA Twins Study | Twins | Supp Tables (processed summary data only) |

---

## 2. NASA OSDR Datasets (GeneLab)

### 2.1 Inspiration4 Datasets

| OSD ID | Sample Type | Omics Available (Processed) |
|--------|------------|----------------------------|
| **OSD-569** | Whole Blood | CBC, bulk RNA-seq (ONT+Illumina), WGS (processed), clonal hematopoiesis, epitranscriptome (m6A) |
| **OSD-570** | PBMCs | snRNA-seq + snATAC-seq (10x Multiome), TCR/BCR V(D)J repertoire |
| **OSD-571** | Blood Plasma | Plasma proteomics (2,992 proteins), EVP proteomics (1,443 proteins), metabolomics (1,135 metabolites), cfRNA |
| **OSD-572** | Skin/Oral/Nasal Swabs | Metagenomics, metatranscriptomics |
| **OSD-573** | Environmental (Dragon capsule) | Environmental metagenomics |
| **OSD-574** | Skin Biopsies | Spatial transcriptomics (NanoString GeoMx), metagenomics |
| **OSD-575** | Blood Serum | Serum biomarkers |
| **OSD-630** | Stool | Gut microbiome |
| **OSD-656** | Urine | Urinary biomarkers |
| **OSD-687** | T-cells | Histone modification profiling |

### 2.2 JAXA CFE Datasets

| OSD ID | Sample Type | Omics Available (Processed) |
|--------|------------|----------------------------|
| **OSD-530** | Human Plasma | cfRNA-seq (normalized counts, DEG results) -- 6 astronauts, 11 timepoints |
| **OSD-532** | Mouse Plasma | cfRNA-seq (raw + processed) -- MHU-1 mission |

### 2.3 Actual Processed Files Available (Verified)

**IMPORTANT**: Our datasets use custom pipelines, NOT GeneLab's standard bulk RNA-seq pipeline.
File names use GLDS IDs (not OSD IDs) and are mostly `.xlsx` format.

**OSD-569 (I4 Whole Blood)** -- GLDS-561 / LSDS-7:
- `LSDS-7_Complete_Blood_Count_CBC_TRANSFORMED.csv` -- Standardized CBC data
- `GLDS-561_long-readRNAseq_Direct_RNA_seq_Gene_Expression_Processed.xlsx` -- ONT RNA-seq DE
- `GLDS-561_directm6Aseq_Direct_RNA_seq_m6A_Processed_Data.xlsx` -- m6A methylation

**OSD-530 (JAXA CFE cfRNA)** -- GLDS-530:
- `GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized.xlsx` (10.7 MB)
- `GLDS-530_rna-seq_..._466genes.xlsx` -- 466 DRR genes (filtered)
- `GLDS-530_rna-seq_..._9group_..._pairwise_analysis_included.xlsx` (11.5 MB)
- + 4 additional xlsx files with SEM and different groupings

**OSD-571 (I4 Plasma Proteomics/Metabolomics)** -- GLDS-563:
- `GLDS-563_proteomics_Plasma_Proteomics_Processed_Data.xlsx`
- `GLDS-563_proteomics_EVP_Proteomics_Processed_Data.xlsx`
- `GLDS-563_metabolomics_Plasma_Metabolomics_Processed_Data.xlsx`
- `GLDS-563_metabolomics_metabolomics_RPPOS-NEG_preprocessed_data.xlsx`
- `GLDS-563_metabolomics_metabolomics_ANPPOS-NEG_preprocessed_data.xlsx`
- + metadata CSVs and preprocessed data files

**OSD-570 (I4 PBMCs Single-Cell)** -- GLDS-562:
- `GLDS-562_snRNA-Seq_PBMC_Gene_Expression_snRNA-seq_Processed_Data.xlsx`
- `GLDS-562_snATAC-Seq_PBMC_Chromatin_Accessibility_snATAC-seq_Processed_Data.xlsx`
- `GLDS-562_scRNA-Seq_VDJ_Results.xlsx`

---

## 3. Processed Data by Omics Modality

### 3.1 Clinical (CBC / CMP)

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-569 | Complete blood count for 4 I4 crew, 6+ timepoints | OSDR open |
| P1 Supp Table 1 | Sample collection details (2,911 samples) | Nature download |
| P10 Supp Tables | Twins Study clinical labs (aggregate/summary) | Science download |

### 3.2 Transcriptomics (Bulk RNA-seq / cfRNA)

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-569 | I4 whole blood bulk RNA-seq (normalized counts, DE) | OSDR open |
| OSD-530 | JAXA CFE cfRNA (6 astronauts, normalized counts) | OSDR open |
| P7 Source Data | Hemoglobin gene expression across I4/JAXA/Twins | Nature download |
| P8 Supp Data 1-6 | JAXA CFE: 466 DRRs, CD36 genes, coregulated clusters, mouse cfRNA | Nature download |
| P1 Supp Table 5 | cfRNA cell-type deconvolution results (Bayes Prism) | Nature download |
| P1 Supp Table 6 | KEGG pathway overrepresentation for DEGs | Nature download |

### 3.3 Single-Cell / Single-Nucleus

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-570 | I4 PBMC snRNA-seq + snATAC-seq (Cell Ranger ARC outputs) | OSDR open |
| P3 Supp Data 1-15 | GO pathways, IPA results, TF motifs, GSEA, drug candidates, conserved signatures | Nature download |
| SOMA Browser | Interactive processed single-cell visualization | Web (open) |

### 3.4 Proteomics

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-571 | I4 plasma proteomics (1,765 proteins after filtering) + EVP proteomics (527 proteins) | OSDR open |
| P2 Source Data | Differential abundance results (limma), protein networks | Nature download |
| SOMA Browser | Interactive proteomics visualization | Web (open) |

### 3.5 Metabolomics

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-571 | I4 plasma metabolomics (1,135 metabolites: 865 ANP + 270 lipid) | OSDR open |
| P2 Source Data | Differential metabolite results | Nature download |
| SOMA Browser | Interactive metabolomics visualization | Web (open) |

### 3.6 Microbiome

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-572 | I4 skin/oral/nasal metagenomics + metatranscriptomics | OSDR open |
| OSD-573 | I4 environmental (Dragon capsule) metagenomics | OSDR open |
| OSD-630 | I4 stool metagenomics | OSDR open |
| P5 Supp Data | Microbial taxa abundances, species tables | Nature download |
| P6 Supp Data 3 | Skin metagenomic + metatranscriptomic taxonomy (826 bacterial, 9,819 viral) | Nature download |
| P1 Supp Table 7 | Microbial taxa coefficient of variation by body site | Nature download |

### 3.7 Spatial Transcriptomics

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-574 | GeoMx DSP processed DCC files + normalized counts (18,676 genes, 95 ROIs) | OSDR open |
| P6 Supp Data 1 | DEG tables (95 up, 121 down genes in skin) | Nature download |
| P6 Supp Data 2 | Pathway enrichment (FGSEA, GSVA, MSigDB Hallmark) | Nature download |
| P6 Source Data | All figure data (26.6 MB xlsx) | Nature download |

### 3.8 Epigenomics / Chromatin

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-570 | snATAC-seq (10x Multiome) processed peak matrices | OSDR open |
| OSD-687 | T-cell histone modification profiling | OSDR open |
| P4 Source Data | Clonal hematopoiesis results, epigenetic changes | Nature download |

### 3.9 Telomere / TERRA

| Source | Data Description | Access |
|--------|-----------------|--------|
| P9 Supp Data 1 | I4 TERRA motif correlations and k-mer analysis | Nature download |
| P9 Supp Data 2 | Twins Study TERRA motif analysis | Nature download |
| P9 Supp Data 4 | Mt. Everest altitude-dependent TERRA data | Nature download |

### 3.10 Immune Repertoire

| Source | Data Description | Access |
|--------|-----------------|--------|
| OSD-570 | TCR/BCR V(D)J clonotype data (Cell Ranger) | OSDR open |
| P3 Supp Data 12 | NASA astronaut cohort cytokine data (n=64) | Nature download |

---

## 4. NASA Twins Study -- Special Handling

The NASA Twins Study is inherently identifiable (only 2 subjects: identical twins, one astronaut). Most individual-level data is restricted.

### What IS freely available (from published paper supplementary):
- **Aggregate summary statistics** from supplementary tables
- **Processed fold-change data** and pathway enrichment results
- **P7 Source Data**: Hemoglobin gene expression comparison (Twins + I4 + JAXA)
- **P9 Supp Data 2**: TERRA motif analysis (aggregate)

### What is RESTRICTED:
- Individual-level omics data (GEO controlled via dbGaP)
- Raw sequencing data
- Detailed clinical records (NASA LSDA, requires formal request)

### Strategy for v2:
Use only the **published aggregate/summary-level** Twins data from supplementary tables. This is sufficient for cross-mission comparison tasks without privacy concerns.

---

## 5. Access Limitations Summary

| Data Category | Access | Notes |
|---------------|--------|-------|
| GeneLab processed files (OSD-*) | **OPEN** | No authentication required |
| Paper supplementary tables | **OPEN** | Direct download from publisher |
| SOMA Browser | **OPEN** | Web-based, no login |
| Source Data files from papers | **OPEN** | Direct download from publisher |
| GEO GSE213808 (mouse) | **OPEN** | Fully public |
| Raw human FASTQ/BAM | **CONTROLLED** | Requires IRB + OSDR application |
| NASA Twins individual data | **CONTROLLED** | dbGaP / LSDA formal request |
| Twins clinical records | **CONTROLLED** | LSDA formal request |

### Key Constraint:
For a fully public benchmark, we use ONLY the "OPEN" rows above. All controlled-access data is excluded.

**Controlled-access note**: Any future controlled track must document access requirements per dataset (OSDR DAR, dbGaP/LSDA) and must not be redistributed in the open track.

---

## 6. Data Fetching Methods

### 6.1 GeneLab API (Processed Data)

**NOTE**: The Biodata Query API (`file.data_type=normalized_counts`) does NOT work
for our datasets -- they use custom pipelines. Use these verified methods instead:

```python
import requests

# Method 1 (VERIFIED): Biodata REST API -- list available files
BASE = "https://visualization.osdr.nasa.gov/biodata/api"
url = f"{BASE}/v2/dataset/OSD-530/files/"
resp = requests.get(url)
files = resp.json()["OSD-530"]["files"]
# Returns dict: {filename: info, ...}

# Method 2 (VERIFIED): GEODE download -- follows S3 redirect
url = "https://osdr.nasa.gov/geode-py/ws/studies/OSD-530/download"
params = {"source": "datamanager",
          "file": "GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized.xlsx"}
resp = requests.get(url, params=params, allow_redirects=True, stream=True)
with open("output.xlsx", "wb") as f:
    for chunk in resp.iter_content(65536):
        f.write(chunk)

# Method 3: AWS S3 (bulk download, no auth, requires aws cli)
# aws s3 sync s3://nasa-osdr/OSD-569/ ./OSD-569/ --no-sign-request \
#     --exclude "*" --include "*.xlsx" --include "*.csv"
```

### 6.2 Supplementary Data from Papers

Supplementary files can be downloaded directly from Nature article pages. The URLs typically follow:
```
https://www.nature.com/articles/{ARTICLE_ID}
# -> Supplementary Information section -> direct download links
```

### 6.3 Interactive Portals

| Portal | URL | Data Types |
|--------|-----|-----------|
| SOMA Browser | https://soma.weill.cornell.edu/apps/SOMA_Browser/ | All I4 omics |
| I4 Multiome | https://soma.weill.cornell.edu/apps/I4_Multiome/ | Single-cell |
| I4 Microbiome | https://soma.weill.cornell.edu/apps/I4_Microbiome/ | Microbiome |

### 6.4 Code Repositories

| Repo | URL | Contents |
|------|-----|----------|
| inspiration4-omics | https://github.com/eliah-o/inspiration4-omics | Multi-omics analysis pipelines |
| SpatialAnalysis | https://github.com/jpark-lab/SpatialAnalysis | Skin spatial transcriptomics |

---

## 7. Recommended Priority for v2 Benchmark

### Tier 1 (Core -- fetch first)
1. **OSD-569** processed: CBC + bulk RNA-seq normalized counts + DE
2. **OSD-530** processed: JAXA CFE cfRNA normalized counts + DE
3. **OSD-571** processed: Plasma proteomics + metabolomics
4. **P2 Source Data**: Proteomics + metabolomics differential results
5. **P7 Source Data**: Cross-mission hemoglobin comparison (I4+JAXA+Twins)
6. **P8 Supp Data 1-6**: JAXA CFE gene lists, DRRs, clusters

### Tier 2 (Expansion)
7. **OSD-570** processed: Single-cell multiome (snRNA+snATAC)
8. **P3 Supp Data 1-15**: Pathway enrichment, TF motifs, drug candidates
9. **OSD-574** processed: Spatial transcriptomics
10. **P6 Supp Data 1-3**: Skin DEGs, pathways, microbiome taxonomy

### Tier 3 (Additional modalities)
11. **OSD-572/573/630**: Microbiome data
12. **P5 Supp Data**: Microbiome species tables
13. **OSD-687**: T-cell histone modifications
14. **P9 Supp Data 1-5**: TERRA/telomere analysis
15. **GEO GSE213808**: Mouse cfRNA (cross-species validation)

---

*Document generated: 2026-02-24*
*This inventory covers all freely accessible processed data for SpaceOmicsBench v2 (Public).*
