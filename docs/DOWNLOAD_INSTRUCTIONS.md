# SpaceOmicsBench v2 -- Data Inventory & Download Status

**Base directory**: `SpaceOmicsBench/v2_public/data/`
**Last updated**: 2026-02-25

---

## Directory Structure

```
data/
├── P01/          # SOMA Atlas supplementary tables (Nature)
├── P02/          # Secretome source data (NatComm)
├── P03/          # Single-cell multi-ome supplementary (NatComm) ✅ COMPLETE
├── P05/          # Microbiome supplementary (NatMicro)
├── P06/          # Spatial skin supplementary + source data (NatComm)
├── P07/          # Hemoglobin cross-mission source data (NatComm)
├── P08/          # JAXA CFE cfRNA supplementary (NatComm)
├── P09/          # TERRA/telomere supplementary (CommBio)
├── P10/          # NASA Twins supplementary tables (Science)
├── clinical/     # CBC, CMP, cytokines, urine (OSDR: OSD-569, OSD-575, OSD-656)
├── transcriptomics/
│   ├── bulk_rnaseq/   # ONT long-read RNA-seq + m6A (OSDR: OSD-569)
│   ├── cfrna/         # JAXA CFE cfRNA counts (OSDR: OSD-530)
│   └── single_cell/   # snRNA-seq, snATAC-seq, VDJ (OSDR: OSD-570)
├── proteomics/        # Plasma + EVP proteomics (OSDR: OSD-571)
├── metabolomics/      # Plasma metabolomics (OSDR: OSD-571)
├── spatial_transcriptomics/  # NanoString GeoMx + skin metagenomics (OSDR: OSD-574)
└── microbiome/
    ├── human/         # Skin/oral/nasal + stool (OSDR: OSD-572, OSD-630)
    └── environmental/ # Dragon capsule (OSDR: OSD-573)
```

---

## Paper Supplementary Data (P01-P10)

### P01: SOMA Atlas (Nature) -- 7 tables + SI PDF
| File | Size | Content |
|------|------|---------|
| `P01_SuppTable1_sample_collection.xlsx` | 77 KB | 2,911 samples banked, 1,194 processed |
| `P01_SuppTable2_OSDR_studies.xlsx` | 13 KB | OSDR database studies comparison |
| `P01_SuppTable3_spaceflight_studies.xlsx` | 33 KB | Spaceflight studies overview |
| `P01_SuppTable4_data_annotations.xlsx` | 183 KB | Raw/processed data annotations |
| `P01_SuppTable5_cfRNA_deconvolution.xlsx` | 47 KB | cfRNA cell-type deconvolution (Bayes Prism) |
| `P01_SuppTable6_KEGG_pathway.xlsx` | 1.6 MB | KEGG pathway overrepresentation for DEGs |
| `P01_SuppTable7_microbial_taxa_CV.xlsx` | 6.6 MB | Microbial taxa CV by body site |
| `P01_supplementary_info.pdf` | 3.5 MB | Extended methods and figures |

### P02: Secretome (NatComm) -- Source Data + SI PDF
| File | Size | Content |
|------|------|---------|
| `P02_source_data.zip` | 3.0 MB | Proteomics + metabolomics DE results (limma logFC, adj.p) |
| `P02_supplementary_info.pdf` | 1.6 MB | Extended methods |

### P03: Single-Cell Multi-ome (NatComm) -- COMPLETE (15 Supp Data files)
| File | Size | Content |
|------|------|---------|
| `Supplementary.Data/Supplementary Data 1.xlsx` | 95 KB | GO pathway enrichment |
| `Supplementary.Data/Supplementary Data 2.xls` | 64 KB | IPA results |
| `Supplementary.Data/Supplementary Data 3.xlsx` | 19 KB | TF motif analysis |
| `Supplementary.Data/Supplementary Data 4.csv` | 12 KB | ssGSEA immune populations |
| `Supplementary.Data/Supplementary Data 5.xlsx` | 230 KB | Murine spaceflight gene sets |
| `Supplementary.Data/Supplementary Data 6.xlsx` | 205 KB | Mouse pathway comparisons |
| `Supplementary.Data/Supplementary Data 7.xlsx` | 3.7 MB | Drug/compound candidates (148) |
| `Supplementary.Data/Supplementary Data 8.xlsx` | 42 KB | Infection-related pathway genes |
| `Supplementary.Data/Supplementary Data 9.xls` | 76 KB | Sex-specific pathway alterations |
| `Supplementary.Data/Supplementary Data 10.xlsx` | 36 KB | CD/HLA gene expression by sex |
| `Supplementary.Data/Supplementary Data 11.xlsx` | 31 KB | TF motif activity by sex |
| `Supplementary.Data/Supplementary Data 12.xlsx` | 16 KB | NASA astronaut cytokines (n=64) |
| `Supplementary.Data/Supplementary Data 13.csv` | 57 MB | Microbiome association regression |
| `Supplementary.Data/Supplementary Data 14.xlsx` | 252 KB | GO terms for microbiome pathways |
| `Supplementary.Data/Supplementary Data 15.xlsx` | 800 KB | Conserved spaceflight signatures |
| `P03_SuppData3_TF_motifs.xlsx` | 27 MB | TF motif analysis (MOESM download) |
| `P03_supplementary_info.pdf` | 14 MB | Extended methods and figures |
| `P03_reporting_summary.pdf` | 90 KB | Peer review reporting summary |

### P05: Microbiome (Nature Microbiology) -- 2 files
| File | Size | Content |
|------|------|---------|
| `P05_SuppData1.xlsx` | 16 KB | Supplementary Data 1 |
| `P05_SuppData2.xlsx` | 90 KB | Supplementary Data 2 |

### P06: Spatial Skin (NatComm) -- 3 Supp Data + Source Data
| File | Size | Content |
|------|------|---------|
| `P06_SuppData1_DEGs.xlsx` | 9.5 MB | DEG tables (log2FC, p-values, q-values) |
| `P06_SuppData2_pathway_enrichment.xlsx` | 7.3 MB | FGSEA, GSVA, MSigDB Hallmark pathways |
| `P06_SuppData3_metagenomics_taxonomy.xlsx` | 1.7 MB | Skin metagenomics + metatranscriptomics taxonomy |
| `P06_source_data.xlsx` | 27 MB | All figure-level data |

### P07: Hemoglobin (NatComm) -- Source Data
| File | Size | Content |
|------|------|---------|
| `P07_source_data_hemoglobin.xlsx` | 2.3 MB | Cross-mission hemoglobin expression (I4 + JAXA CFE + Twins) |

### P08: JAXA CFE cfRNA (NatComm) -- 5 Supp Data (human only)
| File | Size | Content |
|------|------|---------|
| `P08_SuppData1_pairwise_comparisons.xlsx` | 595 KB | All timepoint pairwise contrasts |
| `P08_SuppData2_466_DRRs.xlsx` | 58 KB | 466 differentially represented cfRNAs |
| `P08_SuppData3_coregulated_clusters.xlsx` | 2.9 MB | Correlation analysis, coregulated cfRNA clusters |
| `P08_SuppData4_CD36_genes.xlsx` | 48 KB | 406 CD36-enriched genes |
| `P08_SuppData5_tissue_specificity.xlsx` | 15 KB | Tissue-specificity (GTEx/HPA cross-reference) |

Note: SuppData6 (mouse cfRNA DRRs) removed -- mouse data excluded from benchmark.

### P09: TERRA / Telomere (CommBio) -- 5 Supp Data
| File | Size | Content |
|------|------|---------|
| `P09_SuppData1_I4_TERRA_motifs.xlsx` | 647 KB | I4 TERRA UUAGGG k-mer analysis |
| `P09_SuppData2_Twins_TERRA_motifs.xlsx` | 728 KB | Twins Study TERRA motif analysis |
| `P09_SuppData3_simulated_microgravity.xlsx` | 186 KB | Simulated microgravity experiment data |
| `P09_SuppData4_Everest_TERRA.xlsx` | 539 KB | Mt. Everest altitude-dependent TERRA expression |
| `P09_SuppData5_invitro_stats.xlsx` | 42 KB | In vitro experiment statistics |

### P10: NASA Twins Study (Science) -- 8 tables + SM PDF
| File | Size | Content |
|------|------|---------|
| `P10_SuppTable_S1.xlsx` | 273 KB | Supplementary Table S1 |
| `P10_SuppTable_S2.xlsx` | 33 MB | Supplementary Table S2 |
| `P10_SuppTable_S3.xlsx` | 59 KB | Supplementary Table S3 |
| `P10_SuppTable_S4.xlsx` | 438 KB | Supplementary Table S4 |
| `P10_SuppTable_S5.xlsx` | 3.7 MB | Supplementary Table S5 |
| `P10_SuppTable_S6.xlsx` | 83 KB | Supplementary Table S6 |
| `P10_SuppTable_S8.xlsx` | 72 KB | Supplementary Table S8 |
| `P10_SuppTable_S9.xlsx` | 60 KB | Supplementary Table S9 |
| `P10_supplementary_materials.pdf` | 30 MB | Full supplementary materials |

**Missing**: Table S7. Note: Only aggregate/summary data should be used (no individual-level).

---

## OSDR Data (GeneLab API downloads)

### clinical/ -- 13 files from OSD-569, OSD-575, OSD-656
| Source | Files | Description |
|--------|-------|-------------|
| OSD-569 (LSDS-7) | CBC SUBMITTED + TRANSFORMED | Complete Blood Count (I4) |
| OSD-575 (LSDS-8) | CMP SUBMITTED + TRANSFORMED | Comprehensive Metabolic Panel (I4) |
| OSD-575 (LSDS-8) | Multiplex serum immune/cardio | Serum cytokine/cardiac arrays (I4) |
| OSD-656 (LSDS-64) | Multiplex urine + target info | Urine inflammation NULISAseq (I4) |

### transcriptomics/bulk_rnaseq/ -- 2 files from OSD-569
| File | Size |
|------|------|
| `GLDS-561_long-readRNAseq_Direct_RNA_seq_Gene_Expression_Processed.xlsx` | 118 MB |
| `GLDS-561_directm6Aseq_Direct_RNA_seq_m6A_Processed_Data.xlsx` | 89 MB |

### transcriptomics/cfrna/ -- 7 files from OSD-530 (human only)
| File | Size | Description |
|------|------|-------------|
| `GLDS-530_..._3group_..._scalingnormalized.xlsx` | 10 MB | Main normalized counts (3 groups) |
| `GLDS-530_..._9group_..._pairwise_analysis_included.xlsx` | 11 MB | 9-group pairwise DE |
| `GLDS-530_..._9group_..._SEM.xlsx` | 4.8 MB | 9-group SEM |
| `GLDS-530_..._11group_..._SEM.xlsx` | 4.3 MB | 11-group SEM |
| `GLDS-530_..._3group_totalcount.xlsx` | 2.3 MB | Raw total counts |
| `GLDS-530_..._Input_vs_IP_..._scalingnormalized.xlsx` | 1.8 MB | Input vs IP |
| `GLDS-530_..._466genes.xlsx` | 135 KB | 466 DRR genes (filtered) |

### transcriptomics/single_cell/ -- 3 files from OSD-570
| File | Size |
|------|------|
| `GLDS-562_snRNA-Seq_PBMC_Gene_Expression_snRNA-seq_Processed_Data.xlsx` | 3.2 MB |
| `GLDS-562_snATAC-Seq_PBMC_Chromatin_Accessibility_snATAC-seq_Processed_Data.xlsx` | 9.0 MB |
| `GLDS-562_scRNA-Seq_VDJ_Results.xlsx` | 50 MB |

### proteomics/ -- 6 files from OSD-571
| File | Size |
|------|------|
| `GLDS-563_proteomics_plasma_proteomics_preprocessed_data.tsv` | 58 MB |
| `GLDS-563_proteomics_Plasma_Proteomics_Processed_Data.xlsx` | 315 KB |
| `GLDS-563_proteomics_EVP_Proteomics_Processed_Data.xlsx` | 103 KB |
| `GLDS-563_proteomics_EVPs_proteomics_preprocessed_data.txt` | 509 KB |
| + 2 metadata files | <2 KB |

### metabolomics/ -- 3 files from OSD-571
| File | Size |
|------|------|
| `GLDS-563_metabolomics_Plasma_Metabolomics_Processed_Data.xlsx` | 290 KB |
| `GLDS-563_metabolomics_metabolomics_RPPOS-NEG_preprocessed_data.xlsx` | 100 KB |
| `GLDS-563_metabolomics_metabolomics_ANPPOS-NEG_preprocessed_data.xlsx` | 156 KB |

### spatial_transcriptomics/ -- 13 files from OSD-574
| File | Size |
|------|------|
| `GLDS-566_SpatialTranscriptomics_..._Processed_Data.xlsx` | 1.0 MB |
| 8x skin metagenomics/metatranscriptomics TSV files | 33K-243K each |
| 4x QC report ZIPs | 500K-1.5M each |

### microbiome/human/ -- 47 files from OSD-572, OSD-630
Includes combined taxonomy tables, pathway abundances, gene families, and per-sample assembly files.
Largest: `GLDS-599_GMetagenomics_Gene-families-grouped-by-taxa_...tsv` (48 MB)

### microbiome/environmental/ -- 11 files from OSD-573
Dragon capsule metagenomics: taxonomy coverages, pathway abundances, QC reports.

---

## Removed Items

| Item | Reason |
|------|--------|
| `data/cross_mission/` | No separate data -- cross-mission comparisons use P07, P09, P10 |
| `data/telomere/` | Data is in P09/ supplementary files |
| `data/immune_repertoire/` | VDJ data already in transcriptomics/single_cell/ |
| `data/epigenomics/` | Only had QC report; P04 is not an epigenomics database |
| Mouse data (OSD-532, GEO GSE213808) | Excluded from benchmark scope |
| P08 SuppData6 (mouse cfRNA DRRs) | Mouse data excluded |

---

## Action Items

- [x] ~~P03: Download remaining Supplementary Data files~~ DONE (all 15 files in Supplementary.Data/)
- [x] ~~P04~~ EXCLUDED from benchmark
- [x] ~~P10 Table S7~~ Does not exist in the original paper
- [ ] Verify MOESM-to-SuppTable/SuppData numbering by spot-checking xlsx contents
