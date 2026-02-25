# P08: JAXA CFE Cell-Free RNA (CD36/mtDNA)

- **Paper**: "Release of CD36-associated cell-free mitochondrial DNA and RNA as a hallmark of space environment response"
- **Journal**: Nature Communications, June 2024 (originally 2023)
- **DOI**: 10.1038/s41467-023-41995-z
- **Mission**: JAXA Cell-Free Epigenome Study (6 astronauts, >120 days ISS)

## Data Types

- Human cfRNA RNA-seq (6 astronauts, 11 timepoints)
- Human cfDNA quantification
- CD36-fraction RNA-seq (tissue-specific)
- Mouse cfRNA RNA-seq (MHU-1 mission)

## Supplementary Data Files

| File | Description |
|------|-------------|
| Supp Data 1 | Pairwise comparison results (all timepoint contrasts) |
| Supp Data 2 | 466 differentially represented cfRNAs (DRRs) |
| Supp Data 3 | Correlation analysis and coregulated cfRNA clusters |
| Supp Data 4 | 406 CD36-enriched genes |
| Supp Data 5 | Tissue-specificity analysis (GTEx/HPA cross-reference) |
| Supp Data 6 | Mouse cfRNA DRRs (467 genes, MHU-1 mission) |

## OSDR Datasets

- **OSD-530**: Human cfRNA processed (normalized counts) -- OPEN
- **OSD-532**: Mouse cfRNA (raw + processed) -- OPEN
- **GEO GSE213808**: Mouse cfRNA (quantile-normalized counts, 13.8 MB) -- OPEN

## Access Notes

- Human processed data (OSD-530): **OPEN, no restrictions**
- Human raw FASTQ: **CONTROLLED** (requires ethics approval, contact corresponding author)
- Mouse data (OSD-532, GSE213808): **FULLY OPEN** (both raw + processed)

## Key for Benchmark

Primary source for JAXA CFE cfRNA data. The 466 DRRs and coregulated clusters
provide ground truth for cross-mission concordance tasks with I4 data.
