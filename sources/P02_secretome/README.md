# P02: Secretome Profiling (Proteomics + Metabolomics)

- **Paper**: "Secretome profiling reveals acute changes in oxidative stress, brain homeostasis, and coagulation following short-duration spaceflight"
- **Journal**: Nature Communications, June 2024
- **DOI**: 10.1038/s41467-024-48841-w
- **Mission**: SpaceX Inspiration4

## Data Types

- Plasma proteomics: 2,992 unique proteins (1,765 after filtering), Seer Proteograph + LC-MS/MS DIA
- EVP proteomics: 1,443 unique proteins (527 after filtering), LC-MS/MS DDA
- Plasma metabolomics: 1,135 metabolites (865 ANP + 270 lipid)
- Direct RNA-seq: Oxford Nanopore PromethION (whole blood)

## Files to Download

| File | Size | Description |
|------|------|-------------|
| Source Data (ZIP) | 3 MB | All figure-level processed data tables |
| Supplementary Information | 1.6 MB | Extended figures and methods |

## OSDR Datasets

- **OSD-571**: Plasma proteomics, EVP proteomics, metabolomics, cfRNA
- **OSD-569**: Direct RNA-seq

## Analysis Code

- GitHub: https://github.com/eliah-o/inspiration4-omics

## Key for Benchmark

Primary source for proteomics and metabolomics differential abundance results.
Statistical model: limma (~astronaut + flightStatus), thresholds: adj. p<0.05, |logFC|>1.
