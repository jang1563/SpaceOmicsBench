# P06: Spatial Multi-omics of Human Skin

- **Paper**: "Spatial multi-omics of human skin reveals KRAS and inflammatory responses to spaceflight"
- **Journal**: Nature Communications, June 2024
- **DOI**: 10.1038/s41467-024-48625-2
- **Mission**: SpaceX Inspiration4

## Data Types

- Spatial transcriptomics: NanoString GeoMx DSP WTA (18,676 genes, 95 ROIs)
- Skin metagenomics (826 bacterial, 9,819 viral species)
- Skin metatranscriptomics (88 bacterial, 1,456 viral species)
- PBMC single-cell multiome (referenced from OSD-570)

## Supplementary Data Files

| File | Size | Description |
|------|------|-------------|
| Supp Data 1 | 9.5 MB | DEG tables (log2FC, p-values, q-values) |
| Supp Data 2 | 7.3 MB | Pathway enrichment (FGSEA, GSVA, MSigDB Hallmark) |
| Supp Data 3 | 1.7 MB | Metagenomic + metatranscriptomic taxonomy |
| Source Data | 26.6 MB | All figure data |

## OSDR Datasets

- **OSD-574**: Skin spatial transcriptomics + metagenomics
- **OSD-570**: PBMC multiome

## Code Repositories

- Spatial analysis: https://github.com/jpark-lab/SpatialAnalysis/
- Multi-omics pipelines: https://github.com/eliah-o/inspiration4-omics
- Zenodo archive: https://doi.org/10.5281/zenodo.10016141
