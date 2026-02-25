# SpaceOmicsBench v2 (Public Release)

A comprehensive multi-omics AI benchmark for spaceflight biomedical data,
built entirely from **publicly accessible processed data** -- no IRB-restricted
or individually identifiable data included.

## Missions Covered

| Mission | Duration | Subjects | Data Sources |
|---------|----------|----------|--------------|
| SpaceX Inspiration4 | 3 days (585 km) | 4 civilians | NASA OSDR (OSD-569 to OSD-687) |
| JAXA Cell-Free Epigenome | >120 days ISS | 6 astronauts | NASA OSDR (OSD-530) |
| NASA Twins Study | 340 days ISS | 1 + twin control | Published supplementary data only |

## Omics Modalities (10 types)

1. **Clinical** -- CBC/CMP blood biomarkers
2. **Bulk Transcriptomics** -- RNA-seq (whole blood)
3. **Cell-free RNA** -- Plasma cfRNA-seq
4. **Single-cell Multi-ome** -- snRNA-seq + snATAC-seq (PBMCs)
5. **Proteomics** -- Plasma + EVP proteomics (LC-MS/MS)
6. **Metabolomics** -- Plasma untargeted metabolomics
7. **Spatial Transcriptomics** -- Skin (NanoString GeoMx)
8. **Microbiome** -- Metagenomics + metatranscriptomics (multi-site)
9. **Epigenomics** -- Chromatin accessibility, histone modifications
10. **Telomere/TERRA** -- Telomeric RNA dynamics

## Directory Structure

```
v2_public/
├── README.md                    # This file
├── DATA_INVENTORY.md            # Comprehensive data source documentation
├── data/                        # Core processed data files
│   ├── clinical/                # CBC/CMP biomarkers
│   ├── transcriptomics/
│   │   ├── bulk_rnaseq/         # Whole blood RNA-seq (I4)
│   │   ├── cfrna/               # Cell-free RNA (I4 + JAXA CFE)
│   │   └── single_cell/         # snRNA-seq + snATAC-seq (I4 PBMCs)
│   ├── proteomics/              # Plasma + EVP proteins
│   ├── metabolomics/            # Plasma metabolites
│   ├── spatial_transcriptomics/ # Skin GeoMx data
│   ├── microbiome/
│   │   ├── human/               # Skin, oral, nasal, stool
│   │   └── environmental/       # Dragon capsule
│   ├── epigenomics/             # Chromatin, histone marks
│   ├── telomere/                # TERRA dynamics
│   ├── immune_repertoire/       # TCR/BCR V(D)J
│   └── cross_mission/           # Aligned multi-mission data
├── sources/                     # Raw supplementary files organized by paper
│   ├── P01_SOMA_atlas/          # Nature 2024 (SOMA)
│   ├── P02_secretome/           # Nat Commun 2024 (proteomics+metabolomics)
│   ├── P03_singlecell_multiome/ # Nat Commun 2024 (single-cell)
│   ├── P04_epigenomics/         # Nat Commun 2024 (epigenomics)
│   ├── P05_microbiome/          # Nat Microbiol 2024 (microbiome)
│   ├── P06_spatial_skin/        # Nat Commun 2024 (spatial)
│   ├── P07_hemoglobin/          # Nat Commun 2024 (hemoglobin)
│   ├── P08_JAXA_CFE_cfRNA/      # Nat Commun 2023 (JAXA cfRNA)
│   ├── P09_TERRA/               # Commun Biol 2024 (telomere)
│   └── P10_twins_study/         # Science 2019 (Twins)
├── scripts/                     # Data fetching and processing scripts
├── tasks/                       # Benchmark task definitions
├── splits/                      # Train/test split definitions
├── baselines/                   # Baseline model results
├── evaluation/                  # Evaluation harness and metrics
└── docs/                        # Additional documentation
```

## Data Provenance

All data in this benchmark comes from two types of public sources:

1. **NASA Open Science Data Repository (OSDR)** -- GeneLab-processed omics files
   - API: `https://visualization.osdr.nasa.gov/biodata/api/`
   - AWS S3: `s3://nasa-osdr/` (no auth required)

2. **Published paper supplementary data** -- Processed results from 10 peer-reviewed papers
   - See `DATA_INVENTORY.md` for complete paper list and data mapping

## Quick Start

```bash
# Fetch all public processed data
python scripts/fetch_all_data.py

# Or fetch specific modalities
python scripts/fetch_genelab.py --datasets OSD-569,OSD-530,OSD-571
python scripts/fetch_supplementary.py --papers P01,P02,P08
```

## Citation

If you use this benchmark, please cite the original data papers (see `docs/CITATIONS.bib`).

## License

Benchmark code: MIT License
Data: Subject to original source licenses (NASA OSDR: public domain; Nature papers: CC-BY 4.0)
