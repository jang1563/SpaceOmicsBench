# SpaceOmicsBench v2 — Provenance Summary (Open Track)

This table summarizes the primary sources used to construct the public benchmark tasks.
For complete file inventories and download details, see `DATA_INVENTORY.md` and `docs/DOWNLOAD_INSTRUCTIONS.md`.

| Dataset / Task Group | Source (OSD/GLDS or Paper) | Representative Files Used | Track | Notes |
|---|---|---|---|---|
| Clinical (A1/A2) | OSD-569, OSD-575, OSD-656 | `clinical_cbc.csv`, `clinical_cmp.csv`, cytokine/urine panels | Open | I4 crew CBC/CMP + immune panels |
| cfRNA (B1/B2) | OSD-530, P08 (10.1038/s41467-023-41995-z) | `cfrna_3group_de_noleak.csv`, `cfrna_466drr.csv`, cluster labels | Open | JAXA CFE cfRNA (processed) |
| Proteomics (C1/C2) | OSD-571, P02 (10.1038/s41467-024-48841-w) | `proteomics_plasma_matrix.csv`, `proteomics_*_de_clean.csv` | Open | Plasma + EVP proteomics |
| Metabolomics (D1) | OSD-571, P02 (10.1038/s41467-024-48841-w) | `metabolomics_spaceflight_response.csv` | Open | Plasma metabolomics |
| Spatial (E1–E4) | OSD-574, P06 (10.1038/s41467-024-48625-2) | `gt_spatial_de_*.csv` | Open | Skin GeoMx spatial transcriptomics |
| Microbiome (F1–F5) | OSD-572, OSD-573, OSD-630, P05 (10.1038/s41564-024-01635-8) | `microbiome_*_taxonomy_cpm.csv`, `microbiome_*_pathways_cpm.csv` | Open | Human + environmental metagenomics |
| Multi-Modal (G1) | OSD-569, OSD-571 | `clinical_*`, `proteomics_*`, `metabolomics_*` | Open | PCA fusion of clinical + proteomics + metabolomics |
| Cross-Tissue (H1) | OSD-570, OSD-574 | `conserved_pbmc_to_skin.csv`, `gt_conserved_degs.csv` | Open | PBMC ↔ skin conservation |
| Cross-Mission (I1–I3) | P07 (10.1038/s41467-024-49289-8), P10 (10.1126/science.aau8650) | `cross_mission_*.csv`, hemoglobin/pathway tables | Open (summary only) | Twins data limited to aggregate/summary tables |

**Controlled-access note**: Any sequence-level or identifiable human data are excluded from the open track. Controlled-access materials require source-specific approvals (OSDR DAR, dbGaP/LSDA) and are not redistributed here.
