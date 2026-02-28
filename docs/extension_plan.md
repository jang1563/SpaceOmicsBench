# SpaceOmicsBench v2 — Extension Plan: New Mission Data Integration

**Date:** 2026-02-27
**Status:** All 3 modules implemented. Module 2 in v2_public; Modules 1 & 3 in missions/.

---

## Motivation

The core benchmark evaluates AI models on fixed tasks from I4/Twins/JAXA data.
This extension allows users to bring new spaceflight omics data and compare it
against the benchmark's curated reference signatures for biological interpretation.

---

## 3-Module Architecture

Modules 1 and 3 live in the **separate `missions/` directory** (sibling to v2_public),
keeping v2_public frozen and citable. Module 2 is in v2_public/evaluation/.

```
SpaceOmicsBench/
├── v2_public/                          ← FROZEN core benchmark
│   └── evaluation/
│       └── signature_query.py          # Module 2 (DONE): DE → biological similarity
└── missions/                           ← Extension benchmark (new missions)
    ├── schema/task_schema.json
    └── scripts/
        ├── ingest_osdr.py              # Module 1 (DONE): OSDR → benchmark CSV
        └── add_mission_task.py         # Module 3 (DONE): Register new task + splits
```

---

## Module 2: `evaluation/signature_query.py` (Highest Priority)

### Purpose
Given a user's differential expression (DE) result CSV, compute overlap and
direction concordance against SpaceOmicsBench reference signatures.

### Input Format
CSV file with at minimum:
- `gene` column (HGNC symbol)
- fold-change column: `log2FC`, `logFC`, `log2FoldChange`, or `FC`
- adjusted p-value column: `padj`, `adj_pval`, `FDR`, `adjusted_p_value`

The full DE result (all tested genes, not just significant ones) is preferred
for accurate hypergeometric background.

### Reference Signatures

| ID | Description | Source | N genes | FC available |
|----|-------------|---------|---------|-------------|
| I4_cfRNA_DRR | I4 cfRNA spaceflight-responsive genes (JAXA IHSP) | cfrna_466drr.csv + cfrna_3group_de_noleak.csv | 466 | Yes (edge_pre_vs_flight_fc, log2 scale) |
| I4_Plasma_Proteomics | I4 plasma DE proteins (adj_pval<0.05) | proteomics_plasma_de_clean.csv | 57 | Yes (logFC) |
| I4_PBMC_CD4T | I4 CD4+ T cell DE genes | gt_conserved_degs.csv | 736 | Yes |
| I4_PBMC_CD8T | I4 CD8+ T cell DE genes | gt_conserved_degs.csv | 661 | Yes |
| I4_PBMC_CD14Mono | I4 CD14+ Monocyte DE genes | gt_conserved_degs.csv | 709 | Yes |
| GeneLab_Mouse | GeneLab rodent spaceflight DE (conserved with I4) | gt_conserved_degs.csv | 134 | Yes |
| JAXA_cfRNA | JAXA cfRNA DE genes | gt_conserved_degs.csv | 36 | Yes |
| CrossMission_Conserved | Cross-mission conserved spaceflight genes | cross_mission_gene_de.csv | 814 | Partial (abs only) |

### Metrics Per Signature

1. **n_overlap**: Count of user DE genes in signature
2. **overlap_coef**: n_overlap / min(|user_DE|, |signature|)
3. **jaccard**: n_overlap / |user_DE ∪ signature|
4. **hypergeom_p**: Hypergeometric test (enrichment in user DE vs background)
5. **hypergeom_q**: FDR-corrected p-value (Benjamini-Hochberg)
6. **direction_concordance**: Fraction of overlapping genes with same sign of log2FC
7. **spearman_r**: Spearman correlation of log2FC for overlapping genes
8. **spearman_p**: p-value for Spearman correlation

### Caveats (reported in output)
- I4 has only 4 crew members → cfRNA/proteomics signatures may not generalize
- JAXA cfRNA: group-level DE, not individual-level
- Direction concordance depends on comparable biological comparison (pre vs. post flight)
- Gene symbol matching: HGNC symbols required; no alias resolution implemented

### Output Files
- `signature_query_results.json`: structured comparison data
- `signature_query_report.md`: human-readable markdown report

---

## Module 1: `missions/scripts/ingest_osdr.py` (DONE)

Convert GeneLab standard DE output (`differential_expression_GLbulkRNAseq*.csv`)
to benchmark-compatible format. Key GeneLab columns: Gene.ID, log2FC, adj.P.Val, baseMean.

- Queries OSDR Biodata REST API for file listing
- Auto-detects GeneLab standard DE files
- Downloads via GEODE endpoint (no auth needed for open-access)
- Maps column names to benchmark standard (gene, log2FC, padj, ...)
- Writes provenance sidecar JSON

SOMA dataset (2024, n=2911 samples) uses GeneLab format for Axiom/Polaris missions.
Use `--dry-run` to check available files before committing to download.

---

## Module 3: `missions/scripts/add_mission_task.py` (DONE)

Register a new mission dataset as a benchmark task (Category I pattern: cross-mission prediction).
- Load processed benchmark CSV (output of Module 1 or manual)
- Validate required columns + integer label column
- Build task JSON conforming to missions/schema/task_schema.json (v2-compatible)
- Generate stratified 80/20 split files (5 reps, same logic as v2_public)
- Validate JSON against schema (requires jsonschema)
- Optionally run Module 2 (signature_query.py) for biological comparison

---

## Research Notes

- **SOMA** (Jun 2024): 2,911 samples, I4+Twins+JAXA+Axiom-1/2+Polaris. Uses GeneLab DESeq2 format.
- **GENESTAR**: Biospecimen collection standard for commercial spaceflight (Axiom-2/3).
- **Cross-mission metrics**: Jaccard overlap, direction concordance, Spearman correlation standard in literature.
- **Category I tasks** (I1-I3): Cross-mission prediction pattern already in benchmark — new modules extend this.
