#!/usr/bin/env python3
"""
Preprocess spatial transcriptomics data for SpaceOmicsBench v2.

Input files (from OSDR OSD-574 / GLDS-566):
  - Spatially_Resolved_Transcriptomics_Processed_Data.xlsx: DESeq2 DE for 6 skin layers
  - 8 metagenomics TSV files: skin microbiome taxonomy + pathways (8 samples)

Output files:
  - data/processed/spatial_de_{layer}.csv: DESeq2 results per skin layer (6 files)
  - data/processed/spatial_skin_taxonomy_cpm.csv: skin metagenomics taxonomy (CPM)
  - data/processed/spatial_skin_pathways_cpm.csv: skin metagenomics pathways (CPM)
  - data/processed/spatial_metadata.csv: sample metadata

Samples: 4 crew × 2 timepoints (L-44, R+1) = 8 samples
Skin layers: All, OE (Outer Epidermis), IE (Inner Epidermis), OD (Outer Dermis),
             VA (Vasculature), OE+IE (Epidermis combined)
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
SPATIAL_DIR = BASE_DIR / "data" / "spatial_transcriptomics"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

SKIN_LAYERS = ["All", "OE", "IE", "OD", "VA", "OE+IE"]
LAYER_NAMES = {
    "All": "all_skin",
    "OE": "outer_epidermis",
    "IE": "inner_epidermis",
    "OD": "outer_dermis",
    "VA": "vasculature",
    "OE+IE": "epidermis",
}


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Spatial Transcriptomics Preprocessing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Process DESeq2 results ----
    print("\n--- Reading DESeq2 DE results ---")
    xlsx_path = SPATIAL_DIR / "GLDS-566_SpatialTranscriptomics_Skin_Biopsy_Spatially_Resolved_Transcriptomics_Processed_Data.xlsx"

    xl = pd.ExcelFile(xlsx_path)
    print(f"  Sheets: {xl.sheet_names}")
    assert set(SKIN_LAYERS).issubset(set(xl.sheet_names)), f"Missing sheets: {set(SKIN_LAYERS) - set(xl.sheet_names)}"

    de_results = {}
    for sheet in SKIN_LAYERS:
        # "All" sheet has 7 metadata rows (extra "Data Note 3"), others have 6
        skip = 7 if sheet == "All" else 6
        df = pd.read_excel(xlsx_path, sheet_name=sheet, skiprows=skip)
        # Verify header row was correctly identified
        assert "Gene" in df.columns, f"Sheet '{sheet}': expected 'Gene' column, got {list(df.columns[:4])}"
        # Rename columns
        df = df.rename(columns={
            "Gene": "gene",
            "pvalue": "pval",
            "padj": "adj_pval",
        })
        de_results[sheet] = df
        n_sig = (df["adj_pval"].dropna() < 0.05).sum()
        print(f"  {sheet} ({LAYER_NAMES[sheet]}): {len(df)} genes, {n_sig} significant (adj_pval<0.05)")

    # ---- 2. Process skin metagenomics ----
    print("\n--- Reading skin metagenomics ---")

    # Taxonomy (CPM normalized)
    tax_path = SPATIAL_DIR / "GLDS-566_GMetagenomics_Combined-contig-level-taxonomy-coverages-CPM_GLmetagenomics.tsv"
    tax_df = pd.read_csv(tax_path, sep="\t")
    print(f"  Taxonomy CPM: {tax_df.shape[0]} taxa × {tax_df.shape[1]} cols")

    # Identify sample columns (everything after taxonomy columns)
    tax_cols = ["taxid", "domain", "phylum", "class", "order", "family", "genus", "species"]
    sample_cols = [c for c in tax_df.columns if c not in tax_cols]
    print(f"  Sample columns: {sample_cols}")
    assert len(sample_cols) == 8, f"Expected 8 samples, got {len(sample_cols)}"

    # Pathways (CPM normalized)
    path_cpm = SPATIAL_DIR / "GLDS-566_GMetagenomics_Pathway-abundances-cpm_GLmetagenomics.tsv"
    pathway_df = pd.read_csv(path_cpm, sep="\t")
    print(f"  Pathways CPM: {pathway_df.shape[0]} pathways × {pathway_df.shape[1]} cols")

    # ---- 3. Build metadata ----
    print("\n--- Building metadata ---")
    meta_records = []
    for col in sample_cols:
        # Parse: C001_L-44_DEL or C001_R+1_DEL
        parts = col.split("_")
        crew = parts[0]
        timepoint = parts[1]
        meta_records.append({
            "sample_id": f"{crew}_{timepoint}",
            "sample_col": col,
            "crew": crew,
            "timepoint": timepoint,
            "timepoint_days": -44 if timepoint == "L-44" else 1,
            "phase": "pre_flight" if timepoint == "L-44" else "post_flight",
            "mission": "I4",
            "tissue": "skin",
        })
    meta_df = pd.DataFrame(meta_records).sort_values(["crew", "timepoint_days"]).reset_index(drop=True)
    print(f"  Metadata: {len(meta_df)} samples")

    # ---- 4. Sanity checks ----
    print("\n--- Sanity checks ---")

    # DE: check log2FC ranges for All layer
    all_de = de_results["All"]
    print(f"  All skin DE log2FC: [{all_de['log2FoldChange'].min():.2f}, {all_de['log2FoldChange'].max():.2f}]")

    # Taxonomy: verify all sample values are non-negative (CPM)
    for col in sample_cols:
        vals = tax_df[col].dropna()
        assert vals.min() >= 0, f"Negative CPM in {col}"
    print("  Taxonomy CPM: all values non-negative")

    # No duplicate genes in DE
    for sheet, df in de_results.items():
        dups = df["gene"].duplicated().sum()
        if dups > 0:
            print(f"  WARNING: {sheet} has {dups} duplicate gene names")
        else:
            print(f"  {sheet}: no duplicate genes")

    # ---- 5. Save ----
    print("\n--- Saving outputs ---")

    # Save DE results per layer
    for sheet, df in de_results.items():
        fname = f"spatial_de_{LAYER_NAMES[sheet]}.csv"
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False)
        size_kb = path.stat().st_size / 1024
        print(f"  Saved: {fname} ({size_kb:.1f} KB)")

    # Save metagenomics
    for fname, df in [
        ("spatial_skin_taxonomy_cpm.csv", tax_df),
        ("spatial_skin_pathways_cpm.csv", pathway_df),
        ("spatial_metadata.csv", meta_df),
    ]:
        path = OUTPUT_DIR / fname
        df.to_csv(path, index=False)
        size_kb = path.stat().st_size / 1024
        print(f"  Saved: {fname} ({size_kb:.1f} KB)")

    # ---- 6. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Spatial transcriptomics (GeoMx, NanoString):")
    for sheet in SKIN_LAYERS:
        n = len(de_results[sheet])
        sig = (de_results[sheet]["adj_pval"].dropna() < 0.05).sum()
        print(f"    {LAYER_NAMES[sheet]}: {n} genes, {sig} DE (adj_pval<0.05)")
    print(f"  Skin metagenomics: {tax_df.shape[0]} taxa, {pathway_df.shape[0]} pathways")
    print(f"  Samples: 8 (4 crew × 2 timepoints: L-44, R+1)")
    print(f"  Comparison: (R+1) vs (L-44), DESeq2")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
