#!/usr/bin/env python3
"""
Extract ground truth data from paper supplementary files for SpaceOmicsBench v2.

Source files and ground truth:
  P06 SuppData1: Full DESeq2 results (18,677 genes × 6 skin layers) → spatial DEG ground truth
  P07 Source Data: Hemoglobin/cfRNA cross-mission DE + globin genes → cross-mission ground truth
  P08 SuppData3: Coregulated gene cluster correlation matrix (482 genes)
  P08 SuppData5: Tissue-of-origin enrichment (34 terms)
  P03 SuppData15: Conserved DEGs across studies + GSEA pathways → cross-study signatures

Output files:
  - data/processed/gt_spatial_de_{layer}.csv: Full DESeq2 per layer (6 files, 18677 genes each)
  - data/processed/gt_hemoglobin_de.csv: cfRNA DE results (26,845 genes, Pre/Flight/Post)
  - data/processed/gt_hemoglobin_globin_genes.csv: Globin gene expression across phases
  - data/processed/gt_hemoglobin_i4_expression.csv: I4 hemoglobin gene expression (58 genes)
  - data/processed/gt_hemoglobin_crossmission.csv: Cross-mission gene data (1,950 rows)
  - data/processed/gt_cfrna_clusters.csv: 482-gene correlation matrix + cluster annotations
  - data/processed/gt_cfrna_tissue_enrichment.csv: Tissue-of-origin enrichment (34 terms)
  - data/processed/gt_conserved_degs.csv: Cross-study conserved DEGs (807 genes)
  - data/processed/gt_conserved_pathways_i4_pbmc.csv: I4 PBMC GSEA pathways
  - data/processed/gt_conserved_pathways_i4_skin.csv: I4 skin GSEA pathways
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
P03_DIR = BASE_DIR / "data" / "P03"
P06_DIR = BASE_DIR / "data" / "P06"
P07_DIR = BASE_DIR / "data" / "P07"
P08_DIR = BASE_DIR / "data" / "P08"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

SKIN_LAYERS = {
    "All": "all_skin",
    "OE": "outer_epidermis",
    "IE": "inner_epidermis",
    "OD": "outer_dermis",
    "VA": "vasculature",
    "OE+IE": "epidermis",
}


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- Ground Truth Extraction")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. P06: Full spatial DESeq2 (18,677 genes per layer) ----
    print("\n--- P06 SuppData1: Spatial DEGs (full genome) ---")
    p06_path = P06_DIR / "P06_SuppData1_DEGs.xlsx"

    for sheet, layer_name in SKIN_LAYERS.items():
        df = pd.read_excel(p06_path, sheet_name=sheet)
        # First column is gene name (Unnamed: 0)
        df = df.rename(columns={
            "Unnamed: 0": "gene",
            "pvalue": "pval",
            "padj": "adj_pval",
        })
        assert "gene" in df.columns, f"Missing gene column in {sheet}"
        assert "log2FoldChange" in df.columns, f"Missing log2FoldChange in {sheet}"
        assert len(df) == 18677, f"Expected 18677 genes in {sheet}, got {len(df)}"

        n_sig = (df["adj_pval"].dropna() < 0.05).sum()
        print(f"  {sheet} ({layer_name}): {len(df)} genes, {n_sig} DE (adj_pval<0.05)")

        fname = f"gt_spatial_de_{layer_name}.csv"
        df.to_csv(OUTPUT_DIR / fname, index=False)

    # Verify: P06 full results should be a superset of GLDS-566 processed data
    p06_all = pd.read_csv(OUTPUT_DIR / "gt_spatial_de_all_skin.csv")
    glds_all = pd.read_csv(OUTPUT_DIR / "spatial_de_all_skin.csv")
    overlap = set(p06_all["gene"]) & set(glds_all["gene"])
    print(f"  Overlap with GLDS-566 processed: {len(overlap)}/{len(glds_all)} genes")
    assert len(overlap) == len(glds_all), "GLDS-566 genes not all found in P06"

    # ---- 2. P07: Hemoglobin cross-mission data ----
    print("\n--- P07: Hemoglobin/cfRNA cross-mission data ---")
    p07_path = P07_DIR / "P07_source_data_hemoglobin.xlsx"

    # Figure4: Full cfRNA DE (26,845 genes, pairwise comparisons)
    fig4 = pd.read_excel(p07_path, sheet_name="Figure4")
    fig4 = fig4.rename(columns={"Unnamed: 0": "gene"}) if "Unnamed: 0" in fig4.columns else fig4
    fig4 = fig4.rename(columns={"Feature ID": "gene"}) if "Feature ID" in fig4.columns else fig4
    print(f"  Figure4 (DE): {len(fig4)} genes")
    assert len(fig4) == 26845, f"Expected 26845 genes, got {len(fig4)}"
    fig4.to_csv(OUTPUT_DIR / "gt_hemoglobin_de.csv", index=False)

    # Figure5: Globin gene expression across phases (59 genes)
    fig5 = pd.read_excel(p07_path, sheet_name="Figure5")
    fig5 = fig5.rename(columns={fig5.columns[0]: "gene"})
    print(f"  Figure5 (globin genes): {len(fig5)} genes")
    assert len(fig5) == 59, f"Expected 59 globin genes, got {len(fig5)}"
    fig5.to_csv(OUTPUT_DIR / "gt_hemoglobin_globin_genes.csv", index=False)

    # Figure6: I4 hemoglobin gene expression (58 genes × 4 timepoints)
    fig6 = pd.read_excel(p07_path, sheet_name="Figure6")
    fig6 = fig6.rename(columns={"Unnamed: 0": "gene"}) if "Unnamed: 0" in fig6.columns else fig6
    print(f"  Figure6 (I4 expression): {len(fig6)} genes × {fig6.shape[1]-1} timepoints")
    fig6.to_csv(OUTPUT_DIR / "gt_hemoglobin_i4_expression.csv", index=False)

    # Figure7: Cross-mission gene data (1,950 rows)
    fig7 = pd.read_excel(p07_path, sheet_name="Figure7")
    print(f"  Figure7 (cross-mission): {len(fig7)} rows, cols: {list(fig7.columns)}")
    fig7.to_csv(OUTPUT_DIR / "gt_hemoglobin_crossmission.csv", index=False)

    # ---- 3. P08: cfRNA cluster and tissue data ----
    print("\n--- P08: cfRNA cluster & tissue enrichment ---")

    # SuppData3: Coregulated gene cluster correlation matrix + cluster labels
    p08_raw = pd.read_excel(P08_DIR / "P08_SuppData3_coregulated_clusters.xlsx")
    p08_raw = p08_raw.rename(columns={"Unnamed: 0": "gene"})
    gene_cols = [c for c in p08_raw.columns if c != "gene"]

    # Separate cluster label rows ("Y"/NaN) from correlation rows (floats)
    # Cluster rows have "Y" values; correlation rows have float values
    cluster_rows = []
    corr_rows = []
    for idx, row in p08_raw.iterrows():
        vals = row[gene_cols]
        has_y = (vals == "Y").any()
        if has_y:
            cluster_rows.append(idx)
        else:
            corr_rows.append(idx)

    print(f"  Cluster label rows: {len(cluster_rows)}, Correlation rows: {len(corr_rows)}")

    # 1. Cluster membership matrix: gene × cluster (binary)
    cluster_labels = p08_raw.loc[cluster_rows, ["gene"] + gene_cols].copy()
    for col in gene_cols:
        cluster_labels[col] = (cluster_labels[col] == "Y").astype(int)
    # Transpose: rows=genes, cols=clusters
    cluster_labels_t = cluster_labels.set_index("gene").T
    cluster_labels_t.index.name = "gene"
    cluster_labels_t = cluster_labels_t.reset_index()
    n_assigned = (cluster_labels_t[cluster_labels_t.columns[1:]].sum(axis=1) > 0).sum()
    print(f"  Cluster labels: {len(cluster_labels_t)} genes × {len(cluster_rows)} clusters")
    print(f"  Genes with ≥1 cluster: {n_assigned}/{len(cluster_labels_t)}")
    cluster_labels_t.to_csv(OUTPUT_DIR / "gt_cfrna_cluster_labels.csv", index=False)

    # 2. Gene-gene correlation matrix (466 × 466)
    corr_matrix = p08_raw.loc[corr_rows].copy()
    for col in gene_cols:
        corr_matrix[col] = pd.to_numeric(corr_matrix[col], errors="coerce")
    print(f"  Correlation matrix: {len(corr_matrix)} genes × {len(gene_cols)} genes")
    corr_matrix.to_csv(OUTPUT_DIR / "gt_cfrna_correlation.csv", index=False)

    # 3. Keep combined file for backward compatibility
    p08_clusters = p08_raw.copy()
    print(f"  Combined (legacy): {p08_clusters.shape[0]} rows × {p08_clusters.shape[1]} cols")
    p08_clusters.to_csv(OUTPUT_DIR / "gt_cfrna_clusters.csv", index=False)

    # SuppData5: Tissue-of-origin enrichment
    p08_tissue = pd.read_excel(P08_DIR / "P08_SuppData5_tissue_specificity.xlsx")
    print(f"  Tissue enrichment: {len(p08_tissue)} terms")
    print(f"    Top tissues: {', '.join(p08_tissue['Term'].head(5).tolist())}")
    p08_tissue.to_csv(OUTPUT_DIR / "gt_cfrna_tissue_enrichment.csv", index=False)

    # ---- 4. P03: Conserved signatures across studies ----
    print("\n--- P03 SuppData15: Conserved signatures ---")
    p03_path = P03_DIR / "Supplementary.Data" / "Supplementary Data 15.xlsx"
    xl15 = pd.ExcelFile(p03_path)
    print(f"  Sheets: {xl15.sheet_names}")

    # Sheet 1: DEGs across studies (row 0 is title, row 1 is header)
    degs = pd.read_excel(p03_path, sheet_name="DEGs", skiprows=1)
    if degs.columns[0].startswith("Extended"):
        degs = pd.read_excel(p03_path, sheet_name="DEGs", skiprows=2)
    # Drop unnamed index column
    if "Unnamed: 0" in degs.columns:
        degs = degs.drop(columns=["Unnamed: 0"])
    print(f"  Conserved DEGs: {len(degs)} rows, cols: {list(degs.columns[:8])}")
    degs.to_csv(OUTPUT_DIR / "gt_conserved_degs.csv", index=False)

    # Pathway sheets
    for sheet_name in xl15.sheet_names:
        if sheet_name.startswith("pathways"):
            df = pd.read_excel(p03_path, sheet_name=sheet_name)
            # Drop unnamed index column if present
            if "Unnamed: 0" in df.columns:
                df = df.drop(columns=["Unnamed: 0"])
            n_sig = (df["padj"].dropna() < 0.05).sum() if "padj" in df.columns else "N/A"
            print(f"  {sheet_name}: {len(df)} pathways, {n_sig} significant")
            safe_name = sheet_name.replace(".", "_")
            df.to_csv(OUTPUT_DIR / f"gt_conserved_{safe_name}.csv", index=False)

    # ---- 5. Sanity checks ----
    print("\n--- Sanity checks ---")

    # P06: Full DE should have more genes than GLDS-566 filtered
    print(f"  P06 full genome: 18,677 genes vs GLDS-566 filtered: {len(glds_all)} genes")

    # P07 Figure4: Check fold change columns are numeric
    fc_cols = [c for c in fig4.columns if "Fold Change" in c]
    for c in fc_cols:
        vals = pd.to_numeric(fig4[c], errors="coerce").dropna()
        print(f"  P07 {c}: range [{vals.min():.2f}, {vals.max():.2f}]")

    # P07 Figure5: globin gene values should be positive (normalized means)
    num_cols = [c for c in fig5.columns if c != "gene"]
    vals = fig5[num_cols].values.flatten()
    vals = pd.to_numeric(pd.Series(vals), errors="coerce").dropna()
    if len(vals) > 0:
        print(f"  P07 globin expression range: [{vals.min():.1f}, {vals.max():.1f}]")

    # P08 clusters: correlation matrix should be symmetric-ish
    gene_cols = [c for c in p08_clusters.columns if c != "gene"]
    if len(gene_cols) > 0:
        n_genes = len(gene_cols)
        print(f"  P08 cluster matrix: {p08_clusters.shape[0]} rows × {n_genes} gene columns")

    # P03: conserved DEGs should have meaningful columns
    print(f"  P03 conserved DEGs columns: {list(degs.columns)}")

    # ---- 6. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Ground truth files extracted:")
    print(f"    P06 spatial DE: 6 layers × 18,677 genes (full genome DESeq2)")
    print(f"    P07 hemoglobin DE: {len(fig4)} genes (3 pairwise comparisons)")
    print(f"    P07 globin genes: {len(fig5)} genes with expression values")
    print(f"    P07 I4 expression: {len(fig6)} genes × {fig6.shape[1]-1} timepoints")
    print(f"    P07 cross-mission: {len(fig7)} gene measurements")
    print(f"    P08 clusters: {p08_clusters.shape[0]} genes correlation matrix")
    print(f"    P08 tissue: {len(p08_tissue)} enrichment terms")
    print(f"    P03 conserved DEGs: {len(degs)} entries")
    for sheet_name in xl15.sheet_names:
        if sheet_name.startswith("pathways"):
            safe_name = sheet_name.replace(".", "_")
            df = pd.read_csv(OUTPUT_DIR / f"gt_conserved_{safe_name}.csv")
            print(f"    P03 {sheet_name}: {len(df)} pathways")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
