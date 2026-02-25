#!/usr/bin/env python3
"""
Preprocess cfRNA data for SpaceOmicsBench v2.

Reads JAXA CFE cfRNA processed data from data/transcriptomics/cfrna/ (OSDR OSD-530).
NOTE: These files contain group-level summary statistics (means, SEM, ANOVA, EDGE tests),
NOT per-sample expression counts. The JAXA study processed 64 cfRNA samples from 6 astronauts
into group-level aggregates.

Input files (7 xlsx from GLDS-530):
  - 3-group normalized (26,845 genes): ANOVA + EDGE Pre/Flight/Post
  - 466 DRR genes: Filtered significant genes (FDR<0.05, FC>2, diff>50)
  - 9-group pairwise (26,845 genes): EDGE tests Pre vs Flight1-4, Pre vs Post1-4
  - 9-group SEM: Group means + SEM for Pre, Flight1-4, Post1-4
  - 11-group SEM: Group means + SEM for Pre1-3, Flight1-4, Post1-4
  - Input vs IP CD36 enrichment (22,475 genes)
  - 3-group raw total counts summary (57,773 genes)

Output files:
  - data/processed/cfrna_3group_de.csv: Full 3-group DE results (26,845 genes)
  - data/processed/cfrna_466drr.csv: 466 DRR gene list with statistics
  - data/processed/cfrna_9group_pairwise.csv: 9-group pairwise comparisons
  - data/processed/cfrna_11group_timecourse.csv: 11-group means + SEM
  - data/processed/cfrna_cd36_enrichment.csv: Input vs IP comparison
"""

import re
import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
CFRNA_DIR = BASE_DIR / "data" / "transcriptomics" / "cfrna"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

PYTHON = sys.executable


def clean_column_name(col: str) -> str:
    """Clean column names for CSV-friendly output."""
    col = col.strip()
    col = re.sub(r"\s+", "_", col)
    col = re.sub(r"[(),]", "", col)
    col = re.sub(r"[-:]", "_", col)
    col = re.sub(r"_+", "_", col)
    col = col.strip("_").lower()
    return col


def process_3group_de(cfrna_dir: Path) -> pd.DataFrame:
    """Process 3-group normalized DE results."""
    f = cfrna_dir / "GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized.xlsx"
    df = pd.read_excel(f)

    # Clean column names
    df.columns = [clean_column_name(c) for c in df.columns]

    # Rename for clarity
    # Rename based on actual cleaned column names
    rename = {
        "feature_id": "gene",
        "anova_normalized_values_test_statistic": "anova_F",
        "anova_normalized_values_p_value": "anova_pval",
        "anova_normalized_values_fdr_p_value_correction": "anova_fdr",
        "anova_normalized_values_max_difference": "anova_max_diff",
        "anova_normalized_values_max_fold_change": "anova_max_fc",
        "edge_test_pre_vs_flight_tagwise_dispersions_p_value": "edge_pre_vs_flight_pval",
        "edge_test_pre_vs_flight_tagwise_dispersions_fold_change": "edge_pre_vs_flight_fc",
        "edge_test_pre_vs_flight_tagwise_dispersions_weighted_difference": "edge_pre_vs_flight_diff",
        "edge_test_pre_vs_flight_tagwise_dispersions_fdr_p_value_correction": "edge_pre_vs_flight_fdr",
        "edge_test_pre_vs_post_tagwise_dispersions_p_value": "edge_pre_vs_post_pval",
        "edge_test_pre_vs_post_tagwise_dispersions_fold_change": "edge_pre_vs_post_fc",
        "edge_test_pre_vs_post_tagwise_dispersions_weighted_difference": "edge_pre_vs_post_diff",
        "edge_test_pre_vs_post_tagwise_dispersions_fdr_p_value_correction": "edge_pre_vs_post_fdr",
        "edge_test_flight_vs_post_tagwise_dispersions_p_value": "edge_flight_vs_post_pval",
        "edge_test_flight_vs_post_tagwise_dispersions_fold_change": "edge_flight_vs_post_fc",
        "edge_test_flight_vs_post_tagwise_dispersions_weighted_difference": "edge_flight_vs_post_diff",
        "edge_test_flight_vs_post_tagwise_dispersions_fdr_p_value_correction": "edge_flight_vs_post_fdr",
        "pre_normalized_means": "pre_norm_mean",
        "pre_means": "pre_raw_mean",
        "pre_transformed_means": "pre_transformed_mean",
        "flight_normalized_means": "flight_norm_mean",
        "flight_means": "flight_raw_mean",
        "flight_transformed_means": "flight_transformed_mean",
        "post_normalized_means": "post_norm_mean",
        "post_means": "post_raw_mean",
        "post_transformed_means": "post_transformed_mean",
    }

    # Apply rename for columns that match
    col_map = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=col_map)

    # Select key columns in a meaningful order
    keep_cols = ["gene"]
    # ANOVA results
    keep_cols += [c for c in df.columns if c.startswith("anova_")]
    # EDGE pairwise results
    keep_cols += [c for c in df.columns if c.startswith("edge_")]
    # Group means
    keep_cols += [c for c in df.columns if c.endswith("_mean")]
    # Experiment-level stats
    keep_cols += [c for c in df.columns if c.startswith("experiment_")]

    # Only keep columns that exist
    keep_cols = [c for c in keep_cols if c in df.columns]
    # Remove duplicates while preserving order
    seen = set()
    keep_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]

    df = df[keep_cols]
    return df


def process_466drr(cfrna_dir: Path) -> pd.DataFrame:
    """Process 466 DRR gene list."""
    f = cfrna_dir / "GLDS-530_rna-seq_TGB_050_64samples_3group_totalcount_all0removed_scalingnormalized_ANOVA_FDRpval005_2x_50difference_466genes.xlsx"
    df = pd.read_excel(f)

    df.columns = [clean_column_name(c) for c in df.columns]

    rename = {
        "feature_id": "gene",
        "anova_normalized_values_test_statistic": "anova_F",
        "anova_normalized_values_p_value": "anova_pval",
        "anova_normalized_values_fdr_p_value_correction": "anova_fdr",
        "anova_normalized_values_max_difference": "anova_max_diff",
        "anova_normalized_values_max_fold_change": "anova_max_fc",
        "pre_normalized_means": "pre_norm_mean",
        "pre_means": "pre_raw_mean",
        "pre_transformed_means": "pre_transformed_mean",
        "flight_normalized_means": "flight_norm_mean",
        "flight_means": "flight_raw_mean",
        "flight_transformed_means": "flight_transformed_mean",
        "post_normalized_means": "post_norm_mean",
        "post_means": "post_raw_mean",
        "post_transformed_means": "post_transformed_mean",
    }

    col_map = {k: v for k, v in rename.items() if k in df.columns}
    df = df.rename(columns=col_map)

    return df


def process_9group_pairwise(cfrna_dir: Path) -> pd.DataFrame:
    """Process 9-group pairwise DE results."""
    f = cfrna_dir / "GLDS-530_rna-seq_TGB_050_1_2_64samples_9group_totalcount_all0removed_scalingnormalized_pairwise_analysis_included.xlsx"
    df = pd.read_excel(f)
    df.columns = [clean_column_name(c) for c in df.columns]

    # Rename Feature ID -> gene
    df = df.rename(columns={"feature_id": "gene"})

    # Clean up EDGE test column names for readability
    col_map = {}
    for col in df.columns:
        if col.startswith("edge_test"):
            # Extract comparison and metric
            new = col.replace("edge_test__", "edge_")
            new = new.replace("__tagwise_dispersions___", "_")
            new = new.replace("_p_value", "_pval")
            new = new.replace("_fdr_p_value_correction", "_fdr")
            new = new.replace("_fold_change", "_fc")
            new = new.replace("_weighted_difference", "_diff")
            col_map[col] = new
        elif "normalized_means" in col:
            # Simplify group mean names: "pre___normalized_means" -> "pre_norm_mean"
            new = col.replace("___normalized_means", "_norm_mean")
            col_map[col] = new

    df = df.rename(columns=col_map)
    return df


def process_11group_timecourse(cfrna_dir: Path) -> pd.DataFrame:
    """Process 11-group time-course profiles (means + SEM)."""
    f = cfrna_dir / "GLDS-530_rna-seq_TGB_050_1_2_64samples_11group_totalcount_all0removed_scalingnormalized_SEM.xlsx"
    df = pd.read_excel(f)
    df.columns = [clean_column_name(c) for c in df.columns]

    # Rename Feature ID -> gene
    df = df.rename(columns={"feature_id": "gene"})

    # Clean up column names
    col_map = {}
    for col in df.columns:
        if col == "gene":
            continue
        new = col.replace("___normalized_means", "_mean")
        new = new.replace("___normalized_sem_by_excel", "_sem")
        col_map[col] = new
    df = df.rename(columns=col_map)

    return df


def process_cd36_enrichment(cfrna_dir: Path) -> pd.DataFrame:
    """Process Input vs IP CD36 enrichment results."""
    f = cfrna_dir / "GLDS-530_rna-seq_TGB_063_Input_vs_IP_totalcount_all0removed_scalingnormalized.xlsx"
    df = pd.read_excel(f)
    df.columns = [clean_column_name(c) for c in df.columns]

    df = df.rename(columns={"feature_id": "gene"})

    # Clean EDGE test columns
    col_map = {}
    for col in df.columns:
        if "edge_test" in col:
            new = col.replace("edge_test__", "edge_")
            new = new.replace("__tagwise_dispersions___", "_")
            new = new.replace("_p_value", "_pval")
            new = new.replace("_fdr_p_value_correction", "_fdr")
            new = new.replace("_fold_change", "_fc")
            new = new.replace("_weighted_difference", "_diff")
            col_map[col] = new
        elif "means" in col:
            new = col.replace("___transformed_means", "_transformed_mean")
            new = new.replace("___normalized_means", "_norm_mean")
            col_map[col] = new
    df = df.rename(columns=col_map)

    return df


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 -- cfRNA Data Preprocessing")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Process each file ----
    print("\n--- Processing 3-group DE results ---")
    de_3group = process_3group_de(CFRNA_DIR)
    print(f"  Shape: {de_3group.shape[0]} genes × {de_3group.shape[1]} cols")
    assert de_3group.shape[0] == 26845, f"Expected 26845 genes, got {de_3group.shape[0]}"
    assert de_3group["gene"].nunique() == 26845, "Duplicate gene IDs found"

    print("\n--- Processing 466 DRR genes ---")
    drr_466 = process_466drr(CFRNA_DIR)
    print(f"  Shape: {drr_466.shape[0]} genes × {drr_466.shape[1]} cols")
    assert drr_466.shape[0] == 466, f"Expected 466 genes, got {drr_466.shape[0]}"

    # Verify all 466 DRR genes are in the full 3-group set
    overlap = set(drr_466["gene"]) & set(de_3group["gene"])
    assert len(overlap) == 466, f"Only {len(overlap)}/466 DRR genes found in full set"
    print("  All 466 DRR genes confirmed present in full 26,845-gene set")

    print("\n--- Processing 9-group pairwise ---")
    pairwise_9group = process_9group_pairwise(CFRNA_DIR)
    print(f"  Shape: {pairwise_9group.shape[0]} genes × {pairwise_9group.shape[1]} cols")
    assert pairwise_9group.shape[0] == 26845

    print("\n--- Processing 11-group time-course ---")
    timecourse_11 = process_11group_timecourse(CFRNA_DIR)
    print(f"  Shape: {timecourse_11.shape[0]} genes × {timecourse_11.shape[1]} cols")
    assert timecourse_11.shape[0] == 26845
    # Verify column structure: should have gene + 11 means + 11 SEMs = 23 cols
    assert timecourse_11.shape[1] == 23, f"Expected 23 cols, got {timecourse_11.shape[1]}"

    print("\n--- Processing CD36 enrichment ---")
    cd36 = process_cd36_enrichment(CFRNA_DIR)
    print(f"  Shape: {cd36.shape[0]} genes × {cd36.shape[1]} cols")
    assert cd36.shape[0] == 22475

    # ---- 2. Sanity checks ----
    print("\n--- Sanity checks ---")

    # 466 DRR genes should all have FDR < 0.05
    if "anova_fdr" in drr_466.columns:
        max_fdr = drr_466["anova_fdr"].max()
        print(f"  466 DRR max FDR: {max_fdr:.6f} (should be < 0.05)")
        assert max_fdr < 0.05, f"DRR gene with FDR={max_fdr} >= 0.05"

    # 466 DRR genes should all have |fold change| > 2 (can be negative for downregulated)
    if "anova_max_fc" in drr_466.columns:
        min_abs_fc = drr_466["anova_max_fc"].abs().min()
        print(f"  466 DRR min |fold change|: {min_abs_fc:.2f} (should be > 2.0)")
        assert min_abs_fc >= 2.0, f"DRR gene with |FC|={min_abs_fc} < 2.0"
        n_up = (drr_466["anova_max_fc"] > 0).sum()
        n_down = (drr_466["anova_max_fc"] < 0).sum()
        print(f"  466 DRR: {n_up} upregulated, {n_down} downregulated")

    # 3-group DE should have genes with p-values ranging from very small to ~1
    max_pval = de_3group["anova_pval"].max()
    min_pval = de_3group["anova_pval"].min()
    print(f"  3-group ANOVA p-value range: {min_pval:.2e} to {max_pval:.4f}")

    # 11-group time-course: check means are positive (expression values)
    mean_cols = [c for c in timecourse_11.columns if c.endswith("_mean")]
    for col in mean_cols[:3]:
        vals = timecourse_11[col].dropna()
        print(f"  11-group {col}: min={vals.min():.2f}, max={vals.max():.2f}, median={vals.median():.2f}")

    # CD36: verify Input and IP columns exist and have reasonable values
    print(f"  CD36 enrichment columns: {list(cd36.columns)}")

    # Cross-check: 3-group and 9-group should have same genes
    genes_3g = set(de_3group["gene"])
    genes_9g = set(pairwise_9group["gene"])
    assert genes_3g == genes_9g, f"Gene sets differ: 3-group has {len(genes_3g)}, 9-group has {len(genes_9g)}"
    print("  3-group and 9-group gene sets match perfectly")

    # ---- 3. Save outputs ----
    print("\n--- Saving outputs ---")

    outputs = {
        "cfrna_3group_de.csv": de_3group,
        "cfrna_466drr.csv": drr_466,
        "cfrna_9group_pairwise.csv": pairwise_9group,
        "cfrna_11group_timecourse.csv": timecourse_11,
        "cfrna_cd36_enrichment.csv": cd36,
    }

    for filename, df in outputs.items():
        outpath = OUTPUT_DIR / filename
        df.to_csv(outpath, index=False)
        size_kb = outpath.stat().st_size / 1024
        print(f"  Saved: {filename} ({size_kb:.1f} KB)")

    # ---- 4. Summary ----
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("  Mission: JAXA CFE (Cell-Free Epigenome)")
    print("  Subjects: 6 astronauts, 64 cfRNA samples total")
    print("  Design: Pre-flight / In-flight (4 bins) / Post-flight (4 bins)")
    print()
    print("  3-group DE:        26,845 genes (Pre/Flight/Post ANOVA + EDGE)")
    print("  466 DRR genes:     Filtered significant (FDR<0.05, FC>2, diff>50)")
    print("  9-group pairwise:  26,845 genes (Pre vs Flight1-4, Pre vs Post1-4)")
    print("  11-group profile:  26,845 genes (Pre1-3/Flight1-4/Post1-4 means+SEM)")
    print("  CD36 enrichment:   22,475 genes (Input vs IP)")
    print()
    print("  NOTE: All data is group-level (means/SEM), not per-sample counts.")
    print("  Benchmark tasks B1-B4 use these aggregate statistics.")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
