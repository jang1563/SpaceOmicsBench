#!/usr/bin/env python3
"""
Preprocess cross-mission (I-series) task data.

I1: Hemoglobin Gene DE Prediction
    - Features: 3 fold-change values from Twins transcriptome (gt_hemoglobin_de.csv)
    - Target: whether gene is in hemoglobin/erythropoiesis gene set
    - N=26,845, positives=57

I2: Cross-Mission Pathway Conservation
    - Features: aggregated I4 PBMC pathway enrichment stats
    - Target: whether pathway is also significant in Twins
    - N=452, positives=146

I3: Cross-Mission Gene DE Conservation
    - Features: aggregated Twins blood cell DE features (P10 Supp Table S2)
    - Target: whether gene is also DE in I4 cfRNA (anova FDR < 0.05)
    - N=15,540, positives=814
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data"
SPLITS = ROOT / "splits"
TASKS = ROOT / "tasks"

SEED = 42


# ---------------------------------------------------------------------------
# I1: Hemoglobin Gene DE Prediction
# ---------------------------------------------------------------------------
def preprocess_i1():
    """Build I1 dataset: predict hemoglobin pathway membership from DE fold-changes."""
    de = pd.read_csv(DATA / "gt_hemoglobin_de.csv")
    globin = pd.read_csv(DATA / "gt_hemoglobin_globin_genes.csv")

    globin_genes = set(globin["gene"].values)

    # Features: 3 fold-change columns (exclude p-values to prevent leakage)
    fc_cols = [c for c in de.columns if "Fold Change" in c]
    assert len(fc_cols) == 3, f"Expected 3 FC columns, got {fc_cols}"

    # Label
    de["label"] = de["gene"].isin(globin_genes).astype(int)

    # Output: gene + features + label
    out = de[["gene"] + fc_cols + ["label"]].copy()
    out.columns = ["gene", "fc_pre_vs_flight", "fc_pre_vs_post", "fc_flight_vs_post", "label"]

    pos = out["label"].sum()
    print(f"I1: N={len(out)}, positives={pos} ({pos/len(out)*100:.2f}%)")
    out.to_csv(DATA / "cross_mission_hemoglobin_de.csv", index=False)

    # Generate split
    generate_feature_split(out, "feature_split_I1.json", label_col="label")
    return out


# ---------------------------------------------------------------------------
# I2: Cross-Mission Pathway Conservation
# ---------------------------------------------------------------------------
def preprocess_i2():
    """Build I2 dataset: predict Twins pathway conservation from I4 PBMC stats."""
    i4 = pd.read_csv(DATA / "gt_conserved_pathways_i4_pbmc.csv")
    twins = pd.read_csv(DATA / "gt_conserved_pathways_NASA_Twins.csv")

    twins_pathways = set(twins["Pathway"].unique())

    # Aggregate I4 PBMC per pathway
    agg = i4.groupby("pathway").agg(
        mean_NES=("NES", "mean"),
        std_NES=("NES", "std"),
        mean_ES=("ES", "mean"),
        mean_padj=("padj", "mean"),
        min_padj=("padj", "min"),
        n_celltypes=("celltype", "nunique"),
        mean_size=("size", "mean"),
    ).reset_index()

    # Direction consistency: fraction of cell types with same NES sign as majority
    def direction_consistency(group):
        signs = np.sign(group["NES"].values)
        if len(signs) == 0:
            return 0.5
        majority_sign = 1 if (signs > 0).sum() >= (signs < 0).sum() else -1
        return (signs == majority_sign).mean()

    dir_cons = i4.groupby("pathway").apply(direction_consistency).reset_index()
    dir_cons.columns = ["pathway", "direction_consistency"]
    agg = agg.merge(dir_cons, on="pathway")

    # Fill NaN std with 0 (pathways in only 1 cell type)
    agg["std_NES"] = agg["std_NES"].fillna(0)

    # Label: conserved in Twins
    agg["label"] = agg["pathway"].isin(twins_pathways).astype(int)

    # Feature columns
    feat_cols = ["mean_NES", "std_NES", "mean_ES", "mean_padj", "min_padj",
                 "n_celltypes", "mean_size", "direction_consistency"]

    out = agg[["pathway"] + feat_cols + ["label"]].copy()

    pos = out["label"].sum()
    print(f"I2: N={len(out)}, positives={pos} ({pos/len(out)*100:.1f}%)")
    out.to_csv(DATA / "cross_mission_pathway_features.csv", index=False)

    # Generate split
    generate_feature_split(out, "feature_split_I2.json", label_col="label")
    return out


# ---------------------------------------------------------------------------
# I3: Cross-Mission Gene DE Conservation
# ---------------------------------------------------------------------------
def preprocess_i3():
    """Build I3 dataset: predict I4 cfRNA DE from Twins blood cell DE features."""
    # Load Twins P10 S2 DEG data
    p10_path = RAW / "P10" / "P10_SuppTable_S2.xlsx"
    print("  Loading P10 S2 (this may take a moment)...")
    p10 = pd.read_excel(p10_path, sheet_name="All DEGs", header=1)

    # Aggregate per gene from Twins data
    print("  Aggregating Twins per-gene features...")
    twins_agg = p10.groupby("Gene").agg(
        mean_abs_log2fc=("log2 Fold Change", lambda x: np.abs(x).mean()),
        max_abs_log2fc=("log2 Fold Change", lambda x: np.abs(x).max()),
        mean_base_expr=("base mean expression", "mean"),
        mean_lfcSE=("log-fold change standard  error", "mean"),
        mean_abs_wald=("wald statistic", lambda x: np.abs(x).mean()),
        n_cell_types=("CellType", "nunique"),
        n_contrasts=("Coefficient", "nunique"),
        n_total_deg_entries=("Gene", "count"),
    ).reset_index()
    twins_agg.rename(columns={"Gene": "gene"}, inplace=True)

    # Direction consistency: fraction of entries with same sign as majority
    def gene_dir_consistency(group):
        signs = np.sign(group["log2 Fold Change"].values)
        signs = signs[signs != 0]  # exclude zero
        if len(signs) == 0:
            return 0.5
        majority_sign = 1 if (signs > 0).sum() >= (signs < 0).sum() else -1
        return (signs == majority_sign).mean()

    dir_cons = p10.groupby("Gene").apply(gene_dir_consistency).reset_index()
    dir_cons.columns = ["gene", "direction_consistency"]
    twins_agg = twins_agg.merge(dir_cons, on="gene")

    # Load I4 cfRNA DE for labels
    cfrna = pd.read_csv(DATA / "cfrna_3group_de.csv")
    cfrna["label"] = (cfrna["anova_fdr"] < 0.05).astype(int)

    # Merge on shared gene universe
    merged = twins_agg.merge(cfrna[["gene", "label"]], on="gene", how="inner")

    feat_cols = ["mean_abs_log2fc", "max_abs_log2fc", "mean_base_expr",
                 "mean_lfcSE", "mean_abs_wald", "n_cell_types",
                 "n_contrasts", "n_total_deg_entries", "direction_consistency"]

    out = merged[["gene"] + feat_cols + ["label"]].copy()

    pos = out["label"].sum()
    print(f"I3: N={len(out)}, positives={pos} ({pos/len(out)*100:.1f}%)")
    out.to_csv(DATA / "cross_mission_gene_de.csv", index=False)

    # Generate split
    generate_feature_split(out, "feature_split_I3.json", label_col="label")
    return out


# ---------------------------------------------------------------------------
# Split Generation
# ---------------------------------------------------------------------------
def generate_feature_split(df, filename, label_col="label", n_reps=5,
                           train_frac=0.8):
    """Generate stratified 80/20 feature splits (5 repetitions).

    Output format matches existing splits: flat list of dicts with
    'train_indices' and 'test_indices' keys.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    labels = df[label_col].values
    indices = np.arange(len(df))

    splitter = StratifiedShuffleSplit(
        n_splits=n_reps, train_size=train_frac, random_state=SEED
    )

    folds = []
    for rep, (train_idx, test_idx) in enumerate(splitter.split(indices, labels)):
        folds.append({
            "rep": rep,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })

    out_path = SPLITS / filename
    with open(out_path, "w") as f:
        json.dump(folds, f, indent=2)
    print(f"  Split saved: {out_path} ({n_reps} folds)")


# ---------------------------------------------------------------------------
# Task JSON Generation
# ---------------------------------------------------------------------------
def generate_task_jsons(i1_df, i2_df, i3_df):
    """Generate task definition JSONs for I1, I2, I3."""

    # I1
    i1_task = {
        "task_id": "I1",
        "name": "Cross-Mission Hemoglobin Gene DE Prediction",
        "description": "Predict whether a gene belongs to the hemoglobin/erythropoiesis pathway based on Twins Study transcriptome fold-change features. Tests whether spaceflight-induced gene expression changes are predictive of hemoglobin pathway membership.",
        "category": "cross_mission",
        "category_label": "Cross-Mission",
        "modality": "Transcriptomics",
        "missions": ["NASA Twins Study"],
        "difficulty_tier": "advanced",
        "task_type": "binary_classification",
        "data_files": ["cross_mission_hemoglobin_de.csv"],
        "input_spec": {
            "description": "3 fold-change features from Twins transcriptome DE analysis (Pre vs Flight, Pre vs Post, Flight vs Post)",
            "n_features": 3,
            "feature_columns": ["fc_pre_vs_flight", "fc_pre_vs_post", "fc_flight_vs_post"],
            "id_column": "gene"
        },
        "output_spec": {
            "target_column": "label",
            "target_type": "binary",
            "classes": [0, 1],
            "class_labels": ["non_hemoglobin", "hemoglobin"],
            "class_distribution": {
                "0": int(len(i1_df) - i1_df["label"].sum()),
                "1": int(i1_df["label"].sum())
            }
        },
        "evaluation": {
            "primary_metric": "auprc",
            "secondary_metrics": ["auroc", "f1"]
        },
        "split": "feature_split_I1",
        "n_samples": len(i1_df)
    }

    # I2
    i2_task = {
        "task_id": "I2",
        "name": "Cross-Mission Pathway Conservation",
        "description": "Predict whether an I4 PBMC pathway enrichment is also significant in the NASA Twins Study. Features are aggregated pathway statistics from I4 single-cell PBMC GSEA; target is conservation in Twins blood cell transcriptomics.",
        "category": "cross_mission",
        "category_label": "Cross-Mission",
        "modality": "Transcriptomics",
        "missions": ["Inspiration4", "NASA Twins Study"],
        "difficulty_tier": "advanced",
        "task_type": "binary_classification",
        "data_files": ["cross_mission_pathway_features.csv"],
        "input_spec": {
            "description": "8 aggregated pathway features from I4 PBMC GSEA (mean/std NES, ES, padj, cell type count, pathway size, direction consistency)",
            "n_features": 8,
            "feature_columns": ["mean_NES", "std_NES", "mean_ES", "mean_padj", "min_padj",
                                "n_celltypes", "mean_size", "direction_consistency"],
            "id_column": "pathway"
        },
        "output_spec": {
            "target_column": "label",
            "target_type": "binary",
            "classes": [0, 1],
            "class_labels": ["i4_only", "conserved_in_twins"],
            "class_distribution": {
                "0": int(len(i2_df) - i2_df["label"].sum()),
                "1": int(i2_df["label"].sum())
            }
        },
        "evaluation": {
            "primary_metric": "auroc",
            "secondary_metrics": ["auprc", "f1"]
        },
        "split": "feature_split_I2",
        "n_samples": len(i2_df)
    }

    # I3
    i3_task = {
        "task_id": "I3",
        "name": "Cross-Mission Gene DE Conservation",
        "description": "Predict whether a gene differentially expressed in the NASA Twins Study blood cells is also DE in I4 cell-free RNA. Twins features are aggregated from single-cell DEG analysis across multiple cell types and contrasts; I4 target is based on cfRNA 3-group ANOVA FDR < 0.05.",
        "category": "cross_mission",
        "category_label": "Cross-Mission",
        "modality": "Transcriptomics",
        "missions": ["Inspiration4", "NASA Twins Study"],
        "difficulty_tier": "advanced",
        "task_type": "binary_classification",
        "data_files": ["cross_mission_gene_de.csv"],
        "input_spec": {
            "description": "9 aggregated Twins blood cell DE features per gene (log2FC magnitude, base expression, standard error, Wald statistic, cell type count, contrast count, total DEG entries, direction consistency)",
            "n_features": 9,
            "feature_columns": ["mean_abs_log2fc", "max_abs_log2fc", "mean_base_expr",
                                "mean_lfcSE", "mean_abs_wald", "n_cell_types",
                                "n_contrasts", "n_total_deg_entries", "direction_consistency"],
            "id_column": "gene"
        },
        "output_spec": {
            "target_column": "label",
            "target_type": "binary",
            "classes": [0, 1],
            "class_labels": ["not_de_in_i4", "de_in_i4"],
            "class_distribution": {
                "0": int(len(i3_df) - i3_df["label"].sum()),
                "1": int(i3_df["label"].sum())
            }
        },
        "evaluation": {
            "primary_metric": "auprc",
            "secondary_metrics": ["auroc", "f1"]
        },
        "split": "feature_split_I3",
        "n_samples": len(i3_df)
    }

    for task_id, task_def in [("I1", i1_task), ("I2", i2_task), ("I3", i3_task)]:
        path = TASKS / f"{task_id}.json"
        with open(path, "w") as f:
            json.dump(task_def, f, indent=2)
        print(f"  Task JSON saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Cross-Mission Task Preprocessing")
    print("=" * 60)

    print("\n--- I1: Hemoglobin Gene DE Prediction ---")
    i1 = preprocess_i1()

    print("\n--- I2: Pathway Conservation ---")
    i2 = preprocess_i2()

    print("\n--- I3: Gene DE Conservation ---")
    i3 = preprocess_i3()

    print("\n--- Generating Task JSONs ---")
    generate_task_jsons(i1, i2, i3)

    print("\n" + "=" * 60)
    print("Done. Files created:")
    print(f"  data/processed/cross_mission_hemoglobin_de.csv")
    print(f"  data/processed/cross_mission_pathway_features.csv")
    print(f"  data/processed/cross_mission_gene_de.csv")
    print(f"  splits/feature_split_I1.json")
    print(f"  splits/feature_split_I2.json")
    print(f"  splits/feature_split_I3.json")
    print(f"  tasks/I1.json, I2.json, I3.json")
