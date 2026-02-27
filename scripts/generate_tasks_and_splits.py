#!/usr/bin/env python3
"""
[LEGACY] Generate task definitions (JSON) and split files for SpaceOmicsBench v2.

NOTE: This script is a LEGACY generator retained for reproducibility reference only.
      The canonical task/split files are already committed to the repository under
      tasks/ and splits/. Do NOT run this script to regenerate them unless you
      intentionally want to overwrite existing files (requires --allow-legacy-write).
      For normal use, load tasks/*.json and splits/*.json directly.

21 tasks across 9 categories:
  A (Clinical):      A1 blood panel phase, A2 immune marker phase
  B (cfRNA):         B1 DEG ranking, B2 cluster prediction
  C (Proteomics):    C1 expression→DE, C2 cross-biofluid
  D (Metabolomics):  D1 pathway classification
  E (Spatial):       E1-E4 cross-layer DE prediction (E2/E3 supplementary)
  F (Microbiome):    F1-F5 body site / phase / source classification
  G (Multi-modal):   G1 multi-modal phase
  H (Cross-tissue):  H1 PBMC ↔ skin conservation
  I (Cross-mission): I1-I3 Twins ↔ I4 conservation

Output:
  - tasks/*.json: Task definitions
  - splits/*.json: Split indices
"""

import json
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent
PROC_DIR = BASE_DIR / "data" / "processed"
TASK_DIR = BASE_DIR / "tasks"
SPLIT_DIR = BASE_DIR / "splits"

SEED = 42


# ============================================================
# Helpers
# ============================================================

def stratified_feature_splits(labels, n_reps=5, test_size=0.2, seed=SEED):
    """Generate stratified 80/20 splits for feature-level tasks."""
    rng = np.random.RandomState(seed)
    splits = []
    n = len(labels)
    for rep in range(n_reps):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed + rep)
        # Take the first fold as the split (80/20)
        for train_idx, test_idx in skf.split(np.zeros(n), labels):
            splits.append({
                "rep": rep,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            })
            break
    return splits


def loco_splits(crew_labels):
    """Leave-One-Crew-Out splits."""
    crews = sorted(set(crew_labels))
    splits = []
    for test_crew in crews:
        train_idx = [i for i, c in enumerate(crew_labels) if c != test_crew]
        test_idx = [i for i, c in enumerate(crew_labels) if c == test_crew]
        splits.append({
            "test_crew": test_crew,
            "train_indices": train_idx,
            "test_indices": test_idx,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
        })
    return splits


def save_task(task_id, task_def):
    """Save task definition JSON."""
    path = TASK_DIR / f"{task_id}.json"
    with open(path, "w") as f:
        json.dump(task_def, f, indent=2)
    print(f"  Saved: tasks/{task_id}.json")


def save_split(split_id, split_data):
    """Save split JSON."""
    path = SPLIT_DIR / f"{split_id}.json"
    with open(path, "w") as f:
        json.dump(split_data, f, indent=2)
    print(f"  Saved: splits/{split_id}.json")


# ============================================================
# Task & Split Generators
# ============================================================

def generate_A1():
    """A1: Phase classification from CBC + CMP."""
    cbc = pd.read_csv(PROC_DIR / "clinical_cbc.csv")
    phases = cbc["phase"].tolist()
    crews = cbc["crew"].tolist()

    task = {
        "task_id": "A1",
        "task_name": "Flight Phase Classification (Blood Panel)",
        "task_type": "multi_class_classification",
        "difficulty": "medium",
        "description": "Classify spaceflight phase from standard blood panel (CBC + CMP). 4 crew × 7 timepoints.",
        "data_files": ["clinical_cbc.csv", "clinical_cmp.csv"],
        "input_spec": {
            "features": "20 CBC analytes + 19 CMP analytes",
            "feature_count": 39,
            "meta_columns": ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"],
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "phase",
            "classes": ["pre_flight", "post_flight", "recovery"],
            "class_distribution": {p: phases.count(p) for p in ["pre_flight", "post_flight", "recovery"]},
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1"],
            "confidence_interval": True,
        },
        "split": "loco_clinical",
        "n_samples": len(cbc),
        "notes": ["N=28, small sample size — report 95% CI", "post_flight has only 4 samples (1 per crew)"],
    }
    save_task("A1", task)

    splits = loco_splits(crews)
    save_split("loco_clinical", splits)


def generate_A2():
    """A2: Phase classification from Eve cytokines."""
    eve = pd.read_csv(PROC_DIR / "clinical_cytokines_eve.csv")
    feature_cols = [c for c in eve.columns if c.startswith("eve_")]

    task = {
        "task_id": "A2",
        "task_name": "Flight Phase Classification (Immune Markers)",
        "task_type": "multi_class_classification",
        "difficulty": "medium",
        "description": "Classify spaceflight phase from Eve immune cytokine panel (71 markers). Same samples as A1, different features.",
        "data_files": ["clinical_cytokines_eve.csv"],
        "input_spec": {
            "features": "71 Eve immune cytokine markers",
            "feature_count": len(feature_cols),
            "meta_columns": ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"],
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "phase",
            "classes": ["pre_flight", "post_flight", "recovery"],
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1"],
            "confidence_interval": True,
        },
        "split": "loco_clinical",
        "n_samples": len(eve),
        "notes": ["Same sample set as A1, tests immune panel vs blood panel", "Zero missing values"],
    }
    save_task("A2", task)


def generate_B1():
    """B1: cfRNA DEG ranking."""
    de3 = pd.read_csv(PROC_DIR / "cfrna_3group_de.csv")
    drr = pd.read_csv(PROC_DIR / "cfrna_466drr.csv")
    drr_genes = set(drr["gene"])

    labels = [1 if g in drr_genes else 0 for g in de3["gene"]]

    task = {
        "task_id": "B1",
        "task_name": "Spaceflight-Responsive Gene Ranking (cfRNA)",
        "task_type": "binary_classification",
        "difficulty": "hard",
        "description": "Predict which genes are Differentially Regulated in Response (DRR) to spaceflight using group-level DE statistics from JAXA cfRNA data.",
        "data_files": ["cfrna_3group_de.csv"],
        "ground_truth_files": ["cfrna_466drr.csv"],
        "input_spec": {
            "features": "38 DE statistics per gene (ANOVA, edgeR pairwise, group means)",
            "feature_count": 38,
            "gene_column": "gene",
        },
        "output_spec": {
            "type": "binary",
            "positive_class": "DRR gene",
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "positive_rate": f"{sum(labels)/len(labels)*100:.1f}%",
        },
        "evaluation": {
            "primary_metric": "auprc",
            "secondary_metrics": ["auroc", "ndcg_at_100", "precision_at_50"],
        },
        "split": "feature_split_B1",
        "n_samples": len(de3),
    }
    save_task("B1", task)

    splits = stratified_feature_splits(labels)
    save_split("feature_split_B1", splits)


def generate_B2():
    """B2: Coregulated gene cluster prediction."""
    labels_df = pd.read_csv(PROC_DIR / "gt_cfrna_cluster_labels.csv")
    corr_df = pd.read_csv(PROC_DIR / "gt_cfrna_correlation.csv")
    cluster_cols = [c for c in labels_df.columns if c != "gene"]

    # For stratified split, use "has any cluster" as stratification key
    any_label = (labels_df[cluster_cols].sum(axis=1) > 0).astype(int).tolist()

    task = {
        "task_id": "B2",
        "task_name": "Coregulated Gene Cluster Prediction",
        "task_type": "multilabel_classification",
        "difficulty": "hard",
        "description": "Predict cluster membership of 466 DRR genes from gene-gene correlation patterns. 16 clusters, multi-label.",
        "data_files": ["gt_cfrna_correlation.csv"],
        "ground_truth_files": ["gt_cfrna_cluster_labels.csv"],
        "input_spec": {
            "features": "466-dimensional correlation vector per gene",
            "feature_count": 466,
            "gene_column": "gene",
        },
        "output_spec": {
            "type": "multilabel",
            "n_labels": len(cluster_cols),
            "label_names": cluster_cols,
            "genes_with_labels": sum(any_label),
            "genes_without_labels": len(any_label) - sum(any_label),
        },
        "evaluation": {
            "primary_metric": "micro_f1",
            "secondary_metrics": ["hamming_loss", "per_cluster_auroc"],
        },
        "split": "feature_split_B2",
        "n_samples": len(labels_df),
    }
    save_task("B2", task)

    splits = stratified_feature_splits(any_label)
    save_split("feature_split_B2", splits)


def generate_C1():
    """C1: Protein DE from expression profile."""
    de = pd.read_csv(PROC_DIR / "proteomics_plasma_de_clean.csv")
    mat = pd.read_csv(PROC_DIR / "proteomics_plasma_matrix.csv")
    meta_cols = ["sample_id", "crew", "month_tp", "timepoint", "timepoint_days", "phase", "mission", "tissue"]
    sample_cols = [c for c in mat.columns if c not in meta_cols]

    # Match proteins between matrix and DE
    de_genes = set(de["gene"])
    mat_genes = set(sample_cols)
    overlap = de_genes & mat_genes

    labels = [int(de[de["gene"] == g]["adj_pval"].values[0] < 0.05)
              for g in sorted(overlap) if len(de[de["gene"] == g]) > 0]
    genes = sorted(overlap)

    task = {
        "task_id": "C1",
        "task_name": "Protein DE Prediction from Expression Profile",
        "task_type": "binary_classification",
        "difficulty": "medium",
        "description": "Predict limma differential expression from protein expression across 21 plasma samples. Tests whether ML can learn statistical significance patterns.",
        "data_files": ["proteomics_plasma_matrix.csv", "proteomics_plasma_de_clean.csv"],
        "input_spec": {
            "features": "21 sample expression values per protein",
            "feature_count": 21,
            "note": "Transpose: proteins as instances, samples as features",
        },
        "output_spec": {
            "type": "binary",
            "positive_class": "DE (adj_pval < 0.05)",
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
            "positive_rate": f"{sum(labels)/len(labels)*100:.1f}%",
        },
        "evaluation": {
            "primary_metric": "auroc",
            "secondary_metrics": ["auprc", "f1"],
        },
        "split": "feature_split_C1",
        "n_samples": len(genes),
    }
    save_task("C1", task)

    splits = stratified_feature_splits(labels)
    save_split("feature_split_C1", splits)


def generate_C2():
    """C2: Cross-biofluid protein concordance."""
    pde = pd.read_csv(PROC_DIR / "proteomics_plasma_de_clean.csv")
    ede = pd.read_csv(PROC_DIR / "proteomics_evp_de_clean.csv")

    overlap_genes = sorted(set(pde["gene"]) & set(ede["gene"]))
    merged = pde[pde["gene"].isin(overlap_genes)].merge(
        ede[ede["gene"].isin(overlap_genes)], on="gene", suffixes=("_plasma", "_evp")
    )
    labels = (merged["adj_pval_evp"] < 0.05).astype(int).tolist()

    task = {
        "task_id": "C2",
        "task_name": "Cross-Biofluid Protein DE Concordance",
        "task_type": "binary_classification",
        "difficulty": "hard",
        "description": "Predict EVP (exosome) protein DE from plasma protein DE statistics. Tests cross-biofluid generalization of spaceflight response.",
        "data_files": ["proteomics_plasma_de_clean.csv", "proteomics_evp_de_clean.csv"],
        "input_spec": {
            "features": "4 plasma DE statistics (logFC, AveExpr, t, B)",
            "feature_count": 4,
            "note": "Input from plasma; target from EVP",
        },
        "output_spec": {
            "type": "binary",
            "positive_class": "EVP DE (adj_pval < 0.05)",
            "positive_count": sum(labels),
            "negative_count": len(labels) - sum(labels),
        },
        "evaluation": {
            "primary_metric": "auroc",
            "secondary_metrics": ["auprc", "f1", "direction_concordance"],
        },
        "split": "feature_split_C2",
        "n_samples": len(merged),
        "notes": ["380 overlapping proteins", "Baseline direction concordance: 64.2%"],
    }
    save_task("C2", task)

    splits = stratified_feature_splits(labels)
    save_split("feature_split_C2", splits)


def generate_D1():
    """D1: Metabolite super-pathway classification."""
    met = pd.read_csv(PROC_DIR / "metabolomics_anppos_matrix.csv")
    met = met.dropna(subset=["SuperPathway"])

    # Merge rare classes (<10) into "Other"
    sp_counts = met["SuperPathway"].value_counts()
    rare = sp_counts[sp_counts < 10].index.tolist()
    met["SuperPathway_merged"] = met["SuperPathway"].apply(
        lambda x: "Other" if x in rare else x
    )
    classes = sorted(met["SuperPathway_merged"].unique())
    label_map = {c: i for i, c in enumerate(classes)}
    labels = [label_map[c] for c in met["SuperPathway_merged"]]

    task = {
        "task_id": "D1",
        "task_name": "Metabolite Super-Pathway Classification",
        "task_type": "multi_class_classification",
        "difficulty": "hard",
        "description": "Predict metabolite super-pathway from expression profile and chemical annotations. Rare classes (<10) merged into 'Other'.",
        "data_files": ["metabolomics_anppos_matrix.csv", "metabolomics_de.csv"],
        "input_spec": {
            "features": "24 sample expression values + chemical annotations (Mass, RT) + DE statistics",
            "note": "metabolite_name is the key column",
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "SuperPathway_merged",
            "classes": classes,
            "n_classes": len(classes),
            "class_distribution": {c: labels.count(label_map[c]) for c in classes},
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["per_class_f1", "accuracy"],
        },
        "split": "feature_split_D1",
        "n_samples": len(met),
    }
    save_task("D1", task)

    splits = stratified_feature_splits(labels)
    save_split("feature_split_D1", splits)


def generate_E(layer_id, layer_name, target_layer_name):
    """E1-E4: Cross-layer spatial DE prediction."""
    gt_all = pd.read_csv(PROC_DIR / "gt_spatial_de_all_skin.csv")
    gt_target = pd.read_csv(PROC_DIR / f"gt_spatial_de_{target_layer_name}.csv")

    labels = (gt_target["adj_pval"].fillna(1) < 0.05).astype(int).tolist()
    n_pos = sum(labels)

    task = {
        "task_id": layer_id,
        "task_name": f"Cross-Layer DE Prediction ({target_layer_name})",
        "task_type": "binary_classification",
        "difficulty": "hard" if n_pos >= 30 else "expert",
        "description": f"Predict {target_layer_name} layer-specific DE from global skin (all_skin) DE features. Tests whether aggregate skin response predicts spatially-resolved layer response.",
        "data_files": ["gt_spatial_de_all_skin.csv"],
        "ground_truth_files": [f"gt_spatial_de_{target_layer_name}.csv"],
        "input_spec": {
            "features": "3 features from all_skin DESeq2: baseMean, log2FoldChange, lfcSE",
            "feature_count": 3,
            "gene_column": "gene",
            "note": "stat and pval excluded to avoid trivial leakage",
        },
        "output_spec": {
            "type": "binary",
            "positive_class": f"DE in {target_layer_name} (adj_pval < 0.05)",
            "positive_count": n_pos,
            "negative_count": len(labels) - n_pos,
            "positive_rate": f"{n_pos/len(labels)*100:.2f}%",
        },
        "evaluation": {
            "primary_metric": "auprc",
            "secondary_metrics": ["auroc", "precision_at_50"],
        },
        "split": f"feature_split_{layer_id}",
        "n_samples": len(gt_all),
    }
    save_task(layer_id, task)

    splits = stratified_feature_splits(labels)
    save_split(f"feature_split_{layer_id}", splits)


def generate_F1():
    """F1: Body site classification from taxonomy."""
    meta = pd.read_csv(PROC_DIR / "microbiome_metadata.csv")
    human = meta[meta["source"] == "human"].reset_index(drop=True)
    crews = human["crew"].tolist()
    sites = human["body_site"].tolist()

    task = {
        "task_id": "F1",
        "task_name": "Microbiome Body Site Classification (Taxonomy)",
        "task_type": "multi_class_classification",
        "difficulty": "easy",
        "description": "Classify body site from skin microbiome taxonomy (CPM). 4 crew × 10 body sites × 7 timepoints.",
        "data_files": ["microbiome_human_taxonomy_cpm.csv"],
        "metadata_file": "microbiome_metadata.csv",
        "input_spec": {
            "features": "16,172 taxa (CPM-normalized abundance)",
            "feature_count": 16172,
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "body_site",
            "classes": sorted(set(sites)),
            "n_classes": len(set(sites)),
            "class_distribution": {s: sites.count(s) for s in sorted(set(sites))},
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1"],
        },
        "split": "loco_microbiome_human",
        "n_samples": len(human),
    }
    save_task("F1", task)

    splits = loco_splits(crews)
    save_split("loco_microbiome_human", splits)


def generate_F2():
    """F2: Flight phase detection from taxonomy."""
    meta = pd.read_csv(PROC_DIR / "microbiome_metadata.csv")
    human = meta[meta["source"] == "human"].reset_index(drop=True)
    phases = human["phase"].tolist()

    task = {
        "task_id": "F2",
        "task_name": "Flight Phase Detection (Taxonomy)",
        "task_type": "multi_class_classification",
        "difficulty": "hard",
        "description": "Detect spaceflight phase from skin microbiome taxonomy. Harder than body site — temporal changes are subtler.",
        "data_files": ["microbiome_human_taxonomy_cpm.csv"],
        "metadata_file": "microbiome_metadata.csv",
        "input_spec": {
            "features": "16,172 taxa (CPM-normalized abundance)",
            "feature_count": 16172,
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "phase",
            "classes": sorted(set(phases)),
            "n_classes": len(set(phases)),
            "class_distribution": {p: phases.count(p) for p in sorted(set(phases))},
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1"],
        },
        "split": "loco_microbiome_human",
        "n_samples": len(human),
    }
    save_task("F2", task)


def generate_F3():
    """F3: Human vs environmental microbiome."""
    meta = pd.read_csv(PROC_DIR / "microbiome_metadata.csv")
    human = meta[meta["source"] == "human"]
    env = meta[meta["source"] == "environmental"]

    task = {
        "task_id": "F3",
        "task_name": "Source Classification (Human vs Environmental)",
        "task_type": "binary_classification",
        "difficulty": "easy",
        "description": "Classify microbiome as human crew or Dragon capsule environmental. Requires taxid-based feature alignment (~5,830 shared taxa).",
        "data_files": ["microbiome_human_taxonomy_cpm.csv", "microbiome_env_taxonomy_cpm.csv"],
        "metadata_file": "microbiome_metadata.csv",
        "input_spec": {
            "features": "Shared taxa between human and environmental datasets",
            "alignment": "taxid intersection (~5,830 shared)",
        },
        "output_spec": {
            "type": "binary",
            "classes": ["environmental", "human"],
            "class_distribution": {"human": len(human), "environmental": len(env)},
        },
        "evaluation": {
            "primary_metric": "auroc",
            "secondary_metrics": ["f1", "auprc"],
        },
        "split": "loto_microbiome",
        "n_samples": len(human) + len(env),
        "notes": ["Calibration task — expected easy"],
    }
    save_task("F3", task)

    # Leave-One-Timepoint-Out for F3
    combined_meta = pd.concat([human, env], ignore_index=True)
    timepoints = sorted(combined_meta["timepoint"].unique())
    splits = []
    for tp in timepoints:
        test_idx = combined_meta[combined_meta["timepoint"] == tp].index.tolist()
        train_idx = combined_meta[combined_meta["timepoint"] != tp].index.tolist()
        if len(test_idx) > 0 and len(train_idx) > 0:
            splits.append({
                "test_timepoint": tp,
                "train_indices": train_idx,
                "test_indices": test_idx,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            })
    save_split("loto_microbiome", splits)


def generate_F4():
    """F4: Body site classification from pathways."""
    task = {
        "task_id": "F4",
        "task_name": "Microbiome Body Site Classification (Pathways)",
        "task_type": "multi_class_classification",
        "difficulty": "medium",
        "description": "Classify body site from microbiome functional pathways (567 MetaCyc pathways). Paired with F1 (taxonomy) for feature representation comparison.",
        "data_files": ["microbiome_human_pathways_cpm.csv"],
        "metadata_file": "microbiome_metadata.csv",
        "input_spec": {
            "features": "567 MetaCyc pathways (CPM-normalized)",
            "feature_count": 567,
            "note": "Drop 2 extra samples not in taxonomy metadata (C002_FD3_NAP, C002_R+82_ARM)",
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "body_site",
            "n_classes": 10,
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1"],
        },
        "split": "loco_microbiome_human",
        "n_samples": 275,
        "notes": ["Compare with F1 to assess taxonomy vs pathway features"],
    }
    save_task("F4", task)


def generate_F5():
    """F5: Flight phase detection from pathways."""
    task = {
        "task_id": "F5",
        "task_name": "Flight Phase Detection (Pathways)",
        "task_type": "multi_class_classification",
        "difficulty": "hard",
        "description": "Detect spaceflight phase from microbiome functional pathways. Paired with F2 (taxonomy) for comparison.",
        "data_files": ["microbiome_human_pathways_cpm.csv"],
        "metadata_file": "microbiome_metadata.csv",
        "input_spec": {
            "features": "567 MetaCyc pathways (CPM-normalized)",
            "feature_count": 567,
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "phase",
            "n_classes": 4,
            "classes": ["in_flight", "post_flight", "pre_flight", "recovery"],
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy", "per_class_f1"],
        },
        "split": "loco_microbiome_human",
        "n_samples": 275,
        "notes": ["Compare with F2 to assess taxonomy vs pathway features"],
    }
    save_task("F5", task)


def generate_G1():
    """G1: Multi-modal phase classification."""
    # Find matched samples
    cbc = pd.read_csv(PROC_DIR / "clinical_cbc.csv")
    prot_meta = pd.read_csv(PROC_DIR / "proteomics_metadata.csv")
    prot_plasma = prot_meta[prot_meta["tissue"] == "plasma"]
    met_meta = pd.read_csv(PROC_DIR / "metabolomics_metadata.csv")

    cbc_keys = set(zip(cbc["crew"], cbc["timepoint_days"]))
    prot_keys = set(zip(prot_plasma["crew"], prot_plasma["timepoint_days"]))
    met_keys = set(zip(met_meta["crew"], met_meta["timepoint_days"]))
    matched = sorted(cbc_keys & prot_keys & met_keys)

    matched_crews = [k[0] for k in matched]

    task = {
        "task_id": "G1",
        "task_name": "Multi-Modal Phase Classification",
        "task_type": "multi_class_classification",
        "difficulty": "hard",
        "description": "Classify spaceflight phase using matched clinical + proteomics + metabolomics data. Tests multi-omic fusion benefit over single modality (A1).",
        "data_files": [
            "clinical_cbc.csv", "clinical_cmp.csv",
            "proteomics_plasma_matrix.csv",
            "metabolomics_anppos_matrix.csv",
        ],
        "input_spec": {
            "features": "39 clinical + 2,845 proteins + 454 metabolites",
            "join_key": "crew + timepoint_days",
            "note": "Sample ID formats differ across modalities",
        },
        "output_spec": {
            "type": "categorical",
            "target_column": "phase",
            "classes": ["pre_flight", "post_flight", "recovery"],
        },
        "evaluation": {
            "primary_metric": "macro_f1",
            "secondary_metrics": ["accuracy"],
            "confidence_interval": True,
        },
        "split": "loco_multimodal",
        "n_samples": len(matched),
        "notes": [
            f"21 matched samples (crew × timepoint_days intersection)",
            "Compare with A1 to quantify multi-omic benefit",
        ],
    }
    save_task("G1", task)

    splits = loco_splits(matched_crews)
    save_split("loco_multimodal", splits)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Legacy task/split generator. "
            "Current benchmark tasks/splits are maintained separately; "
            "do not run this script unless you intentionally want legacy regeneration."
        )
    )
    parser.add_argument(
        "--allow-legacy-write",
        action="store_true",
        help="Allow writing task/split JSON files from this legacy generator.",
    )
    args = parser.parse_args()

    if not args.allow_legacy_write:
        print("Refusing to run legacy generator without --allow-legacy-write.")
        print("Use current benchmark assets in tasks/ and splits/ directly.")
        return 2

    print("=" * 60)
    print("SpaceOmicsBench v2 -- Legacy Task & Split Generation")
    print("=" * 60)

    TASK_DIR.mkdir(parents=True, exist_ok=True)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Category A: Clinical
    print("\n--- Category A: Clinical ---")
    generate_A1()
    generate_A2()

    # Category B: cfRNA
    print("\n--- Category B: cfRNA ---")
    generate_B1()
    generate_B2()

    # Category C: Proteomics
    print("\n--- Category C: Proteomics ---")
    generate_C1()
    generate_C2()

    # Category D: Metabolomics
    print("\n--- Category D: Metabolomics ---")
    generate_D1()

    # Category E: Spatial cross-layer
    print("\n--- Category E: Spatial Cross-Layer ---")
    e_layers = [
        ("E1", "outer_epidermis"),
        ("E2", "inner_epidermis"),
        ("E3", "outer_dermis"),
        ("E4", "epidermis"),
    ]
    for layer_id, layer_name in e_layers:
        generate_E(layer_id, f"Cross-Layer: {layer_name}", layer_name)

    # Category F: Microbiome
    print("\n--- Category F: Microbiome ---")
    generate_F1()
    generate_F2()
    generate_F3()
    generate_F4()
    generate_F5()

    # Category G: Multi-modal
    print("\n--- Category G: Multi-Modal ---")
    generate_G1()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    task_files = sorted(TASK_DIR.glob("*.json"))
    split_files = sorted(SPLIT_DIR.glob("*.json"))
    print(f"  Tasks: {len(task_files)} JSON files")
    for tf in task_files:
        with open(tf) as f:
            t = json.load(f)
        print(f"    {t['task_id']}: {t['task_name']} (N={t['n_samples']}, {t['difficulty']})")
    print(f"  Splits: {len(split_files)} JSON files")
    for sf in split_files:
        print(f"    {sf.name}")
    print()

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
