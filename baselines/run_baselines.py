#!/usr/bin/env python3
"""
SpaceOmicsBench v2 — ML Baselines

Runs baseline models on all 18 tasks:
  1. Random: random predictions matching class distribution
  2. Majority: always predict majority class
  3. LogReg: Logistic Regression (L2, class-weighted)
  4. RF: Random Forest (100 trees, class-weighted)
  5. MLP: Multi-layer Perceptron (256→128→64, dropout-like via alpha)

Additional analyses:
  - B1 feature ablation (effect-size only vs no-effect-size)
  - G1 actual multi-modal fusion (PCA per modality)
  - Normalized composite score
  - Per-fold detailed results for LOCO tasks

Output: baselines/baseline_results.json
"""

import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    print("Warning: xgboost not available, skipping XGBoost baseline")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
TASK_DIR = BASE_DIR / "tasks"
SPLIT_DIR = BASE_DIR / "splits"
OUT_DIR = BASE_DIR / "baselines"

SEED = 42

# Tasks included in main benchmark (E2/E3 are supplementary)
MAIN_TASKS = [
    "A1", "A2", "B1", "B2", "C1", "C2", "D1",
    "E1", "E4",
    "F1", "F2", "F3", "F4", "F5",
    "G1", "H1",
    "I1", "I2", "I3",
]
SUPPLEMENTARY_TASKS = ["E2", "E3"]
ALL_TASKS = MAIN_TASKS + SUPPLEMENTARY_TASKS


# ─── Data loaders ──────────────────────────────────────────────────────────


def _parse_formula(formula_str):
    """Parse chemical formula into atom counts (C, H, N, O, S, P)."""
    atoms = {"C": 0, "H": 0, "N": 0, "O": 0, "S": 0, "P": 0}
    if not isinstance(formula_str, str) or pd.isna(formula_str):
        return list(atoms.values())
    for symbol in atoms:
        match = re.findall(rf"{symbol}(\d*)", formula_str)
        for m in match:
            atoms[symbol] += int(m) if m else 1
    return list(atoms.values())


def load_task_data(task_id):
    """Load features (X) and labels (y) for a task.

    Returns:
        X: np.ndarray (N, D)
        y: np.ndarray (N,) for binary/multiclass, (N, L) for multilabel
        task: dict (task JSON)
    """
    # B1 ablation variants use B1.json
    json_id = "B1" if task_id.startswith("B1_") else task_id
    task_path = TASK_DIR / f"{json_id}.json"
    with open(task_path) as f:
        task = json.load(f)

    if task_id == "A1":
        cbc = pd.read_csv(DATA_DIR / "clinical_cbc.csv")
        cmp = pd.read_csv(DATA_DIR / "clinical_cmp.csv")
        meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"]
        X_cbc = cbc.drop(columns=[c for c in meta_cols if c in cbc.columns], errors="ignore")
        X_cmp = cmp.drop(columns=[c for c in meta_cols if c in cmp.columns], errors="ignore")
        X = pd.concat([X_cbc, X_cmp], axis=1).values.astype(float)
        y = cbc["phase"].values

    elif task_id == "A2":
        cyt = pd.read_csv(DATA_DIR / "clinical_cytokines_eve.csv")
        meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"]
        X = cyt.drop(columns=[c for c in meta_cols if c in cyt.columns], errors="ignore").values.astype(float)
        y = cyt["phase"].values

    elif task_id.startswith("B1"):
        de = pd.read_csv(DATA_DIR / "cfrna_3group_de_noleak.csv")
        drr = pd.read_csv(DATA_DIR / "cfrna_466drr.csv")
        drr_genes = set(drr.iloc[:, 0])
        y = np.array([1 if g in drr_genes else 0 for g in de["gene"]], dtype=int)

        if task_id == "B1":
            X = de.drop(columns=["gene"]).values.astype(float)
        elif task_id == "B1_effect_only":
            # Ablation: only effect-size features
            effect_cols = [c for c in de.columns if c != "gene" and
                           any(k in c for k in ["_fc", "_diff", "max_diff", "max_fc"])]
            X = de[effect_cols].values.astype(float)
        elif task_id == "B1_no_effect":
            # Ablation: exclude effect-size features
            effect_patterns = ["_fc", "_diff", "max_diff", "max_fc"]
            no_effect_cols = [c for c in de.columns if c != "gene" and
                              not any(k in c for k in effect_patterns)]
            X = de[no_effect_cols].values.astype(float)
        task["task_id"] = "B1"  # Use B1 split for all variants

    elif task_id == "B2":
        corr = pd.read_csv(DATA_DIR / "gt_cfrna_correlation.csv")
        labels = pd.read_csv(DATA_DIR / "gt_cfrna_cluster_labels.csv")
        gene_col = corr.columns[0]
        X = corr.drop(columns=[gene_col]).values.astype(float)
        label_cols = [c for c in labels.columns if c != labels.columns[0]]
        y = labels[label_cols].values.astype(int)

    elif task_id == "C1":
        # Redesigned: sample-level proteomics phase classification
        matrix = pd.read_csv(DATA_DIR / "proteomics_plasma_matrix.csv")
        meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days",
                     "phase", "mission", "month_tp"]
        protein_cols = [c for c in matrix.columns if c not in meta_cols]
        X_raw = matrix[protein_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
        # PCA to handle p>>n (2,845 proteins → 10 components)
        pca = PCA(n_components=min(10, X_raw.shape[0] - 1), random_state=SEED)
        X = pca.fit_transform(X_raw)
        y = matrix["phase"].values

    elif task_id == "C2":
        plasma = pd.read_csv(DATA_DIR / "proteomics_plasma_de_clean.csv")
        evp = pd.read_csv(DATA_DIR / "proteomics_evp_de_clean.csv")
        p_col = "protein" if "protein" in plasma.columns else plasma.columns[0]
        e_col = "protein" if "protein" in evp.columns else evp.columns[0]
        overlap = sorted(set(plasma[p_col]) & set(evp[e_col]))
        plasma_sub = plasma[plasma[p_col].isin(overlap)].set_index(p_col).loc[overlap]
        evp_sub = evp[evp[e_col].isin(overlap)].set_index(e_col).loc[overlap]
        X = plasma_sub[["logFC", "AveExpr", "t", "B"]].values.astype(float)
        y = (evp_sub["adj_pval"] < 0.05).astype(int).values

    elif task_id == "D1":
        df = pd.read_csv(DATA_DIR / "metabolomics_spaceflight_response.csv")
        # Numerical features
        num_cols = ["Mass", "RT", "annotation_confidence"]
        X_num = df[num_cols].fillna(0).values.astype(float)
        parts = [X_num]
        # SuperPathway one-hot
        if "SuperPathway" in df.columns:
            parts.append(pd.get_dummies(df["SuperPathway"], prefix="SP").values.astype(float))
        # SubPathway one-hot
        if "SubPathway" in df.columns:
            parts.append(pd.get_dummies(df["SubPathway"], prefix="sub").values.astype(float))
        # Formula → atom counts (C, H, N, O, S, P)
        if "Formula" in df.columns:
            atom_features = np.array([_parse_formula(f) for f in df["Formula"]])
            parts.append(atom_features.astype(float))
        X = np.hstack(parts)
        y = df["is_spaceflight_de"].astype(int).values

    elif task_id.startswith("E"):
        layer_map = {"E1": "outer_epidermis", "E2": "inner_epidermis",
                     "E3": "outer_dermis", "E4": "epidermis"}
        layer = layer_map[task_id]
        all_skin = pd.read_csv(DATA_DIR / "gt_spatial_de_all_skin.csv")
        layer_de = pd.read_csv(DATA_DIR / f"gt_spatial_de_{layer}.csv")
        X = all_skin[["baseMean", "log2FoldChange", "lfcSE"]].values.astype(float)
        y = (layer_de["adj_pval"].fillna(1.0) < 0.05).astype(int).values

    elif task_id in ("F1", "F2"):
        tax = pd.read_csv(DATA_DIR / "microbiome_human_taxonomy_cpm.csv")
        md = pd.read_csv(DATA_DIR / "microbiome_metadata.csv")
        human = md[md["source"] == "human"].reset_index(drop=True)
        tax_cols = ["taxid", "domain", "phylum", "class", "order", "family", "genus", "species"]
        sample_cols = [c for c in tax.columns if c not in tax_cols]
        X = tax[sample_cols].T.values.astype(float)
        y = human["body_site"].values if task_id == "F1" else human["phase"].values

    elif task_id == "F3":
        human_tax = pd.read_csv(DATA_DIR / "microbiome_human_taxonomy_cpm.csv")
        env_tax = pd.read_csv(DATA_DIR / "microbiome_env_taxonomy_cpm.csv")
        shared_taxids = set(human_tax["taxid"]) & set(env_tax["taxid"])
        h_sub = human_tax[human_tax["taxid"].isin(shared_taxids)].set_index("taxid")
        e_sub = env_tax[env_tax["taxid"].isin(shared_taxids)].set_index("taxid")
        tax_cols = ["domain", "phylum", "class", "order", "family", "genus", "species"]
        h_samples = [c for c in h_sub.columns if c not in tax_cols]
        e_samples = [c for c in e_sub.columns if c not in tax_cols]
        shared_idx = sorted(shared_taxids)
        h_vals = h_sub.loc[shared_idx, h_samples].T.values.astype(float)
        e_vals = e_sub.loc[shared_idx, e_samples].T.values.astype(float)
        X = np.vstack([h_vals, e_vals])
        y = np.array([1] * len(h_samples) + [0] * len(e_samples), dtype=int)

    elif task_id in ("F4", "F5"):
        path = pd.read_csv(DATA_DIR / "microbiome_human_pathways_cpm.csv")
        md = pd.read_csv(DATA_DIR / "microbiome_metadata.csv")
        human = md[md["source"] == "human"].reset_index(drop=True)
        pathway_col = path.columns[0]
        sample_cols = [c for c in path.columns if c != pathway_col]
        valid_samples = [c for c in sample_cols if c in set(human["sample_id"].values) or
                         any(c.startswith(crew) for crew in ["C001", "C002", "C003", "C004"])]
        X = path[valid_samples[:275]].T.values.astype(float)
        y = human["body_site"].values[:X.shape[0]] if task_id == "F4" else human["phase"].values[:X.shape[0]]

    elif task_id == "G1":
        # Multi-modal: clinical + PCA(proteomics) + PCA(metabolomics)
        cbc = pd.read_csv(DATA_DIR / "clinical_cbc.csv")
        cmp = pd.read_csv(DATA_DIR / "clinical_cmp.csv")
        prot_matrix = pd.read_csv(DATA_DIR / "proteomics_plasma_matrix.csv")
        met_matrix = pd.read_csv(DATA_DIR / "metabolomics_anppos_matrix.csv")
        prot_meta = pd.read_csv(DATA_DIR / "proteomics_metadata.csv")
        met_meta = pd.read_csv(DATA_DIR / "metabolomics_metadata.csv")

        # Find matched timepoints across all 3 modalities
        prot_plasma = prot_meta[prot_meta["tissue"] == "plasma"]
        cbc_keys = set(zip(cbc["crew"], cbc["timepoint_days"]))
        prot_keys = set(zip(prot_plasma["crew"], prot_plasma["timepoint_days"]))
        met_keys = set(zip(met_meta["crew"], met_meta["timepoint_days"]))
        matched = sorted(cbc_keys & prot_keys & met_keys)
        matched_set = set(matched)

        # Clinical features
        meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"]
        mask_cbc = [(row.crew, row.timepoint_days) in matched_set for row in cbc.itertuples()]
        X_cbc = cbc[mask_cbc].drop(columns=[c for c in meta_cols if c in cbc.columns]).values.astype(float)
        X_cmp = cmp[mask_cbc].drop(columns=[c for c in meta_cols if c in cmp.columns]).values.astype(float)
        X_clinical = np.hstack([X_cbc, X_cmp])

        # Proteomics features (PCA)
        prot_meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days",
                          "phase", "mission", "month_tp"]
        prot_protein_cols = [c for c in prot_matrix.columns if c not in prot_meta_cols]
        prot_vals = prot_matrix[prot_protein_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        prot_matched_rows = [
            i for i, row in prot_matrix.iterrows()
            if (row["crew"], row["timepoint_days"]) in matched_set
        ]
        X_prot_raw = prot_vals.iloc[prot_matched_rows].values
        n_prot_pca = min(8, X_prot_raw.shape[0] - 1, X_prot_raw.shape[1])
        X_prot = PCA(n_components=n_prot_pca, random_state=SEED).fit_transform(X_prot_raw)

        # Metabolomics features (PCA)
        met_feature_cols = [c for c in met_matrix.columns
                            if c not in ["SuperPathway", "SubPathway", "metabolite_name",
                                         "annotation_confidence", "Formula", "Mass", "RT",
                                         "CAS ID", "Mode", "KEGG", "HMDB"]]
        met_vals = met_matrix[met_feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        # Transpose: samples as rows
        met_sample_ids = met_feature_cols
        met_t = met_vals.T
        met_matched_rows = []
        for sid in met_sample_ids:
            for crew, td in matched:
                if sid.startswith(crew):
                    met_matched_rows.append(sid)
                    break
        if len(met_matched_rows) >= X_clinical.shape[0]:
            met_matched_rows = met_matched_rows[:X_clinical.shape[0]]
            X_met_raw = met_t.loc[met_matched_rows].values
            n_met_pca = min(8, X_met_raw.shape[0] - 1, X_met_raw.shape[1])
            X_met = PCA(n_components=n_met_pca, random_state=SEED).fit_transform(X_met_raw)
            X = np.hstack([X_clinical, X_prot, X_met])
        else:
            X = np.hstack([X_clinical, X_prot])

        y = cbc[mask_cbc]["phase"].values

    elif task_id == "H1":
        df = pd.read_csv(DATA_DIR / "conserved_pbmc_to_skin.csv")
        feature_cols = ["CD4_T", "CD8_T", "other_T", "B", "NK", "CD14_Mono", "CD16_Mono", "DC", "other"]
        X = df[feature_cols].values.astype(float)
        y = df["skin_de"].astype(int).values

    elif task_id == "I1":
        df = pd.read_csv(DATA_DIR / "cross_mission_hemoglobin_de.csv")
        X = df[["fc_pre_vs_flight", "fc_pre_vs_post", "fc_flight_vs_post"]].values.astype(float)
        y = df["label"].astype(int).values

    elif task_id == "I2":
        df = pd.read_csv(DATA_DIR / "cross_mission_pathway_features.csv")
        feat_cols = ["mean_NES", "std_NES", "mean_ES", "mean_padj", "min_padj",
                     "n_celltypes", "mean_size", "direction_consistency"]
        X = df[feat_cols].values.astype(float)
        y = df["label"].astype(int).values

    elif task_id == "I3":
        df = pd.read_csv(DATA_DIR / "cross_mission_gene_de.csv")
        feat_cols = ["mean_abs_log2fc", "max_abs_log2fc", "mean_base_expr",
                     "mean_lfcSE", "mean_abs_wald", "n_cell_types",
                     "n_contrasts", "n_total_deg_entries", "direction_consistency"]
        X = df[feat_cols].values.astype(float)
        y = df["label"].astype(int).values

    else:
        raise ValueError(f"Unknown task: {task_id}")

    return X, y, task


def load_splits(task):
    """Load split indices for a task."""
    split_path = SPLIT_DIR / f"{task['split']}.json"
    with open(split_path) as f:
        return json.load(f)


# ─── Baseline models ───────────────────────────────────────────────────────


def get_baselines(task_type, n_features=10):
    """Get baseline models for a task type."""
    models = {}

    models["random"] = None  # handled separately
    models["majority"] = None  # handled separately

    if task_type in ("binary_classification", "multi_class_classification",
                     "multi_class_feature_classification"):
        models["logreg"] = LogisticRegression(
            max_iter=1000, C=1.0, class_weight="balanced",
            random_state=SEED, solver="lbfgs",
        )
        models["rf"] = RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=SEED, n_jobs=-1,
        )
        models["mlp"] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=500, random_state=SEED,
            early_stopping=True, validation_fraction=0.15,
            alpha=0.01, learning_rate_init=0.001,
        )
        if HAS_XGB:
            models["xgboost"] = XGBClassifier(
                n_estimators=100, max_depth=6,
                random_state=SEED, eval_metric="logloss", verbosity=0,
            )

    elif task_type == "multilabel_classification":
        models["multilabel_logreg"] = "multilabel_logreg"
        models["multilabel_rf"] = "multilabel_rf"
        models["multilabel_mlp"] = "multilabel_mlp"

    return models


# ─── Evaluation ────────────────────────────────────────────────────────────


def evaluate_predictions(y_true, y_pred, y_score, task_type, labels=None):
    """Compute metrics for predictions."""
    result = {}

    if task_type in ("binary_classification",):
        try:
            result["auroc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            result["auroc"] = float("nan")
        try:
            result["auprc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            result["auprc"] = float("nan")
        result["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    elif task_type in ("multi_class_classification", "multi_class_feature_classification"):
        result["accuracy"] = float(accuracy_score(y_true, y_pred))
        result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    elif task_type == "multilabel_classification":
        result["micro_f1"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
        result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return result


def run_single_task(task_id, verbose=True):
    """Run all baselines on a single task."""
    if verbose:
        print(f"\n{'='*40}")
        print(f"  Task {task_id}")
        print(f"{'='*40}")

    X, y, task = load_task_data(task_id)
    # Use the actual task_id for split (B1 variants share B1 split)
    split_task_id = task.get("task_id", task_id)
    split_path = SPLIT_DIR / f"{task['split']}.json"
    with open(split_path) as f:
        splits = json.load(f)
    task_type = task["task_type"]

    # Handle NaN in features
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    # Encode labels for multiclass
    le = None
    if task_type in ("multi_class_classification", "multi_class_feature_classification"):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        y_encoded = y

    is_multilabel = (task_type == "multilabel_classification")
    n_classes = len(np.unique(y_encoded)) if not is_multilabel else y.shape[1]

    baselines = get_baselines(task_type, n_features=X.shape[1])
    results = {}

    for model_name, model in baselines.items():
        split_results = []
        per_fold_details = []

        for split_idx, split in enumerate(splits):
            train_idx = np.array(split["train_indices"])
            test_idx = np.array(split["test_indices"])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if model_name == "random":
                rng = np.random.RandomState(SEED + split_idx)
                if is_multilabel:
                    probs = y_train.mean(axis=0)
                    y_pred = (rng.random((len(y_test), y.shape[1])) < probs).astype(int)
                    y_score = rng.random((len(y_test), y.shape[1]))
                else:
                    # Class-proportional random (not uniform)
                    counts = Counter(y_train.tolist())
                    classes = sorted(counts.keys())
                    probs = np.array([counts[c] for c in classes], dtype=float)
                    probs /= probs.sum()
                    y_pred = rng.choice(classes, size=len(y_test), p=probs)
                    y_score = rng.random(len(y_test))

            elif model_name == "majority":
                if is_multilabel:
                    majority = (y_train.mean(axis=0) > 0.5).astype(int)
                    y_pred = np.tile(majority, (len(y_test), 1))
                    y_score = np.tile(y_train.mean(axis=0), (len(y_test), 1))
                else:
                    majority_class = Counter(y_train.tolist()).most_common(1)[0][0]
                    y_pred = np.full(len(y_test), majority_class)
                    y_score = np.zeros(len(y_test))

            elif model_name in ("multilabel_logreg", "multilabel_rf", "multilabel_mlp"):
                from sklearn.multiclass import OneVsRestClassifier
                if "logreg" in model_name:
                    base = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED)
                elif "rf" in model_name:
                    base = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1)
                else:
                    base = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=SEED,
                                         early_stopping=True, alpha=0.01)
                clf = OneVsRestClassifier(base)
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                y_score = clf.predict_proba(X_test_s) if hasattr(clf, "predict_proba") else y_pred.astype(float)

            else:
                # Standard sklearn model
                clf = model.__class__(**model.get_params())
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                if hasattr(clf, "predict_proba"):
                    proba = clf.predict_proba(X_test_s)
                    if proba.shape[1] == 2:
                        y_score = proba[:, 1]
                    else:
                        y_score = proba
                elif hasattr(clf, "decision_function"):
                    y_score = clf.decision_function(X_test_s)
                else:
                    y_score = y_pred.astype(float)

            metrics = evaluate_predictions(y_test, y_pred, y_score, task_type)
            split_results.append(metrics)

            # Per-fold details for LOCO splits
            fold_detail = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
            if "test_crew" in split:
                fold_detail["test_crew"] = split["test_crew"]
            per_fold_details.append(fold_detail)

        # Aggregate across splits
        agg = {}
        if split_results:
            keys = [k for k in split_results[0] if isinstance(split_results[0][k], (int, float))]
            for k in keys:
                vals = [s[k] for s in split_results if not np.isnan(s.get(k, float("nan")))]
                if vals:
                    agg[k] = {"mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4)}

        results[model_name] = agg
        if per_fold_details and "test_crew" in per_fold_details[0]:
            results[f"{model_name}_per_fold"] = per_fold_details

        # Print primary metric
        primary = task["evaluation"]["primary_metric"]
        if primary in agg:
            if verbose:
                print(f"  {model_name:20s}: {primary}={agg[primary]['mean']:.4f} +/- {agg[primary]['std']:.4f}")

    return results


# ─── Composite score ──────────────────────────────────────────────────────


def compute_composite(all_results, random_baselines):
    """Compute normalized composite score.

    Formula: normalized_i = (score_i - random_i) / (1.0 - random_i)
    Category score = mean(normalized tasks in category)
    Composite = mean(category scores)
    """
    # Categories with sub-task grouping for F
    CATEGORIES = {
        "A_clinical": ["A1", "A2"],
        "B_cfrna": ["B1", "B2"],
        "C_proteomics": ["C1", "C2"],
        "D_metabolomics": ["D1"],
        "E_spatial": ["E1", "E4"],  # E2/E3 supplementary excluded
        "F_bodysite": ["F1", "F4"],  # Grouped sub-tasks
        "F_phase": ["F2", "F5"],
        "F_source": ["F3"],
        "G_multimodal": ["G1"],
        "H_crosstissue": ["H1"],
        "I_crossmission": ["I1", "I2", "I3"],
    }

    # Collect all model names (exclude per_fold keys)
    all_models = set()
    for task_id, res in all_results.items():
        if task_id.startswith("_") or "error" in res:
            continue
        all_models.update(k for k in res if not k.endswith("_per_fold"))

    composite = {}
    for model_name in sorted(all_models):
        cat_scores = {}
        for cat, tasks in CATEGORIES.items():
            task_scores = []
            for tid in tasks:
                if tid not in all_results or "error" in all_results[tid]:
                    continue
                if model_name not in all_results[tid]:
                    continue
                task_path = TASK_DIR / f"{tid}.json"
                with open(task_path) as f:
                    task = json.load(f)
                primary = task["evaluation"]["primary_metric"]
                if primary in all_results[tid][model_name]:
                    score = all_results[tid][model_name][primary]["mean"]
                    # Get random baseline for normalization
                    rand_score = random_baselines.get(tid, 0.0)
                    if rand_score < 1.0:
                        normalized = (score - rand_score) / (1.0 - rand_score)
                    else:
                        normalized = 0.0
                    task_scores.append(max(0.0, normalized))  # Floor at 0
            if task_scores:
                cat_scores[cat] = round(np.mean(task_scores), 4)
        if cat_scores:
            composite[model_name] = {
                "category_scores": cat_scores,
                "composite": round(np.mean(list(cat_scores.values())), 4),
                "n_categories": len(cat_scores),
            }

    return composite


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 — ML Baselines")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run main + supplementary tasks
    for task_id in ALL_TASKS:
        try:
            results = run_single_task(task_id)
            all_results[task_id] = results
        except Exception as e:
            print(f"  {task_id}: ERROR -- {e}")
            import traceback
            traceback.print_exc()
            all_results[task_id] = {"error": str(e)}

    # B1 ablation study
    print("\n" + "=" * 60)
    print("  B1 Feature Ablation Study")
    print("=" * 60)
    for variant in ["B1_effect_only", "B1_no_effect"]:
        try:
            results = run_single_task(variant)
            all_results[variant] = results
        except Exception as e:
            print(f"  {variant}: ERROR -- {e}")
            all_results[variant] = {"error": str(e)}

    # Collect random baselines for normalization
    random_baselines = {}
    for task_id in ALL_TASKS:
        if task_id in all_results and "error" not in all_results[task_id]:
            if "random" in all_results[task_id]:
                task_path = TASK_DIR / f"{task_id}.json"
                with open(task_path) as f:
                    task = json.load(f)
                primary = task["evaluation"]["primary_metric"]
                if primary in all_results[task_id]["random"]:
                    random_baselines[task_id] = all_results[task_id]["random"][primary]["mean"]

    # Compute composite scores (normalized)
    composite = compute_composite(all_results, random_baselines)
    all_results["_composite"] = composite
    all_results["_random_baselines"] = random_baselines

    # Save results
    out_path = OUT_DIR / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY (main tasks)")
    print("=" * 70)
    header = f"{'Task':>5} {'Tier':>10} {'Primary':>8} {'Random':>8} {'Major':>8} {'LogReg':>8} {'RF':>8} {'MLP':>8}"
    print(header)
    print("-" * 70)

    for task_id in ALL_TASKS:
        if task_id not in all_results or "error" in all_results[task_id]:
            supp = " (S)" if task_id in SUPPLEMENTARY_TASKS else ""
            print(f"{task_id:>5}{supp} ERROR")
            continue

        task_path = TASK_DIR / f"{task_id}.json"
        with open(task_path) as f:
            task = json.load(f)
        primary = task["evaluation"]["primary_metric"]
        tier = task.get("difficulty_tier", "?")[:4]
        supp = "*" if task_id in SUPPLEMENTARY_TASKS else " "

        row = f"{task_id:>4}{supp} {tier:>10} {primary:>8}"
        for model in ["random", "majority", "logreg", "rf", "mlp",
                       "multilabel_logreg", "multilabel_rf", "multilabel_mlp"]:
            if model in all_results[task_id] and primary in all_results[task_id][model]:
                val = all_results[task_id][model][primary]["mean"]
                row += f" {val:>8.4f}"
        print(row)

    # B1 ablation
    print("\n" + "=" * 70)
    print("B1 ABLATION STUDY")
    print("=" * 70)
    for variant in ["B1", "B1_effect_only", "B1_no_effect"]:
        if variant in all_results and "error" not in all_results[variant]:
            row = f"  {variant:20s}:"
            for model in ["logreg", "rf", "mlp"]:
                if model in all_results[variant] and "auprc" in all_results[variant][model]:
                    val = all_results[variant][model]["auprc"]["mean"]
                    row += f"  {model}={val:.4f}"
            print(row)

    # Composite scores
    print("\n" + "=" * 70)
    print("NORMALIZED COMPOSITE SCORES")
    print("  Formula: (score - random) / (1 - random), averaged by category")
    print("=" * 70)
    for model, data in sorted(composite.items(), key=lambda x: -x[1]["composite"]):
        if model in ("random", "majority"):
            continue
        cats = " | ".join(f"{c}={v:.3f}" for c, v in sorted(data["category_scores"].items()))
        print(f"  {model:20s}: {data['composite']:.4f}  ({cats})")


if __name__ == "__main__":
    sys.exit(main() or 0)
