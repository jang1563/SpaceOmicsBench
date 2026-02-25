#!/usr/bin/env python3
"""
SpaceOmicsBench v2 — ML Baselines

Runs 5 baseline models on all 18 tasks:
  1. Random: random predictions matching class distribution
  2. Majority: always predict majority class
  3. LogReg: Logistic Regression (L2, class-weighted)
  4. RF: Random Forest (100 trees, class-weighted)
  5. XGBoost: XGBClassifier (class-weighted)

Output: baselines/baseline_results.json
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
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


# ─── Data loaders ──────────────────────────────────────────────────────────


def load_task_data(task_id):
    """Load features (X) and labels (y) for a task.

    Returns:
        X: np.ndarray (N, D)
        y: np.ndarray (N,) for binary/multiclass, (N, L) for multilabel
        task: dict (task JSON)
    """
    task_path = TASK_DIR / f"{task_id}.json"
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

    elif task_id == "B1":
        de = pd.read_csv(DATA_DIR / "cfrna_3group_de_noleak.csv")
        drr = pd.read_csv(DATA_DIR / "cfrna_466drr.csv")
        drr_genes = set(drr.iloc[:, 0])
        X = de.drop(columns=["gene"]).values.astype(float)
        y = np.array([1 if g in drr_genes else 0 for g in de["gene"]], dtype=int)

    elif task_id == "B2":
        corr = pd.read_csv(DATA_DIR / "gt_cfrna_correlation.csv")
        labels = pd.read_csv(DATA_DIR / "gt_cfrna_cluster_labels.csv")
        gene_col = corr.columns[0]
        X = corr.drop(columns=[gene_col]).values.astype(float)
        label_cols = [c for c in labels.columns if c != labels.columns[0]]
        y = labels[label_cols].values.astype(int)

    elif task_id == "C1":
        matrix = pd.read_csv(DATA_DIR / "proteomics_plasma_matrix.csv")
        de = pd.read_csv(DATA_DIR / "proteomics_plasma_de_clean.csv")
        # Align proteins — each row = protein, columns = sample expression values
        meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days",
                     "phase", "mission", "month_tp"]
        sample_cols = [c for c in matrix.columns if c not in meta_cols]
        X = matrix[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0).T.values
        y = (de["adj_pval"] < 0.05).astype(int).values

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
        # Encode categorical features
        feature_cols = ["Mass", "RT", "annotation_confidence"]
        X_num = df[feature_cols].fillna(0).values.astype(float)
        # One-hot SuperPathway
        if "SuperPathway" in df.columns:
            sp_dummies = pd.get_dummies(df["SuperPathway"], prefix="SP").values.astype(float)
            X = np.hstack([X_num, sp_dummies])
        else:
            X = X_num
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
        # Match sample order to metadata
        X = tax[sample_cols].T.values.astype(float)
        if task_id == "F1":
            y = human["body_site"].values
        else:
            y = human["phase"].values

    elif task_id == "F3":
        human_tax = pd.read_csv(DATA_DIR / "microbiome_human_taxonomy_cpm.csv")
        env_tax = pd.read_csv(DATA_DIR / "microbiome_env_taxonomy_cpm.csv")
        md = pd.read_csv(DATA_DIR / "microbiome_metadata.csv")
        # Align by taxid
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
        # Only keep samples in human metadata (275)
        valid_samples = [c for c in sample_cols if c in set(human["sample_id"].values) or
                         any(c.startswith(crew) for crew in ["C001", "C002", "C003", "C004"])]
        X = path[valid_samples[:275]].T.values.astype(float)
        if task_id == "F4":
            y = human["body_site"].values[:X.shape[0]]
        else:
            y = human["phase"].values[:X.shape[0]]

    elif task_id == "G1":
        cbc = pd.read_csv(DATA_DIR / "clinical_cbc.csv")
        cmp = pd.read_csv(DATA_DIR / "clinical_cmp.csv")
        prot_meta = pd.read_csv(DATA_DIR / "proteomics_metadata.csv")
        prot_plasma = prot_meta[prot_meta["tissue"] == "plasma"]
        met_meta = pd.read_csv(DATA_DIR / "metabolomics_metadata.csv")
        # Find matched samples
        cbc_keys = set(zip(cbc["crew"], cbc["timepoint_days"]))
        prot_keys = set(zip(prot_plasma["crew"], prot_plasma["timepoint_days"]))
        met_keys = set(zip(met_meta["crew"], met_meta["timepoint_days"]))
        matched = sorted(cbc_keys & prot_keys & met_keys)
        matched_set = set(matched)
        mask = [(row.crew, row.timepoint_days) in matched_set for row in cbc.itertuples()]
        meta_cols = ["sample_id", "crew", "tissue", "timepoint", "timepoint_days", "phase", "mission"]
        X_cbc = cbc[mask].drop(columns=[c for c in meta_cols if c in cbc.columns]).values.astype(float)
        X_cmp = cmp[mask].drop(columns=[c for c in meta_cols if c in cmp.columns]).values.astype(float)
        X = np.hstack([X_cbc, X_cmp])  # Just clinical for G1 baseline (full multimodal is complex)
        y = cbc[mask]["phase"].values

    elif task_id == "H1":
        df = pd.read_csv(DATA_DIR / "conserved_pbmc_to_skin.csv")
        feature_cols = ["CD4_T", "CD8_T", "other_T", "B", "NK", "CD14_Mono", "CD16_Mono", "DC", "other"]
        X = df[feature_cols].values.astype(float)
        y = df["skin_de"].astype(int).values

    else:
        raise ValueError(f"Unknown task: {task_id}")

    return X, y, task


def load_splits(task):
    """Load split indices for a task."""
    split_path = SPLIT_DIR / f"{task['split']}.json"
    with open(split_path) as f:
        return json.load(f)


# ─── Baseline models ───────────────────────────────────────────────────────


def get_baselines(task_type, n_classes=2):
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
        if HAS_XGB:
            models["xgboost"] = XGBClassifier(
                n_estimators=100, max_depth=6,
                random_state=SEED, use_label_encoder=False,
                eval_metric="logloss", verbosity=0,
            )

    elif task_type == "multilabel_classification":
        # For multi-label, use binary relevance with each model
        models["multilabel_logreg"] = "multilabel_logreg"
        models["multilabel_rf"] = "multilabel_rf"

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
    splits = load_splits(task)
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

    baselines = get_baselines(task_type, n_classes)
    results = {}

    for model_name, model in baselines.items():
        split_results = []

        for split in splits:
            train_idx = np.array(split["train_indices"])
            test_idx = np.array(split["test_indices"])

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            if model_name == "random":
                rng = np.random.RandomState(SEED)
                if is_multilabel:
                    # Random multilabel: predict each label with its marginal probability
                    probs = y_train.mean(axis=0)
                    y_pred = (rng.random((len(y_test), y.shape[1])) < probs).astype(int)
                    y_score = rng.random((len(y_test), y.shape[1]))
                else:
                    unique_classes = np.unique(y_train)
                    y_pred = rng.choice(unique_classes, size=len(y_test))
                    y_score = rng.random(len(y_test))

            elif model_name == "majority":
                if is_multilabel:
                    majority = (y_train.mean(axis=0) > 0.5).astype(int)
                    y_pred = np.tile(majority, (len(y_test), 1))
                    y_score = np.tile(y_train.mean(axis=0), (len(y_test), 1))
                else:
                    from collections import Counter
                    majority_class = Counter(y_train).most_common(1)[0][0]
                    y_pred = np.full(len(y_test), majority_class)
                    y_score = np.zeros(len(y_test))

            elif model_name == "multilabel_logreg":
                from sklearn.multiclass import OneVsRestClassifier
                clf = OneVsRestClassifier(LogisticRegression(
                    max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED))
                clf.fit(X_train_s, y_train)
                y_pred = clf.predict(X_test_s)
                y_score = clf.predict_proba(X_test_s) if hasattr(clf, "predict_proba") else y_pred.astype(float)

            elif model_name == "multilabel_rf":
                from sklearn.multiclass import OneVsRestClassifier
                clf = OneVsRestClassifier(RandomForestClassifier(
                    n_estimators=100, class_weight="balanced", random_state=SEED, n_jobs=-1))
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

        # Aggregate across splits
        agg = {}
        if split_results:
            keys = [k for k in split_results[0] if isinstance(split_results[0][k], (int, float))]
            for k in keys:
                vals = [s[k] for s in split_results if not np.isnan(s.get(k, float("nan")))]
                if vals:
                    agg[k] = {"mean": round(np.mean(vals), 4), "std": round(np.std(vals), 4)}

        results[model_name] = agg

        # Print primary metric
        primary = task["evaluation"]["primary_metric"]
        if primary in agg:
            if verbose:
                print(f"  {model_name:10s}: {primary}={agg[primary]['mean']:.4f} ± {agg[primary]['std']:.4f}")

    return results


# ─── Main ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 60)
    print("SpaceOmicsBench v2 — ML Baselines")
    print("=" * 60)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_tasks = [
        "A1", "A2", "B1", "B2", "C1", "C2", "D1",
        "E1", "E2", "E3", "E4",
        "F1", "F2", "F3", "F4", "F5",
        "G1", "H1",
    ]

    all_results = {}
    for task_id in all_tasks:
        try:
            results = run_single_task(task_id)
            all_results[task_id] = results
        except Exception as e:
            print(f"  {task_id}: ERROR — {e}")
            import traceback
            traceback.print_exc()
            all_results[task_id] = {"error": str(e)}

    # Compute composite scores
    composite = compute_composite(all_results)
    all_results["_composite"] = composite

    # Save results
    out_path = OUT_DIR / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {out_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Task':>5} {'Primary':>12} {'Random':>10} {'Majority':>10} {'LogReg':>10} {'RF':>10} {'XGBoost':>10}")
    print("-" * 67)

    for task_id in all_tasks:
        if task_id not in all_results or "error" in all_results[task_id]:
            print(f"{task_id:>5} ERROR")
            continue

        task_path = TASK_DIR / f"{task_id}.json"
        with open(task_path) as f:
            task = json.load(f)
        primary = task["evaluation"]["primary_metric"]

        row = f"{task_id:>5} {primary:>12}"
        for model in ["random", "majority", "logreg", "rf", "xgboost",
                      "multilabel_logreg", "multilabel_rf"]:
            if model in all_results[task_id] and primary in all_results[task_id][model]:
                val = all_results[task_id][model][primary]["mean"]
                row += f" {val:>10.4f}"
            else:
                row += f" {'—':>10}"
        print(row)

    # Print composite scores
    print("\n" + "=" * 60)
    print("COMPOSITE SCORES (category-average → overall-average)")
    print("=" * 60)
    for model, data in sorted(composite.items()):
        cats = " | ".join(f"{c}={v:.3f}" for c, v in sorted(data["category_scores"].items()))
        print(f"  {model:20s}: {data['composite']:.4f}  ({cats})")


def compute_composite(all_results):
    """Compute composite score: category-average → overall-average.

    Equal weight per modality regardless of task count.
    """
    CATEGORIES = {
        "A": ["A1", "A2"],
        "B": ["B1", "B2"],
        "C": ["C1", "C2"],
        "D": ["D1"],
        "E": ["E1", "E2", "E3", "E4"],
        "F": ["F1", "F2", "F3", "F4", "F5"],
        "G": ["G1"],
        "H": ["H1"],
    }

    # Collect all model names across tasks
    all_models = set()
    for task_id, res in all_results.items():
        if "error" not in res:
            all_models.update(res.keys())

    composite = {}
    for model in sorted(all_models):
        cat_scores = {}
        for cat, tasks in CATEGORIES.items():
            task_scores = []
            for tid in tasks:
                if tid not in all_results or "error" in all_results[tid]:
                    continue
                if model not in all_results[tid]:
                    continue
                task_path = TASK_DIR / f"{tid}.json"
                with open(task_path) as f:
                    task = json.load(f)
                primary = task["evaluation"]["primary_metric"]
                if primary in all_results[tid][model]:
                    task_scores.append(all_results[tid][model][primary]["mean"])
            if task_scores:
                cat_scores[cat] = round(np.mean(task_scores), 4)
        if cat_scores:
            composite[model] = {
                "category_scores": cat_scores,
                "composite": round(np.mean(list(cat_scores.values())), 4),
                "n_categories": len(cat_scores),
            }

    return composite


if __name__ == "__main__":
    sys.exit(main() or 0)
