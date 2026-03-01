#!/usr/bin/env python3
"""
SpaceOmicsBench v2 — Evaluation Harness

Loads task definitions and split files, applies a model's predictions,
and computes all relevant metrics per task.

Usage:
  python eval_harness.py --task all --predictions results/my_model/
  python eval_harness.py --task A1 --predictions results/my_model/ --output results/my_model/
  python eval_harness.py --dry-run          # verify all tasks & splits load correctly

Prediction file format:
  For each task, provide a JSON file named {task_id}.json in the predictions directory:
    - Sample-level classification: {"predictions": [{"split_idx": 0, "y_pred": [...]}]}
    - Feature-level binary: {"predictions": [{"rep": 0, "y_score": [...]}]}
    - Feature-level multi-label: {"predictions": [{"rep": 0, "y_pred": [[...]], "y_score": [[...]]}]}
    - Feature-level multi-class: {"predictions": [{"rep": 0, "y_pred": [...]}]}
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import (
    binary_metrics,
    bootstrap_ci,
    classification_metrics,
    direction_concordance,
    multiclass_metrics,
    multilabel_metrics,
    ranking_metrics,
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
TASK_DIR = BASE_DIR / "tasks"
SPLIT_DIR = BASE_DIR / "splits"

# All 19 main task IDs
MAIN_TASKS = [
    "A1", "A2",
    "B1", "B2",
    "C1", "C2",
    "D1",
    "E1", "E4",
    "F1", "F2", "F3", "F4", "F5",
    "G1",
    "H1",
    "I1", "I2", "I3",
]
SUPPLEMENTARY_TASKS = ["E2", "E3"]  # frontier, metric instability
ALL_TASKS = MAIN_TASKS + SUPPLEMENTARY_TASKS


class SpaceOmicsBenchEvaluator:
    """Main evaluator for SpaceOmicsBench v2."""

    def __init__(self, data_dir=None, task_dir=None, split_dir=None):
        self.data_dir = Path(data_dir) if data_dir else DATA_DIR
        self.task_dir = Path(task_dir) if task_dir else TASK_DIR
        self.split_dir = Path(split_dir) if split_dir else SPLIT_DIR

        self.tasks = {}
        self.splits = {}

    def load_task(self, task_id):
        """Load a task definition JSON."""
        path = self.task_dir / f"{task_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Task definition not found: {path}")
        with open(path) as f:
            self.tasks[task_id] = json.load(f)
        return self.tasks[task_id]

    def load_split(self, split_name):
        """Load a split JSON (cached)."""
        if split_name in self.splits:
            return self.splits[split_name]
        path = self.split_dir / f"{split_name}.json"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        with open(path) as f:
            self.splits[split_name] = json.load(f)
        return self.splits[split_name]

    def load_ground_truth(self, task_id):
        """Load ground truth labels for a task.

        Returns:
            labels: np.ndarray of labels (1D for binary/multiclass, 2D for multilabel)
            meta: dict with extra info (class names, gene names, etc.)
        """
        task = self.tasks[task_id]
        meta = {}

        if task_id in ("A1", "A2"):
            return self._load_clinical_labels(task_id, meta)
        elif task_id == "B1":
            return self._load_b1_labels(meta)
        elif task_id == "B2":
            return self._load_b2_labels(meta)
        elif task_id == "C1":
            return self._load_c1_labels(meta)
        elif task_id == "C2":
            return self._load_c2_labels(meta)
        elif task_id == "D1":
            return self._load_d1_labels(meta)
        elif task_id.startswith("E"):
            return self._load_spatial_labels(task_id, meta)
        elif task_id in ("F1", "F2", "F4", "F5"):
            return self._load_microbiome_labels(task_id, meta)
        elif task_id == "F3":
            return self._load_f3_labels(meta)
        elif task_id == "G1":
            return self._load_g1_labels(meta)
        elif task_id == "H1":
            return self._load_h1_labels(meta)
        elif task_id in ("I1", "I2", "I3"):
            return self._load_cross_mission_labels(task_id, meta)
        else:
            raise ValueError(f"Unknown task: {task_id}")

    # ─── Label loaders ─────────────────────────────────────────────

    def _load_clinical_labels(self, task_id, meta):
        cbc = pd.read_csv(self.data_dir / "clinical_cbc.csv")
        labels = cbc["phase"].values
        meta["crews"] = cbc["crew"].values.tolist()
        meta["classes"] = ["pre_flight", "post_flight", "recovery"]
        if task_id == "A2":
            meta["data_file"] = "clinical_cytokines_eve.csv"
        return labels, meta

    def _load_b1_labels(self, meta):
        de = pd.read_csv(self.data_dir / "cfrna_3group_de_noleak.csv")
        drr = pd.read_csv(self.data_dir / "cfrna_466drr.csv")
        drr_genes = set(drr.iloc[:, 0].values)
        labels = np.array([1 if g in drr_genes else 0 for g in de.iloc[:, 0]], dtype=int)
        meta["genes"] = de.iloc[:, 0].values.tolist()
        return labels, meta

    def _load_b2_labels(self, meta):
        cluster_labels = pd.read_csv(self.data_dir / "gt_cfrna_cluster_labels.csv")
        gene_col = cluster_labels.columns[0]
        label_cols = [c for c in cluster_labels.columns if c != gene_col]
        labels = cluster_labels[label_cols].values.astype(int)
        meta["genes"] = cluster_labels[gene_col].values.tolist()
        meta["cluster_names"] = label_cols
        return labels, meta

    def _load_c1_labels(self, meta):
        # Redesigned: sample-level proteomics phase classification
        matrix = pd.read_csv(self.data_dir / "proteomics_plasma_matrix.csv")
        labels = matrix["phase"].values
        meta["crews"] = matrix["crew"].values.tolist()
        meta["classes"] = ["pre_flight", "post_flight", "recovery"]
        return labels, meta

    def _load_c2_labels(self, meta):
        plasma = pd.read_csv(self.data_dir / "proteomics_plasma_de_clean.csv")
        evp = pd.read_csv(self.data_dir / "proteomics_evp_de_clean.csv")
        # Find overlapping proteins
        p_col = "protein" if "protein" in plasma.columns else plasma.columns[0]
        e_col = "protein" if "protein" in evp.columns else evp.columns[0]
        overlap = set(plasma[p_col]) & set(evp[e_col])
        plasma_sub = plasma[plasma[p_col].isin(overlap)].set_index(p_col)
        evp_sub = evp[evp[e_col].isin(overlap)].set_index(e_col)
        # Align by protein
        common = sorted(overlap)
        labels = (evp_sub.loc[common, "adj_pval"] < 0.05).astype(int).values
        meta["proteins"] = common
        meta["plasma_logfc"] = plasma_sub.loc[common, "logFC"].values.tolist()
        meta["evp_logfc"] = evp_sub.loc[common, "logFC"].values.tolist()
        return labels, meta

    def _load_d1_labels(self, meta):
        # Metabolite spaceflight response prediction
        df = pd.read_csv(self.data_dir / "metabolomics_spaceflight_response.csv")
        labels = df["is_spaceflight_de"].astype(int).values
        meta["metabolites"] = df["metabolite"].values.tolist()
        return labels, meta

    def _load_spatial_labels(self, task_id, meta):
        layer_map = {"E1": "outer_epidermis", "E2": "inner_epidermis",
                     "E3": "outer_dermis", "E4": "epidermis"}
        layer = layer_map[task_id]
        layer_de = pd.read_csv(self.data_dir / f"gt_spatial_de_{layer}.csv")
        labels = (layer_de["adj_pval"].fillna(1.0) < 0.05).astype(int).values
        meta["genes"] = layer_de["gene"].values.tolist()
        meta["layer"] = layer
        return labels, meta

    def _load_microbiome_labels(self, task_id, meta):
        md = pd.read_csv(self.data_dir / "microbiome_metadata.csv")
        human = md[md["source"] == "human"].reset_index(drop=True)
        if task_id in ("F1", "F4"):
            labels = human["body_site"].values
            meta["classes"] = sorted(set(labels))
        elif task_id in ("F2", "F5"):
            labels = human["phase"].values
            meta["classes"] = ["pre_flight", "in_flight", "post_flight", "recovery"]
        meta["crews"] = human["crew"].values.tolist()
        return labels, meta

    def _load_f3_labels(self, meta):
        md = pd.read_csv(self.data_dir / "microbiome_metadata.csv")
        # Human + environmental samples
        mask = md["source"].isin(["human", "environmental"])
        sub = md[mask].reset_index(drop=True)
        labels = (sub["source"] == "human").astype(int).values
        meta["classes"] = ["environmental", "human"]
        meta["timepoints"] = sub["timepoint"].values.tolist()
        return labels, meta

    def _load_g1_labels(self, meta):
        # Multi-modal: only matched samples across clinical + proteomics + metabolomics
        cbc = pd.read_csv(self.data_dir / "clinical_cbc.csv")
        prot_meta = pd.read_csv(self.data_dir / "proteomics_metadata.csv")
        prot_plasma = prot_meta[prot_meta["tissue"] == "plasma"]
        met_meta = pd.read_csv(self.data_dir / "metabolomics_metadata.csv")

        cbc_keys = set(zip(cbc["crew"], cbc["timepoint_days"]))
        prot_keys = set(zip(prot_plasma["crew"], prot_plasma["timepoint_days"]))
        met_keys = set(zip(met_meta["crew"], met_meta["timepoint_days"]))
        matched = sorted(cbc_keys & prot_keys & met_keys)
        matched_set = set(matched)

        mask = [
            (row.crew, row.timepoint_days) in matched_set
            for row in cbc.itertuples()
        ]
        cbc_matched = cbc[mask].reset_index(drop=True)
        labels = cbc_matched["phase"].values
        meta["crews"] = cbc_matched["crew"].values.tolist()
        meta["classes"] = ["pre_flight", "post_flight", "recovery"]
        return labels, meta

    def _load_h1_labels(self, meta):
        df = pd.read_csv(self.data_dir / "conserved_pbmc_to_skin.csv")
        labels = df["skin_de"].astype(int).values
        meta["genes"] = df["human_gene"].values.tolist()
        return labels, meta

    def _load_cross_mission_labels(self, task_id, meta):
        file_map = {
            "I1": "cross_mission_hemoglobin_de.csv",
            "I2": "cross_mission_pathway_features.csv",
            "I3": "cross_mission_gene_de.csv",
        }
        id_col_map = {"I1": "gene", "I2": "pathway", "I3": "gene"}
        df = pd.read_csv(self.data_dir / file_map[task_id])
        labels = df["label"].astype(int).values
        meta["ids"] = df[id_col_map[task_id]].values.tolist()
        return labels, meta

    # ─── Evaluation dispatcher ─────────────────────────────────────

    def evaluate(self, task_id, predictions, n_bootstrap=1000):
        """Evaluate predictions for a single task.

        Args:
            task_id: task identifier (e.g., "A1")
            predictions: list of dicts, one per split/rep
            n_bootstrap: number of bootstrap samples for CI

        Returns:
            dict with per-split and aggregated metrics
        """
        task = self.tasks.get(task_id) or self.load_task(task_id)
        labels, meta = self.load_ground_truth(task_id)
        splits = self.load_split(task["split"])

        task_type = task["task_type"]
        primary = task["evaluation"]["primary_metric"]

        results = {
            "task_id": task_id,
            "task_name": task["task_name"],
            "task_type": task_type,
            "n_samples": task["n_samples"],
            "primary_metric": primary,
            "per_split": [],
        }

        for i, split in enumerate(splits):
            test_idx = split["test_indices"]
            pred = predictions[i] if i < len(predictions) else None
            if pred is None:
                continue

            y_true_test = labels[test_idx] if labels.ndim == 1 else labels[test_idx]

            if task_type == "multi_class_classification":
                y_pred = np.asarray(pred["y_pred"])
                split_metrics = classification_metrics(
                    y_true_test, y_pred,
                    labels=meta.get("classes"),
                )

            elif task_type == "binary_classification":
                y_score = np.asarray(pred["y_score"])
                split_metrics = binary_metrics(y_true_test, y_score)
                rank = ranking_metrics(y_true_test, y_score)
                split_metrics.update(rank)

            elif task_type == "multilabel_classification":
                y_pred = np.asarray(pred["y_pred"])
                y_score = np.asarray(pred.get("y_score")) if "y_score" in pred else None
                split_metrics = multilabel_metrics(
                    y_true_test, y_pred, y_score,
                    label_names=meta.get("cluster_names"),
                )

            elif task_type == "multi_class_feature_classification":
                y_pred = np.asarray(pred["y_pred"])
                split_metrics = multiclass_metrics(
                    y_true_test, y_pred,
                    labels=meta.get("classes"),
                )

            else:
                raise ValueError(f"Unknown task_type: {task_type}")

            # Add split info
            split_info = {k: v for k, v in split.items()
                          if k not in ("train_indices", "test_indices")}
            split_metrics["split_info"] = split_info
            results["per_split"].append(split_metrics)

        # Aggregate across splits
        if results["per_split"]:
            results["aggregate"] = self._aggregate_splits(results["per_split"], primary)

        # Direction concordance for C2
        if task_id == "C2" and "plasma_logfc" in meta and "evp_logfc" in meta:
            results["direction_concordance"] = direction_concordance(
                meta["plasma_logfc"], meta["evp_logfc"]
            )

        return results

    def _aggregate_splits(self, per_split, primary_metric):
        """Aggregate metrics across splits/reps."""
        agg = {}
        # Collect numeric metrics
        all_keys = set()
        for s in per_split:
            all_keys.update(k for k, v in s.items()
                            if isinstance(v, (int, float)) and not np.isnan(v))
        for key in sorted(all_keys):
            vals = [s[key] for s in per_split if key in s and not np.isnan(s[key])]
            if vals:
                agg[key] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                }
        if primary_metric in agg:
            agg["primary"] = agg[primary_metric]
        return agg

    # ─── Dry run ───────────────────────────────────────────────────

    def dry_run(self, task_ids=None):
        """Verify all tasks and splits load correctly without predictions."""
        if task_ids is None:
            task_ids = ALL_TASKS

        print("=" * 60)
        print("SpaceOmicsBench v2 — Dry Run")
        print("=" * 60)

        errors = []
        for tid in task_ids:
            try:
                task = self.load_task(tid)
                labels, meta = self.load_ground_truth(tid)
                splits = self.load_split(task["split"])

                # Verify dimensions
                if labels.ndim == 1:
                    n_labels = len(labels)
                else:
                    n_labels = labels.shape[0]

                n_expected = task["n_samples"]
                # For LOCO/LOTO splits, check total coverage
                all_test = set()
                all_train = set()
                for s in splits:
                    all_test.update(s["test_indices"])
                    all_train.update(s["train_indices"])
                total_covered = len(all_test | all_train)

                status = "OK"
                notes = []

                if n_labels != n_expected:
                    notes.append(f"label count {n_labels} != expected {n_expected}")

                if total_covered != n_labels:
                    notes.append(f"split coverage {total_covered} != {n_labels}")

                if labels.ndim == 1:
                    unique = np.unique(labels[~pd.isna(labels)] if hasattr(labels[0], '__len__') is False else labels)
                    class_info = f"{len(unique)} classes"
                    if len(unique) == 2:
                        pos_rate = (labels == 1).mean() if labels.dtype in (int, np.int64) else "N/A"
                        class_info = f"binary, {pos_rate:.1%} positive"
                else:
                    class_info = f"multilabel {labels.shape}"

                n_splits = len(splits)
                split_type = task["split"]

                if notes:
                    status = "WARN"

                print(f"  [{status}] {tid}: {task['task_name']}")
                print(f"       N={n_labels}, {class_info}, {n_splits} splits ({split_type})")
                if notes:
                    for note in notes:
                        print(f"       ⚠ {note}")

            except Exception as e:
                errors.append((tid, str(e)))
                print(f"  [FAIL] {tid}: {e}")

        print()
        if errors:
            print(f"ERRORS: {len(errors)} tasks failed")
            for tid, err in errors:
                print(f"  {tid}: {err}")
            return False
        else:
            print(f"All {len(task_ids)} tasks verified successfully.")
            return True

    # ─── Full benchmark ───────────────────────────────────────────

    def run_full_benchmark(self, predictions_dir, task_ids=None, output_path=None):
        """Run evaluation on all tasks.

        Args:
            predictions_dir: directory with {task_id}.json prediction files
            task_ids: list of task IDs (default: all)
            output_path: where to save results JSON
        """
        pred_dir = Path(predictions_dir)
        if task_ids is None:
            task_ids = ALL_TASKS

        all_results = {
            "benchmark": "SpaceOmicsBench",
            "version": "2.0",
            "n_tasks": len(task_ids),
            "results": {},
        }

        for tid in task_ids:
            pred_path = pred_dir / f"{tid}.json"
            if not pred_path.exists():
                print(f"  Skip {tid}: no predictions file")
                continue

            with open(pred_path) as f:
                predictions = json.load(f)["predictions"]

            try:
                result = self.evaluate(tid, predictions)
                all_results["results"][tid] = result
                primary = result.get("aggregate", {}).get("primary", {})
                if primary:
                    print(f"  {tid}: {result['primary_metric']}="
                          f"{primary['mean']:.4f} ± {primary['std']:.4f}")
                else:
                    print(f"  {tid}: evaluated (no aggregate)")
            except Exception as e:
                print(f"  {tid}: ERROR — {e}")
                all_results["results"][tid] = {"error": str(e)}

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)
            print(f"\nResults saved to: {output_path}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description="SpaceOmicsBench v2 Evaluator")
    parser.add_argument("--task", default="all",
                        help="Task ID or 'all' (default: all)")
    parser.add_argument("--predictions", type=str, default=None,
                        help="Directory with prediction JSON files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--dry-run", action="store_true",
                        help="Verify tasks & splits without predictions")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--task-dir", type=str, default=None)
    parser.add_argument("--split-dir", type=str, default=None)

    args = parser.parse_args()

    evaluator = SpaceOmicsBenchEvaluator(
        data_dir=args.data_dir,
        task_dir=args.task_dir,
        split_dir=args.split_dir,
    )

    if args.dry_run:
        task_ids = ALL_TASKS if args.task == "all" else [args.task]
        success = evaluator.dry_run(task_ids)
        sys.exit(0 if success else 1)

    if args.predictions is None:
        parser.error("--predictions is required unless --dry-run")

    task_ids = ALL_TASKS if args.task == "all" else [args.task]
    evaluator.run_full_benchmark(args.predictions, task_ids, args.output)


if __name__ == "__main__":
    main()
