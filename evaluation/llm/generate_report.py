#!/usr/bin/env python3
"""
SpaceOmicsBench v2 - LLM Evaluation Report Generator
=====================================================

Generates a comprehensive markdown report from scored LLM evaluation results.
Supports single-model reports and multi-model comparison tables.

Usage:
    # Single model report
    python generate_report.py results/scored_eval_claude-sonnet-4-20250514_*.json

    # Multi-model comparison
    python generate_report.py results/scored_eval_*.json --compare

    # Custom output
    python generate_report.py results/scored_*.json --output report.md
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
ANNOTATION_SCHEMA_FILE = SCRIPT_DIR / "annotation_schema.json"


def _load_dimension_weights() -> Dict[str, float]:
    """Load dimension weights from annotation_schema.json (single source of truth)."""
    with open(ANNOTATION_SCHEMA_FILE) as f:
        schema = json.load(f)
    return {dim: cfg["weight"] for dim, cfg in schema["rating_dimensions"].items()}


DIMENSION_WEIGHTS = _load_dimension_weights()

DIFFICULTY_ORDER = ["easy", "medium", "hard", "expert"]
MODALITY_ORDER = [
    "clinical", "transcriptomics", "proteomics", "metabolomics",
    "spatial", "microbiome", "cross_mission", "multi_omics", "methods",
]


def load_scored_file(fpath: str) -> Dict[str, Any]:
    """Load a scored results JSON file."""
    with open(fpath) as f:
        return json.load(f)


def extract_model_name(data: Dict) -> str:
    """Extract a short model name from metadata."""
    model = data.get("metadata", {}).get("model", "unknown")
    # Shorten common model names for display
    shortcuts = {
        "claude-sonnet-4-20250514": "Claude Sonnet 4",
        "claude-opus-4-20250514": "Claude Opus 4",
        "gpt-4o-2024-08-06": "GPT-4o",
        "gpt-4o-mini-2024-07-18": "GPT-4o Mini",
    }
    return shortcuts.get(model, model)


def compute_aggregates(results: List[Dict]) -> Dict[str, Any]:
    """Compute per-dimension, per-modality, per-difficulty aggregates."""
    agg = {}

    ok = [r for r in results if r.get("scores", {}).get("success")]
    if not ok:
        return agg

    # Overall dimension averages
    for dim in DIMENSION_WEIGHTS:
        vals = [r["scores"][dim] for r in ok if dim in r["scores"]]
        agg[f"overall_{dim}"] = round(sum(vals) / len(vals), 3) if vals else None

    # Overall weighted
    ws = [r["scores"]["weighted_score"] for r in ok if "weighted_score" in r["scores"]]
    agg["overall_weighted"] = round(sum(ws) / len(ws), 3) if ws else None

    # By difficulty
    agg["by_difficulty"] = {}
    for diff in DIFFICULTY_ORDER:
        subset = [r for r in ok if r.get("difficulty") == diff and "weighted_score" in r["scores"]]
        if subset:
            vals = [r["scores"]["weighted_score"] for r in subset]
            agg["by_difficulty"][diff] = {
                "mean": round(sum(vals) / len(vals), 3),
                "n": len(subset),
                "min": round(min(vals), 3),
                "max": round(max(vals), 3),
            }

    # By modality
    agg["by_modality"] = {}
    for mod in MODALITY_ORDER:
        subset = [r for r in ok if r.get("modality") == mod and "weighted_score" in r["scores"]]
        if subset:
            vals = [r["scores"]["weighted_score"] for r in subset]
            agg["by_modality"][mod] = {
                "mean": round(sum(vals) / len(vals), 3),
                "n": len(subset),
            }

    # Flags
    agg["flags"] = {}
    for flag in ["hallucination", "factual_error", "harmful_recommendation",
                  "exceeds_data_scope", "novel_insight"]:
        agg["flags"][flag] = sum(
            1 for r in ok if r.get("scores", {}).get("flags", {}).get(flag, False))

    agg["n_scored"] = len(ok)
    agg["n_total"] = len(results)

    return agg


def generate_single_report(data: Dict, agg: Dict) -> str:
    """Generate markdown report for a single model."""
    meta = data.get("metadata", {})
    model = meta.get("model", "unknown")
    ts = meta.get("scoring_timestamp", "")
    n = agg.get("n_scored", 0)
    total = agg.get("n_total", 0)

    lines = [
        f"# SpaceOmicsBench v2 - LLM Evaluation Report",
        f"",
        f"**Model:** {model}",
        f"**Date:** {ts[:10] if ts else 'N/A'}",
        f"**Questions scored:** {n}/{total}",
        f"",
        f"---",
        f"",
        f"## Overall Scores",
        f"",
        f"| Dimension | Weight | Score |",
        f"|-----------|--------|-------|",
    ]
    for dim, w in DIMENSION_WEIGHTS.items():
        v = agg.get(f"overall_{dim}")
        score_str = f"{v:.2f}" if v is not None else "N/A"
        lines.append(f"| {dim.replace('_', ' ').title()} | {w:.2f} | {score_str}/5 |")

    ws = agg.get("overall_weighted")
    lines.append(f"| **Weighted Total** | **1.00** | **{ws:.2f}/5** |" if ws else "")
    lines.append("")

    # By difficulty
    lines.extend([
        "## Scores by Difficulty",
        "",
        "| Difficulty | N | Mean | Min | Max |",
        "|------------|---|------|-----|-----|",
    ])
    for diff in DIFFICULTY_ORDER:
        d = agg.get("by_difficulty", {}).get(diff)
        if d:
            lines.append(f"| {diff.title()} | {d['n']} | {d['mean']:.2f} | {d['min']:.2f} | {d['max']:.2f} |")
    lines.append("")

    # By modality
    lines.extend([
        "## Scores by Modality",
        "",
        "| Modality | N | Mean Score |",
        "|----------|---|------------|",
    ])
    for mod in MODALITY_ORDER:
        d = agg.get("by_modality", {}).get(mod)
        if d:
            lines.append(f"| {mod.replace('_', ' ').title()} | {d['n']} | {d['mean']:.2f} |")
    lines.append("")

    # Flags
    flags = agg.get("flags", {})
    lines.extend([
        "## Quality Flags",
        "",
        "| Flag | Count |",
        "|------|-------|",
    ])
    for flag, count in flags.items():
        lines.append(f"| {flag.replace('_', ' ').title()} | {count} |")
    lines.append("")

    # Per-question detail
    results = data.get("results", [])
    ok = [r for r in results if r.get("scores", {}).get("success")]
    if ok:
        lines.extend([
            "## Per-Question Results",
            "",
            "| ID | Modality | Difficulty | Score | Flags |",
            "|----|----------|------------|-------|-------|",
        ])
        for r in sorted(ok, key=lambda x: x.get("question_id", "")):
            qid = r.get("question_id", "?")
            mod = r.get("modality", "?")
            diff = r.get("difficulty", "?")
            ws = r["scores"].get("weighted_score", 0)
            flag_list = []
            for f, v in r["scores"].get("flags", {}).items():
                if v:
                    flag_list.append(f)
            flag_str = ", ".join(flag_list) if flag_list else "-"
            lines.append(f"| {qid} | {mod} | {diff} | {ws:.2f} | {flag_str} |")
        lines.append("")

    return "\n".join(lines)


def generate_comparison_report(models: List[Dict]) -> str:
    """Generate a multi-model comparison report."""
    lines = [
        "# SpaceOmicsBench v2 - Multi-Model Comparison Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Models compared:** {len(models)}",
        "",
        "---",
        "",
    ]

    model_names = []
    model_aggs = []
    for m in models:
        name = extract_model_name(m["data"])
        agg = compute_aggregates(m["data"].get("results", []))
        model_names.append(name)
        model_aggs.append(agg)

    # Sanitize model names for markdown table (pipe chars break table formatting)
    safe_names = [n.replace("|", "/") for n in model_names]

    # Overall comparison
    header = "| Dimension | " + " | ".join(safe_names) + " |"
    sep = "|-----------|" + "|".join(["-------"] * len(safe_names)) + "|"
    lines.extend(["## Overall Dimension Scores", "", header, sep])

    for dim in DIMENSION_WEIGHTS:
        row = f"| {dim.replace('_', ' ').title()} |"
        for agg in model_aggs:
            v = agg.get(f"overall_{dim}")
            row += f" {v:.2f} |" if v else " N/A |"
        lines.append(row)

    row = "| **Weighted** |"
    for agg in model_aggs:
        v = agg.get("overall_weighted")
        row += f" **{v:.2f}** |" if v else " N/A |"
    lines.append(row)
    lines.append("")

    # By difficulty comparison
    header = "| Difficulty | " + " | ".join(safe_names) + " |"
    sep = "|------------|" + "|".join(["-------"] * len(safe_names)) + "|"
    lines.extend(["## By Difficulty", "", header, sep])
    for diff in DIFFICULTY_ORDER:
        row = f"| {diff.title()} |"
        for agg in model_aggs:
            d = agg.get("by_difficulty", {}).get(diff)
            row += f" {d['mean']:.2f} (n={d['n']}) |" if d else " - |"
        lines.append(row)
    lines.append("")

    # By modality comparison
    header = "| Modality | " + " | ".join(safe_names) + " |"
    sep = "|----------|" + "|".join(["-------"] * len(safe_names)) + "|"
    lines.extend(["## By Modality", "", header, sep])
    for mod in MODALITY_ORDER:
        row = f"| {mod.replace('_', ' ').title()} |"
        for agg in model_aggs:
            d = agg.get("by_modality", {}).get(mod)
            row += f" {d['mean']:.2f} |" if d else " - |"
        lines.append(row)
    lines.append("")

    # Flag comparison
    header = "| Flag | " + " | ".join(safe_names) + " |"
    sep = "|------|" + "|".join(["-------"] * len(safe_names)) + "|"
    lines.extend(["## Quality Flags", "", header, sep])
    for flag in ["hallucination", "factual_error", "novel_insight", "harmful_recommendation", "exceeds_data_scope"]:
        row = f"| {flag.replace('_', ' ').title()} |"
        for agg in model_aggs:
            row += f" {agg.get('flags', {}).get(flag, 0)} |"
        lines.append(row)
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate SpaceOmicsBench LLM evaluation report")
    parser.add_argument("input_files", nargs="+", help="Scored results JSON file(s)")
    parser.add_argument("--output", default=None, help="Output markdown file")
    parser.add_argument("--compare", action="store_true",
                        help="Generate multi-model comparison (requires 2+ files)")

    args = parser.parse_args()

    if args.compare and len(args.input_files) >= 2:
        models = []
        for fpath in args.input_files:
            data = load_scored_file(fpath)
            models.append({"file": fpath, "data": data})
        report = generate_comparison_report(models)
        outfile = args.output or "comparison_report.md"
    else:
        data = load_scored_file(args.input_files[0])
        agg = compute_aggregates(data.get("results", []))
        report = generate_single_report(data, agg)
        outfile = args.output or f"report_{Path(args.input_files[0]).stem}.md"

    with open(outfile, "w") as f:
        f.write(report)

    print(f"Report saved to: {outfile}")
    print(f"({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
