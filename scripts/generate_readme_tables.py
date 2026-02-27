#!/usr/bin/env python3
"""
Generate README baseline tables from baselines/baseline_results.json.

Usage:
  python scripts/generate_readme_tables.py --print
  python scripts/generate_readme_tables.py --update-readme README.md
"""

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TASK_DIR = BASE_DIR / "tasks"
BASELINE_FILE = BASE_DIR / "baselines" / "baseline_results.json"

MAIN_TASKS = [
    "A1", "A2", "B1", "B2", "C1", "C2", "D1",
    "E1", "E4", "F1", "F2", "F3", "F4", "F5",
    "G1", "H1", "I1", "I2", "I3",
]

MODEL_KEYS = [
    ("random", "Random"),
    ("majority", "Majority"),
    ("logreg", "LogReg"),
    ("rf", "RF"),
    ("mlp", "MLP"),
    ("xgboost", "XGBoost"),
    ("lightgbm", "LightGBM"),
]


def _fmt(v):
    if v is None:
        return "—"
    return f"{v:.3f}"


def _best(values):
    best_idx = None
    best_val = None
    for i, v in enumerate(values):
        if v is None:
            continue
        if best_val is None or v > best_val:
            best_val = v
            best_idx = i
    return best_idx


def build_baseline_table():
    baselines = json.loads(BASELINE_FILE.read_text())
    rows = []
    for tid in MAIN_TASKS:
        task = json.loads((TASK_DIR / f"{tid}.json").read_text())
        tier = task.get("difficulty_tier", "unknown").title()
        metric = task.get("evaluation", {}).get("primary_metric", "metric")

        scores = []
        for key, _label in MODEL_KEYS:
            entry = baselines.get(tid, {}).get(key)
            if entry is None:
                scores.append(None)
            else:
                scores.append(entry.get(metric, {}).get("mean"))

        best_idx = _best(scores)
        formatted = []
        for i, v in enumerate(scores):
            if v is None:
                formatted.append("—")
            else:
                val = _fmt(v)
                if i == best_idx:
                    val = f"**{val}**"
                formatted.append(val)

        row = f"| {tid} | {tier} | {metric.upper()} | " + " | ".join(formatted) + " |"
        rows.append(row)

    header = "| Task | Tier | Metric | Random | Majority | LogReg | RF | MLP | XGBoost | LightGBM |"
    sep = "|------|------|--------|--------|----------|--------|----|-----|---------|----------|"
    return "\n".join([header, sep] + rows)


def build_composite_table():
    baselines = json.loads(BASELINE_FILE.read_text())
    composite = baselines.get("_composite", {})

    def top3(categories):
        pairs = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        return ", ".join(f"{k} ({v:.3f})" for k, v in pairs[:3])

    rows = []
    for key, label in [
        ("rf", "RF"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("logreg", "LogReg"),
        ("mlp", "MLP"),
    ]:
        entry = composite.get(key)
        if not entry:
            continue
        comp = entry.get("composite", 0.0)
        best = top3(entry.get("category_scores", {}))
        rows.append((label, comp, best))

    rows.sort(key=lambda x: x[1], reverse=True)

    header = "| Model | Composite | Best Categories |"
    sep = "|-------|-----------|-----------------|"
    lines = [header, sep]
    for i, (label, comp, best) in enumerate(rows):
        val = f"{comp:.3f}"
        if i == 0:
            val = f"**{val}**"
        lines.append(f"| {label} | {val} | {best} |")
    return "\n".join(lines)


def update_readme(path: Path):
    text = path.read_text()

    def replace_block(start, end, content):
        if start not in text or end not in text:
            raise ValueError(f"Missing markers: {start} / {end}")
        before, rest = text.split(start, 1)
        _old, after = rest.split(end, 1)
        return before + start + "\n\n" + content + "\n\n" + end + after

    new_text = text
    new_text = replace_block("<!-- BEGIN BASELINE_TABLE -->", "<!-- END BASELINE_TABLE -->", build_baseline_table())
    new_text = replace_block("<!-- BEGIN COMPOSITE_TABLE -->", "<!-- END COMPOSITE_TABLE -->", build_composite_table())
    path.write_text(new_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print", action="store_true", help="Print tables to stdout")
    parser.add_argument("--update-readme", type=str, help="Path to README.md to update in-place")
    args = parser.parse_args()

    baseline_table = build_baseline_table()
    composite_table = build_composite_table()

    if args.print or not args.update_readme:
        print("# Baseline Results Table\n")
        print(baseline_table)
        print("\n# Composite Scores Table\n")
        print(composite_table)

    if args.update_readme:
        update_readme(Path(args.update_readme))


if __name__ == "__main__":
    main()
