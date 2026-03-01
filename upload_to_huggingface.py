#!/usr/bin/env python3
"""
Upload SpaceOmicsBench v2.1 to HuggingFace Hub as a Dataset.

Usage:
    python upload_to_huggingface.py --token hf_xxxxx
    python upload_to_huggingface.py  # uses HF_TOKEN env var or cached login
"""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
REPO_ID = "jang1563/SpaceOmicsBench"


DATASET_CARD = """\
---
license: cc-by-nc-4.0
task_categories:
  - tabular-classification
  - tabular-regression
  - question-answering
language:
  - en
tags:
  - biology
  - genomics
  - proteomics
  - metabolomics
  - metagenomics
  - spaceflight
  - benchmarking
  - multi-omics
  - astronaut
  - biomedical
pretty_name: SpaceOmicsBench
size_categories:
  - 1K<n<10K
---

# SpaceOmicsBench

A multi-omics AI benchmark for spaceflight biomedical data, featuring **21 ML tasks** across **9 modalities** and a **100-question LLM evaluation** framework.

Data sources: SpaceX Inspiration4 (I4) civilian astronaut mission, NASA Twins Study, and JAXA Cell-Free Epigenome (CFE) study. All benchmark tables are derived from OSDR public releases and/or published supplementary tables.

[![GitHub](https://img.shields.io/badge/GitHub-SpaceOmicsBench-181717?logo=github)](https://github.com/jang1563/SpaceOmicsBench)
[![LLM Leaderboard](https://img.shields.io/badge/LLM_Leaderboard-Interactive_Viz-a78bfa)](https://jang1563.github.io/SpaceOmicsBench/llm_leaderboard.html)

## Dataset Summary

| | |
|---|---|
| **ML Tasks** | 21 tasks (19 main + 2 supplementary) |
| **LLM Evaluation** | 100 questions, 5-dimension Claude-as-judge scoring, 9 models |
| **Modalities** | Clinical, cfRNA, Proteomics, Metabolomics, Spatial Transcriptomics, Microbiome, Multi-modal, Cross-tissue, Cross-mission |
| **Difficulty Tiers** | Calibration / Standard / Advanced / Frontier |
| **Missions** | Inspiration4, NASA Twins, JAXA CFE |
| **Evaluation Schemes** | LOCO, LOTO, 80/20 feature splits (5 reps) |
| **ML Baselines** | Random, Majority, LogReg, RF, MLP, XGBoost, LightGBM |

## Repository Structure

```
SpaceOmicsBench/
├── data/processed/        # Benchmark CSV tables (65+ files)
├── tasks/                 # ML task definitions (JSON)
├── splits/                # Train/test splits (JSON)
├── evaluation/llm/        # LLM question bank (100 questions)
├── results/v2.1/          # Scored LLM results (9 models, v2.1)
└── baselines/             # Baseline results (JSON)
```

## LLM Leaderboard (v2.1)

| Model | Overall (1-5) |
|-------|:---:|
| Claude Sonnet 4.6 | 4.62 |
| Claude Haiku 4.5 | 4.41 |
| DeepSeek-V3 | 4.34 |
| Claude Sonnet 4 | 4.03 |
| Gemini 2.5 Flash | 4.00 |
| GPT-4o Mini | 3.32 |
| Llama-3.3-70B (Groq) | 3.31 |
| Llama-3.3-70B (Together) | 3.31 |
| GPT-4o | 3.30 |

Judge: Claude Sonnet 4.6. See full breakdown at the [interactive leaderboard](https://jang1563.github.io/SpaceOmicsBench/llm_leaderboard.html).

## Citation

```bibtex
@misc{kim2025spaceomicsbench,
  title={SpaceOmicsBench: A Multi-Omics AI Benchmark for Spaceflight Biomedical Data},
  author={Kim, JangKeun},
  year={2025},
  url={https://github.com/jang1563/SpaceOmicsBench}
}
```

## License

- **Code** (scripts, evaluation framework, baselines): [MIT License](https://github.com/jang1563/SpaceOmicsBench/blob/main/LICENSE)
- **Benchmark data** (processed tables, task definitions, question bank, scored results): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) — free for academic/research use; commercial use requires a separate license.

Copyright (c) 2025 JangKeun Kim. For commercial licensing inquiries: silveray1563@gmail.com
"""


def get_upload_files():
    """Return list of (local_path, repo_path) tuples to upload."""
    files = []

    # data/processed/*.csv
    processed_dir = PROJECT_ROOT / "data" / "processed"
    for csv_file in sorted(processed_dir.glob("*.csv")):
        files.append((csv_file, f"data/processed/{csv_file.name}"))

    # tasks/*.json (top-level only, skip legacy/)
    tasks_dir = PROJECT_ROOT / "tasks"
    for task_file in sorted(tasks_dir.glob("*.json")):
        files.append((task_file, f"tasks/{task_file.name}"))

    # splits/*.json (top-level only, skip legacy/)
    splits_dir = PROJECT_ROOT / "splits"
    for split_file in sorted(splits_dir.glob("*.json")):
        files.append((split_file, f"splits/{split_file.name}"))

    # evaluation/llm/question_bank.json
    qb_path = PROJECT_ROOT / "evaluation" / "llm" / "question_bank.json"
    if qb_path.exists():
        files.append((qb_path, "evaluation/llm/question_bank.json"))

    # results/v2.1/*.json
    results_dir = PROJECT_ROOT / "results" / "v2.1"
    for result_file in sorted(results_dir.glob("*.json")):
        files.append((result_file, f"results/v2.1/{result_file.name}"))

    # baselines/baseline_results.json
    baseline_path = PROJECT_ROOT / "baselines" / "baseline_results.json"
    if baseline_path.exists():
        files.append((baseline_path, "baselines/baseline_results.json"))

    return files


def main():
    parser = argparse.ArgumentParser(description="Upload SpaceOmicsBench to HuggingFace Hub")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token (or set HF_TOKEN env var)")
    parser.add_argument("--repo-id", type=str, default=REPO_ID, help=f"HF repo ID (default: {REPO_ID})")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--dry-run", action="store_true", help="List files without uploading")
    args = parser.parse_args()

    # Resolve token
    token = args.token or os.environ.get("HF_TOKEN")

    try:
        from huggingface_hub import HfApi, login
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    # Collect files
    files = get_upload_files()
    total_size = sum(p.stat().st_size for p, _ in files)

    print(f"SpaceOmicsBench → {args.repo_id}")
    print(f"Files: {len(files)}, Total size: {total_size / 1024 / 1024:.1f} MB")
    print()

    if args.dry_run:
        print("DRY RUN — files that would be uploaded:")
        for local_path, repo_path in files:
            size_kb = local_path.stat().st_size / 1024
            print(f"  {repo_path:60s}  {size_kb:7.1f} KB")
        return

    # Login
    if token:
        login(token=token)
    else:
        print("No token provided. Trying cached login...")

    api = HfApi()

    # Verify auth
    try:
        user = api.whoami()
        print(f"Logged in as: {user['name']}")
    except Exception as e:
        print(f"ERROR: Not authenticated. Run with --token hf_xxx or set HF_TOKEN env var.")
        print(f"       Get token at: https://huggingface.co/settings/tokens")
        sys.exit(1)

    # Create repo if needed
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        print(f"Repository ready: https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"ERROR creating repo: {e}")
        sys.exit(1)

    # Upload dataset card
    print("\nUploading dataset card...")
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message="Add dataset card",
    )

    # Upload files in batches
    print(f"\nUploading {len(files)} files...")
    BATCH_SIZE = 20
    for i in range(0, len(files), BATCH_SIZE):
        batch = files[i : i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  Batch {batch_num}/{total_batches}: {batch[0][1]} ... {batch[-1][1]}")

        from huggingface_hub import CommitOperationAdd

        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))
            for local_path, repo_path in batch
        ]

        api.create_commit(
            repo_id=args.repo_id,
            repo_type="dataset",
            operations=operations,
            commit_message=f"Upload SpaceOmicsBench v2.1 files (batch {batch_num}/{total_batches})",
        )

    print(f"\nDone! Dataset uploaded to:")
    print(f"  https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
